"""
batch.py -- the single batched resolved-screen read, backend-optimal on numpy (CPU) or torch (GPU),
with an explicit cheap/expensive cost split.

This is the single path for reading a stack of same-ordered-length frames
``X: (B, T, F)`` -- B screens, each ``T`` (ordered) x ``F`` (feature) -- through the library's
shared floor + whiten + null-provider contract, exact, and identical on both backends (numpy in ->
CPU; a torch tensor in -> its device, on-GPU, no host copy).

The resolved read is a spectral projector onto the states above the noise floor
---------------------------------------------------------------------------------
Reading a signal as a physical operator, ``C = screen^T screen`` (the feature Gram) has the noise
floor ``t = floor^2`` as a spectral edge (the Tracy-Widom / Marchenko-Pastur bulk edge -- a
mobility edge separating the noise bulk from the signal spikes).  The resolved sector is the
Fermi projector ``P = 1(C > t)`` onto the eigenstates above that edge, and every read is that one
operator: ``K_signal = tr(P)`` (the integrated density of states above the edge), ``energy[b,t] =
s_t P s_t^T`` (row t's energy inside the signal subspace), and the resolved ``projector`` is ``P``.
This is the same operator that counts electrons above a Fermi level -- the resolved read is the
Fermi projection of the signal operator at the noise edge.  There is no solver knob: the projector
is computed by the backend-optimal exact realisation, chosen automatically.

Two backends, one exact answer (byte-identical K_signal)
--------------------------------------------------------
  CPU (numpy / torch-CPU)   LAPACK SVD of the (N, Fe) screen -- fast + exact, bit-identical to
                            ``Projection``.
  GPU (torch-CUDA)          the Fermi projector via the matrix sign function (Newton-Schulz --
                            all cuBLAS matmuls), because every batched dense cuSOLVER factorisation
                            (SVD / eigh / LDL) is ~1-2 s on CUDA.  Iterated to convergence
                            (``sign^2 = I``): no tuning, a numerical tolerance, not a knob.
Both are exact, so ``K_signal`` (the integer resolved-mode count) and the resolved-mode set are
byte-identical across CPU and GPU.  The continuous ``energy`` agrees to floating-point precision
(cuBLAS and MKL use different reduction orders, so continuous values match to floating-point
precision across backends, not bit-for-bit).

Cost tiers (cheap is always available; expensive is on-demand + independent)
---------------------------------------------------------------------------
  Tier 0  (cheap, eager)      fold + whiten + floor -> K_signal, sigma_top, noise_floor.  The
                              survey gate; run on every frame.  No projector / no energy.
  Tier 1  (expensive)         per-row ``energy`` (and the ``projector`` when asked).  Computed only
                              for the frames that cleared the gate (pass ``subset=``); pure and
                              thread-safe, so the caller dispatches it to a worker thread / stream.

Fold groups run on a thread pool on CPU (numpy releases the GIL in LAPACK), overlapping the
per-width factorisations; the whole read is pure, so the caller can also run it off-thread.

The fold is a substrate choice, not a backend one (``fold="auto"`` default)
--------------------------------------------------------------------------
The feature-axis fold area-averages adjacent feature cells; it is valid only for a dense, smooth
continuum (frequency / wavelength / space).  For a sparse / choppy feature axis -- an unordered
learned basis (LLM KV ``head_dim``, embeddings) or a spike-dominated ordered signal -- averaging
destroys structure.

The decision is :func:`entropy.fold_width` -- PAPER Def 2.2, concentration AND continuity -- the
same call the per-frame ``geometry`` makes, so ``fold="auto"`` reads exactly what a per-frame
``Projection`` reads, on either backend.  It is one criterion in one place: a second locality
gate in front of it could only disagree with it, and did.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import environment as _env
from .entropy import MAD_SCALE
from .projection import fold_target_batch, normalize_batch, project_batch
from .null_providers import debias_denominator, screen_floor_sq, apply_floor

# Newton-Schulz matrix-sign convergence (the one numerical tolerance -- a convergence criterion,
# not a substrate knob).  ``sign^2 = I`` at convergence; ``_NS_TOL`` on max|Y^2 - I| is well below
# what could flip the rounded integer trace (the count), and ``_NS_MAX`` is a safety cap.
_NS_TOL = 1e-3
_NS_MAX = 100

# A singleton, bounded thread pool shared by every read, sized to the container's allocated cores.
# Concurrent calls submit their fold-group tasks here, so total worker threads stay <= cores no
# matter how many reads run in parallel -- no oversubscription / starvation (each call spawning its
# own core-count pool would collectively thrash).  Lazily created (double-checked lock).
_POOL = None
_POOL_LOCK = threading.Lock()


def _shared_pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(max_workers=max(1, _env.available_cpus()),
                                           thread_name_prefix="entroptics-batch")
    return _POOL


def _is_oom(e: Exception) -> bool:
    """A CUDA out-of-memory error (from concurrent GPU contention or a mis-estimate)."""
    try:
        import torch
        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(e).lower()


def _empty_cuda_cache():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class ResourceLimits:
    """A caller's resource envelope for a :func:`resolved_batch` read -- the default is all available
    resources; set any field to dial consumption back (shared box / multi-tenant / bounded VRAM).

    ``threads``    -- max CPU worker threads for the fold-group pool (``None`` = the container's
                      allocated cores, cgroup-aware, not the host count).
    ``memory_gb``  -- working-set cap.  ``None`` (default) = adaptive: re-measure the point-in-time
                      free memory of the target backend before each chunk and size it to what is
                      actually free then (free VRAM on GPU -- a hard wall; free RAM on CPU -- avoid
                      paging); more frees up -> later chunks grow, less -> they shrink.  A value pins
                      a fixed budget (guaranteed-consistent chunking, no starving under transient
                      pressure).  Chunking is output-transparent (byte-identical regardless of chunk
                      count) -- it varies only timing / peak memory, never results.
    ``gpu``        -- allow the GPU (``True`` default); ``False`` keeps the read on the CPU even when
                      a CUDA device (or CUDA-resident input) is present."""
    threads:   int | None = None
    memory_gb: float | None = None
    gpu:       bool = True

    @classmethod
    def coerce(cls, x) -> "ResourceLimits":
        if x is None:
            return cls()
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            return cls(**x)
        raise TypeError(f"limits must be a ResourceLimits, a dict, or None; got {type(x)}")



@dataclass
class ResolvedBatch:
    """One batched resolved-screen read (output of :func:`resolved_batch`).

    ``K_signal`` is a numpy ``(B,)`` int array on both backends and is byte-identical across CPU and
    GPU (an exact integer count).  ``sigma_top`` / ``noise_floor`` are ``(B,)`` on the input's
    backend; ``energy`` is ``(B, T)`` on the backend (agrees to fp precision across backends, not
    bit-identical -- different BLAS reduction orders).  ``projector`` (only when ``basis=True``) is
    the resolved spectral projector ``P = 1(C > floor^2)`` -- ``(B, F, F)`` when every frame shares
    one fold width, else a per-frame list."""
    K_signal:    "np.ndarray"          # (B,) int numpy -- byte-identical CPU/GPU (exact count)
    sigma_top:   "np.ndarray"          # (B,)      -- top singular value
    noise_floor: "np.ndarray"          # (B,)      -- the derived screen floor
    energy:      "np.ndarray" = None   # (B, T)    -- per-row resolved-subspace energy (Tier 1)
    projector:   "object" = None       # (B, F, F) or list -- the resolved Fermi projector (Tier 1)


# ══════════════════════════════════════════════════════════════════════════════
# The per-group compute core (one uniform folded screen -> the read)
# ══════════════════════════════════════════════════════════════════════════════

def _floor_batch(xp, screen, N, Fe, far, null, seed):
    """The screen noise floor for a uniform ``(Bg, N, Fe)`` folded screen -- the derived ``mp``
    edge batched via the shared primitives (:func:`debias_denominator` / :func:`screen_floor_sq`),
    or a caller ``null`` provider applied per frame.  Returns ``(Bg,)`` on ``xp``."""
    if null is None:
        row_energy = _env.sum_ax(xp, xp.abs(screen) ** 2, 2)                 # (Bg, N)
        sigma2 = _env.median_ax(xp, row_energy, 1) / debias_denominator(N, Fe) + 1e-30
        return xp.sqrt(screen_floor_sq(sigma2, N, Fe, far))                  # (Bg,)
    # a caller-suppliable provider is the occasional path -> per frame, in numpy (needs the spectrum)
    scr = np.asarray(_env.to_numpy(screen))
    sv = np.linalg.svd(scr, compute_uv=False)
    fl = np.array([apply_floor(null, spectrum=sv[j], data=scr[j], shape=(N, Fe),
                               far=far, kind="projection", seed=seed) for j in range(int(scr.shape[0]))])
    return _env.asdtype_of(screen, fl.astype(float))


def _resolve_group(xp, screen, *, far, null, seed, want_energy, want_projector):
    """Read one uniform folded screen ``(Bg, N, Fe)`` -- the resolved spectral projector above the
    floor.  Auto-dispatched to the backend-optimal exact realisation (no solver knob): LAPACK SVD
    off-CUDA (bit-identical to ``Projection``), the Fermi matrix-sign projector on CUDA (all cuBLAS
    matmuls, since batched dense cuSOLVER is ~1-2 s).  Both give the same exact ``K_signal``."""
    N, Fe = int(screen.shape[1]), int(screen.shape[2])
    floor = _floor_batch(xp, screen, N, Fe, far, null, seed)
    if _env.is_cuda(screen):
        return _fermi_resolve(xp, screen, floor, want_energy=want_energy, want_projector=want_projector)
    return _svd_resolve(xp, screen, floor, want_energy=want_energy, want_projector=want_projector)


def _svd_resolve(xp, screen, floor, *, want_energy, want_projector):
    """Exact resolved read via LAPACK SVD of the ``(Bg, N, Fe)`` screen (CPU / off-CUDA).  Tier 0
    uses ``svdvals`` (bit-identical to ``Projection``); the projector/energy tier the full SVD."""
    expensive = bool(want_energy or want_projector)
    if not expensive:
        S = xp.linalg.svdvals(screen)
        keep = (S > floor[:, None])
        K = _env.sum_ax(xp, keep.astype(S.dtype) if not _env.is_torch(xp) else keep.to(S.dtype), 1)
        return dict(K=K, floor=floor, sigma_top=S[:, 0], energy=None, projector=None)
    U, S, Vh = xp.linalg.svd(screen, full_matrices=False)                  # (Bg,N,R)(Bg,R)(Bg,R,Fe)
    keepf = (S > floor[:, None])
    keep = keepf.to(S.dtype) if _env.is_torch(xp) else keepf.astype(S.dtype)   # (Bg, R)
    K = _env.sum_ax(xp, keep, 1)
    energy = _env.sum_ax(xp, (U ** 2) * (keep * S ** 2)[:, None, :], 2) if want_energy else None
    projector = None
    if want_projector:
        V = _env.movedim(xp, Vh, 1, 2)                                     # (Bg, Fe, R)
        Vk = V * keep[:, None, :]                                          # zero non-signal columns
        projector = Vk @ _env.movedim(xp, V, 1, 2)                         # P = sum_signal v v^T
    return dict(K=K, floor=floor, sigma_top=S[:, 0], energy=energy, projector=projector)


def _spectral_norm(xp, A, *, iters=6):
    """Top ``|eigenvalue|`` of a batch of symmetric ``(Bg, F, F)`` matrices via power iteration
    (matmul-only) -- an upper-bound scale for the sign iteration and the ``sigma_top`` read."""
    Bg, F = int(A.shape[0]), int(A.shape[-1])
    v = _env.randn_like(xp, (Bg, F, 1), ref=A, seed=0)
    v = v / _env.clampmin(xp, _env.vnorm_ax(xp, v, 1), 1e-30)
    for _ in range(int(iters)):
        v = A @ v
        v = v / _env.clampmin(xp, _env.vnorm_ax(xp, v, 1), 1e-30)
    return _env.vnorm_ax(xp, A @ v, 1)[:, 0, 0]                            # (Bg,)


def _fermi_resolve(xp, screen, floor, *, want_energy, want_projector):
    """Exact resolved read via the Fermi projector ``P = 1(C > floor^2)`` (CUDA) -- the matrix sign
    function by Newton-Schulz, all cuBLAS matmuls (batched dense cuSOLVER is ~1-2 s on CUDA).

    ``C = screen^T screen`` (Gram); shift to the edge ``A = C - floor^2 I`` (signal eigenvalues > 0,
    noise < 0); scale into the sign-iteration's convergence disc and iterate ``Y <- (3Y - Y^3)/2``
    to ``sign(A)`` (converged when ``sign^2 = I``).  Then ``K_signal = tr((I+sign)/2)`` -- the
    integrated density of states above the edge, exact (equals the CPU SVD count byte-for-byte),
    ``energy[b,t] = s_t P s_t^T``, and ``projector = P``.  No knob: the iteration runs to a
    numerical convergence tolerance."""
    F = int(screen.shape[2])
    C = _env.movedim(xp, screen, 1, 2) @ screen                            # (Bg, F, F) Gram
    K, sign = _fermi_sign_from_gram(xp, C, floor ** 2)
    sigma_top = xp.sqrt(_env.clampmin(xp, _spectral_norm(xp, C), 0.0))
    energy = projector = None
    if want_energy or want_projector:
        P = 0.5 * (_env.eye(xp, F, ref=screen) + sign)                    # the Fermi projector
        if want_energy:
            energy = _env.sum_ax(xp, (screen @ P) * screen, 2)           # s_t P s_t^T per row
        if want_projector:
            projector = P
    return dict(K=K, floor=floor, sigma_top=sigma_top, energy=energy, projector=projector)


def _fermi_sign_from_gram(xp, C, floor2):
    """The Fermi-projector core, from a Gram ``C`` ``(B, F, F)`` and per-screen squared floor
    ``floor2`` ``(B,)``: the matrix sign of ``A = C - floor2*I`` by Newton-Schulz (all cuBLAS
    matmuls), iterated to ``sign^2 = I``.  Returns ``(K (B,), sign (B, F, F))``, with
    ``K = tr((I+sign)/2)`` the exact resolved-mode count (integrated density of states above the
    edge).  Shared by :func:`_fermi_resolve` (forms ``C`` from a screen) and
    :class:`ResolvedScreenBatch` (accumulates ``C`` over a stream)."""
    F = int(C.shape[-1])
    I = _env.eye(xp, F, ref=C)
    A = C - floor2[:, None, None] * I                                     # shift to the spectral edge
    sn = 1.1 * _spectral_norm(xp, A)
    alpha = sn.clamp(min=1e-30) if _env.is_torch(xp) else np.clip(sn, 1e-30, None)
    Y = A / alpha[:, None, None]                                          # eigenvalues into (-1, 1)
    for _ in range(_NS_MAX):                                              # Newton-Schulz -> sign(A)
        Y2 = Y @ Y
        if _env.amax_abs(xp, Y2 - I) < _NS_TOL:                          # sign^2 = I at convergence
            break
        Y = 1.5 * Y - 0.5 * (Y2 @ Y)
    tr = _env.sum_ax(xp, _env.diagonal(xp, Y), 1)                        # tr(sign) = 2K - F
    K = _env.round_int(xp, 0.5 * (F + tr))                               # (B,) exact integer count
    return K, Y


# ══════════════════════════════════════════════════════════════════════════════
# CPU vs GPU: the discriminators, and an (optional) auto-recommendation
# ══════════════════════════════════════════════════════════════════════════════
# The library dispatches on where the tensor already lives (numpy -> CPU SVD; torch-CUDA -> the
# Fermi matrix-sign projector).  Both are exact; the choice is purely cost, driven by three axes:
#
#   Fe  (resolvable detail = folded feature width)  -- cubic.  GPU work ~ niter * B * Fe^3 (matmul);
#        CPU work ~ B * Fe^2 * min(N, Fe) (SVD).  The dominant term; the fold reduces it.
#   B   (batch = number of screens)                 -- the GPU parallelism axis.  GPU does the whole
#        batch as one fused matmul; CPU loops (LAPACK + a thread pool ~ core count).  GPU's edge
#        grows with B.
#   T   (context window = ordered length)           -- linear and cheap.  It is reduced into the
#        Fe x Fe Gram before the expensive part, so a long context is nearly free and does not drive
#        the choice; the VRAM wall is B * Fe^2 (batch x detail^2), not T.
#
# Plus a H2D transfer (B*T*F elements) if the data starts on CPU.  Rule of thumb: GPU when the data
# already lives there and Fe is small with B large (the LLM-KV regime); CPU otherwise (small B, a
# one-off read, data on CPU, or very large Fe).


def _chunk_rows(budget_gb, T: int, F: int, bytes_per: int, want_proj: bool) -> int | None:
    """Rows per chunk so the working set fits ``budget_gb`` GB (``None`` = no chunking).  Footprint
    per frame ~ input ``T*F`` + folded screen ``T*F`` + Gram ``F*F`` (+ projector ``F*F``), with 3x
    headroom for transient matmul buffers."""
    if not budget_gb:
        return None
    per_frame = (2 * int(T) * int(F) + (2 if want_proj else 1) * int(F) * int(F)) * int(bytes_per) * 3
    return max(1, int(float(budget_gb) * 0.8 * 1e9 / max(per_frame, 1)))


def _targets_gpu(X, device, lim, B, T, F) -> bool:
    """Whether this read will land on CUDA (to size the memory budget for the right backend)."""
    if not lim.gpu:
        return False
    if _env.is_cuda(X):
        return True
    if device in ("cuda", "gpu"):
        return True
    if device == "auto":
        try:
            import torch
            if not torch.cuda.is_available():
                return False
        except Exception:
            return False
        return recommend_backend(B, T, F, data_on_gpu=False)[0] == "gpu"
    return False                                          # "cpu", None (passive numpy), or a CPU device


def recommend_backend(B: int, T: int, F: int, *, data_on_gpu: bool = False,
                      cuda_available: bool | None = None) -> tuple[str, str]:
    """Recommend ``"cpu"`` or ``"gpu"`` for a :func:`resolved_batch` read of a ``(B, T, F)`` stack
    (``F`` the folded feature width), with a one-line reason -- guidance for placing the tensor
    (``resolved_batch`` itself dispatches on where the data already lives).  From the cost axes
    above; no fitted constants -- it fires GPU only in the regime where the cubic ``Fe`` is small
    and the parallel batch ``B`` is large enough to beat the H2D transfer + launch overhead."""
    if data_on_gpu:
        return "gpu", "data already resident on GPU (no host copy) -> GPU"
    if cuda_available is None:
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    if not cuda_available:
        return "cpu", "no CUDA device -> CPU (exact LAPACK SVD)"
    # CPU-resident data: offloading pays only when the batched matmul win beats the H2D transfer.
    # GPU shines at small folded Fe (cheap cubic) with a large batch B; large Fe makes Fe^3 bite.
    if int(F) <= 256 and int(B) >= 64:
        return "gpu", f"small Fe={int(F)} (cheap cubic) x large B={int(B)} (parallel) -> GPU"
    if int(F) >= 1024:
        return "cpu", f"large Fe={int(F)} (Fe^3 dominates) -> CPU"
    if int(B) < 16:
        return "cpu", f"small batch B={int(B)} (GPU launch/iteration not amortized) -> CPU"
    return "cpu", "moderate size + CPU-resident data (H2D transfer not repaid) -> CPU"


# ══════════════════════════════════════════════════════════════════════════════
# The public batched read (groups by fold width, reassembles in input order)
# ══════════════════════════════════════════════════════════════════════════════

def _concat_batches(parts: list, want_energy: bool, want_proj: bool) -> "ResolvedBatch":
    """Concatenate chunked :class:`ResolvedBatch` results back into one (memory-bounded read)."""
    xp = _env.ns(parts[0].sigma_top)
    K = np.concatenate([np.asarray(p.K_signal) for p in parts])
    sig = _env.cat0(xp, [p.sigma_top for p in parts])
    fl = _env.cat0(xp, [p.noise_floor for p in parts])
    en = _env.cat0(xp, [p.energy for p in parts]) if want_energy else None
    proj = None
    if want_proj:
        proj = []
        for p in parts:
            pj = p.projector
            proj.extend(pj if isinstance(pj, list) else [pj[j] for j in range(int(pj.shape[0]))])
        if proj and all(hasattr(a, "shape") and a.shape == proj[0].shape for a in proj):
            proj = _env.stack(xp, proj, 0)
    return ResolvedBatch(K_signal=K, sigma_top=sig, noise_floor=fl, energy=en, projector=proj)


def resolved_batch(X, *, fold="auto", far: float = 0.05, null=None, seed: int = 0,
                   energy: bool = False, basis: bool = False, subset=None,
                   device="auto", limits=None, _allow_chunk: bool = True) -> ResolvedBatch:
    """Read a stack ``X: (B, T, F)`` onto the resolved screen -- the one batched path.

    Tier 0 (always): ``K_signal``, ``sigma_top``, ``noise_floor``.  Tier 1 (only when ``energy``
    and/or ``basis`` is set): the per-row resolved-subspace ``energy`` and the resolved spectral
    ``projector``.  Pass ``subset`` (indices) to run Tier 1 on only the frames that cleared the gate
    -- boring frames never pay for the projection.

    The resolved read is the Fermi projector onto the states above the noise floor, computed by the
    backend-optimal exact realisation (no solver knob): LAPACK SVD off-CUDA (bit-identical to
    ``Projection``), the matrix-sign projector on CUDA.  ``K_signal`` is byte-identical across CPU and
    GPU (exact integer count); ``energy`` agrees to fp precision.

    ``fold`` (feature-axis resolution mode): ``"auto"`` (default) and ``True`` both apply the
    entropy-matched fold decision, :func:`entropy.fold_width` per frame -- concentration AND
    continuity, PAPER Def 2.2 -- which folds a dense/smooth concentrated continuum and leaves a
    sparse/choppy or unordered basis (KV head_dim, embeddings) native of its own accord.  They are
    one read, not two: the decision is per frame either way, so there is nothing for ``True`` to
    force that ``"auto"`` has not already asked.  ``False`` pins native resolution.
    ``null``: the noise-floor provider (``None`` = derived ``mp``).  numpy in -> CPU (fold
    groups run on a thread pool); a torch tensor in -> its device (pass ``device=`` to move a numpy
    stack onto a GPU once).  ``device`` selects where the read runs and defaults to ``"auto"`` -- the
    engine optimises placement every call from its environment + the signal detail
    (:func:`recommend_backend`): GPU-resident data stays put (locality); a CPU stack is offloaded to
    CUDA only when the cost model favours it (small folded Fe x large B), and the outputs are moved
    back to numpy so a numpy call returns numpy.  Pass ``None`` for the passive "run where the data
    lives" behaviour, or ``"cpu"`` / ``"cuda"`` / a torch device to force it.

    ``limits`` is the caller's resource envelope (:class:`ResourceLimits` or a dict: ``threads`` /
    ``memory_gb`` / ``gpu``); the default uses all available resources.  ``memory_gb`` processes the
    batch in memory-bounded chunks; ``gpu=False`` keeps the read on the CPU; ``threads`` caps the
    fold-group pool (default = the container's allocated cores, not the host's).  Deterministic."""
    lim = ResourceLimits.coerce(limits)
    if subset is not None:
        X = X[list(subset)]
    if len(getattr(X, "shape", ())) != 3:
        raise ValueError(f"X must be (B, T, F); got shape {getattr(X, 'shape', None)}")
    B0, T0, F0 = (int(v) for v in X.shape)
    bytes_per = 4 if _env.precision() == 32 else 8
    # Memory-bounded, adaptive chunking (never a stale global budget).  Before each chunk, the
    # point-in-time free memory of the target backend is re-measured and the next chunk is sized to
    # current availability -- if memory frees up mid-batch later chunks grow, if it shrinks they
    # shrink, so each chunk uses what is actually free (free VRAM on GPU is a hard wall; free RAM on
    # CPU avoids paging).  Chunking is output-transparent (byte-identical regardless of chunk count),
    # so this varies only timing / peak memory, never results.  An explicit ``memory_gb`` pins a fixed
    # budget (guaranteed-consistent chunking; no starving under transient pressure) and always wins.
    if _allow_chunk:
        on_gpu = _targets_gpu(X, device, lim, B0, T0, F0)

        def _budget():
            return lim.memory_gb if lim.memory_gb is not None else _env.available_memory_gb(on_gpu=on_gpu)

        b0 = _chunk_rows(_budget(), T0, F0, bytes_per, basis)
        if on_gpu or (b0 and b0 < B0):    # GPU: always loop so OOM-backoff protects; CPU: only if bounded
            parts, i = [], 0
            while i < B0:
                b = _chunk_rows(_budget(), T0, F0, bytes_per, basis) or (B0 - i)   # re-measure per chunk
                b = max(1, min(int(b), B0 - i))
                while True:                                                        # OOM-backoff (contention)
                    try:
                        parts.append(resolved_batch(X[i:i + b], fold=fold, far=far, null=null, seed=seed,
                                                    energy=energy, basis=basis, device=device,
                                                    limits=lim, _allow_chunk=False))
                        break
                    except Exception as e:
                        if b > 1 and _is_oom(e):
                            b = max(1, b // 2); _empty_cuda_cache(); continue   # halve + retry
                        raise
                i += b
            return _concat_batches(parts, energy, basis)

    xp = _env.ns(X)
    in_numpy = not _env.is_torch(xp)
    Xt = X
    moved = False                                         # offloaded a numpy stack onto CUDA
    if not lim.gpu:                                       # resource envelope forbids the GPU
        if _env.is_cuda(X):
            Xt = _env.to_numpy(X); xp = np; in_numpy = True
    elif device == "auto":
        if not _env.is_cuda(X):
            s = tuple(int(v) for v in X.shape)
            if recommend_backend(s[0], s[1], s[2], data_on_gpu=False)[0] == "gpu":
                import torch
                Xt = torch.as_tensor(np.asarray(_env.to_numpy(X))).cuda()   # offload CPU -> GPU
                xp = torch; moved = True
    elif device in ("cuda", "gpu"):
        import torch
        Xt = torch.as_tensor(np.asarray(_env.to_numpy(X))).cuda() if not _env.is_cuda(X) else X
        xp = torch; moved = not _env.is_cuda(X) and in_numpy
    elif device == "cpu":
        Xt = _env.to_numpy(X); xp = np
    elif device is not None and _env.is_torch(xp):
        Xt = Xt.to(device)
    Xt = _env.asnum(Xt)
    B, T, F = int(Xt.shape[0]), int(Xt.shape[1]), int(Xt.shape[2])

    data = normalize_batch(xp, Xt)                                        # (B, T, F) whitened
    if fold is False:
        F_eff = np.full(B, F, dtype=int)                                   # forced native resolution
    else:
        # "auto" and True are the same read: `fold_width` (PAPER Def 2.2) already decides per
        # frame, returning F untouched where the frame must not fold.  This is the call the
        # per-frame `geometry` makes, which is what keeps the batch identical to `Projection`.
        F_eff = fold_target_batch(xp, Xt, far=far)

    # outputs in input order
    K_signal = np.zeros(B, dtype=np.int64)
    sigma_top = _env.zeros(xp, (B,), ref=(Xt if _env.is_torch(xp) else None))
    noise_floor = _env.zeros(xp, (B,), ref=(Xt if _env.is_torch(xp) else None))
    energy_out = _env.zeros(xp, (B, T), ref=(Xt if _env.is_torch(xp) else None)) if energy else None
    proj_list: list = [None] * B

    widths = [int(w) for w in np.unique(F_eff)]

    def _group(Fe):
        sel = np.where(F_eff == Fe)[0]
        screen = project_batch(xp, data[list(sel)], Fe)                   # (Bg, T, Fe)
        r = _resolve_group(xp, screen, far=far, null=null, seed=seed,
                           want_energy=energy, want_projector=basis)
        return sel, r

    # Fold groups are independent -> run them on a thread pool on CPU (numpy releases the GIL in
    # LAPACK, so the per-width factorisations overlap across cores).  On CUDA they share the device
    # stream, so run sequentially.  Reassembly is done single-threaded (no races on the outputs).
    max_threads = lim.threads if lim.threads is not None else _env.available_cpus()
    if not _env.is_torch(xp) and len(widths) > 1 and max_threads > 1:
        results = list(_shared_pool().map(_group, widths))   # singleton bounded pool (no oversubscription)
    else:
        results = [_group(Fe) for Fe in widths]

    for sel, r in results:
        K_signal[sel] = np.asarray(_env.to_numpy(r["K"])).astype(np.int64)
        _scatter(xp, sigma_top, sel, r["sigma_top"])
        _scatter(xp, noise_floor, sel, r["floor"])
        if energy:
            _scatter(xp, energy_out, sel, r["energy"])
        if basis:
            for j, b in enumerate(sel):
                proj_list[b] = r["projector"][j]

    projector = _stack_or_list(xp, proj_list, len(widths) == 1) if basis else None

    if moved and in_numpy:                               # offloaded from numpy -> return numpy
        sigma_top = _env.to_numpy(sigma_top)
        noise_floor = _env.to_numpy(noise_floor)
        if energy_out is not None:
            energy_out = _env.to_numpy(energy_out)
        if projector is not None:
            projector = ([_env.to_numpy(p) for p in projector] if isinstance(projector, list)
                         else _env.to_numpy(projector))
    return ResolvedBatch(K_signal=K_signal, sigma_top=sigma_top, noise_floor=noise_floor,
                         energy=energy_out, projector=projector)


# ══════════════════════════════════════════════════════════════════════════════
# Stateful resolved screen -- the revisited-screen sibling (LLM KV across turns)
# ══════════════════════════════════════════════════════════════════════════════
# The batch functions above are pure (no memoization -- required for the "expensive is
# independent" rule and for cross-caller privacy).  A revisited screen (an LLM KV head read on
# many turns, tokens appended each turn) instead keeps state: the feature Gram ``C = S^T S`` is
# additive, so appended rows are a rank-k update, and the eigenbasis need only refresh every few
# appends.  ``ResolvedScreen`` is that stateful object -- caller-owned (no global registry): the
# caller keys ``{(session, layer, head) -> ResolvedScreen}`` and owns eviction / privacy, so there
# is no cross-caller overlap by construction (ownership, not keying, is the isolation guarantee).

def _frozen_whiten(xp, X):
    """Per-channel robust MAD whiten stats (median, shrunk MAD) frozen from a warmup block ``X``
    ``(N, F)`` -- the same math as ``entropy.normalize``, kept so a growing stream is whitened by a
    stable per-channel scale (the median/MAD of a growing axis is not incrementally updatable)."""
    from .entropy import mad_stats
    med, mad_eff, _ = mad_stats(xp, _env.asnum(X))
    return med, xp.where(mad_eff > 0.0, mad_eff, xp.ones_like(mad_eff))


class ResolvedScreen:
    """A stateful, resumable resolved screen for a revisited screen (e.g. an LLM KV attention head
    across turns): append rows (tokens) incrementally, refresh the resolved basis lazily, and read
    ``K_signal`` / per-row ``energy`` without recomputing from scratch.

    The feature Gram ``C = S^T S`` (whitened rows) is accumulated by rank-k updates; the eigenbasis
    is refreshed every ``refresh_every`` appends (or on demand) -- the only expensive step, amortised
    -- and the stale basis is reused for ``energy`` in between.  Whitening uses stats frozen from the
    first ``warmup`` rows (a growing axis' median/MAD is not additive), so the read converges to the
    batch :func:`resolved_batch` for a stationary stream; pass ``whiten=False`` if rows are already
    normalised.  Native feature resolution only (``fold=False``:
    the revisit case is unordered bases -- KV head_dim / embeddings).

    Caller-owned: hold one per screen and key/evict them yourself; the library keeps no global state.
    ``forgetting`` in (0,1] fades old rows (both the Gram and the floor's row-energy window) for a
    non-stationary stream.  ``state()`` / :meth:`from_state` resume across sessions."""

    def __init__(self, F, *, far: float = 0.05, null=None, seed: int = 0,
                 refresh_every: int = 32, warmup: int = 64, whiten: bool = True,
                 forgetting: float = 1.0):
        self.F = int(F)
        self.far = float(far); self.null = null; self.seed = int(seed)
        self.refresh_every = int(refresh_every); self.warmup = int(warmup)
        self.whiten = bool(whiten); self.forgetting = float(forgetting)
        self.T = 0                       # rows accumulated
        self._C = None                   # (F, F) whitened feature Gram
        self._rowen = []                 # per-row whitened energies (for the screen-floor sigma2)
        self._warm = []                  # warmup rows buffer (until whiten stats freeze)
        self._med = self._mad = None     # frozen whiten stats
        self._V = self._eval = self._keep = None   # cached resolved basis / eigenvalues / keep-mask
        self._since = 0                  # appends since the last basis refresh
        self._lock = threading.RLock()   # guard concurrent update/refresh/read on one instance

    # ── whiten (frozen stats) ─────────────────────────────────────────────────
    def _ensure_stats(self, xp, rows):
        if not self.whiten or self._med is not None:
            return
        self._warm.append(rows)
        if sum(int(r.shape[0]) for r in self._warm) >= self.warmup:
            X = _env.cat0(xp, self._warm)
            self._med, self._mad = _frozen_whiten(xp, X)
            self._warm = []

    def _apply_whiten(self, xp, rows):
        if not self.whiten or self._med is None:
            return rows
        return (rows - self._med[None, :]) / self._mad[None, :]

    # ── streaming update ──────────────────────────────────────────────────────
    def update(self, rows):
        """Append ``rows`` ``(k, F)`` (new tokens) -- a rank-k Gram update; refreshes the basis
        every ``refresh_every`` appends.  Backend-agnostic (torch rows stay on-device).  Thread-safe:
        an internal lock serialises concurrent ``update`` / ``refresh`` / reads on one instance, so
        two parallel appends cannot clobber the Gram (the batch functions are separately pure)."""
        xp = _env.ns(rows)
        R = _env.asnum(rows)
        if R.ndim == 1:
            R = R[None, :]
        with self._lock:
            self._ensure_stats(xp, R)
            Sw = self._apply_whiten(xp, R)
            G = Sw.T @ Sw                                        # (F, F) rank-k Gram update
            lam = self.forgetting
            self._C = G if self._C is None else (lam * self._C + G)
            re = _env.sum_ax(xp, Sw ** 2, 1)                      # (k,) per-row energies
            self._rowen.append(re)
            self.T = int(round(lam * self.T)) + int(R.shape[0])
            self._since += int(R.shape[0])
            if self._since >= self.refresh_every:
                self.refresh()
        return self

    def refresh(self):
        """Recompute the resolved basis from the current Gram (the amortised expensive step)."""
        with self._lock:
            if self._C is None:
                return self
            xp = _env.ns(self._C)
            w, V = xp.linalg.eigh(self._C)
            w = _env.flip2(xp, w); V = _env.flip2(xp, V)         # descending
            self._eval = _env.clampmin(xp, w, 0.0)
            self._V = V
            floor_sq = self._floor_sq(xp)
            self._keep = (self._eval > floor_sq)
            self._since = 0
        return self

    def _floor_sq(self, xp):
        """The screen noise floor SQUARED (correlation units, to threshold the eigenvalues).

        ``null=None`` is the derived ``mp`` screen edge off the tracked row energies.  A caller
        provider is evaluated through :func:`apply_floor` on the cut point it thresholds, then
        squared -- a provider scores in singular-value units, the Gram is read in their squares.
        The revisited screen keeps no rows, so the provider is handed the resolved spectrum
        (``sqrt`` of the eigenvalues) as its sample, which is what it thresholds against."""
        re = _env.cat0(xp, self._rowen) if self._rowen else _env.zeros(xp, (1,), ref=self._C)
        if self.null is None:
            sigma2 = _env.median1d(xp, re) / debias_denominator(self.T, self.F) + 1e-30
            return screen_floor_sq(sigma2, self.T, self.F, self.far)
        spectrum = np.sqrt(np.asarray(_env.to_numpy(_env.clampmin(xp, self._eval, 0.0))))
        fl = apply_floor(self.null, spectrum=spectrum, data=None,
                         shape=(self.T, self.F), far=self.far, kind="projection", seed=self.seed)
        return _env.asdtype_of(self._C, np.asarray(float(fl) ** 2))

    # ── reads ─────────────────────────────────────────────────────────────────
    def _fresh(self):
        with self._lock:
            if self._V is None:
                self.refresh()

    @property
    def K_signal(self) -> int:
        """Modes above the noise floor on the accumulated screen (refreshes the basis if stale)."""
        self._fresh()
        if self._keep is None:
            return 0
        return int(_env.sum_ax(_env.ns(self._keep),
                               self._keep.to(self._eval.dtype) if _env.is_torch(_env.ns(self._keep))
                               else self._keep.astype(float)))

    def energy(self, rows):
        """Per-row resolved-subspace energy of ``rows`` ``(k, F)`` against the current basis --
        ``energy[t] = sum_{k<K} (S.v_k)^2[t]`` -- reusing the (lazily-refreshed) basis, no resolve."""
        self._fresh()
        xp = _env.ns(rows)
        R = _env.asnum(rows)
        if R.ndim == 1:
            R = R[None, :]
        Sw = self._apply_whiten(xp, R)
        keep = (self._keep.to(Sw.dtype) if _env.is_torch(xp) else self._keep.astype(Sw.dtype))
        P = Sw @ self._V                                          # (k, F) projections S.v
        return _env.sum_ax(xp, (P ** 2) * keep[None, :], 1)       # (k,)

    def state(self) -> dict:
        """Export the full state (resume/splice across sessions)."""
        return dict(F=self.F, far=self.far, seed=self.seed, refresh_every=self.refresh_every,
                    warmup=self.warmup, whiten=self.whiten, forgetting=self.forgetting,
                    T=self.T, C=_env.to_numpy(self._C) if self._C is not None else None,
                    rowen=[_env.to_numpy(r) for r in self._rowen],
                    med=_env.to_numpy(self._med) if self._med is not None else None,
                    mad=_env.to_numpy(self._mad) if self._mad is not None else None)

    @classmethod
    def from_state(cls, s: dict) -> "ResolvedScreen":
        """Resume a :class:`ResolvedScreen` from :meth:`state` (numpy; move to a device by feeding
        the next ``update`` a torch tensor)."""
        obj = cls(s["F"], far=s["far"], seed=s["seed"], refresh_every=s["refresh_every"],
                  warmup=s["warmup"], whiten=s["whiten"], forgetting=s["forgetting"])
        obj.T = int(s["T"])
        obj._C = None if s["C"] is None else np.asarray(s["C"])
        obj._rowen = [np.asarray(r) for r in s["rowen"]]
        obj._med = None if s["med"] is None else np.asarray(s["med"])
        obj._mad = None if s["mad"] is None else np.asarray(s["mad"])
        return obj


def _cat_tokens(xp, parts):
    """Concatenate ``(B, k_i, ...)`` blocks along the token axis (1)."""
    return xp.cat(parts, dim=1) if _env.is_torch(xp) else np.concatenate(parts, axis=1)


def _frozen_whiten_batch(xp, X):
    """Per-screen robust MAD whiten stats (median, shrunk MAD) frozen from a warmup block ``X``
    ``(B, W, F)`` -- the batched, per-screen form of :func:`_frozen_whiten` (James-Stein shrink of
    each screen's per-channel MAD toward its pooled scale)."""
    from .entropy import MAD_LOGVAR, resolution_floor
    data = _env.asnum(X)
    med = _env.median_ax(xp, data, 1)                                   # (B, F)
    mad = _env.median_ax(xp, xp.abs(data - med[:, None, :]), 1) * MAD_SCALE   # (B, F)
    N = int(data.shape[1])
    typ = _env.median_ax(xp, mad, 1)                                    # (B,) per-screen pooled scale
    pos = mad > resolution_floor(xp, typ, mad)[:, None]                 # (B, F) -- see mad_stats
    lm = xp.log(xp.where(pos, mad, xp.ones_like(mad)))
    lm0 = xp.log(xp.where(typ > 0.0, typ, xp.ones_like(typ)))
    V = _env.std_ax(xp, lm, 1) ** 2                                     # (B,)
    ratio = (MAD_LOGVAR / max(N, 1)) / _env.clampmin(xp, V, 1e-300)
    w = _env.cliprange(xp, 1.0 - ratio, 0.0, 1.0)
    w = xp.where(V > 0.0, w, xp.zeros_like(w))
    mad_eff = xp.where(pos, xp.exp(lm0[:, None] + w[:, None] * (lm - lm0[:, None])),
                       xp.zeros_like(mad))                             # (B, F)
    return med, xp.where(mad_eff > 0.0, mad_eff, xp.ones_like(mad_eff))


class ResolvedScreenBatch:
    """The scalable stateful resolved screen -- ``B`` revisited screens (e.g. every layer x head of an
    LLM KV cache -- often thousands) held as one ``(B, F, F)`` Gram with a batched lazy refresh and
    batched ``K_signal`` / ``energy`` reads.  Thousands of individual :class:`ResolvedScreen` objects
    (each its own lock, Gram, and ``eigh``) do not scale for live serving; this keeps one tensor and
    one batched refresh (the Fermi matrix-sign projector on CUDA -- matmul-only, the fast path where
    batched dense cuSOLVER is ~1-2 s).

    Append ``(B, k, F)`` blocks (``k`` new tokens for each of the ``B`` screens) with :meth:`update`;
    the Grams accumulate by a batched rank-``k`` update, the projector refreshes every
    ``refresh_every`` appends, and :attr:`K_signal` / :meth:`energy` read all ``B`` at once.  Whitening
    uses per-screen stats frozen from the first ``warmup`` rows (a growing axis' median/MAD is not
    additive); ``whiten=False`` if rows are pre-normalised.  Caller-owned; thread-safe (one lock).
    ``forgetting`` in (0,1] fades old rows."""

    def __init__(self, B, F, *, far: float = 0.05, null=None, seed: int = 0,
                 refresh_every: int = 32, warmup: int = 64,
                 whiten: bool = True, forgetting: float = 1.0):
        self.B = int(B); self.F = int(F); self.far = float(far)
        self.null = null; self.seed = int(seed)
        self.refresh_every = int(refresh_every); self.warmup = int(warmup)
        self.whiten = bool(whiten); self.forgetting = float(forgetting)
        self.T = 0                       # rows accumulated per screen (uniform)
        self._C = None                   # (B, F, F) whitened feature Grams
        self._rowen = []                 # list of (B, k) per-row whitened energies (floor sigma2)
        self._warm = []                  # warmup row blocks (B, k, F) until whiten stats freeze
        self._med = self._mad = None     # (B, F) frozen whiten stats
        self._K = self._P = None         # cached batched K_signal (B,) / projector (B, F, F)
        self._since = 0
        self._lock = threading.RLock()

    def _ensure_stats(self, xp, rows):
        if not self.whiten or self._med is not None:
            return
        self._warm.append(rows)
        if sum(int(r.shape[1]) for r in self._warm) >= self.warmup:
            self._med, self._mad = _frozen_whiten_batch(xp, _cat_tokens(xp, self._warm))
            self._warm = []

    def _apply_whiten(self, xp, rows):
        if not self.whiten or self._med is None:
            return rows
        return (rows - self._med[:, None, :]) / self._mad[:, None, :]

    def update(self, rows) -> "ResolvedScreenBatch":
        """Append ``rows`` ``(B, k, F)`` -- ``k`` new tokens for each of the ``B`` screens -- as a
        batched rank-``k`` Gram update; refreshes the projector every ``refresh_every`` appends.
        Backend-agnostic (torch rows stay on-device).  Thread-safe."""
        xp = _env.ns(rows)
        R = _env.asnum(rows)
        if R.ndim == 2:                                          # (B, F) -> one token per screen
            R = R[:, None, :]
        with self._lock:
            self._ensure_stats(xp, R)
            Sw = self._apply_whiten(xp, R)                       # (B, k, F)
            G = _env.movedim(xp, Sw, 1, 2) @ Sw                  # (B, F, F) batched rank-k update
            lam = self.forgetting
            self._C = G if self._C is None else (lam * self._C + G)
            self._rowen.append(_env.sum_ax(xp, Sw ** 2, 2))      # (B, k) per-row energies
            self.T = int(round(lam * self.T)) + int(R.shape[1])
            self._since += int(R.shape[1])
            if self._since >= self.refresh_every:
                self.refresh()
        return self

    def refresh(self) -> "ResolvedScreenBatch":
        """Recompute the batched projector + K_signal from the current Grams (amortised expensive
        step) -- one batched Fermi matrix-sign solve over all ``B`` screens."""
        with self._lock:
            if self._C is None:
                return self
            xp = _env.ns(self._C)
            re = _cat_tokens(xp, self._rowen) if self._rowen else _env.zeros(xp, (self.B, 1), ref=self._C)
            if self.null is None:
                sigma2 = _env.median_ax(xp, re, 1) / debias_denominator(self.T, self.F) + 1e-30   # (B,)
                floor2 = screen_floor_sq(sigma2, self.T, self.F, self.far)                        # (B,) squared
            else:
                # A caller provider is the occasional path, so it runs per screen in numpy on
                # that screen's own resolved spectrum -- the same contract `_floor_batch` and
                # `ResolvedScreen` honour, squared into the Gram's units.
                ev = np.asarray(_env.to_numpy(_env.clampmin(xp, xp.linalg.eigvalsh(self._C), 0.0)))
                fl = np.array([apply_floor(self.null, spectrum=np.sqrt(ev[j][::-1]), data=None,
                                           shape=(self.T, self.F), far=self.far,
                                           kind="projection", seed=self.seed) ** 2
                               for j in range(self.B)])
                floor2 = _env.asdtype_of(self._C, fl.astype(float))
            self._K, sign = _fermi_sign_from_gram(xp, self._C, floor2)
            self._P = 0.5 * (_env.eye(xp, self.F, ref=self._C) + sign)
            self._since = 0
        return self

    def _fresh(self):
        with self._lock:
            if self._K is None:
                self.refresh()

    @property
    def K_signal(self):
        """``(B,)`` numpy int array -- resolved-mode count per screen (refreshes if stale)."""
        self._fresh()
        if self._K is None:
            return np.zeros(self.B, dtype=np.int64)
        return np.asarray(_env.to_numpy(self._K)).astype(np.int64)

    def energy(self, rows):
        """Per-row resolved-subspace energy of query ``rows`` ``(B, k, F)`` against the current
        batched projector -- ``energy[b,t] = s_t P_b s_t^T`` -- returned ``(B, k)`` on the backend."""
        self._fresh()
        xp = _env.ns(rows)
        R = _env.asnum(rows)
        if R.ndim == 2:
            R = R[:, None, :]
        Sw = self._apply_whiten(xp, R)
        return _env.sum_ax(xp, (Sw @ self._P) * Sw, 2)          # (B, k)

    def state(self) -> dict:
        """Export the full batched state (resume across sessions; numpy)."""
        return dict(B=self.B, F=self.F, far=self.far, refresh_every=self.refresh_every,
                    warmup=self.warmup, whiten=self.whiten, forgetting=self.forgetting, T=self.T,
                    C=_env.to_numpy(self._C) if self._C is not None else None,
                    rowen=[_env.to_numpy(r) for r in self._rowen],
                    med=_env.to_numpy(self._med) if self._med is not None else None,
                    mad=_env.to_numpy(self._mad) if self._mad is not None else None)

    @classmethod
    def from_state(cls, s: dict) -> "ResolvedScreenBatch":
        obj = cls(s["B"], s["F"], far=s["far"], refresh_every=s["refresh_every"],
                  warmup=s["warmup"], whiten=s["whiten"], forgetting=s["forgetting"])
        obj.T = int(s["T"])
        obj._C = None if s["C"] is None else np.asarray(s["C"])
        obj._rowen = [np.asarray(r) for r in s["rowen"]]
        obj._med = None if s["med"] is None else np.asarray(s["med"])
        obj._mad = None if s["mad"] is None else np.asarray(s["mad"])
        return obj


def _scatter(xp, out, sel, vals):
    """Write ``vals`` (group order) into ``out`` at original positions ``sel``."""
    if _env.is_torch(xp):
        import torch
        out[torch.as_tensor(np.asarray(sel), device=out.device)] = vals
    else:
        out[sel] = vals


def _stack_or_list(xp, items, uniform):
    """Stack per-frame arrays into one tensor when every frame shares a shape (``uniform``),
    else return the per-frame list (ragged fold widths)."""
    if uniform and items and all(it is not None for it in items):
        return _env.stack(xp, items, 0)
    return items
