"""
dynamics.py -- the streaming DYNAMICAL OPERATOR of Entroptics (online DMD / Koopman).

The screen matrix is a state TRAJECTORY; the dynamical operator underneath it is the
one-step propagator A with  x_{t+1} = A x_t.  This module estimates A RECURSIVELY
from the first frame onward (online DMD) and reads its EXACT per-mode decay rates
and frequencies from its eigenvalues mu_k:

    alpha_k = -log|mu_k|     decay rate per step  (>=0 stable; 0 undamped; <0 growth)
    beta_k  = arg(mu_k)      frequency per step   (rad/step)

These are the EXACT decay rates the entropy-width read only approximates.  The
eigenvalue spectrum spans SHORT range (max alpha) to LONG range (min alpha) in one
fit.  The decay is C(tau) = sum_k P_k mu_k^tau, so the operator EXTRAPOLATES it to any
lag -- long-range values from a short observation.

Streaming & splicing.  Start on the FIRST frame and propagate.  ``forgetting = 1``
accumulates the full history (asymptotic long-run rates); ``< 1`` tracks non-stationary
rates.  The accumulators Pxx = sum x_t x_t^H, Pyx = sum x_{t+1} x_t^H are the complete
sufficient statistics and (at forgetting=1) ADDITIVE -> ``state()``/``from_state()``
resume, ``merge()`` splices, ``seed()`` warm-starts, ``tensors()`` extracts them.

ONE code path, backend-agnostic (no torch shadow): every operation dispatches to numpy
OR torch based on the array type it is fed, so the SAME code runs on CPU (numpy) or
GPU (torch tensors stay on-device).  Real- and complex-safe.

  Dynamics(F)                  -- the streaming operator; .update(x) per frame from frame 0.
  Dynamics.rates()             -- DecayRates: long_range / short_range / dominant + spectrum.
  Dynamics.reconstruct_decay(L)-- C(tau) rebuilt from the spectrum (exact, extrapolatable).
  Dynamics.state()/from_state()-- export/import the full state (splice / resume).
  Dynamics.merge(other)        -- splice two segments (exact at forgetting=1).
  Dynamics.tensors()           -- extract the full (complex) operator tensors.
  dynamics(W)                  -- batch: stream all rows of W (T, F), return the operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import ns as _ns   # shared numpy/torch dispatch (one source of truth)


# ── backend dispatch (numpy or torch, one code path) ─────────────────────────

class _Backend:
    """Thin numpy/torch dispatch bound to a reference array (namespace + device).
    Only the handful of ops whose numpy/torch spellings differ are wrapped here;
    the rest use ``self.xp`` (identical spelling in both)."""

    def __init__(self, ref):
        self.xp = _ns(ref)
        self.torch = self.xp is not np
        self.dev = getattr(ref, "device", None) if self.torch else None
        self.rdtype = self.xp.float64
        self.cdtype = self.xp.complex128

    def zeros(self, shape, *, complex=False):
        dt = self.cdtype if complex else self.rdtype
        return (self.xp.zeros(shape, dtype=dt, device=self.dev) if self.torch
                else np.zeros(shape, dtype=dt))

    def eye(self, n, *, complex=False):
        dt = self.cdtype if complex else self.rdtype
        return (self.xp.eye(n, dtype=dt, device=self.dev) if self.torch
                else np.eye(n, dtype=dt))

    def vec(self, x):
        """Bring x into this backend as a 1-D array on the right device."""
        if self.torch:
            import torch
            t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
            return t.to(device=self.dev).reshape(-1)
        return np.asarray(x).reshape(-1)

    def finite(self, x):
        """Zero out non-finite (NaN/Inf) entries: MISSING DATA (RFI, dropped cells)
        contributes nothing to the accumulators, so a coherent signal on the VALID cells is
        still resolved -- sparse is not incoherent.  Backend-agnostic (numpy/torch, complex-safe)."""
        xp = self.xp
        return xp.where(xp.isfinite(x), x, xp.zeros_like(x))

    def astype2d(self, X):
        """Materialise X as a 2-D (T, F) array on this backend/device (a 1-D frame -> (1, F))."""
        if self.torch:
            import torch
            t = X if isinstance(X, torch.Tensor) else torch.as_tensor(np.asarray(X))
            t = t.to(device=self.dev)
            return t.reshape(1, -1) if t.ndim == 1 else t
        a = np.asarray(X)
        return a.reshape(1, -1) if a.ndim == 1 else a

    def iscomplex(self, x):
        return self.xp.is_complex(x) if self.torch else np.iscomplexobj(x)

    def astype(self, x, complex):
        dt = self.cdtype if complex else self.rdtype
        return x.to(dt) if self.torch else x.astype(dt)

    def copy(self, x):
        return x.clone() if self.torch else x.copy()

    def real(self, x):
        return self.xp.real(x)

    def clampmin(self, v, lo):
        return v.clamp(min=lo) if self.torch else np.clip(v, lo, None)

    def argsort_desc(self, v):
        return v.argsort(descending=True) if self.torch else np.argsort(v)[::-1]

    def arange(self, n, *, complex=False):
        dt = self.cdtype if complex else self.rdtype
        return (self.xp.arange(n, dtype=dt, device=self.dev) if self.torch
                else np.arange(n, dtype=dt))

    def pinv(self, M, rcond=None):
        """Moore-Penrose pseudoinverse of the Hermitian M (numpy ``rcond`` / torch
        ``rtol``); ``hermitian=True`` since M is a Gram accumulator."""
        if self.torch:
            return (self.xp.linalg.pinv(M, hermitian=True) if rcond is None
                    else self.xp.linalg.pinv(M, rtol=rcond, hermitian=True))
        return (np.linalg.pinv(M, hermitian=True) if rcond is None
                else np.linalg.pinv(M, rcond=rcond, hermitian=True))


@dataclass
class DecayRates:
    """Exact per-mode decay rates from the dynamical operator's eigenvalues.
    Arrays are in the operator's backend (numpy or torch, on-device)."""
    mu:          object   # eigenvalues mu_k of the propagator (the DMD/Koopman spectrum)
    alpha:       object   # decay rate per mode = -log|mu_k| (dominant-first)
    beta:        object   # frequency per mode  = arg(mu_k)  (rad/step)
    long_range:  float    # slowest decay = min(alpha) -- the long-range rate
    short_range: float    # fastest decay = max(alpha) -- the short-range rate
    dominant:    float    # decay rate of the |mu|-largest (dominant) mode
    n_modes:     int      # number of resolved modes
    n_frames:    int      # frames seen so far


@dataclass
class DynamicsState:
    """The FULL state of a Dynamics operator -- the complete tensors and counts,
    sufficient to resume, splice, or reconstruct the operator exactly.  These are
    ALL the long-range parameters."""
    Pxx:        object        # sum x_t x_t^H       (F x F, complex or real; numpy or torch)
    Pyx:        object        # sum x_{t+1} x_t^H   (F x F)
    first:      object | None  # first frame seen (for adjacent-splice boundaries)
    prev:       object | None  # last frame seen  (for a seamless resume)
    forgetting: float
    n_frames:   int
    n_pairs:    int
    Px:         object | None = None   # sum of left states (the mean, for centred reads)


class Dynamics:
    """Streaming dynamical operator (online DMD / Koopman) on a sequence of feature
    vectors x_t in R^F (or C^F).  Feed one frame per ``update(x)`` from frame 0.
    Backend-agnostic: numpy arrays run on CPU, torch tensors run on their device.

    forgetting : 1.0 accumulates ALL history (asymptotic rates + exact splicing);
                 < 1 gives effective memory 1/(1-forgetting) to TRACK drift.
    rank       : cap on resolved modes (reduced DMD in the POD subspace); None ->
                 data-derived (modes above a tiny energy floor, up to frames seen).
    """

    def __init__(self, n_features: int, *, forgetting: float = 1.0,
                 rank: int | None = None, far: float = 0.05, null=None):
        self.F = int(n_features)
        self.lam = float(forgetting)
        self.rank = rank
        # far/null: the detection operating point for the well-posed DMD truncation in the
        # UNDER-SAMPLED regime (n_pairs < 2F; see _signal_rank).  The truncation is a detection
        # decision (which feature modes are signal), so -- per the null-provider contract -- the
        # caller owns its risk level and null, not a hard-wired constant; default = the derived
        # mp floor at far=0.05, so well-sampled records (which never truncate) are unaffected.
        self._far = float(far)
        self._null = null
        self._b: _Backend | None = None
        self._complex = False
        self.Pxx = None
        self.Pyx = None
        self.Px = None         # running sum of the left states -> the mean, for CENTERED reads
        self._first = None
        self._prev = None
        self.n_frames = 0
        self.n_pairs = 0
        self._red = None       # cached RAW _reduced() result; invalidated on every mutation
        self._red_c = None     # cached CONNECTED _reduced_c() result
        self._srank = None     # cached resolved signal rank (DMD truncation, well-posed at T<F)

    # ── streaming update ──────────────────────────────────────────────────────
    def update(self, x) -> "Dynamics":
        """Feed one frame (an F-vector; numpy or torch)."""
        self._red = self._red_c = self._srank = None
        if self._b is None:
            self._b = _Backend(x)
            self.Pxx = self._b.zeros((self.F, self.F))
            self.Pyx = self._b.zeros((self.F, self.F))
            self.Px = self._b.zeros((self.F,))
        b = self._b
        x = b.vec(x)
        if int(x.shape[0]) != self.F:
            raise ValueError(f"expected a {self.F}-vector, got {int(x.shape[0])}")
        if b.iscomplex(x) and not self._complex:   # promote real -> complex in place
            self._complex = True
            self.Pxx = b.astype(self.Pxx, True)
            self.Pyx = b.astype(self.Pyx, True)
            self.Px = b.astype(self.Px, True)
            if self._first is not None:
                self._first = b.astype(self._first, True)
            if self._prev is not None:
                self._prev = b.astype(self._prev, True)
        x = b.finite(b.astype(x, self._complex))   # RFI / missing cells -> 0 (contribute nothing)
        self.n_frames += 1
        if self._first is None:
            self._first = b.copy(x)
        if self._prev is not None:
            self.Pxx = self.lam * self.Pxx + b.xp.outer(self._prev, b.xp.conj(self._prev))
            self.Pyx = self.lam * self.Pyx + b.xp.outer(x, b.xp.conj(self._prev))
            self.Px = self.lam * self.Px + self._prev     # running sum of left states (the mean)
            self.n_pairs += 1
        self._prev = x
        return self

    def update_block(self, X) -> "Dynamics":
        """Ingest a BLOCK of frames ``X`` (rows = frames, ``T x F``) in ONE vectorised
        pass: the accumulators become two matmuls instead of a Python per-frame loop -- the
        dominant constant-factor speedup for batch ingest, and BLAS/GPU-friendly.  Exact at
        ``forgetting=1`` (the default; equal to the per-frame recurrence up to float
        summation order); for ``forgetting<1`` it falls back to the per-frame recurrence.
        Backend-agnostic: numpy on CPU, torch on its device (stays GPU-resident)."""
        b0 = self._b if self._b is not None else _Backend(X)
        Xb = b0.astype2d(X)                                    # (T, F) on the operator's backend
        T = int(Xb.shape[0])
        if T == 0:
            return self
        if int(Xb.shape[1]) != self.F:
            raise ValueError(f"expected blocks of {self.F}-vectors, got {int(Xb.shape[1])}")
        if self.lam != 1.0:                                    # rare: exact per-frame recurrence
            for t in range(T):
                self.update(Xb[t])
            return self
        self._red = self._red_c = self._srank = None
        if self._b is None:
            self._b = b0
            self.Pxx = b0.zeros((self.F, self.F))
            self.Pyx = b0.zeros((self.F, self.F))
            self.Px = b0.zeros((self.F,))
        b, xp = self._b, self._b.xp
        if b.iscomplex(Xb) and not self._complex:              # promote real -> complex in place
            self._complex = True
            self.Pxx = b.astype(self.Pxx, True); self.Pyx = b.astype(self.Pyx, True)
            self.Px = b.astype(self.Px, True)
            if self._first is not None: self._first = b.astype(self._first, True)
            if self._prev is not None: self._prev = b.astype(self._prev, True)
        Xb = b.finite(b.astype(Xb, self._complex))             # RFI / missing cells -> 0 (contribute nothing)
        if self._prev is not None:                             # boundary transition prev -> X[0]
            p = self._prev
            self.Pxx = self.Pxx + xp.outer(p, xp.conj(p))
            self.Pyx = self.Pyx + xp.outer(Xb[0], xp.conj(p))
            self.Px = self.Px + p
            self.n_pairs += 1
        if self._first is None:
            self._first = b.copy(Xb[0])
        if T >= 2:                                             # within-block transitions (vectorised)
            L = Xb[:-1]                                        # left states x_0..x_{T-2}
            Rr = Xb[1:]                                        # right states x_1..x_{T-1}
            self.Pxx = self.Pxx + L.T @ xp.conj(L)            # sum_t x_t x_t^H
            self.Pyx = self.Pyx + Rr.T @ xp.conj(L)           # sum_t x_{t+1} x_t^H
            self.Px = self.Px + xp.sum(L, axis=0)             # sum_t x_t (the mean, for centering)
            self.n_pairs += (T - 1)
        self.n_frames += T
        self._prev = b.copy(Xb[-1])
        return self

    def _centered(self):
        """The CONNECTED (mean-subtracted) accumulators ``(P_{xx}-\\bar x\\bar x^H,
        P_{yx}-\\bar x\\bar x^H)`` with ``\\bar x = P_x/n`` -- the fluctuation covariances.
        Removing the mean strips a constant per-channel offset (a bandpass / DC): a fixed
        offset is a DMD fixed point ($\\mu=1$) that would otherwise read as a persistent
        mode.  The left/right means coincide to O(1/n), so one correction centres both.  The
        COHERENCE / OPTICS reads (feature spectrum, forgetting margin, reconstructed decay)
        run on these -- matching the screen's per-channel centring and the connected decay of
        section 4; the exact-recovery ``rates()`` keeps the RAW states (Theorem 9.2)."""
        b, xp = self._b, self._b.xp
        n = max(int(self.n_pairs), 1)
        corr = xp.outer(self.Px, xp.conj(self.Px)) / n
        return self.Pxx - corr, self.Pyx - corr

    def _signal_rank(self) -> int:
        """The resolved SIGNAL dimension -- the count of connected feature-correlation
        eigenvalues above the derived (mp) floor.  The DMD is truncated to it (below), so the
        operator fits ONLY the well-determined signal modes and stays well-posed when T < F
        (short cutouts, rank-deficient), rather than overfitting the noise bulk into spurious
        |mu| > 1.  A denoising truncation tied to the noise floor."""
        if self._srank is None:
            from .null_providers import apply_floor
            ev = self._feature_evals()
            edge = apply_floor(self._null, spectrum=ev, data=None, shape=(self.n_frames, self.F),
                               far=self._far, kind="bulk")   # caller's operating point (default mp @ 0.05)
            self._srank = int(self._b.xp.sum(ev > edge))
        return self._srank

    # ── reduced propagator (POD-subspace DMD) ─────────────────────────────────
    def _reduce(self, Pxx, Pyx):
        """The reduced propagator ``(A_tilde, Vr, wr)`` from a pair of covariances (raw or
        connected).  Truncated to the resolved SIGNAL rank (``_signal_rank``) so the fit is
        over-determined and well-posed at T < F.  Backend-agnostic.

        Memory-frugal for a WIDE ``F`` (state = pooled spatial volume ``L^d``): the rank is fixed
        from the eigenvalue spectrum alone (no ``F x F`` eigenvector matrix), then ONLY the top-``r``
        eigenvectors are taken via a partial (subset) eigensolve.  EXACT -- identical to the full
        ``eigh`` truncated to ``r`` -- but peak memory is ``O(F^2)`` (Pxx + r vectors) instead of
        ``O(F^2)`` twice (Pxx + the full ``V``), so ``F ~ 3e4`` (e.g. ``L=32``) fits where the full
        ``eigh`` OOMs."""
        b, xp = self._b, self._b.xp
        F = int(Pxx.shape[0])
        w = b.clampmin(b.real(xp.linalg.eigvalsh(Pxx)), 0.0)   # eigenvalues ONLY (no F x F V)
        order = b.argsort_desc(w)                              # descending energy
        w = w[order]
        if int(w.shape[0]) == 0 or float(w[0]) <= 0:
            return b.zeros((0, 0), complex=self._complex), b.zeros((F, 0)), w[:0]
        floor = float(w[0]) * 1e-10                            # data-derived rank (drop null modes)
        r = int((w > floor).sum())
        r = max(1, min(r, self.n_pairs, self.F))
        if self.n_pairs < 2 * self.F:                          # UNDER-SAMPLED (n_pairs < 2F): the DMD
            r = min(r, max(1, self._signal_rank()))            #   would overfit -> truncate to the
            # resolved signal rank so the fit is over-determined (|mu|>1 spurious modes gone).
            # Well-sampled data keeps the full numerical rank, so exact recovery is untouched.
        if self.rank is not None:
            r = min(r, int(self.rank))
        Vr, wr = self._top_eigvecs(Pxx, r), w[:r]              # top-r eigenvectors, partial solve
        return (Vr.conj().T @ Pyx @ Vr) / wr[None, :], Vr, wr

    def _top_eigvecs(self, Pxx, r):
        """The top-``r`` eigenvectors of a Hermitian PSD matrix via a PARTIAL (subset) eigensolve --
        never forms the full ``F x F`` eigenvector matrix.  numpy path uses LAPACK's subset driver
        (``scipy.linalg.eigh(subset_by_index=...)``, the relatively-robust ``evr``); any other
        backend falls back to the full ``eigh`` (unchanged behaviour).  Exact."""
        b, xp = self._b, self._b.xp
        F = int(Pxx.shape[0])
        try:
            import numpy as _np, scipy.linalg as _sla
            if isinstance(Pxx, _np.ndarray):
                _, V = _sla.eigh(Pxx, subset_by_index=[F - r, F - 1], driver="evr")  # ascending
                return b.astype(_np.ascontiguousarray(V[:, ::-1]), self._complex)     # -> descending
        except Exception:
            pass
        w, V = xp.linalg.eigh(Pxx)                             # fallback: full eigh (other backends)
        return V[:, b.argsort_desc(b.real(w))][:, :r]

    def _reduced(self):
        """RAW reduced propagator -- for exact system-ID rate recovery (Theorem 9.2)."""
        if self._red is None:                            # cached until the next mutation
            self._red = self._reduce(self.Pxx, self.Pyx)
        return self._red

    def _reduced_c(self):
        """CONNECTED reduced propagator -- the fluctuation dynamics (mean removed), for the
        forgetting margin and the reconstructed decay."""
        if self._red_c is None:
            self._red_c = self._reduce(*self._centered())
        return self._red_c

    def propagator(self):
        """The current reduced one-step propagator A_tilde (r x r) in the POD basis."""
        return self._reduced()[0]

    def propagator_full(self, *, rcond: float | None = None):
        """The FULL-space one-step propagator A (F x F), x_{t+1} ~= A x_t, computed
        as A = Pyx Pxx^+ (Moore-Penrose; ``rcond`` drops directions with energy below
        rcond x the largest).  In the operator's backend (numpy or torch, on-device);
        the zero operator before any transition has been seen."""
        b = self._b
        if b is None or self.n_pairs == 0:
            return np.zeros((self.F, self.F))
        return self.Pyx @ b.pinv(self.Pxx, rcond)

    def predict(self, x, *, rcond: float | None = None):
        """One-step forecast A x for a state ``x`` (an F-vector; numpy or torch),
        A the full propagator (propagator_full).  Complex-safe; returned in the
        operator's backend."""
        if self._b is None or self.n_pairs == 0:
            return np.zeros(self.F)
        A = self.propagator_full(rcond=rcond)
        xv = self._b.astype(self._b.vec(x), self._complex)
        return A @ xv

    # ── exact decay rates ─────────────────────────────────────────────────────
    def rates(self) -> DecayRates:
        """Exact per-mode decay rates alpha_k = -log|mu_k| and frequencies
        beta_k = arg(mu_k) from the propagator eigenvalues, dominant (|mu|-largest)
        first.  ``long_range`` = slowest (min alpha), ``short_range`` = fastest."""
        if self._b is None:
            z = np.zeros(0)
            return DecayRates(z, z, z, 0.0, 0.0, 0.0, 0, self.n_frames)
        b, xp = self._b, self._b.xp
        A_tilde = self._reduced()[0]
        if int(A_tilde.shape[0]) == 0:
            z = b.zeros(0)
            return DecayRates(z, z, z, 0.0, 0.0, 0.0, 0, self.n_frames)
        mu = xp.linalg.eigvals(A_tilde)
        mag = xp.abs(mu)
        order = b.argsort_desc(mag)                      # dominant (largest |mu|) first
        mu, mag = mu[order], mag[order]
        alpha = -xp.log(b.clampmin(mag, 1e-300))         # decay rate (>=0 stable; <0 growth)
        beta = xp.angle(mu)                              # frequency
        return DecayRates(
            mu=mu, alpha=alpha, beta=beta,
            long_range=float(xp.min(alpha)), short_range=float(xp.max(alpha)),
            dominant=float(alpha[0]), n_modes=int(mu.shape[0]), n_frames=self.n_frames,
        )

    # ── decay reconstruction (exact, extrapolatable to any lag) ───────────────
    def reconstruct_decay(self, max_lag: int):
        """Rebuild C(tau) = sum_k P_k mu_k^tau from the operator spectrum for
        tau = 0..max_lag-1 -- EXTRAPOLATES beyond the observed window.  Normalised
        so C(0) = 1.  Returned in the operator's backend."""
        b = self._b
        if b is None:
            return np.ones(1)
        xp = b.xp
        A_tilde, _, wr = self._reduced_c()          # connected decay (matches section-4 C(tau))
        if int(A_tilde.shape[0]) == 0:
            return b.zeros(1) + 1.0
        mu, T = xp.linalg.eig(A_tilde)
        Tinv = xp.linalg.pinv(T)
        cov_modal = Tinv @ b.astype(xp.diag(wr), True) @ Tinv.conj().T
        P = b.clampmin(b.real(xp.diag(cov_modal)), 0.0)         # modal power (real)
        taus = b.arange(int(max_lag))                          # real lags 0,1,2,...
        powers = mu[None, :] ** taus[:, None]                  # complex ** real -> (L, r)
        C = b.real(powers @ b.astype(P, True))
        c0 = float(C[0])
        return C / c0 if c0 != 0 else C

    # ── operator-derived reads (streaming replacements for the batch reads) ────
    def forgetting(self, *, tol: float = 1e-9) -> dict:
        """The aperture-FORGETTING read, straight off the operator spectrum: the DMD margin
        ``max_k|mu_k|`` and whether the screen forgets (``margin < 1``).  A persistent
        unit-circle mode (``|mu| = 1``) is the one that does NOT forget.  This is face (iii)
        of forgetting (a spectral margin => decay, summable, Cesaro mean -> 0), the aperture
        form of the extraction-bound axiom -- read incrementally, no O(T^2) autocorrelation.
        On the CONNECTED (mean-subtracted) dynamics, so a constant bandpass / DC does not read
        as a persistent mode (see ``_centered``)."""
        if self._b is None:
            return dict(margin=0.0, forgets=True, n_modes=0)
        xp = self._b.xp
        A_c = self._reduced_c()[0]
        if int(A_c.shape[0]) == 0:
            return dict(margin=0.0, forgets=True, n_modes=0)
        mu = xp.linalg.eigvals(A_c)
        margin = float(xp.max(xp.abs(mu)))
        return dict(margin=margin, forgets=bool(margin < 1.0 - tol), n_modes=int(mu.shape[0]))

    def connected_decay_rate(self) -> float:
        """alpha_1 = -log|mu_1|: the decay rate of the DOMINANT (slowest, |mu|-largest) mode of
        the CONNECTED (mean-subtracted) operator spectrum -- the fluctuation dynamics, centred as
        in ``forgetting`` and ``reconstruct_decay`` (see ``_centered``).  A forward operator read,
        deterministic in the operator eigenvalues.  ``0`` when no mode is resolved."""
        if self._b is None:
            return 0.0
        b, xp = self._b, self._b.xp
        A_c = self._reduced_c()[0]
        if int(A_c.shape[0]) == 0:
            return 0.0
        mag = xp.max(xp.abs(xp.linalg.eigvals(A_c)))       # |mu_1| = dominant (largest) magnitude
        return float(-xp.log(b.clampmin(mag, 1e-300)))

    def _feature_evals(self, *, k: int | None = None, oversample: int = 8,
                       n_power: int = 2, seed: int = 0):
        """Descending eigenvalues of the unit-diagonal feature correlation from ``Pxx``, on
        the operator's backend (numpy or torch, on-device).  ``k=None`` is the
        eigendecomposition (O(F^3)); ``k`` returns only the top ``~k`` via a randomized
        range-finder with ``n_power`` subspace iterations [Halko 2011] in **O(F^2 (k+p))** --
        the feature-side lever for a wide, LOW-RANK ``F`` with a spectral separation (a few signal
        modes over a noise bulk; the bulk below the floor never needs resolving). Deterministic per ``seed``."""
        b, xp = self._b, self._b.xp
        Cov = self._centered()[0]                                # connected (mean-subtracted) covariance
        d = xp.sqrt(b.clampmin(b.real(xp.diag(Cov)), 1e-30))
        R = Cov / xp.outer(d, d)                                 # unit-diagonal correlation
        if k is None or int(k) + int(oversample) >= self.F:
            ev = b.clampmin(b.real(xp.linalg.eigvalsh(R)), 0.0)
            return ev[b.argsort_desc(ev)]                        # exact, descending
        ell = int(k) + int(oversample)
        Om = np.random.default_rng(seed).standard_normal((self.F, ell))
        Y = R @ b.astype(b.astype2d(Om), self._complex)          # random sketch, onto backend/device
        for _ in range(int(n_power)):                            # subspace iteration sharpens the top modes
            Q, _ = xp.linalg.qr(Y)
            Y = R @ Q
        Q, _ = xp.linalg.qr(Y)                                   # F x ell orthonormal range
        Bm = Q.conj().T @ R @ Q                                  # ell x ell compression
        w = b.clampmin(b.real(xp.linalg.eigvalsh(Bm)), 0.0)
        return w[b.argsort_desc(w)]                              # top-ell approx eigenvalues, descending

    def resolved(self, *, null=None, far: float = 0.05, seed: int = 0, k: int | None = None) -> int:
        """Streaming ``K_signal``: the number of feature modes above the noise floor, from
        the accumulated feature covariance ``Pxx`` (its unit-diagonal correlation
        eigenvalues) scored by the null PROVIDER -- the operator form of the screen's
        resolved dimension (cut point ``"bulk"``; only closed-form providers apply, the
        accumulator holds no raw samples).  O(F^3) once, no O(T^2) screen SVD; pass ``k``
        (an upper bound on the expected mode count) for the O(F^2 k) randomized path on a
        wide ``F``.  Backend-agnostic: the eigendecomposition and count stay on-device."""
        from .null_providers import apply_floor
        if self._b is None or self.n_pairs < 1:
            return 0
        xp = self._b.xp
        ev = self._feature_evals(k=k, seed=seed)
        edge = apply_floor(null, spectrum=ev, data=None, shape=(self.n_frames, self.F),
                           far=far, kind="bulk", seed=seed)
        return int(xp.sum(ev > edge))

    def significance(self):
        """Per-mode EVIDENCE of the feature spectrum against the noise null (the operator
        form of ``screen.mode_significance``): the standardized Tracy-Widom deviate
        ``g_k = (T*lambda_k - mu)/sigma_J`` of each ``Pxx`` correlation eigenvalue and its
        tail probability ``p_k = P(TW1 > g_k)``.  ``resolved() == #(p_k < far)`` at ``mp``;
        the read exposes the evidence, the caller sets the false-alarm level."""
        from .screen import ModeSignificance
        from .null_providers import johnstone, tw1_sf
        from .environment import to_numpy
        if self._b is None or self.n_pairs < 1:
            e = np.zeros(0)
            return ModeSignificance(deviate=e, pvalue=e)
        ev = np.asarray(to_numpy(self._feature_evals()), dtype=float)
        mu, sig_J = johnstone(self.n_frames, self.F)             # (T, N) = (frames, features)
        g = (self.n_frames * ev - mu) / sig_J
        p = np.array([tw1_sf(float(x)) for x in g])
        return ModeSignificance(deviate=g, pvalue=p)

    def phi_F(self) -> float:
        """Feature FILL fraction ``2^H(feature-correlation eigenvalues)/F`` from ``Pxx`` --
        the operator form of the screen's ``phi_F`` (high = disorder, low = coherence).
        O(F^3) once (needs the full spectrum); backend-agnostic."""
        from .entropy import shannon_bits
        if self._b is None or self.n_pairs < 1:
            return 0.0
        return float(2.0 ** shannon_bits(self._feature_evals()) / self.F)

    def feature_entropy(self) -> float:
        """``H_F`` -- Shannon entropy (bits) of the feature power marginal from ``diag(Pxx)``
        (``= sum_t |x_{t,f}|^2``): the operator form of geometry's ``H_F``.  O(F)."""
        from .entropy import shannon_bits
        if self._b is None:
            return 0.0
        b, xp = self._b, self._b.xp
        return float(shannon_bits(b.clampmin(b.real(xp.diag(self.Pxx)), 0.0)))

    # ── state export / import (all long-range params) + splicing ──────────────
    def state(self) -> DynamicsState:
        """Export the FULL operator state (all tensors + counts) -- resume or splice."""
        b = self._b
        cp = (lambda a: None if a is None else b.copy(a)) if b else (lambda a: a)
        return DynamicsState(
            Pxx=cp(self.Pxx), Pyx=cp(self.Pyx), first=cp(self._first), prev=cp(self._prev),
            forgetting=self.lam, n_frames=self.n_frames, n_pairs=self.n_pairs, Px=cp(self.Px))

    @classmethod
    def from_state(cls, s: DynamicsState, *, rank: int | None = None,
                   far: float = 0.05, null=None) -> "Dynamics":
        """Reconstruct a Dynamics from an exported state -- resume exactly.  ``far``/``null``
        set the under-sampled truncation operating point (not part of the accumulator state)."""
        F = int(s.Pxx.shape[0])
        dyn = cls(F, forgetting=s.forgetting, rank=rank, far=far, null=null)
        dyn._b = _Backend(s.Pxx)
        dyn.Pxx = dyn._b.copy(s.Pxx)
        dyn.Pyx = dyn._b.copy(s.Pyx)
        dyn._complex = dyn._b.iscomplex(s.Pxx)
        dyn._first = None if s.first is None else dyn._b.copy(s.first)
        dyn._prev = None if s.prev is None else dyn._b.copy(s.prev)
        dyn.n_frames = int(s.n_frames)
        dyn.n_pairs = int(s.n_pairs)
        sPx = getattr(s, "Px", None)
        dyn.Px = dyn._b.copy(sPx) if sPx is not None else dyn._b.zeros((F,))   # backward-compat
        return dyn

    def merge(self, other: "Dynamics", *, adjacent: bool = False) -> "Dynamics":
        """Splice two operators by SUMMING their (forgetting=1) accumulators -- the
        operator of the CONCATENATED stream.  ``adjacent``: ``other``'s stream
        immediately follows ``self``'s, so the boundary transition is added too."""
        if self.lam != 1.0 or other.lam != 1.0:
            raise ValueError("exact merge requires forgetting=1 on both operators")
        if self._b is None:                      # empty (+) X = X -- nothing accumulated yet
            return other if other._b is None else Dynamics.from_state(other.state(), rank=self.rank)
        if other._b is None:
            return Dynamics.from_state(self.state(), rank=self.rank)
        b = self._b
        out = Dynamics(self.F, forgetting=1.0, rank=self.rank, far=self._far, null=self._null)
        out._b = b
        out._complex = self._complex or other._complex
        out.Pxx = b.astype(self.Pxx, out._complex) + b.astype(other.Pxx, out._complex)
        out.Pyx = b.astype(self.Pyx, out._complex) + b.astype(other.Pyx, out._complex)
        sPx = self.Px if self.Px is not None else b.zeros((self.F,))
        oPx = other.Px if other.Px is not None else b.zeros((self.F,))
        out.Px = b.astype(sPx, out._complex) + b.astype(oPx, out._complex)
        out.n_frames = self.n_frames + other.n_frames
        out.n_pairs = self.n_pairs + other.n_pairs
        out._first = None if self._first is None else b.astype(self._first, out._complex)
        out._prev = other._prev if other._prev is not None else self._prev
        if out._prev is not None:
            out._prev = b.astype(out._prev, out._complex)
        if adjacent and self._prev is not None and other._first is not None:
            p = b.astype(self._prev, out._complex)
            f = b.astype(other._first, out._complex)
            out.Pxx = out.Pxx + b.xp.outer(p, b.xp.conj(p))
            out.Pyx = out.Pyx + b.xp.outer(f, b.xp.conj(p))
            out.Px = out.Px + p                                    # boundary left state -> the mean
            out.n_pairs += 1
        return out

    def seed(self, A_prior, *, weight: float = 1.0) -> "Dynamics":
        """Warm-start from a prior propagator ``A_prior`` (F x F) with confidence
        ``weight``: Pxx += weight*I, Pyx += weight*A_prior, so the initial propagator
        is A_prior and real transitions refine it."""
        if self._b is None:
            self._b = _Backend(A_prior)
            self.Pxx = self._b.zeros((self.F, self.F))
            self.Pyx = self._b.zeros((self.F, self.F))
            self.Px = self._b.zeros((self.F,))
        b = self._b
        if b.iscomplex(A_prior) and not self._complex:
            self._complex = True
            self.Pxx = b.astype(self.Pxx, True)
            self.Pyx = b.astype(self.Pyx, True)
            self.Px = b.astype(self.Px, True)
        w = float(weight)
        self._red = self._red_c = self._srank = None
        self.Pxx = self.Pxx + w * b.eye(self.F, complex=self._complex)
        self.Pyx = self.Pyx + w * b.astype(A_prior, self._complex)
        self.n_pairs += int(round(w))
        return self

    # ── full (complex) tensor extraction ──────────────────────────────────────
    def tensors(self) -> dict:
        """Extract the FULL operator tensors: the accumulators, the reduced
        propagator, its eigen-decomposition (eigenvalues + DMD modes in F-space), and
        the POD modes/energies -- for inspection, storage, or splicing."""
        b, xp = self._b, self._b.xp
        A_tilde, Vr, wr = self._reduced()
        if int(A_tilde.shape[0]):
            mu, T = xp.linalg.eig(A_tilde)
            modes = Vr @ T                               # DMD modes in the full F-space
        else:
            mu = b.zeros(0, complex=True)
            modes = b.zeros((self.F, 0), complex=True)
        return {
            "Pxx": self.Pxx, "Pyx": self.Pyx,
            "A_tilde": A_tilde, "pod_modes": Vr, "pod_energy": wr,
            "eigenvalues": mu, "modes": modes,
            "n_frames": self.n_frames, "n_pairs": self.n_pairs,
        }


def dynamics(W, *, forgetting: float = 1.0, rank: int | None = None,
             far: float = 0.05, null=None) -> Dynamics:
    """Batch convenience: ingest all rows of ``W`` (T, F) into a fresh
    :class:`Dynamics` and return the fitted operator -- via the VECTORISED ``update_block``
    (two matmuls, no per-frame Python loop; BLAS/GPU-friendly).  Exact at
    ``forgetting=1``.  ``W`` numpy -> CPU; ``W`` torch -> its device.  Rows are the states x_t.
    ``far``/``null`` set the under-sampled DMD-truncation operating point (default mp @ 0.05)."""
    if W.ndim != 2:
        raise ValueError("W must be 2-D (T, F)")
    return Dynamics(int(W.shape[1]), forgetting=forgetting, rank=rank,
                    far=far, null=null).update_block(W)


# ── the reflection-positive MOMENT PENCIL: the transfer spectrum of a scalar correlation
#    SEQUENCE (Prony / Hankel-DMD), the scalar-series companion to the covariance DMD above ──
@dataclass
class HankelSpectrum:
    """Transfer/Koopman eigenvalues read from a scalar correlation sequence's own moments.
    ``evals`` are sorted descending -- the leading one is the slowest (gap) mode."""
    evals:     object   # transfer eigenvalues lambda_k of the moment pencil, descending
    isolation: float    # lambda_1 / lambda_2 -- how isolated the leading (gap) mode is
    psd:       float    # H0 conditioning (min/max eigenvalue): >= 0 iff PSD (0 = singular/rank-deficient)
    n:         int      # moment order (the pencil is (n+1) x (n+1))

    @property
    def leading(self) -> float:
        """The dominant transfer eigenvalue lambda_1 (the slowest / gap mode)."""
        e = np.asarray(self.evals)
        return float(e[0]) if e.size else float("nan")

    @property
    def rate(self) -> float:
        """The gap = -log(lambda_1) (decay rate of the slowest mode); NaN if lambda_1 not in (0,1)."""
        lam = self.leading
        return -float(np.log(lam)) if 0.0 < lam < 1.0 else float("nan")


def hankel_spectrum(c, n: int, *, rcond: float = 1e-6) -> HankelSpectrum:
    """The transfer-operator spectrum of a real correlation sequence via the reflection-positive
    MOMENT PENCIL (a.k.a. Prony / Hankel-DMD / matrix pencil).

    ``c`` is a correlation sequence ``c[tau] = <O(0) O(tau)>`` at integer lags ``tau = 0..K`` (real,
    an autocovariance).  Form the Hankel moment matrices

        H0[i,j] = c[i+j],   H1[i,j] = c[i+j+1]        (i, j = 0..n)

    and solve the SYMMETRIC generalized eigenproblem ``H1 v = lambda H0 v`` by whitening on H0's
    positive spectrum (its SPD square root): ``M = H0^{-1/2} H1 H0^{-1/2}``, ``lambda = eigvalsh(M)``.
    The ``lambda_k`` ARE the transfer/one-step-propagator eigenvalues; the leading ``lambda_1 =
    e^{-rate}`` is the slowest (gap) mode.  H0 is a Gram / moment matrix, positive semidefinite by
    reflection positivity in the limit; ``psd`` (min/max H0 eigenvalue) reports how well finite
    statistics realise that.

    This is the SCALAR-SEQUENCE companion to the vector-covariance DMD of :class:`Dynamics`: the same
    forward transfer spectrum, read from a correlation sequence's own moments rather than a state
    trajectory.  Domain-agnostic -- ``c`` is any real correlation sequence.

    Scanning ``n`` exposes the moment-order systematic: a well-isolated leading mode is stable in
    ``n`` while ``psd`` stays positive.  Report the band across ``n``; do not pick one favourable
    order.  (See :func:`jackknife` for an error bar, since the pencil has no closed-form interval.)"""
    c = np.asarray(c, dtype=float).ravel()
    if n < 1:
        raise ValueError("moment order n must be >= 1")
    if c.size < 2 * n + 2:
        raise ValueError(f"need >= {2 * n + 2} correlation lags for moment order n={n}, got {c.size}")
    if c[0] != 0:
        c = c / c[0]                                   # normalise C(0)=1 (the pencil is ratio-invariant)
    idx = np.add.outer(np.arange(n + 1), np.arange(n + 1))
    H0 = c[idx]
    H1 = c[idx + 1]
    w, V = np.linalg.eigh(0.5 * (H0 + H0.T))
    wmax = float(w.max()) if w.size else 0.0
    keep = w > rcond * wmax
    if not bool(keep.any()):
        return HankelSpectrum(evals=np.zeros(0), isolation=float("nan"), psd=0.0, n=n)
    Vr = V[:, keep] / np.sqrt(w[keep])
    M = Vr.T @ H1 @ Vr
    ev = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
    iso = float(ev[0] / ev[1]) if ev.size > 1 and ev[1] > 0 else float("inf")
    psd = float(w.min() / wmax) if wmax > 0 else 0.0
    return HankelSpectrum(evals=ev, isolation=iso, psd=psd, n=n)


def jackknife(samples, read, *, n_bins: int | None = None):
    """Delete-one(-bin) JACKKNIFE point estimate and standard error of a scalar ``read``.

    ``samples``: an ``(N, ...)`` array or length-``N`` sequence.  ``read``: ``callable(subset) -> float``,
    evaluated on the full set and on each delete-one(-bin) subset (the subset is passed in the SAME
    form as ``samples``).  With ``n_bins`` the ``N`` samples are split into that many contiguous bins,
    each deleted in turn (delete-one-bin, cheaper for large ``N``); without it, delete-one-sample.
    Returns ``(estimate_on_full, se)`` with

        se = sqrt( (G-1)/G * sum_g (theta_(g) - mean_g)^2 ),   G = number of groups.

    Domain-agnostic resampling for reads with NO closed-form interval (e.g. :func:`hankel_spectrum`)."""
    is_arr = hasattr(samples, "shape")
    arr = samples if is_arr else list(samples)
    N = int(arr.shape[0]) if is_arr else len(arr)
    if N < 2:
        raise ValueError("jackknife needs >= 2 samples")
    G = N if n_bins is None else min(int(n_bins), N)
    groups = np.array_split(np.arange(N), G)

    def _sub(drop):
        keep = np.setdiff1d(np.arange(N), drop)
        return arr[keep] if is_arr else [arr[i] for i in keep]

    full = float(read(arr))
    theta = np.array([float(read(_sub(g))) for g in groups])
    se = float(np.sqrt((G - 1) / G * np.sum((theta - theta.mean()) ** 2)))
    return full, se
