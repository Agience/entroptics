"""
dynamics.py -- the streaming dynamical operator of Entroptics (online DMD / Koopman).

The screen matrix is a state trajectory; the dynamical operator underneath it is the
one-step propagator A with  x_{t+1} = A x_t.  This module estimates A recursively
from the first frame onward (online DMD) and reads its exact per-mode decay rates
and frequencies from its eigenvalues mu_k:

    alpha_k = -log|mu_k|     decay rate per step  (>=0 stable; 0 undamped; <0 growth)
    beta_k  = arg(mu_k)      frequency per step   (rad/step)

These are the exact decay rates the entropy-width read only approximates.  The
eigenvalue spectrum spans short range (max alpha) to long range (min alpha) in one
fit.  The decay is C(tau) = sum_k P_k mu_k^tau, so the operator extrapolates it to any
lag -- long-range values from a short observation.

Streaming & splicing.  Start on the first frame and propagate.  ``forgetting = 1``
accumulates the full history (asymptotic long-run rates); ``< 1`` tracks non-stationary
rates.  The accumulators Pxx = sum x_t x_t^H, Pyx = sum x_{t+1} x_t^H are the complete
sufficient statistics and (at forgetting=1) additive -> ``state()``/``from_state()``
resume, ``merge()`` splices, ``seed()`` warm-starts, ``tensors()`` extracts them.

One code path, backend-agnostic (no torch shadow): every operation dispatches to numpy
or torch based on the array type it is fed, so the same code runs on CPU (numpy) or
GPU (torch tensors stay on-device).  Real- and complex-safe.

  Dynamics(F)                  -- the streaming operator; .update(x) per frame from frame 0.
  Dynamics.rates()             -- DecayRates: long_range / short_range / dominant + spectrum.
  Dynamics.reconstruct_decay(L)-- C(tau) rebuilt from the spectrum (exact, extrapolatable).
  Dynamics.modes()             -- ModePowers: each mode's power P_k in that C(tau), largest first.
  Dynamics.state()/from_state()-- export/import the full state (splice / resume).
  Dynamics.merge(other)        -- splice two segments (exact at forgetting=1).
  Dynamics.tensors()           -- extract the full (complex) operator tensors.
  dynamics(W)                  -- batch: stream all rows of W (T, F), return the operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import math
import numpy as np

from .environment import ns as _ns   # shared numpy/torch dispatch (one source of truth)
from .environment import to_numpy
from .entropy import shannon_bits, macheps
from .null_providers import apply_floor, johnstone, tw1_sf
from .projection import ModeSignificance


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
        """Zero out non-finite (NaN/Inf) entries: missing data (RFI, dropped cells)
        contributes nothing to the accumulators, so a coherent signal on the valid cells is
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
class ModePowers:
    """Each connected mode's power in the decay C(tau) = sum_k P_k mu_k^tau -- the same computation
    as ``Dynamics.reconstruct_decay`` -- sorted by power, largest first.
    Arrays are in the operator's backend (numpy or torch, on-device)."""
    mu:     object   # eigenvalues mu_k of the connected propagator
    alpha:  object   # decay rate per mode = -log|mu_k|
    beta:   object   # frequency per mode  = arg(mu_k)  (rad/step)
    power:  object   # P_k >= 0: the mode's contribution to C(0), in the accumulator's units (a sum over frames)
    share:  object   # P_k / sum_k P_k

    def spectrum(self, f) -> np.ndarray:
        """The Fourier view of the operator: the power spectral density at frequencies ``f``
        (cycles per step, any values, any resolution) of the decay these modes describe,

            S(f) = sum_k P_k (L_k(f) + L_k(-f)) / 2,
            L_k(f) = (1 - r_k^2) / |1 - r_k e^{i (beta_k - 2 pi f)}|^2,   r_k = exp(-|alpha_k|).

        It is the transform of ``C(tau) = Re sum_k P_k mu_k^|tau|`` -- the decay
        ``Dynamics.reconstruct_decay`` rebuilds -- so it is even in ``f`` and integrates over one
        period to ``sum_k P_k = C(0)``.  Evaluated from the modes, not from samples, so it has no
        bins and no leakage, and each line's width is its decay rate ``alpha_k``.

        A mode on or outside the unit circle (``|mu_k| >= 1``) has no decaying correlation to
        transform; it is drawn at the reflected radius ``exp(-|alpha_k|)``, a line whose width is
        ``|alpha_k|``, so the view is continuous in ``|mu_k|`` across the circle.  A mode exactly on
        the circle is a line of zero width: 0 away from its frequency, NaN on it."""
        f = np.asarray(f, dtype=float)
        mu = np.asarray(to_numpy(self.mu)).astype(complex).ravel()
        P = np.asarray(to_numpy(self.power), float).ravel()
        if mu.size == 0:
            return np.zeros(f.shape)
        with np.errstate(divide="ignore"):
            r = np.exp(-np.abs(np.log(np.abs(mu))))[:, None]                  # mu = 0 -> r = 0
        b = np.angle(mu)[:, None]
        w = 2.0 * np.pi * f.ravel()[None, :]

        def lorentz(sign):
            with np.errstate(divide="ignore", invalid="ignore"):
                return (1.0 - r ** 2) / np.abs(1.0 - r * np.exp(1j * (b - sign * w))) ** 2

        S = P @ (0.5 * (lorentz(1.0) + lorentz(-1.0)))
        return S.reshape(f.shape)


@dataclass
class DynamicsState:
    """The full state of a Dynamics operator -- the complete tensors and counts,
    sufficient to resume, splice, or reconstruct the operator exactly.  These are
    all the long-range parameters."""
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

    forgetting : 1.0 accumulates all history (asymptotic rates + exact splicing);
                 < 1 gives effective memory 1/(1-forgetting) to track drift.
    rank       : cap on resolved modes (reduced DMD in the POD subspace); None ->
                 data-derived (modes above a tiny energy floor, up to frames seen).
    """

    def __init__(self, n_features: int, *, forgetting: float = 1.0,
                 rank: int | None = None, far: float = 0.05, null=None):
        self.F = int(n_features)
        self.lam = float(forgetting)
        self.rank = rank
        # far/null: the detection operating point for the well-posed DMD truncation in the
        # under-sampled regime (n_pairs < 2F; see _signal_rank).  The truncation is a detection
        # decision (which feature modes are signal), so -- per the null-provider contract -- the
        # caller owns its risk level and null, not a hard-wired constant; default = the derived
        # mp floor at far=0.05, so well-sampled records (which never truncate) are unaffected.
        self._far = float(far)
        self._null = null
        self._b: _Backend | None = None
        self._complex = False
        self.Pxx = None
        self.Pyx = None
        self.Pinv = None     # Pxx^+ carried forward at O(F^2) -- see _carry_inverse
        self.Px = None         # running sum of the left states -> the mean, for centered reads
        self._first = None
        self._prev = None
        self.n_frames = 0
        self.n_pairs = 0
        self._red = None       # cached raw _reduced() result; invalidated on every mutation
        self._red_c = None     # cached connected _reduced_c() result
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
        absent = ~b.xp.isfinite(b.xp.abs(x))
        x = b.finite(b.astype(x, self._complex))   # RFI / missing cells -> 0 (contribute nothing)
        if self.Pinv is not None and self._prev is not None and bool(absent.any()):
            # An operator is read off PAIRS, and a state with a hole in it is not the state the
            # system was in -- a zero there reads as decay toward zero (see carry_over_gaps).  A
            # block can be completed from its own record; a single frame cannot, so the operator
            # STANDING NOW predicts the hole: x_t = A x_{t-1} with A = Pyx Pxx^+.  Pxx^+ is carried
            # forward below at the same O(F^2) the accumulators cost, so the predictor is always
            # current and there is no interval to choose.
            pred = self.Pyx @ (self.Pinv @ self._prev)
            x = b.xp.where(absent, pred, x)
        self.n_frames += 1
        if self._first is None:
            self._first = b.copy(x)
        if self._prev is not None:
            self.Pxx = self.lam * self.Pxx + b.xp.outer(self._prev, b.xp.conj(self._prev))
            self.Pyx = self.lam * self.Pyx + b.xp.outer(x, b.xp.conj(self._prev))
            self.Px = self.lam * self.Px + self._prev     # running sum of left states (the mean)
            self.n_pairs += 1
            self._carry_inverse(b, self._prev)
        self._prev = x
        return self

    def _carry_inverse(self, b, u) -> None:
        """Carry ``Pxx^+`` forward through this frame's rank-1 update, at O(F^2).

        ``Pxx`` grows by ``lam*Pxx + u u^H``, and the inverse of a scaled rank-1 update is the
        Sherman-Morrison identity, so the predictor the fill above needs costs no more than the
        accumulator it is read from -- no recomputation interval to pick, and nothing stale.

        It is seeded once the stream OVER-DETERMINES the fit (``n_pairs >= 2F``), which is the
        same point the rank truncation already uses; before that there is no inverse to carry and
        a frame with a hole simply contributes what it has.  Sherman-Morrison drifts as it is
        carried -- measured at 1.2e-10 after 59,000 pairs -- which is some seven orders below the
        estimator's own sampling error at that count, and the reads still take an exact ``pinv``;
        only the imputed cells see the carried one."""
        xp = b.xp
        if self.Pinv is None:
            if self.n_pairs >= 2 * self.F:
                self.Pinv = b.pinv(self.Pxx, None)
            return
        if self.lam != 1.0:
            self.Pinv = self.Pinv / self.lam
        Mu = self.Pinv @ u
        d = xp.sum(xp.conj(u) * Mu) + 1.0     # kept in the array's dtype: a Python complex here
        if float(abs(to_numpy(d))) > 0.0:     # would promote a real inverse and break the matmul
            self.Pinv = self.Pinv - xp.outer(Mu, xp.conj(Mu)) / d

    def update_block(self, X, *, adjacent: bool = True) -> "Dynamics":
        """Ingest a block of frames ``X`` (rows = frames, ``T x F``) in one vectorised
        pass: the accumulators become two matmuls instead of a Python per-frame loop -- the
        dominant constant-factor speedup for batch ingest, and BLAS/GPU-friendly.  Exact at
        ``forgetting=1`` (the default; equal to the per-frame recurrence up to float
        summation order); for ``forgetting<1`` it falls back to the per-frame recurrence.
        Backend-agnostic: numpy on CPU, torch on its device (stays GPU-resident).

        ``adjacent=False`` says this block is a NEW RUN, not a continuation: no transition pair is
        formed between the previous block's last frame and this block's first.  It is the same
        statement :meth:`merge` already takes under the same name, and it is what an ENSEMBLE is --
        many independent trajectories of one system, where a pair spanning two of them is not a
        transition the system ever made.

        Without it, an ensemble can only be accumulated by merging one operator per member, and a
        merge holds source, source and destination at once: three dense ``F x F`` pairs.  Streaming
        the members into a single accumulator instead is the SAME operator (verified bit-identical,
        ``max|dPxx| = 0``) at one operator's memory -- which at ``F = 32768`` is the difference
        between 24 GB and about 72 GB, i.e. between fitting on a host and not."""
        b0 = self._b if self._b is not None else _Backend(X)
        Xb = b0.astype2d(X)                                    # (T, F) on the operator's backend
        T = int(Xb.shape[0])
        if T == 0:
            return self
        if int(Xb.shape[1]) != self.F:
            raise ValueError(f"expected blocks of {self.F}-vectors, got {int(Xb.shape[1])}")
        Xb = carry_over_gaps(Xb)             # a block IS a record: a hole in it is carried by the
                                             # record's own operator, never by a zero.  Ahead of
                                             # the forgetting fallback below, so both ingest paths
                                             # see the same completed block.
        if not adjacent:
            self._prev = None                  # a new run: no transition spans the join
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
        self.Pinv = None          # block ingest moved Pxx wholesale: reseed
        return self

    def _centered(self):
        """The connected (mean-subtracted) accumulators ``(P_{xx}-\\bar x\\bar x^H,
        P_{yx}-\\bar x\\bar x^H)`` with ``\\bar x = P_x/n`` -- the fluctuation covariances.
        Removing the mean strips a constant per-channel offset (a bandpass / DC): a fixed
        offset is a DMD fixed point ($\\mu=1$) that would otherwise read as a persistent
        mode.  The left/right means coincide to O(1/n), so one correction centres both.  The
        coherence / optics reads (feature spectrum, forgetting margin, reconstructed decay)
        run on these -- matching the screen's per-channel centring and the connected decay of
        section 4; the exact-recovery ``rates()`` keeps the raw states (Theorem 9.2)."""
        b, xp = self._b, self._b.xp
        n = max(int(self.n_pairs), 1)
        corr = xp.outer(self.Px, xp.conj(self.Px)) / n
        return self.Pxx - corr, self.Pyx - corr

    def _signal_rank(self) -> int:
        """The resolved signal dimension -- the count of connected feature-correlation
        eigenvalues above the derived (mp) floor.  The DMD is truncated to it (below), so the
        operator fits only the well-determined signal modes and stays well-posed when T < F
        (short cutouts, rank-deficient), so the noise bulk cannot be overfitted into spurious
        |mu| > 1.  A denoising truncation tied to the noise floor."""
        if self._srank is None:
            ev = self._feature_evals()
            edge = apply_floor(self._null, spectrum=ev, data=None, shape=(self.n_frames, self.F),
                               far=self._far, kind="bulk")   # caller's operating point (default mp @ 0.05)
            self._srank = int(self._b.xp.sum(ev > edge))
        return self._srank

    def floor_contrast(self) -> float | None:
        """``lambda_1 / edge`` -- the dominant feature-correlation eigenvalue over the derived
        noise floor.  ``None`` when the operator carries no spectrum to read it from.

        The same ``apply_floor`` call and the same caller operating point (``far``) that
        ``_signal_rank`` counts modes against, so the floor a mode is measured against and the
        floor a mode is said to have decayed to are one number, not two that can drift.  Published
        because the reciprocal is the only non-arbitrary ``eps`` for "this mode has been
        forgotten": a mode is gone when its amplitude reaches the level this spectrum cannot tell
        from noise.  ``Aperture._coherence_horizon`` is the caller."""
        if self._b is None or self.n_pairs < 1:
            return None
        ev = self._feature_evals()
        if int(ev.shape[0]) == 0:
            return None
        edge = apply_floor(self._null, spectrum=ev, data=None, shape=(self.n_frames, self.F),
                           far=self._far, kind="bulk")
        top = float(self._b.xp.max(ev))
        edge = float(edge)
        if not (edge > 0.0) or not math.isfinite(top):
            return None
        return top / edge

    # ── reduced propagator (POD-subspace DMD) ─────────────────────────────────
    def _reduce(self, Pxx, Pyx):
        """The reduced propagator ``(A_tilde, Vr, wr)`` from a pair of covariances (raw or
        connected).  Truncated to the resolved signal rank (``_signal_rank``) so the fit is
        over-determined and well-posed at T < F.  Backend-agnostic.

        Memory-frugal for a wide ``F`` (state = pooled spatial volume ``L^d``): the rank is fixed
        from the eigenvalue spectrum alone (no ``F x F`` eigenvector matrix), then only the top-``r``
        eigenvectors are taken via a partial (subset) eigensolve.  Exact -- identical to the full
        ``eigh`` truncated to ``r`` -- but peak memory is ``O(F^2)`` (Pxx + r vectors) instead of
        ``O(F^2)`` twice (Pxx + the full ``V``), so ``F ~ 3e4`` (e.g. ``L=32``) fits where the full
        ``eigh`` OOMs."""
        b, xp = self._b, self._b.xp
        F = int(Pxx.shape[0])
        capped = self._rank_capped_eig(Pxx)                    # exact when rank(Pxx) << F; else None
        if capped is not None:
            w, _Vcap = capped
            w = b.clampmin(w, 0.0)
        else:
            _Vcap = None
            w = b.clampmin(b.real(xp.linalg.eigvalsh(Pxx)), 0.0)   # eigenvalues only (no F x F V)
            order = b.argsort_desc(w)                              # descending energy
            w = w[order]
        if int(w.shape[0]) == 0 or float(w[0]) <= 0:
            return b.zeros((0, 0), complex=self._complex), b.zeros((F, 0)), w[:0]
        floor = float(w[0]) * 1e-10                            # data-derived rank (drop null modes)
        r = int((w > floor).sum())
        r = max(1, min(r, self.n_pairs, self.F))
        if self.n_pairs < 2 * self.F:                          # under-sampled (n_pairs < 2F): the DMD
            r = min(r, max(1, self._signal_rank()))            #   would overfit -> truncate to the
            # resolved signal rank so the fit is over-determined (|mu|>1 spurious modes gone).
            # Well-sampled data keeps the full numerical rank, so exact recovery is untouched.
        if self.rank is not None:
            r = min(r, int(self.rank))
        # `n_pairs >= 2F` is a COUNT, and counting pairs is not the same as having independent
        # information: an ensemble spliced from many short trajectories reaches 2F pairs while each
        # trajectory still carries only its own few steps, so the fit can absorb the noise bulk and
        # come back with a mode that GROWS.  Measured on spliced SU(2) at L=12 (F=1728, n_pairs=4416,
        # so the guard above declined to truncate): |mu_1| = 1.37, i.e. 37% amplification per step,
        # which `connected_decay_rate` then had to refuse as NaN.  The count cannot detect that, but
        # the fitted operator can be asked directly -- so it is, and a growing fit falls back to the
        # resolved signal rank, which is what the truncation was for.
        r = self._rank_without_growth(Pxx, Pyx, w, r, _Vcap)
        # the sketch already produced the eigenvectors alongside the spectrum, so the separate
        # subset solve is only needed on the full path
        Vr = (b.astype(_Vcap[:, :r], self._complex) if _Vcap is not None
              else self._top_eigvecs(Pxx, r))
        wr = w[:r]
        return (Vr.conj().T @ Pyx @ Vr) / wr[None, :], Vr, wr

    def _rank_without_growth(self, Pxx, Pyx, w, r, V=None):
        """``r``, reduced to the resolved signal rank if the rank-``r`` fit has a growing mode.

        A decaying system's propagator has ``|mu| <= 1``.  A fit that returns ``|mu| > 1`` has
        modelled noise, and the cure is the denoising truncation the caller may have skipped on a
        pair COUNT.  Asking the operator is cheap -- an ``r x r`` eigenvalue solve on a matrix that
        has just been formed anyway -- and it is a direct test of the thing that matters, where the
        count is a proxy that can be wrong.  Returns ``r`` unchanged when the fit already decays,
        so a well-determined fit (exact recovery, Theorem 9.2) is never touched."""
        b, xp = self._b, self._b.xp
        sig = max(1, int(self._signal_rank()))
        if r <= sig:
            return r                                   # already at or below the resolved rank
        # Reuse the eigenvectors the caller already has where it has them: a second F x F subset
        # solve here costs another O(F^3) AND another F x r matrix, and peak memory is what decides
        # whether this runs at all -- an earlier build of this check was OOM-killed for want of it.
        Vr = V[:, :r] if V is not None else self._top_eigvecs(Pxx, r)
        A = (Vr.conj().T @ Pyx @ Vr) / w[:r][None, :]
        mu = xp.linalg.eigvals(A)
        tol = int(mu.shape[0]) * macheps(xp, mu)
        if float(xp.max(xp.abs(mu))) <= 1.0 + tol:
            return r                                   # decays: the unrestricted fit stands
        return sig

    def _rank_capped_eig(self, P):
        """``(w_desc, V_desc)`` of a Hermitian PSD accumulator, or ``None`` to take the full path.

        ``P = sum_t x_t x_t^H`` is a sum of ``n_pairs`` rank-1 terms, so ``rank(P) <= n_pairs``.
        When the stream is shorter than the state is wide -- an ensemble of short configurations
        against a pooled spatial volume, ``F = L^d`` -- most of ``P``'s spectrum is exactly zero and
        the full ``eigvalsh`` spends ``O(F^3)`` computing those zeros.  A range finder over
        ``k > rank`` captures the entire non-null space, so the non-zero spectrum comes back EXACT,
        not approximated: at ``F=4096`` with 64 rows the same eigenvalues arrive ~57x faster, and at
        ``F=8192`` ~82x.

        Exactness is CHECKED, not assumed: ``sum(w)`` must equal ``tr(P)``, which is the sum of ALL
        eigenvalues, so any mode the sketch failed to capture shows up as a trace defect (measured:
        1e-16 when the range is complete, 5e-1 when it is not).  A failed check falls back to the
        full eigendecomposition, so a bad draw costs time and never correctness.  The sketch is
        seeded, so a given input always gives the same answer.

        numpy only; any other backend returns ``None`` and takes the full path, as ``_top_eigvecs``
        does for the same reason."""
        import numpy as _np
        if not isinstance(P, _np.ndarray) or _np.iscomplexobj(P):
            return None                      # complex accumulators keep the full, well-trodden path
        F = int(P.shape[0])
        k = min(F, max(1, int(self.n_pairs)) + 8)
        if k >= F // 2:
            return None                      # not rank-deficient enough to be worth a sketch
        Om = _np.random.default_rng(0).standard_normal((F, k))
        Q, _ = _np.linalg.qr(P @ Om)
        B = Q.T @ (P @ Q)
        wB, UB = _np.linalg.eigh(0.5 * (B + B.T))
        tr = float(_np.trace(P))
        if not (tr > 0.0) or abs(float(wB.sum()) - tr) > 1e-10 * abs(tr):
            return None                      # range incomplete -> full eigendecomposition
        order = _np.argsort(wB)[::-1]
        return wB[order], _np.ascontiguousarray(Q @ UB[:, order])

    def _top_eigvecs(self, Pxx, r):
        """The top-``r`` eigenvectors of a Hermitian PSD matrix via a partial (subset) eigensolve --
        never forms the full ``F x F`` eigenvector matrix.  numpy path uses LAPACK's subset driver
        (``scipy.linalg.eigh(subset_by_index=...)``, the relatively-robust ``evr``); any other
        backend falls back to the full ``eigh``, returning the same top-``r`` eigenvectors.
        Exact either way."""
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
        """Raw reduced propagator -- for exact system-ID rate recovery (Theorem 9.2)."""
        if self._red is None:                            # cached until the next mutation
            self._red = self._reduce(self.Pxx, self.Pyx)
        return self._red

    def _reduced_c(self):
        """Connected reduced propagator -- the fluctuation dynamics (mean removed), for the
        forgetting margin and the reconstructed decay."""
        if self._red_c is None:
            self._red_c = self._reduce(*self._centered())
        return self._red_c

    def propagator(self):
        """The current reduced one-step propagator A_tilde (r x r) in the POD basis."""
        return self._reduced()[0]

    def propagator_full(self, *, rcond: float | None = None):
        """The full-space one-step propagator A (F x F), x_{t+1} ~= A x_t, computed
        as A = Pyx Pxx^+ (Moore-Penrose; ``rcond`` drops directions with energy below
        rcond x the largest).  In the operator's backend (numpy or torch, on-device);
        the zero operator before any transition has been seen."""
        b = self._b
        if b is None or self.n_pairs == 0:
            return np.zeros((self.F, self.F))
        return self.Pyx @ b.pinv(self.Pxx, rcond)

    def predict(self, x, *, steps: int = 1, rcond: float | None = None):
        """Forecast the state ``steps`` ahead.  ``steps=1`` is the one-step ``A x`` (the full
        propagator ``propagator_full``).  ``steps=h > 1`` is the spectral h-step forecast
        ``x_h = sum_k phi_k mu_k^h b_k`` from the reduced DMD eigenvalues -- exact powers of the
        eigenvalues, so re-applying ``A`` in a loop (which amplifies error and costs h matmuls)
        is avoided: it is one eigensolve of the small reduced operator, then ``mu^h``.  Numerically
        stable (a decaying ``|mu|<1`` mode stays bounded).  Complex-safe; operator backend."""
        if self._b is None or self.n_pairs == 0:
            return np.zeros(self.F)
        if int(steps) <= 1:
            A = self.propagator_full(rcond=rcond)
            xv = self._b.astype(self._b.vec(x), self._complex)
            return A @ xv
        return self._spectral_forecast(self._b.vec(x), [int(steps)])[0]

    def rollout(self, x, horizon: int, *, include_current: bool = False):
        """The predicted trajectory ``[x_1, ..., x_h]`` (``horizon x F``) from ``x``, via the reduced
        DMD spectrum -- ``x_j = sum_k phi_k mu_k^j b_k`` with exact eigenvalue powers (no ``A^j``
        error amplification; the whole horizon in one eigensolve).  ``include_current`` prepends the
        reconstructed ``x_0`` (``horizon+1`` rows).  Operator backend; complex-safe."""
        if self._b is None or self.n_pairs == 0:
            return np.zeros((int(horizon), self.F))
        lo = 0 if include_current else 1
        return self._spectral_forecast(self._b.vec(x), list(range(lo, int(horizon) + 1)))

    def _spectral_forecast(self, xv, horizons):
        """``x_h = sum_k phi_k mu_k^h b_k`` at each integer horizon in ``horizons`` (>=0), from the
        reduced raw DMD spectrum: project ``x`` onto the POD basis, read the mode amplitudes
        ``b = W^{-1} Vr^H x``, and evolve each mode by its exact eigenvalue power ``mu_k^h``.  Returns
        ``(len(horizons), F)`` in the operator backend (real if the dynamics are real)."""
        b, xp = self._b, self._b.xp
        A_tilde, Vr, _ = self._reduced()                     # raw reduced propagator (forward pred)
        r = int(A_tilde.shape[0]); H = len(horizons)
        if r == 0:
            return b.zeros((H, self.F), complex=self._complex)
        mu, W = xp.linalg.eig(b.astype(A_tilde, True))       # A_tilde = W diag(mu) W^{-1}
        a0 = Vr.conj().T @ b.astype(xv, True)                # POD projection of the initial state (r,)
        amp = xp.linalg.solve(W, a0)                         # mode amplitudes b_k = (W^{-1} a0)  (r,)
        hs = b.astype(b.astype2d(np.asarray(horizons, dtype=float)).reshape(-1), True)   # (H,) complex
        logmu = xp.log(b.clampmin(xp.abs(mu), 1e-300) * xp.exp(1j * xp.angle(mu)))       # (r,) safe log
        muH = xp.exp(hs[:, None] * logmu[None, :])           # (H, r) = mu^h, exact
        aH = (muH * amp[None, :]) @ W.T                      # (H, r): a_h = W (mu^h ⊙ b)
        xH = aH @ Vr.T                                       # (H, F): x_h = Vr a_h
        return xH if self._complex else b.real(xH)

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

    # ── modal power: the P_k of C(tau) = sum_k P_k mu_k^tau ───────────────────
    def _modal_power(self):
        """The connected modes mu_k and their powers P_k >= 0 in the section-4 decay
        C(tau) = sum_k P_k mu_k^tau, or None before the first pair."""
        b = self._b
        if b is None:
            return None
        xp = b.xp
        A_tilde, _, wr = self._reduced_c()          # connected decay (matches section-4 C(tau))
        if int(A_tilde.shape[0]) == 0:
            return None
        mu, T = xp.linalg.eig(A_tilde)
        Tinv = xp.linalg.pinv(T)
        cov_modal = Tinv @ b.astype(xp.diag(wr), True) @ Tinv.conj().T
        P = b.clampmin(b.real(xp.diag(cov_modal)), 0.0)         # modal power (real)
        return mu, P

    def modes(self) -> ModePowers:
        """Each connected mode's power P_k in the decay C(tau) = sum_k P_k mu_k^tau -- the
        same computation as ``reconstruct_decay`` -- sorted by power, largest first, with its
        share P_k / sum P, rate alpha_k = -log|mu_k| and frequency beta_k = arg(mu_k).  Ranking
        by power puts the record's own signal modes first; how many modes are signal is the
        count's job, not this one's."""
        mp = self._modal_power()
        if mp is None:
            z = np.zeros(0) if self._b is None else self._b.zeros(0)
            return ModePowers(z, z, z, z, z)
        b = self._b
        xp = b.xp
        mu, P = mp
        order = b.argsort_desc(P)
        mu, P = mu[order], P[order]
        total = float(P.sum())
        share = P / total if total > 0 else P * 0.0
        return ModePowers(mu=mu, alpha=-xp.log(b.clampmin(xp.abs(mu), 1e-300)),
                          beta=xp.angle(mu), power=P, share=share)

    # ── decay reconstruction (exact, extrapolatable to any lag) ───────────────
    def reconstruct_decay(self, max_lag: int):
        """Rebuild C(tau) = sum_k P_k mu_k^tau from the operator spectrum for
        tau = 0..max_lag-1 -- extrapolates beyond the observed window.  Normalised
        so C(0) = 1.  Returned in the operator's backend.  The P_k are ``modes().power``."""
        b = self._b
        if b is None:
            return np.ones(1)
        mp = self._modal_power()
        if mp is None:
            return b.zeros(1) + 1.0
        mu, P = mp
        taus = b.arange(int(max_lag))                          # real lags 0,1,2,...
        powers = mu[None, :] ** taus[:, None]                  # complex ** real -> (L, r)
        C = b.real(powers @ b.astype(P, True))
        c0 = float(C[0])
        return C / c0 if c0 != 0 else C

    # ── operator-derived reads (streaming replacements for the batch reads) ────
    def forgetting(self, *, tol: float | None = None) -> dict:
        """The aperture-forgetting read, straight off the operator spectrum: the DMD margin
        ``max_k|mu_k|`` and whether the screen forgets (``margin < 1``).  A persistent
        unit-circle mode (``|mu| = 1``) is the one that does not forget.  This is face (iii)
        of forgetting (a spectral margin => decay, summable, Cesaro mean -> 0), the aperture
        form of the extraction-bound axiom -- read incrementally, no O(T^2) autocorrelation.
        On the connected (mean-subtracted) dynamics, so a constant bandpass / DC does not read
        as a persistent mode (see ``_centered``)."""
        if self._b is None:
            return dict(margin=0.0, forgets=True, n_modes=0)
        xp = self._b.xp
        A_c = self._reduced_c()[0]
        if int(A_c.shape[0]) == 0:
            return dict(margin=0.0, forgets=True, n_modes=0)
        mu = xp.linalg.eigvals(A_c)
        margin = float(xp.max(xp.abs(mu)))
        # "on the unit circle" is a question about arithmetic, not about the signal: an eig of an
        # r x r operator returns a magnitude good to about r*eps, and a margin inside that is 1.
        if tol is None:
            tol = int(mu.shape[0]) * macheps(self._b.xp, mu)
        return dict(margin=margin, forgets=bool(margin < 1.0 - tol), n_modes=int(mu.shape[0]))

    def connected_decay_rate(self) -> float:
        """alpha_1 = -log|mu_1|: the decay rate of the dominant (slowest, |mu|-largest) mode of
        the connected (mean-subtracted) operator spectrum -- the fluctuation dynamics, centred as
        in ``forgetting`` and ``reconstruct_decay`` (see ``_centered``).  A forward operator read,
        deterministic in the operator eigenvalues.  ``0`` when no mode is resolved.

        **NaN when the fitted operator grows.**  ``|mu_1| > 1`` is a mode that amplifies with every
        step, which a decaying (reflection-positive) system does not have -- it means the fit has
        absorbed noise, not that the signal has a negative decay rate.  Returning ``-log|mu_1|``
        there hands back a negative number shaped exactly like a rate, and it reads downstream as a
        measured gap: a spliced ensemble of short trajectories produced ``-0.31``, ``-0.28`` and
        ``-0.09`` this way, printed in a certification table beside real rates.  A read that did not
        resolve a rate must not be reported as one, so it comes back NaN and propagates as NaN.

        Being ON the unit circle is a different statement and stays a rate of ``0``: a persistent
        mode does not decay, and ``margin == 1`` within the arithmetic's own resolution is the same
        question ``forgetting`` asks, taken at the same ``r * eps`` tolerance so the two cannot
        disagree about where the unit circle is."""
        if self._b is None:
            return 0.0
        b, xp = self._b, self._b.xp
        A_c = self._reduced_c()[0]
        if int(A_c.shape[0]) == 0:
            return 0.0
        mu = xp.linalg.eigvals(A_c)
        mag = float(xp.max(xp.abs(mu)))                    # |mu_1| = dominant (largest) magnitude
        tol = int(mu.shape[0]) * macheps(xp, mu)           # same resolution `forgetting` uses
        if mag > 1.0 + tol:
            return float("nan")                            # growing mode: no rate was resolved
        if mag >= 1.0 - tol:
            return 0.0                                     # on the unit circle: persistent, no decay
        # `mag` is a Python float from the line above, not a backend array, so it must not go
        # through `b.clampmin`/`xp.log`: on the torch backend those are tensor methods and this
        # raised AttributeError for every resolved rate. math.log is exact on a float and agrees
        # with the numpy path it replaces.
        return float(-math.log(max(mag, 1e-300)))

    def _feature_evals(self, *, k: int | None = None, oversample: int = 8,
                       n_power: int = 2, seed: int = 0):
        """Descending eigenvalues of the unit-diagonal feature correlation from ``Pxx``, on
        the operator's backend (numpy or torch, on-device).  ``k=None`` is the
        eigendecomposition (O(F^3)); ``k`` returns only the top ``~k`` via a randomized
        range-finder with ``n_power`` subspace iterations [Halko 2011] in **O(F^2 (k+p))** --
        the feature-side lever for a wide, low-rank ``F`` with a spectral separation (a few signal
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
        eigenvalues) scored by the null provider -- the operator form of the screen's
        resolved dimension (cut point ``"bulk"``; only closed-form providers apply, the
        accumulator holds no raw samples).  O(F^3) once, no O(T^2) screen SVD; pass ``k``
        (an upper bound on the expected mode count) for the O(F^2 k) randomized path on a
        wide ``F``.  Backend-agnostic: the eigendecomposition and count stay on-device."""
        if self._b is None or self.n_pairs < 1:
            return 0
        xp = self._b.xp
        ev = self._feature_evals(k=k, seed=seed)
        edge = apply_floor(null, spectrum=ev, data=None, shape=(self.n_frames, self.F),
                           far=far, kind="bulk", seed=seed)
        return int(xp.sum(ev > edge))

    def significance(self):
        """Per-mode evidence of the feature spectrum against the noise null (the operator
        form of ``screen.mode_significance``): the standardized Tracy-Widom deviate
        ``g_k = (T*lambda_k - mu)/sigma_J`` of each ``Pxx`` correlation eigenvalue and its
        tail probability ``p_k = P(TW1 > g_k)``.  ``resolved() == #(p_k < far)`` at ``mp``;
        the read exposes the evidence, the caller sets the false-alarm level."""
        if self._b is None or self.n_pairs < 1:
            e = np.zeros(0)
            return ModeSignificance(deviate=e, pvalue=e)
        ev = np.asarray(to_numpy(self._feature_evals()), dtype=float)
        mu, sig_J = johnstone(self.n_frames, self.F)             # (T, N) = (frames, features)
        g = (self.n_frames * ev - mu) / sig_J
        p = np.array([tw1_sf(float(x)) for x in g])
        return ModeSignificance(deviate=g, pvalue=p)

    def phi_F(self) -> float:
        """Feature fill fraction ``2^H(feature-correlation eigenvalues)/F`` from ``Pxx`` --
        the operator form of the screen's ``phi_F`` (high = disorder, low = coherence).
        O(F^3) once (needs the full spectrum); backend-agnostic."""
        if self._b is None or self.n_pairs < 1:
            return 0.0
        return float(2.0 ** shannon_bits(self._feature_evals()) / self.F)

    def feature_entropy(self) -> float:
        """``H_F`` -- Shannon entropy (bits) of the feature power marginal from ``diag(Pxx)``
        (``= sum_t |x_{t,f}|^2``): the operator form of geometry's ``H_F``.  O(F)."""
        if self._b is None:
            return 0.0
        b, xp = self._b, self._b.xp
        return float(shannon_bits(b.clampmin(b.real(xp.diag(self.Pxx)), 0.0)))

    # ── state export / import (all long-range params) + splicing ──────────────
    def state(self) -> DynamicsState:
        """Export the full operator state (all tensors + counts) -- resume or splice."""
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
        dyn.Pinv = None           # restored accumulators: reseed
        dyn.n_frames = int(s.n_frames)
        dyn.n_pairs = int(s.n_pairs)
        sPx = getattr(s, "Px", None)
        dyn.Px = dyn._b.copy(sPx) if sPx is not None else dyn._b.zeros((F,))   # no left-state history yet -> zero
        return dyn

    def merge(self, other: "Dynamics", *, adjacent: bool = False) -> "Dynamics":
        """Splice two operators by summing their (forgetting=1) accumulators -- the
        operator of the concatenated stream.  ``adjacent``: ``other``'s stream
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
        self.Pinv = None          # Pxx moved outside the recurrence: reseed
        self.Pyx = self.Pyx + w * b.astype(A_prior, self._complex)
        self.n_pairs += int(round(w))
        return self

    # ── full (complex) tensor extraction ──────────────────────────────────────
    def tensors(self) -> dict:
        """Extract the full operator tensors: the accumulators, the reduced
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
    :class:`Dynamics` and return the fitted operator -- via the vectorised ``update_block``
    (two matmuls, no per-frame Python loop; BLAS/GPU-friendly).  Exact at
    ``forgetting=1``.  ``W`` numpy -> CPU; ``W`` torch -> its device.  Rows are the states x_t.
    ``far``/``null`` set the under-sampled DMD-truncation operating point (default mp @ 0.05)."""
    if W.ndim != 2:
        raise ValueError("W must be 2-D (T, F)")
    return Dynamics(int(W.shape[1]), forgetting=forgetting, rank=rank,
                    far=far, null=null).update_block(W)


def carry_over_gaps(W, *, iters: int = 64):
    """Fill a record's unobserved cells from the record's OWN one-step operator, to a fixed point.

    Every other read here treats a missing cell as absent.  An operator cannot: it is read off
    PAIRS of states, and a state with a hole in it is not the state the system was in.  Zeroing
    the hole makes the transition into it look like decay toward zero, and no reweighting of the
    accumulated sums repairs that -- the pairs themselves are wrong.  Measured on a planted
    operator, the zeroed read of the slowest rate rose from 0.0195 to 0.4597 as dropout went from
    0 to 35%, a factor of 23, always in the same direction.

    What a hole CAN be filled with is what the signal itself says belongs there: ``x_t`` is
    ``A x_{t-1}``, so the operator standing on the observed cells predicts the missing ones, and
    re-reading the operator from the completed record gives a better predictor, to a fixed point.
    With the fill, the same read is 0.0195 at no dropout and 0.0202 at 35% -- the residual is the
    estimator's own finite-sample offset, which does not grow with the dropout.  This invents
    nothing on a record with no operator to speak of: on white noise the slowest rate stays fast
    (no persistent mode appears) at every dropout level.

    A record with nothing missing is returned unchanged, so a caller without gaps pays one
    ``isfinite`` scan and nothing else."""
    xp = _ns(W)
    miss = ~xp.isfinite(xp.abs(W))
    if not bool(miss.any()):
        return W                                   # nothing missing: one scan, then untouched
    # Backend-agnostic throughout: a record that arrived on a device goes back on it.
    Z = xp.where(miss, xp.zeros_like(W), W)
    scale = float(abs(to_numpy(xp.max(xp.abs(Z))))) or 1.0
    tol = scale * int(Z.shape[0]) * macheps(xp, Z)      # the arithmetic's own resolution
    for _ in range(int(iters)):
        L, R = Z[:-1], Z[1:]
        A = (R.T @ xp.conj(L)) @ _pinv_of(xp, L.T @ xp.conj(L))
        pred = xp.zeros_like(Z)
        pred[1:] = Z[:-1] @ A.T
        nxt = xp.where(miss, pred, Z)
        if float(abs(to_numpy(xp.max(xp.abs(nxt - Z))))) <= tol:
            return nxt
        Z = nxt
    return Z


def _pinv_of(xp, M):
    """Moore-Penrose pseudoinverse of a Hermitian Gram, on either backend."""
    return xp.linalg.pinv(M, hermitian=True)


# ── the reflection-positive moment pencil: the transfer spectrum of a scalar correlation
#    sequence (Prony / Hankel-DMD), the scalar-series companion to the covariance DMD above ──
@dataclass
class HankelSpectrum:
    """Transfer/Koopman eigenvalues read from a scalar correlation sequence's own moments.
    ``evals`` are sorted descending -- the leading one is the slowest, the SPECTRAL-gap mode.
    ("Gap" elsewhere in this library means a missing cell; here it is the spectral gap.)"""
    evals:     object   # transfer eigenvalues lambda_k of the moment pencil, descending
    isolation: float    # lambda_1 / lambda_2 -- how isolated the leading (spectral-gap) mode is
    psd:       float    # H0 conditioning (min/max eigenvalue): >= 0 iff PSD (0 = singular/rank-deficient)
    n:         int      # moment order (the pencil is (n+1) x (n+1))

    @property
    def leading(self) -> float:
        """The dominant transfer eigenvalue lambda_1 (the slowest, spectral-gap mode)."""
        e = np.asarray(self.evals)
        return float(e[0]) if e.size else float("nan")

    @property
    def rate(self) -> float:
        """The SPECTRAL gap = -log(lambda_1) (decay rate of the slowest mode); NaN if lambda_1
        not in (0,1).  Not a missing cell -- see ``carry_over_gaps`` for that sense."""
        lam = self.leading
        return -float(np.log(lam)) if 0.0 < lam < 1.0 else float("nan")


def hankel_spectrum(c, n: int, *, rcond: float | None = None) -> HankelSpectrum:
    """The transfer-operator spectrum of a real correlation sequence via the reflection-positive
    moment pencil (a.k.a. Prony / Hankel-DMD / matrix pencil).

    ``c`` is a correlation sequence ``c[tau] = <O(0) O(tau)>`` at integer lags ``tau = 0..K`` (real,
    an autocovariance).  Form the Hankel moment matrices

        H0[i,j] = c[i+j],   H1[i,j] = c[i+j+1]        (i, j = 0..n)

    and solve the symmetric generalized eigenproblem ``H1 v = lambda H0 v`` by whitening on H0's
    positive spectrum (its SPD square root): ``M = H0^{-1/2} H1 H0^{-1/2}``, ``lambda = eigvalsh(M)``.
    The ``lambda_k`` are the transfer/one-step-propagator eigenvalues; the leading ``lambda_1 =
    e^{-rate}`` is the slowest, spectral-gap mode.  H0 is a Gram / moment matrix, positive semidefinite by
    reflection positivity in the limit; ``psd`` (min/max H0 eigenvalue) reports how well finite
    statistics realise that.

    This is the scalar-sequence companion to the vector-covariance DMD of :class:`Dynamics`: the same
    forward transfer spectrum, read from a correlation sequence's own moments, with no state
    trajectory.  Domain-agnostic -- ``c`` is any real correlation sequence.

    Scanning ``n`` exposes the moment-order systematic: a well-isolated leading mode is stable in
    ``n`` while ``psd`` stays positive.  Report the band across ``n``; do not pick one favourable
    order.  (See :func:`jackknife` for an error bar, since the pencil has no closed-form interval.)
    ``rcond``: see :func:`matrix_pencil` -- by default H0's directions are cut where the record's own
    noise shows."""
    c = np.asarray(c, dtype=float).ravel()
    if n < 1:
        raise ValueError("moment order n must be >= 1")
    if c.size < 2 * n + 2:
        raise ValueError(f"need >= {2 * n + 2} correlation lags for moment order n={n}, got {c.size}")
    if c[0] != 0:
        c = c / c[0]                                   # normalise C(0)=1 (the pencil is ratio-invariant)
    idx = np.add.outer(np.arange(n + 1), np.arange(n + 1))
    mp = matrix_pencil(c[idx], c[idx + 1], rcond=rcond)
    if mp.evals.size == 0:
        return HankelSpectrum(evals=np.zeros(0), isolation=float("nan"), psd=0.0, n=n)
    ev = mp.evals
    iso = float(ev[0] / ev[1]) if ev.size > 1 and ev[1] > 0 else float("inf")
    return HankelSpectrum(evals=ev, isolation=iso, psd=mp.psd, n=n)


@dataclass
class MatrixPencil:
    """The generalised eigenproblem ``C1 v = lambda C0 v`` of a correlator matrix pair."""
    evals:   np.ndarray   # transfer eigenvalues, descending
    vectors: np.ndarray   # (n, k) generalised eigenvectors, columns in the order of evals
    psd:     float        # C0 conditioning, min/max eigenvalue: >= 0 iff PSD (< 0: lost positivity)


def matrix_pencil(C0, C1, *, rcond: float | None = None) -> MatrixPencil:
    """The generalised eigenproblem of a measured pair of correlator matrices, ``C1 v = lambda C0 v``
    -- the same construction as :func:`hankel_spectrum`, which calls this on its Hankel moments.
    Both matrices are read through their symmetric parts: a correlator matrix of a time-reversal
    symmetric measurement is symmetric, and finite statistics break that only by noise.  For a
    ``C1`` that is not symmetric the eigenvalues are those of ``(C1 + C1^T)/2`` against ``C0``, not
    of ``C1`` itself.  Real matrices only; a complex pair is refused.

    ``C0`` (the zero-step correlator matrix, symmetrised) is whitened on the part of its spectrum
    the measurement resolves.  ``C0`` is positive semidefinite in the limit, and finite statistics
    break that only by noise, so its most negative eigenvalue is the size of the noise the record
    itself shows: directions with eigenvalue at or below ``|min(w, 0)|`` -- or below round-off,
    ``max(w) n eps``, when ``C0`` is positive -- cannot be told from that noise and are dropped.
    Nothing is chosen: the cut is read off ``C0``.  A caller who knows the noise may pass ``rcond``
    and cut at ``rcond * max(w)`` instead.  With ``V_r = V_keep / sqrt(w_keep)``,
    ``M = V_r^T C1 V_r`` (symmetrised) is eigen-decomposed; the eigenvalues are the transfer
    eigenvalues, descending, and ``V_r y`` the generalised eigenvectors (a Ritz / variational vector
    per level).  ``psd`` is ``C0``'s min/max eigenvalue: negative when finite statistics have lost
    positivity, and the directions that carried it are the ones dropped."""
    if np.iscomplexobj(C0) or np.iscomplexobj(C1):
        raise ValueError("matrix_pencil reads real correlator matrices; got a complex one")
    C0 = np.asarray(C0, dtype=float)
    C1 = np.asarray(C1, dtype=float)
    if C0.ndim != 2 or C0.shape[0] != C0.shape[1] or C1.shape != C0.shape:
        raise ValueError(f"C0 and C1 must be square and the same shape; got {C0.shape}, {C1.shape}")
    w, V = np.linalg.eigh(0.5 * (C0 + C0.T))
    wmax = float(w.max()) if w.size else 0.0
    if rcond is None:
        cut = max(-min(float(w.min()), 0.0), wmax * w.size * np.finfo(float).eps) if w.size else 0.0
    else:
        cut = float(rcond) * wmax
    keep = w > cut
    if not bool(keep.any()):
        return MatrixPencil(evals=np.zeros(0), vectors=np.zeros((C0.shape[0], 0)), psd=0.0)
    Vr = V[:, keep] / np.sqrt(w[keep])
    M = Vr.T @ C1 @ Vr
    Ms = 0.5 * (M + M.T)
    ev = np.sort(np.linalg.eigvalsh(Ms))[::-1]             # the moment pencil's own eigenvalue path
    lam, Y = np.linalg.eigh(Ms)
    psd = float(w.min() / wmax) if wmax > 0 else 0.0
    return MatrixPencil(evals=ev, vectors=Vr @ Y[:, np.argsort(lam)[::-1]], psd=psd)


def jackknife(samples, read, *, n_bins: int | None = None, groups=None):
    """Delete-one(-group) jackknife point estimate and standard error of a ``read``.

    ``samples``: an ``(N, ...)`` array or length-``N`` sequence.  ``read``: ``callable(subset)``,
    evaluated on the full set and on each delete-one-group subset (the subset is passed in the same
    form as ``samples``, rows in their original order).  The groups are, in order of precedence:
    ``groups`` -- explicit index arrays (independent streams, Markov chains, blocks; unequal sizes and
    non-contiguous membership allowed; a sample in no group is never deleted); ``n_bins`` -- that
    many contiguous bins; neither -- delete-one-sample.  Returns ``(estimate_on_full, se)`` with

        se = sqrt( (G-1)/G * sum_g (theta_(g) - mean_g)^2 ),   G = number of groups.

    A scalar read returns floats.  A read returning an array is resampled element-wise, and
    ``estimate_on_full`` and ``se`` come back with its shape (a correlator profile gets one SE per
    lag).  A non-finite replicate makes that element's ``se`` non-finite; nothing is dropped.
    Domain-agnostic resampling for reads with no closed-form interval (e.g. :func:`hankel_spectrum`)."""
    is_arr = hasattr(samples, "shape")
    arr = samples if is_arr else list(samples)
    N = int(arr.shape[0]) if is_arr else len(arr)
    if N < 2:
        raise ValueError("jackknife needs >= 2 samples")
    if groups is not None and n_bins is not None:
        raise ValueError("jackknife takes groups or n_bins, not both")
    if groups is None:
        G = N if n_bins is None else min(int(n_bins), N)
        groups = np.array_split(np.arange(N), G)
    else:
        groups = [np.asarray(g, dtype=int).ravel() for g in groups]
        G = len(groups)
        if G < 2:
            raise ValueError("jackknife needs >= 2 groups")
        if any(g.size == 0 for g in groups):
            raise ValueError("jackknife groups must not be empty")
        members = np.concatenate(groups)
        if members.min() < 0 or members.max() >= N:
            raise ValueError("jackknife group index out of range")
        if np.unique(members).size != members.size:
            raise ValueError("jackknife groups must not share or repeat an index")

    def _sub(drop):
        keep = np.setdiff1d(np.arange(N), drop)
        return arr[keep] if is_arr else [arr[i] for i in keep]

    first = read(arr)
    if np.ndim(first) == 0:
        full = float(first)
        theta = np.array([float(read(_sub(g))) for g in groups])
        se = float(np.sqrt((G - 1) / G * np.sum((theta - theta.mean()) ** 2)))
        return full, se
    full = np.asarray(first)
    theta = np.stack([np.asarray(read(_sub(g))) for g in groups])
    se = np.sqrt((G - 1) / G * np.sum(np.abs(theta - theta.mean(axis=0)) ** 2, axis=0))
    return full, se


def bootstrap(samples, read, *, draws: int | None = None, rng=0, indices=None) -> np.ndarray:
    """Resampling with replacement: the replicates of ``read`` on resamples of ``samples``.

    ``samples``: an ``(N, ...)`` array or length-``N`` sequence (a list of planes works).  ``read``:
    ``callable(subset) -> float or array``, given each resample in the same form as ``samples``.
    ``draws``: the number of replicates -- the caller's choice; the library has no default.
    ``rng``: a ``numpy.random.Generator``, used as is so a caller's stream continues across calls,
    or a seed.  For ``b = 0 .. draws-1``, in order, ``idx = rng.integers(0, N, N)`` and
    ``theta_b = read(samples[idx])``: that draw pattern is part of the contract, so a caller's
    hand loop over the same stream gives the same replicates bit for bit.  ``indices``: prebuilt
    index arrays, one per replicate, for callers that share one set of draws across several reads;
    it replaces ``draws`` and ``rng``.

    Returns the replicates, shape ``(draws,)`` for a scalar read or ``(draws, *shape)`` for an array
    read.  The caller takes their spread or quantiles: the library supplies no level."""
    is_arr = hasattr(samples, "shape")
    arr = samples if is_arr else list(samples)
    N = int(arr.shape[0]) if is_arr else len(arr)
    if N < 1:
        raise ValueError("bootstrap needs at least one sample")
    if (draws is None) == (indices is None):
        raise ValueError("bootstrap takes exactly one of draws or indices")
    if indices is None:
        if int(draws) < 1:
            raise ValueError("bootstrap needs draws >= 1")
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        indices = (gen.integers(0, N, N) for _ in range(int(draws)))

    def _take(idx):
        return arr[idx] if is_arr else [arr[i] for i in idx]

    reps = [np.asarray(read(_take(np.asarray(idx)))) for idx in indices]
    if not reps:
        raise ValueError("bootstrap needs at least one replicate")
    return np.stack(reps)
