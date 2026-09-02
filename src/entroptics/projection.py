"""
projection.py -- the projection side of Entroptics.  A signal is read onto its
entropy-matched screen.  Read-only: the screen is a measurement of the signal
(its modes, the count above the floor, the per-mode footprints), never written
back out -- a clean view is a filter of the data (project onto the resolved
modes), never a synthesis.

Where ``aperture.Aperture`` reads the optics (the information *about* a structure,
read-only), ``Projection`` holds the projection (the information *within* it):
the signal folded onto its own entropy-matched screen, the screen's SVD modes,
the count of modes standing above the noise floor, and the per-mode footprints
those modes carry.

Standalone: numpy only.  Fully parameter-free: the scale comes from
``entropy.geometry`` and the noise floor from the derived Tracy-Widom edge (no
fitted or substrate-calibrated constant), a derived edge conditional on the
iid-Gaussian bulk null.  The per-mode ``footprints`` read exposes the shape of
each resolved mode (broadband vs localized) that the scalar floor cannot see.

    from entroptics import Projection
    sc = Projection(W)        # read the signal onto the screen
    sc.coherence          # is there ordered structure? (deterministic z-score)
    sc.K_signal           # how many modes stand above the noise floor

Per-axis geometry uses the _T (ordered) / _F (feature) subscripts:
  delta_T = ordered window width, delta_F = feature bin width, n_T / n_F mode counts.

Primitives
----------
  project(data, delta_T, delta_F) -- rescale both axes onto the screen (T,F)->(N,F_eff).
  noise_floor(S)           -- the derived Johnstone / Tracy-Widom singular-value noise floor.
  mode_significance(S)     -- per-mode evidence (TW1 deviate + p-value); K_signal = #(p<far).
  coherence(S)             -- ordered-axis coherence z-score (closed-form permutation null).
  footprints(U,S,Vt,K)     -- per-mode localization: fill of each resolved mode's vectors.
  beam                     -- the projection as a Beam (its footprints are its modes).
  read(W) / ProjectionRead     -- the full read as a plain dataclass.
  Projection                   -- the same read as an object.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import environment as _env
from .entropy import (geometry, normalize, downsample, upsample, shannon_bits, fold_width,
                      MAD_SCALE, MAD_LOGVAR, _shrink_mad)
# The noise floor is a caller-suppliable null provider (a FloorContext -> float callback);
# null_providers ships the derived defaults (mp/robust) + the plumbing.  The shared
# primitives are re-exported here under their private names (``_tw1_sf`` etc.) so a caller
# that imports them from ``projection`` directly keeps working unchanged.
from .beam import Beam
from .null_providers import (                          # noqa: F401 (re-exported)
    _TW1_UPPER_Q, johnstone as _johnstone, tw1_quantile as _tw1_quantile,
    tw1_sf as _tw1_sf, noise_sigma2 as _noise_sigma2, apply_floor as _apply_floor,
    debias_denominator as _debias_denominator, screen_floor_sq as _screen_floor_sq,
)


# ══════════════════════════════════════════════════════════════════════════════
# Projection projection + noise floor + coherence
# ══════════════════════════════════════════════════════════════════════════════

def _fold_axis(A: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Down/upsample one axis, excluding missing (NaN) cells from the average so a
    masked cell never leaks a zero into a fold.  A coarsened cell is the mean over
    its valid constituents (weighted by valid area); a cell with no valid data
    stays NaN (finalised to 0 by ``project``).  Backend-agnostic (numpy or torch)."""
    xp = _env.ns(A)
    n_in = int(A.shape[axis])
    if n_out == n_in:
        return A
    if n_out > n_in:
        return upsample(A, n_out, axis)                      # refine (hold)
    valid = xp.isfinite(A)
    if bool(valid.all()):
        return downsample(A, n_out, axis)                    # no missing -> plain area-mean
    Az = xp.where(valid, A, A * 0)
    num = downsample(Az, n_out, axis)                        # area-mean of valid values (zeros elsewhere)
    den = downsample(_env.asnum(valid), n_out, axis)          # valid area fraction
    return num / den                                         # mean over valid cells; all-missing -> NaN


def project(data: np.ndarray, delta_T: float, delta_F: float) -> np.ndarray:
    """Fold a whitened waterfall onto the screen -- rescale both axes to the
    entropy-matched grid in one call: (T, F) -> (N = round(T/delta_T),
    F_eff = round(F/delta_F)).

    Each axis independently downsamples (delta > 1 -> coarsen) or upsamples
    (delta < 1 -> refine).  Both axes are scaled here (``normalize`` only whitens),
    so normalization and rescaling stay two clean, orthogonal steps.  Missing
    (NaN) cells from a mask are excluded from the fold average; a screen cell with
    no valid data left is mean-imputed (0) -- the minimum the SVD requires."""
    T, F = data.shape
    N = max(1, int(round(T / delta_T)))
    F_eff = max(1, int(round(F / delta_F)))
    out = _fold_axis(data, N, axis=0)
    out = _fold_axis(out, F_eff, axis=1)
    xp = _env.ns(out)
    out = xp.where(xp.isfinite(xp.abs(out)), out, xp.zeros_like(out))    # all-missing cell -> 0
    return _env.as_compute(xp, out)   # pin the screen to the set precision (float64 default => no-op)


def noise_floor(screen: np.ndarray, *, far: float = 0.05, null=None,
                s: np.ndarray | None = None, seed: int = 0) -> float:
    """Singular-value noise floor: the level above which a singular value is
    structure, not noise -- the scalar a null provider returns for this screen.
    ``K_signal = #(S > floor)``.

    ``null`` is a null-provider callback ``FloorContext -> float`` (see
    :mod:`null_providers`); ``None`` uses the derived default ``mp`` (the finite-size
    Johnstone / Tracy-Widom edge: ``sqrt(sigma^2*(mu + q*sigma_J))`` with the de-biased
    per-cell variance ``sigma^2`` and the universal TW1 quantile ``q`` for ``far``).  Pass
    ``null_providers.robust``, ``null_providers.permutation()``, or your own provider
    (a local reference / physics null) to score the floor differently -- it is evaluated
    on this screen, so under a per-window / streaming read it is local, not global.  ``s``
    is the precomputed singular spectrum (reused where a provider needs it; ``mp`` does
    not); resampling providers are deterministic per ``seed``.  Backend-agnostic."""
    if len(screen.shape) != 2 or int(screen.shape[0]) * int(screen.shape[1]) == 0:
        return float("inf")
    N, F = int(screen.shape[0]), live_columns(screen)
    if N <= 0 or F <= 0:
        return float("inf")
    return _apply_floor(null, spectrum=s, data=screen, shape=(N, F),
                        far=far, kind="projection", seed=seed)


def live_columns(screen) -> int:
    """The number of feature channels the floor's null is sized by: columns that are not
    identically zero.

    Not ``screen.shape[1]``: the floor is the edge of an N x F noise ensemble, so F decides
    where it sits, and a column of exact zeros is not a channel of that ensemble. Two routes
    produce them, both upstream in this module: :func:`project` mean-imputes an all-missing
    screen cell to ``0``, and :func:`normalize` returns a channel it could not scale (no spread) as zeros. Either way the column carries no observation.

    Counting them biases the floor in two opposing directions at once, which is worse than
    either alone because the errors do not announce themselves by cancelling: the de-biased
    per-cell variance divides the median row energy by a denominator in F, so dead columns add
    no energy but do add F and ``sigma^2`` comes out too small (the floor sinks, noise reads as
    signal), while ``johnstone(N, F)`` returns the edge of a wider ensemble than was measured
    (the floor lifts, signal reads as noise).

    Floored at 1 so the Johnstone shape stays non-degenerate; a screen with nothing live is a
    collapse for a caller to detect, not a width to invent."""
    Z = np.asarray(_env.to_numpy(screen))
    if Z.ndim != 2 or Z.shape[1] == 0:
        return int(Z.shape[1]) if Z.ndim == 2 else 0
    return max(1, int(np.count_nonzero(np.any(np.abs(Z) != 0.0, axis=0))))


@dataclass
class ModeSignificance:
    """Per-singular-value evidence against the noise null, carrying no threshold: the
    standardized Tracy-Widom deviate and the tail probability of every singular value.
    The resolved dimension at any false-alarm level ``far`` is ``#(pvalue < far)`` -- the
    read reports the evidence, the reader supplies the decision level."""
    deviate: np.ndarray   # g_k = (s_k^2/sigma^2 - mu)/sigma_J, the standardized TW1 deviate
    pvalue:  np.ndarray   # p_k = P(TW1 > g_k), the upper-tail probability (Chiani approx)


def mode_significance(screen: np.ndarray, s: np.ndarray | None = None) -> ModeSignificance:
    """Per-mode significance of the screen's singular spectrum against the derived noise
    null, free of any threshold: for each singular value ``s_k`` the standardized
    Tracy-Widom deviate ``g_k = (s_k^2/sigma^2 - mu)/sigma_J`` and its tail probability
    ``p_k = P(TW1 > g_k)``.  This is the evidence the noise floor thresholds: the resolved
    count ``K_signal`` equals ``#(p_k < far)``, so the false-alarm level ``far`` is applied
    by the reader, not baked into the read.  The identity ``K_signal == #(p_k < far)`` is
    exact when the floor quantile is inverted from this same survival function (any non-
    tabulated ``far``); at the tabulated levels (e.g. ``far=0.05``, ``q=0.9793``) the floor
    uses the exact table quantile while ``p_k`` uses the Chiani approximation, so a mode whose
    ``g_k`` sits within the ~7e-3 approximation gap of the quantile can differ by one count.
    Deterministic; numpy-only."""
    xp = _env.ns(screen)
    # Same width the floor is sized by (`live_columns`) -- the documented identity
    # `K_signal == #(p_k < far)` is between these two reads, so they must agree on F or the
    # evidence and the threshold are computed against different ensembles.
    N, F = int(screen.shape[0]), live_columns(screen)
    sv = xp.linalg.svd(screen, compute_uv=False) if s is None else s
    sv = np.asarray(_env.to_numpy(sv), dtype=float)
    if N <= 0 or F <= 0 or sv.size == 0:
        empty = sv[:0].copy()
        return ModeSignificance(deviate=empty, pvalue=empty)
    mu, sig_J = _johnstone(N, F)
    sigma2 = _noise_sigma2(xp, screen, N, F)
    g = (sv ** 2 / sigma2 - mu) / sig_J
    p = np.array([_tw1_sf(float(gk)) for gk in g])
    return ModeSignificance(deviate=g, pvalue=p)


def coherence(screen: np.ndarray, lag: int = 1) -> float:
    """Ordered-axis coherence of the screen at ``lag`` -- a deterministic z-score
    against the exact row-permutation null (closed form; no sampling, no RNG).

    The statistic is A = mean_i Re<row_i, row_{i+lag}>^2: how alike, on average, are
    rows ``lag`` apart (squared inner products, so a common row scale cancels and A
    is scale-invariant).  Under a uniformly random row permutation each compared pair
    becomes a uniformly random pair of distinct rows, so E_pi[A] = mu, the mean of R =
    Re(S Sᴴ)^2 over ordered off-diagonal pairs (Theorem 5.2, exact).  The z-score
    standardises A by its exact permutation standard deviation:

        coherence = (A - mu) / sqrt(Var_pi[A])

    Var_pi[A] is the exact Cliff-Ord / Mantel second moment: the M = N - lag
    superdiagonal terms are not independent (consecutive terms at lag=1 share a row),
    so a naive var/(N-lag) mis-standardises.  It is assembled in closed form from the
    graph moments of R -- S1 = sum R, S2 = sum R^2, U = sum of squared row-sums (all
    over off-diagonal pairs) -- via the single-term variance and the two-term
    expectations for pairs that share one index (E_share) or are disjoint (E_disj):

        E_share = (U - S2) / (N(N-1)(N-2))
        E_disj  = (S1^2 - 4U + 2S2) / (N(N-1)(N-2)(N-3))
        Var[A]  = [ M(mu2 - mu^2) + n_share(E_share - mu^2)
                                  + n_disj (E_disj  - mu^2) ] / M^2

    with mu2 = S2/(N(N-1)), n_share = 2 max(0, N-2 lag) sharing ordered pairs and
    n_disj = M(M-1) - n_share disjoint ones.  This gives a z with exactly unit
    permutation variance (validated against brute-force permutation), so the null is
    calibrated (mean 0, sd 1) across shapes.

    Returns a z-score: > 0 = rows ``lag`` apart are more alike than a random
    reordering (ordered structure); ~ 0 = indistinguishable; < 0 = anti-ordered.
    Deterministic -> bit-identical on numpy and torch.  Backend-agnostic; O(N^2 F)
    via one Gram matmul (cheaper than the screen SVD already computed).
    """
    xp = _env.ns(screen)
    N = int(screen.shape[0])
    if N < 2 * lag + 2 or lag < 1 or N < 4:
        return 0.0
    G = screen @ xp.conj(screen).T                 # (N, N) Hermitian row Gram
    R = xp.real(G) ** 2                            # squared real inner products (symmetric)
    d = xp.diagonal(R)                             # main diagonal = ||row||^4
    S1 = float(_env.sum_ax(xp, R)) - float(_env.sum_ax(xp, d))            # sum over off-diag pairs
    S2 = float(_env.sum_ax(xp, R * R)) - float(_env.sum_ax(xp, d * d))    # sum of squares, off-diag
    rowsum = _env.sum_ax(xp, R, 1) - d             # per-row sum excluding the diagonal (u_a)
    U = float(_env.sum_ax(xp, rowsum * rowsum))    # sum of squared off-diagonal row sums
    Dp = N * (N - 1)
    mu = S1 / Dp                                    # exact null mean (Theorem 5.2)
    mu2 = S2 / Dp
    mu_sq = mu * mu
    E_share = (U - S2) / (N * (N - 1) * (N - 2))
    E_disj = (S1 * S1 - 4.0 * U + 2.0 * S2) / (N * (N - 1) * (N - 2) * (N - 3))
    M = N - lag
    n_share = 2 * max(0, N - 2 * lag)              # ordered pairs of terms sharing one index
    n_disj = M * (M - 1) - n_share
    var_A = (M * (mu2 - mu_sq)
             + n_share * (E_share - mu_sq)
             + n_disj * (E_disj - mu_sq)) / (M * M)
    if var_A < 1e-24:
        return 0.0
    A = float(_env.sum_ax(xp, xp.diagonal(R, lag))) / float(M)   # lag-th superdiagonal mean
    return (A - mu) / math.sqrt(var_A)


# ══════════════════════════════════════════════════════════════════════════════
# Per-mode localization -- the footprint (shape) of each resolved mode
# ══════════════════════════════════════════════════════════════════════════════

def _vector_fill(xp, v) -> float:
    """Fill fraction of a (unit) vector: ``2^H(|v|^2) / len(v)`` in (0,1] -- 1 when
    the mode is spread uniformly over the axis, -> 1/len when it localizes on a
    single coordinate.  The entropic fill of section 3 applied to one mode vector."""
    n = int(v.shape[0])
    if n == 0:
        return 0.0
    return 2.0 ** shannon_bits(xp.abs(v) ** 2) / n


def footprints(U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
               k_signal: int) -> list[Beam]:
    """Per-mode read: each of the ``k_signal`` modes standing above the noise floor, as a leaf
    :class:`beam.Beam`.

    A footprint is an extracted signal.  Each mode carries its ordered-axis amplitude
    (``profile``, the left singular vector scaled by its singular value), its feature-axis
    direction (``basis``, the right singular vector), the fill fraction of each (``phi_T``,
    ``phi_F``) and their product (``etendue``, the phase-space area it occupies).  Because it
    is a beam, ``frame`` lays it back out as the rank-1 signal it is, so a single mode can be
    filtered off and passed along.  Summing the modes' frames reconstructs the resolved sector.

    The fills read the shape the scalar floor is blind to -- a broadband transient
    (``phi_F ~ 1``, ``phi_T`` small) against narrowband persistent structure (``phi_F`` small,
    ``phi_T ~ 1``) at the same singular value.  Deterministic; backend-agnostic."""
    xp = _env.ns(S)
    out: list[Beam] = []
    for k in range(int(k_signal)):
        pt = _vector_fill(xp, U[:, k])
        pf = _vector_fill(xp, Vt[k])
        amp = np.asarray(_env.to_numpy(U[:, k])) * float(S[k])
        out.append(Beam(lens="", index=k, energy=float(S[k]) ** 2, flow=np.abs(amp) ** 2,
                        basis=np.asarray(_env.to_numpy(Vt[k])).reshape(-1, 1),
                        profile=amp.reshape(-1, 1), _fills=(pt, pf), _modes=[]))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# The full read -- as a dataclass (read) and as an object (Projection)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectionRead:
    """The full screen read as a plain record (output of ``read``)."""
    delta_T:     float   # ordered window width (entropy-matched scale)
    delta_F:     float   # feature bin width (entropy-matched scale)
    coherence:   float   # ordered-axis coherence z-score (>~2 => significant structure)
    K_signal:    int     # SVD modes above the noise floor (the resolved sector)
    H_screen:    float   # Shannon entropy (bits) of the signal-mode weights
    sigma_top:   float   # top singular value of the screen
    noise_floor: float   # the singular-value noise floor


class Projection:
    """The projection of a signal onto its entropy-matched screen.

    Construct from a 2-D array ``W`` (T, F): axis-0 the ordered axis, axis-1 the
    feature axis.  On construction the signal's own entropy geometry sets the
    scale, the waterfall is folded onto the screen, and the screen's SVD structure
    is read.  Read-only: there is no write-back -- a denoised view is a filter of
    the data (project onto the resolved modes), computed by the caller.

    Attributes
    ----------
    W, mask                   : the input and its bad-cell mask
    H_T, H_F                  : per-axis Shannon entropy (bits)
    n_T, n_F                  : per-axis effective mode counts
    delta_T, delta_F          : per-axis matched cell scales (window / bin width)
    screen                    : (N, F_eff) the projected screen tensor
    U, S, Vt                  : SVD factors of the screen (the modes within)
    noise_floor               : the singular-value noise floor
    sigma_top                 : top singular value
    K_signal                  : # singular values above the noise floor (resolved modes)
    H_screen                  : Shannon entropy (bits) of the signal-mode weights
    coherence                 : ordered-axis coherence z-score (closed-form permutation null)
    beam                      : the projection AS a Beam (its footprints are its modes)
    """

    def __init__(self, W, mask=None, *, far: float = 0.05, null=None, seed: int = 0):
        xp = _env.ns(W)
        if len(getattr(W, "shape", ())) != 2:
            raise ValueError(f"Projection expects a 2-D array (T, F); got {getattr(W, 'shape', None)}")
        if min(int(v) for v in W.shape) == 0:
            raise ValueError(f"Projection expects a screen with both axes non-empty; got "
                             f"{tuple(int(v) for v in W.shape)}. An axis of length 0 carries no cell "
                             f"to read, which is not the same as an axis whose cells are all missing "
                             f"-- pass those as NaN or behind a mask.")
        nan = ~xp.isfinite(xp.abs(W))
        if mask is not None or bool(nan.any()):
            # masked / gapped -> numpy path (robust normalize); rare on the GPU stream.
            W, xp = _env.to_numpy(W), np
            nan = ~np.isfinite(np.abs(W))
            if nan.any():
                mask = (mask | nan) if mask is not None else nan
        self.W = W
        self.mask = mask
        self.T, self.F = int(W.shape[0]), int(W.shape[1])

        # Fully-dead rows/cols (every cell missing) carry no information: they are ignored
        # (dropped); imputing zeros -- zeros inject a false "flat" block of
        # structure and inflate the aspect ratio, distorting the SVD / noise floor.
        # Scattered missing cells stay and are handled by the fold's valid-cell mean;
        # the dropped lines are simply excluded from the read.
        self._live_rows = self._live_cols = None
        if mask is not None:
            lr = ~np.all(mask, axis=1); lc = ~np.all(mask, axis=0)
            if lr.any() and lc.any() and not (lr.all() and lc.all()):
                self._live_rows, self._live_cols = lr, lc
                W = W[np.ix_(lr, lc)]
                mask = mask[np.ix_(lr, lc)]
                if not mask.any():
                    mask = None

        geom = geometry(W, mask)
        self.H_T = geom["H_T"]
        self.H_F = geom["H_F"]
        self.n_T = geom["n_T"]
        self.n_F = geom["n_F"]
        self.delta_T = geom["delta_T"]
        self.delta_F = geom["delta_F"]
        self._geom = geom

        data = normalize(W, mask)
        self.screen = project(data, self.delta_T, self.delta_F)

        self.null = null
        self._xp = xp; self._far = float(far); self._seed = int(seed)
        self._U = self._Vt = None      # full SVD basis -- lazy (heavy; only beam / footprints)
        self._coh = None               # coherence -- lazy (a separate lag-1 pass)
        # Lightweight signal decision: only the singular values drive the noise floor and
        # K_signal, so use ``svdvals`` (no U / Vt). The full SVD basis, the beam, and
        # the coherence are heavy and deferred to first access -- a streaming monitor that
        # reads K_signal / has_signal never pays for them (the high-speed capture path).
        self.S = _env.svdvals(xp, self.screen)
        self.noise_floor = noise_floor(self.screen, far=far, null=null, s=self.S, seed=seed)
        self.sigma_top = float(self.S[0]) if int(self.S.shape[0]) else 0.0
        sig = self.S[self.S > self.noise_floor]
        self.K_signal = int(sig.shape[0])
        self.H_screen = shannon_bits(sig ** 2) if int(sig.shape[0]) else 0.0

    def refloor(self, null):
        """This same projection, floored by a different ``null`` provider.

        ``null`` reaches exactly one line of ``__init__`` -- ``noise_floor(..., s=self.S)`` -- and
        that line is handed the spectrum rather than the data. So the provider chooses WHERE the
        floor sits in an already-computed spectrum; it does not change the decomposition, the
        screen, or the normalization that produced either. Rebuilding a whole ``Projection`` to
        change it recomputes an identical ``svdvals``.

        ⚑ Measured 2026-08-27 on a 64-patch sweep at ``patch=256``: ``sweep(null="local")`` built
        80 projections for 64 patches -- one per patch, plus a second for each of the 16 that
        passed the coherence gate -- and SVD was 66% of the run (0.855s of 1.295s). The second
        projection of a coherent patch is that redundancy, and it is what this removes.

        WHAT IS SHARED, and why each is safe to share:

          * ``screen``, ``S``, ``sigma_top``, ``delta_T/F``, ``_geom`` -- all produced upstream of
            the floor, from the data alone.
          * ``_U`` / ``_Vt`` -- the full basis is a factorization of ``screen``; the floor only
            decides how many of its modes count as resolved.
          * ``_coh`` -- ``coherence`` is a lag-1 pass over ``screen`` and its own docstring says it
            is "not part of the K_signal signal decision".

        WHAT IS RECOMPUTED: ``noise_floor`` and the two reads that stand on it, ``K_signal`` and
        ``H_screen``. ``footprints`` and ``beam`` are properties that re-read ``K_signal`` at
        access, so they follow with nothing to invalidate.

        This is an identity, not an approximation: ``p.refloor(n)`` reads the same as
        ``Projection(W, mask, far=..., null=n, seed=...)`` on every public field, and
        ``tests/test_projection_refloor.py`` asserts exactly that rather than trusting this
        paragraph.
        """
        other = object.__new__(type(self))
        other.__dict__.update(self.__dict__)
        other.null = null
        other.noise_floor = noise_floor(self.screen, far=self._far, null=null,
                                        s=self.S, seed=self._seed)
        sig = self.S[self.S > other.noise_floor]
        other.K_signal = int(sig.shape[0])
        other.H_screen = shannon_bits(sig ** 2) if int(sig.shape[0]) else 0.0
        return other

    # ── heavy outputs: lazy (first access only; the monitoring path never touches them) ──
    def _basis(self):
        """Full SVD basis (U, Vt) -- computed once, on demand (beam / footprints only)."""
        if self._U is None:
            self._U, _, self._Vt = self._xp.linalg.svd(self.screen, full_matrices=False)
        return self._U, self._Vt

    @property
    def U(self):
        """Left singular vectors of the screen (ordered-axis modes); lazy."""
        return self._basis()[0]

    @property
    def Vt(self):
        """Right singular vectors of the screen (feature-axis modes, row-wise); lazy."""
        return self._basis()[1]

    @property
    def has_signal(self) -> bool:
        """True iff the top singular value clears the noise floor (`K_signal > 0`) -- the cheap
        monitor read. For an SVD-free gate on a raw frame (before building a Projection), use the
        module-level `probe_signal(W, ...)`."""
        return self.sigma_top > self.noise_floor

    @property
    def coherence(self) -> float:
        """Ordered-axis coherence z-score (lag-1 permutation null). Lazy -- a separate pass, not
        part of the K_signal signal decision."""
        if self._coh is None:
            self._coh = coherence(self.screen, lag=1)
        return self._coh

    @property
    def N(self) -> int:
        """Ordered-axis length of the screen (rows)."""
        return self.screen.shape[0]

    @property
    def F_eff(self) -> int:
        """Folded feature-axis length of the screen (columns)."""
        return self.screen.shape[1]

    def _screen_fills(self):
        """The projected screen's per-axis fills; ``reads`` imports this module, so the import
        is deferred to the call."""
        from .reads import phi_T, phi_F
        return float(phi_T(self.screen)), float(phi_F(self.screen))

    @property
    def beam(self) -> Beam:
        """This projection AS a :class:`beam.Beam` -- the resolved modes it carries, with its
        footprints as that beam's ``modes``.

        The projection's factorization and a beam are the same pair under two names: amplitude
        on the modes (``profile``) times the mode directions (``basis``).  Reading it as a beam
        gives one type for "what a signal carries", floor-truncated like every other resolved
        read, and it composes with everything a beam composes with -- ``frame`` lays it back
        out, and a mode splits off and places on a screen."""
        k = int(self.K_signal)
        U, Vt = self._basis()
        Un = np.asarray(_env.to_numpy(U))[:, :k]
        Sn = np.asarray(_env.to_numpy(self.S))[:k]
        prof = Un * Sn
        return Beam(lens="", index=-1, energy=float((Sn ** 2).sum()),
                    flow=np.abs(prof).sum(axis=1) ** 2 if k else np.zeros(int(self.N)),
                    basis=np.asarray(_env.to_numpy(Vt))[:k].T,
                    profile=prof,
                    _fills=self._screen_fills,
                    _modes=lambda: self.footprints)

    @property
    def footprints(self) -> list[Beam]:
        """Per-mode localization fingerprints for the ``K_signal`` resolved modes:
        each mode's ordered/feature fill fractions and per-mode etendue (see
        :func:`footprints`).  Empty when ``K_signal == 0``.  Reads the shape of
        each mode above the floor -- broadband signal vs a localized (narrowband or
        compact) blob -- that ``K_signal`` alone cannot distinguish."""
        return footprints(self.U, self.S, self.Vt, self.K_signal)

    @property
    def significance(self) -> ModeSignificance:
        """Per-mode evidence against the noise null (see :func:`mode_significance`):
        the standardized Tracy-Widom deviate and tail probability ``p_k`` of every
        singular value.  With the default ``mp`` provider, ``K_signal == #(p_k < far)``
        (exact for a non-tabulated ``far`` inverted from the same survival function; at a
        tabulated ``far`` it can differ by one within the Chiani approximation gap -- see
        ``mode_significance``) -- the read exposes the evidence, the false-alarm level is the caller's.
        (The p-values are always the analytic ``mp`` evidence; under a different ``null``
        provider ``K_signal`` follows that provider instead, so the identity is specific
        to the ``mp`` default.)"""
        return mode_significance(self.screen, self.S)

    def read(self) -> ProjectionRead:
        """The full read as a plain :class:`ProjectionRead` record."""
        return ProjectionRead(delta_T=self.delta_T, delta_F=self.delta_F,
                          coherence=self.coherence, K_signal=self.K_signal,
                          H_screen=self.H_screen, sigma_top=self.sigma_top,
                          noise_floor=self.noise_floor)

    def tensor(self, d: int | None = None, *, rank: tuple | None = None) -> dict:
        """Delay-embedded Tucker (HOSVD) of this signal at native resolution -- the
        within-window fine structure the averaged screen SVD loses.  ``d`` is the
        delay-window width.  See tensor.tensor_embed."""
        from .tensor import tensor_read    # deferred import (tensor path is optional)
        return tensor_read(self.W, self.mask, d, rank=rank)

    def aperture(self):
        """The companion :class:`aperture.Aperture` for the same signal -- the
        optics of this projection."""
        from .aperture import Aperture       # deferred import (avoids a cycle)
        return Aperture(self.W, self.mask)

    def __repr__(self) -> str:
        return (f"Projection(shape={self.W.shape}, N={self.N}, F_eff={self.F_eff}, "
                f"K_signal={self.K_signal}, coherence={self.coherence:.2f})")


def read(W: np.ndarray, mask: np.ndarray | None = None, *, far: float = 0.05,
         null=None, seed: int = 0) -> ProjectionRead:
    """Read a signal onto the screen and return the full :class:`ProjectionRead`:
    geometry -> normalize -> project -> coherence -> noise-floor modes.  Fully
    parameter-free by default: the entropy-matched fold and the derived ``mp`` noise floor
    carry no fitted constant; ``far`` is the floor's significance level (default 5%).
    ``null`` is a null-provider callback selecting the noise floor (``None`` = derived
    default ``mp``; see :func:`noise_floor` and :mod:`null_providers`).  Deterministic (a
    resampling provider is deterministic per ``seed``).  (= ``Projection(W, ...).read()``.)"""
    return Projection(W, mask=mask, far=far, null=null, seed=seed).read()


def _sigma_top_upper(screen) -> float:
    """A rigorous, SVD-free upper bound on the top singular value of ``screen``:
    ``sqrt(||screen||_1 * ||screen||_inf)`` (max column abs-sum times max row abs-sum) -- the
    spectral norm is bounded by the geometric mean of the 1- and inf- operator norms.  O(NF)."""
    xp = _env.ns(screen)
    A = xp.abs(screen)
    col = float(A.sum(axis=0).max()) if int(screen.shape[1]) else 0.0
    row = float(A.sum(axis=1).max()) if int(screen.shape[0]) else 0.0
    return math.sqrt(col * row)


def probe_signal(W, mask=None, *, far: float = 0.05, null=None, seed: int = 0) -> bool:
    """SVD-free signal gate for high-speed / streaming capture.  Builds the entropy-matched fold
    and the noise floor (both O(NF); no SVD for the derived / reference nulls), then tests a
    rigorous upper bound on the screen's top singular value against the floor.  Returns True iff
    signal may be present -- so a monitor can decide whether to build the full :class:`Projection`
    (and pay the SVD / embedding / coherence) without one.  Conservative: never False when
    ``Projection(W, ...).K_signal > 0`` (the bound only over-estimates the true top singular value),
    so gating on it cannot drop a real detection.  A masked / gapped frame returns True (defer to
    the full Projection).  ``null`` scores the floor as in :func:`noise_floor`; a sampled provider
    that needs the spectrum computes one internally (no saving for those)."""
    xp = _env.ns(W)
    if mask is not None or not bool(xp.all(xp.isfinite(xp.abs(W)))):
        return True                                    # masked / gapped -> defer to the full Projection
    geom = geometry(W, mask)
    screen = project(normalize(W, mask), geom["delta_T"], geom["delta_F"])
    floor = noise_floor(screen, far=far, null=null, s=None, seed=seed)
    return _sigma_top_upper(screen) > floor


# ══════════════════════════════════════════════════════════════════════════════
# Batched monitor path -- the throughput lever for ensembles (numpy / CPU)
# ══════════════════════════════════════════════════════════════════════════════
# The per-frame Python loop (wrapper plane reduction, ensemble traces) pays call + object
# overhead per frame and vectorizes nothing.  ``read_batch`` folds + svd-values + floors a
# stack of same-shape frames in one vectorized pass -- bit-identical to reading each frame
# with ``Projection`` (the fold is per-column, the per-channel medians are per-frame, so batching
# only changes the loop nesting, never a float).  It is the small-F ensemble lever the GPU
# cannot provide (cuSOLVER has no occupancy on many tiny SVDs).  numpy only: frames that are
# masked / non-finite / complex / a different shape fall back to the per-frame ``Projection``.

@dataclass
class BatchRead:
    """One frame's lightweight monitor read from :func:`read_batch` -- the same
    ``K_signal`` / ``sigma_top`` / ``noise_floor`` / singular values ``S`` a per-frame
    :class:`Projection` produces (bit-identical)."""
    K_signal:    int
    sigma_top:   float
    noise_floor: float
    S:           np.ndarray


# ══════════════════════════════════════════════════════════════════════════════
# THE BATCHED-READ CONTRACT
# ══════════════════════════════════════════════════════════════════════════════
# `fold_target_batch` / `normalize_batch` / `project_batch` are the three steps a per-frame
# `Projection` takes -- decide the fold, whiten, resample onto the matched grid -- exposed as a
# stack-at-a-time contract.  They are PUBLIC because `batch.py` is built on them: it is the one
# consumer that does not construct a `Projection`, so these are the surface that keeps the batched
# read identical to the per-frame one.
#
# A change here changes `resolved_batch` silently.  That is not hypothetical: `batch.py` once put
# its own fold gate in front of `fold_target_batch`, and the default batched read stopped agreeing
# with `Projection` on ~1 frame in 240 while every test still passed.  Anything that reads a stack
# goes through these three and nothing else.

def fold_target_batch(xp, stack, *, far: float = 0.05) -> np.ndarray:
    """Per-frame feature-fold target ``F_eff`` (== ``geometry``'s ``n_F``) for a real, finite
    ``(B, N, F)`` stack -- ``entropy.geometry``'s ``H_F`` + fold guard, batched.  Backend-agnostic
    (the reduction runs on ``xp``; the fold target is materialised to a numpy int array for the
    per-group control flow).  ``H_F`` is read per frame via the same ``shannon_bits`` ``geometry``
    uses, so the fold width matches the per-frame ``Projection`` exactly on either backend."""
    B, N, F = int(stack.shape[0]), int(stack.shape[1]), int(stack.shape[2])
    P = _env.asnum(xp.abs(stack)); P = P * P                       # |W|^2 (compute precision)
    marg = _env.sum_ax(xp, P, 1)                                   # (B, F) feature power marginal
    tot = _env.sum_ax(xp, marg, 1)                                 # (B,) on xp
    # Batched shannon_bits over the feature marginal (no per-frame loop -- a per-frame call would
    # serialise B tiny reductions on the GPU).  Same formula as ``entropy.shannon_bits``:
    # H = -sum p log2 p, p = marg / sum(marg), with the p>0 guard and [1e-12,1] clip in the log.
    safe_tot = _env.clampmin(xp, tot, 1e-30)
    p = marg / safe_tot[:, None]
    plog = xp.where(p > 0, p * xp.log2(_env.cliprange(xp, p, 1e-12, 1.0)), xp.zeros_like(p))
    H = -_env.sum_ax(xp, plog, 1)                                  # (B,) on xp
    log2F = float(np.log2(F))
    tot_np = np.asarray(_env.to_numpy(tot))
    H_F = np.where(tot_np > 1e-30, np.asarray(_env.to_numpy(H)), log2F)   # P_total<=0 -> maximal entropy
    # The same `fold_width` the per-frame `geometry` calls -- including its continuity gate --
    # so the batched read stays bit-identical to reading each frame with `Projection`.  Only the
    # H_F reduction is batched; the decision itself is per frame because continuity is.
    F_eff = np.empty(B, dtype=int)
    for b in range(B):
        F_eff[b] = fold_width(float(H_F[b]), N, F, stack[b], far=far)[0]
    return F_eff


def normalize_batch(xp, stack):
    """Per-channel robust MAD whiten of a real, finite ``(B, N, F)`` stack -- ``entropy.normalize``'s
    clean path over axis 1, fully batched (no per-frame Python loop) for the common case where every
    channel is present, bit-identical on numpy to the per-frame ``normalize``.  The James-Stein
    ``_shrink_mad`` reduces exactly to a batched form when ``pos`` is all-True per frame (full median
    / population log-MAD variance); only frames that actually have a dead channel (below the resolution floor)
    fall back to the per-frame ``_shrink_mad``.  So a clean GPU stack pays no per-frame launches."""
    from .entropy import resolution_floor
    data = _env.asnum(stack)
    med = _env.median_ax(xp, data, 1, keep=True)                 # (B,1,F) per-channel median over N
    centered = data - med
    mad = _env.median_ax(xp, xp.abs(centered), 1) * MAD_SCALE     # (B, F)
    B, F, N = int(mad.shape[0]), int(mad.shape[1]), int(data.shape[1])
    typical = _env.median_ax(xp, mad, 1)                         # (B,) full per-frame median
    pos = mad > resolution_floor(xp, typical, mad)[:, None]      # (B, F) -- see mad_stats
    # ── batched shrink (exact where every channel is positive) ──
    lm = xp.log(xp.where(pos, mad, xp.ones_like(mad)))           # (B, F) log-MAD where there is one
    lm0 = xp.log(xp.where(typical > 0.0, typical, xp.ones_like(typical)))    # (B,)
    V_obs = _env.std_ax(xp, lm, 1) ** 2                          # (B,) population var of log-MAD
    ratio = (MAD_LOGVAR / max(N, 1)) / _env.clampmin(xp, V_obs, 1e-300)
    w = _env.cliprange(xp, 1.0 - ratio, 0.0, 1.0)
    w = xp.where(V_obs > 0.0, w, xp.zeros_like(w))               # V_obs<=0 -> no shrink weight
    mad_eff = xp.where(pos, xp.exp(lm0[:, None] + w[:, None] * (lm - lm0[:, None])),
                       xp.zeros_like(mad))                                  # (B, F)
    # ── per-frame fallback only for frames with a dead channel (rare on a clean stack) ──
    # detect bad frames in one batched reduction (a per-frame ``pos[b].all()`` would serialise B
    # GPU syncs); the Python loop below then runs only for the frames that need it.
    allpos = _env.sum_ax(xp, (~pos).to(mad.dtype) if _env.is_torch(xp) else (~pos).astype(mad.dtype), 1)
    bad = np.nonzero(np.asarray(_env.to_numpy(allpos)) > 0)[0]
    for b in bad:
        m = mad[b]
        spread = m > 0
        typ = float(_env.median1d(xp, m[spread])) if bool(spread.any()) else 0.0
        mad_eff[b] = _shrink_mad(xp, m, m > resolution_floor(xp, typ, m), typ, N)
    safe = mad_eff > 0.0
    div = xp.where(safe, mad_eff, xp.ones_like(mad_eff))[:, None, :]
    zero = _env.zeros(xp, tuple(int(s) for s in data.shape),
                      ref=(data if _env.is_torch(xp) else None))
    return xp.where(safe[:, None, :], centered / div, zero)


def project_batch(xp, data, F_eff: int):
    """Fold a whitened real ``(B, N, F)`` stack to ``(B, N, F_eff)`` -- ``project`` over the
    feature axis (axis 2); the ordered axis is never folded (delta_T=1).  Backend-agnostic."""
    out = _fold_axis(data, F_eff, axis=2)
    out = xp.where(xp.isfinite(xp.abs(out)), out, xp.zeros_like(out))
    return _env.as_compute(xp, out)


def read_batch(frames, *, far: float = 0.05, null=None, seed: int = 0) -> list[BatchRead]:
    """Read a batch of same-shape frames onto their screens in one vectorized pass and return
    a per-frame :class:`BatchRead` (``K_signal`` / ``sigma_top`` / ``noise_floor`` / ``S``),
    **bit-identical** to ``[Projection(f).read()-equivalent for f in frames]`` but amortizing the
    per-frame fold + object overhead (the ensemble throughput lever at small ``F``).  Frames
    are grouped by their feature-fold target so each group folds to one width and svd-values in
    one ``linalg.svd`` call.  ``null`` scores the floor as in :func:`noise_floor` (``None`` =
    derived ``mp``, batched; any other provider is applied per frame -- still one fold/SVD pass).
    A frame that is masked / non-finite / complex, or a differing shape, falls back to a
    per-frame :class:`Projection`.  numpy / CPU (the GPU is slower on many tiny SVDs; for a single
    large frame use ``Projection`` with ``set_precision(32)`` on a torch-cuda tensor)."""
    frames = [np.asarray(f) for f in frames]
    out: list = [None] * len(frames)
    shape0 = next((f.shape for f in frames if f.ndim == 2 and not np.iscomplexobj(f)), None)
    batchable = []
    if shape0 is not None:                                       # same-shape real frames: one
        same = [i for i, f in enumerate(frames)                  # stacked, vectorized finiteness check
                if f.ndim == 2 and not np.iscomplexobj(f) and f.shape == shape0]
        st = np.stack([frames[i] for i in same])
        finite = np.isfinite(st).reshape(len(same), -1).all(axis=1)
        batchable = [same[k] for k in range(len(same)) if bool(finite[k])]
    bset = set(batchable)
    for i, f in enumerate(frames):                              # everything else -> per-frame Projection
        if i not in bset:
            sc = Projection(f, far=far, null=null, seed=seed)
            out[i] = BatchRead(sc.K_signal, sc.sigma_top, float(sc.noise_floor), sc.S)
    if batchable:
        idx = np.array(batchable)
        stack = np.stack([frames[i] for i in batchable])          # (B, N, F)
        N, F = shape0
        F_eff_all = fold_target_batch(np, stack, far=far)
        data = normalize_batch(np, stack)                                    # (B, N, F)
        for Fe in np.unique(F_eff_all):                                       # group by fold width
            sel = np.where(F_eff_all == Fe)[0]
            screen = project_batch(np, data[sel], int(Fe))                   # (Bg, N, Fe)
            S = np.linalg.svd(screen, compute_uv=False)                       # (Bg, min(N,Fe))
            if null is None:                                                  # batched mp floor
                floors = _mp_floor_batch(screen, S, N, int(Fe), far)
            else:
                floors = np.array([noise_floor(screen[j], far=far, null=null, s=S[j], seed=seed)
                                   for j in range(len(sel))])
            for j, k in enumerate(sel):
                s = S[j]; fl = float(floors[j])
                out[int(idx[k])] = BatchRead(int((s > fl).sum()), float(s[0]) if s.shape[0] else 0.0, fl, s)
    return out


def _mp_floor_batch(screen: np.ndarray, S: np.ndarray, N: int, F_eff: int, far: float) -> np.ndarray:
    """The derived ``mp`` screen floor ``sqrt(sigma^2 * (mu + q*sigma_J))`` for a ``(B, N, F_eff)``
    batch -- the same edge :func:`noise_floor` computes per frame, vectorized over the batch.
    Shares the de-biasing denominator and the Johnstone edge with the per-frame ``mp`` provider
    (via ``null_providers``) so the batch and per-frame floors cannot drift."""
    row_energy = (np.abs(screen) ** 2).sum(axis=2)                            # (B, N)
    sigma2 = np.median(row_energy, axis=1) / _debias_denominator(N, F_eff) + 1e-30   # (B,)
    return np.sqrt(_screen_floor_sq(sigma2, N, F_eff, far))
