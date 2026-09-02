"""
screen.py -- the PROJECTION side of Entroptics.  You READ a signal onto its
entropy-matched screen.  Read-only: the screen is a MEASUREMENT of the signal
(its modes, the count above the floor, the per-mode footprints), never written
back out -- a clean view is a FILTER of the data (project onto the resolved
modes), never a synthesis.

Where ``aperture.Aperture`` reads the OPTICS (the information *about* a structure,
read-only), ``screen.Screen`` holds the PROJECTION (the information *within* it):
the signal folded onto its own entropy-matched screen, the screen's SVD modes,
the count of modes standing above the noise floor, and the continuous embedding.

Standalone: numpy only.  Fully parameter-free: the scale comes from
``entropy.geometry`` and the noise floor from the DERIVED Tracy-Widom edge (no
fitted or substrate-calibrated constant), a derived edge conditional on the
iid-Gaussian bulk null.  The per-mode ``footprints`` read exposes the SHAPE of
each resolved mode (broadband vs localized) that the scalar floor cannot see.

    from entroptics import Screen
    sc = Screen(W)        # read the signal onto the screen
    sc.coherence          # is there ordered structure? (deterministic z-score)
    sc.K_signal           # how many modes stand above the noise floor

Per-axis geometry uses the _T (ordered) / _F (feature) subscripts:
  delta_T = ordered window width, delta_F = feature bin width, n_T / n_F mode counts.

Primitives
----------
  project(data, delta_T, delta_F) -- rescale BOTH axes onto the screen (T,F)->(N,F_eff).
  noise_floor(S)           -- the derived Johnstone / Tracy-Widom singular-value noise floor.
  mode_significance(S)     -- per-mode evidence (TW1 deviate + p-value); K_signal = #(p<far).
  coherence(S)             -- ordered-axis coherence z-score (closed-form permutation null).
  footprints(U,S,Vt,K)     -- per-mode localization: fill of each resolved mode's vectors.
  embed(S)                 -- SVD embedding of the screen (continuous coords + basis; lossless).
  read(W) / ScreenRead     -- the full read as a plain dataclass.
  Screen                   -- the same read as an object.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import environment as _env
from .entropy import geometry, normalize, downsample, upsample, shannon_bits, MAD_SCALE, _shrink_mad
# The noise floor is a caller-suppliable null PROVIDER (a FloorContext -> float callback);
# null_providers ships the derived defaults (mp/robust) + the plumbing.  The shared
# primitives are imported under their old private names so the rest of this module -- and
# callers that import _tw1_sf / _TW1_UPPER_Q from screen -- keep working unchanged.
from .null_providers import (                          # noqa: F401 (re-exported)
    _TW1_UPPER_Q, johnstone as _johnstone, tw1_quantile as _tw1_quantile,
    tw1_sf as _tw1_sf, noise_sigma2 as _noise_sigma2, apply_floor as _apply_floor,
)


# ══════════════════════════════════════════════════════════════════════════════
# Screen projection + noise floor + coherence
# ══════════════════════════════════════════════════════════════════════════════

def _fold_axis(A: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Down/upsample one axis, EXCLUDING missing (NaN) cells from the average so a
    masked cell never leaks a zero into a fold.  A coarsened cell is the mean over
    its VALID constituents (weighted by valid area); a cell with no valid data
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
    return num / den                                         # mean over VALID cells; all-missing -> NaN


def project(data: np.ndarray, delta_T: float, delta_F: float) -> np.ndarray:
    """Fold a whitened waterfall onto the screen -- RESCALE BOTH axes to the
    entropy-matched grid in one call: (T, F) -> (N = round(T/delta_T),
    F_eff = round(F/delta_F)).

    Each axis independently downsamples (delta > 1 -> coarsen) or upsamples
    (delta < 1 -> refine).  Both axes are scaled here (``normalize`` only whitens),
    so normalization and rescaling stay two clean, orthogonal steps.  Missing
    (NaN) cells from a mask are EXCLUDED from the fold average; a screen cell with
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
    """Singular-value NOISE FLOOR: the level above which a singular value is
    structure, not noise -- the scalar a NULL PROVIDER returns for this screen.
    ``K_signal = #(S > floor)``.

    ``null`` is a null-provider callback ``FloorContext -> float`` (see
    :mod:`null_providers`); ``None`` uses the derived default ``mp`` (the finite-size
    Johnstone / Tracy-Widom edge: ``sqrt(sigma^2*(mu + q*sigma_J))`` with the de-biased
    per-cell variance ``sigma^2`` and the universal TW1 quantile ``q`` for ``far``).  Pass
    ``null_providers.robust``, ``null_providers.permutation()``, or your OWN provider
    (a local reference / physics null) to score the floor differently -- it is evaluated
    on THIS screen, so under a per-window / streaming read it is local, not global.  ``s``
    is the precomputed singular spectrum (reused where a provider needs it; ``mp`` does
    not); resampling providers are deterministic per ``seed``.  Backend-agnostic."""
    if len(screen.shape) != 2 or int(screen.shape[0]) * int(screen.shape[1]) == 0:
        return float("inf")
    N, F = int(screen.shape[0]), int(screen.shape[1])
    if N <= 0 or F <= 0:
        return float("inf")
    return _apply_floor(null, spectrum=s, data=screen, shape=(N, F),
                        far=far, kind="screen", seed=seed)


@dataclass
class ModeSignificance:
    """Per-singular-value EVIDENCE against the noise null, carrying no threshold: the
    standardized Tracy-Widom deviate and the tail probability of every singular value.
    The resolved dimension at any false-alarm level ``far`` is ``#(pvalue < far)`` -- the
    read reports the evidence, the reader supplies the decision level."""
    deviate: np.ndarray   # g_k = (s_k^2/sigma^2 - mu)/sigma_J, the standardized TW1 deviate
    pvalue:  np.ndarray   # p_k = P(TW1 > g_k), the upper-tail probability (Chiani approx)


def mode_significance(screen: np.ndarray, s: np.ndarray | None = None) -> ModeSignificance:
    """Per-mode significance of the screen's singular spectrum against the derived noise
    null, FREE OF ANY THRESHOLD: for each singular value ``s_k`` the standardized
    Tracy-Widom deviate ``g_k = (s_k^2/sigma^2 - mu)/sigma_J`` and its tail probability
    ``p_k = P(TW1 > g_k)``.  This is the evidence the noise floor thresholds: the resolved
    count ``K_signal`` equals ``#(p_k < far)``, so the false-alarm level ``far`` is applied
    by the reader, not baked into the read.  The identity ``K_signal == #(p_k < far)`` is
    EXACT when the floor quantile is inverted from this same survival function (any non-
    tabulated ``far``); at the tabulated levels (e.g. ``far=0.05``, ``q=0.9793``) the floor
    uses the exact table quantile while ``p_k`` uses the Chiani approximation, so a mode whose
    ``g_k`` sits within the ~7e-3 approximation gap of the quantile can differ by one count.
    Deterministic; numpy-only."""
    xp = _env.ns(screen)
    N, F = int(screen.shape[0]), int(screen.shape[1])
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
    """Ordered-axis COHERENCE of the screen at ``lag`` -- a DETERMINISTIC z-score
    against the EXACT row-permutation null (closed form; no sampling, no RNG).

    The statistic is A = mean_i Re<row_i, row_{i+lag}>^2: how alike, on average, are
    rows ``lag`` apart (squared inner products, so a common row scale cancels and A
    is scale-invariant).  Under a uniformly random row permutation each compared pair
    becomes a uniformly random pair of DISTINCT rows, so E_pi[A] = mu, the mean of R =
    Re(S Sᴴ)^2 over ordered off-diagonal pairs (Theorem 5.2, exact).  The z-score
    standardises A by its EXACT permutation standard deviation:

        coherence = (A - mu) / sqrt(Var_pi[A])

    Var_pi[A] is the exact Cliff-Ord / Mantel second moment: the M = N - lag
    superdiagonal terms are NOT independent (consecutive terms at lag=1 share a row),
    so a naive var/(N-lag) mis-standardises.  It is assembled in closed form from the
    graph moments of R -- S1 = sum R, S2 = sum R^2, U = sum of squared row-sums (all
    over off-diagonal pairs) -- via the single-term variance and the two-term
    expectations for pairs that share one index (E_share) or are disjoint (E_disj):

        E_share = (U - S2) / (N(N-1)(N-2))
        E_disj  = (S1^2 - 4U + 2S2) / (N(N-1)(N-2)(N-3))
        Var[A]  = [ M(mu2 - mu^2) + n_share(E_share - mu^2)
                                  + n_disj (E_disj  - mu^2) ] / M^2

    with mu2 = S2/(N(N-1)), n_share = 2 max(0, N-2 lag) sharing ordered pairs and
    n_disj = M(M-1) - n_share disjoint ones.  This gives a z with EXACTLY unit
    permutation variance (validated against brute-force permutation), so the null is
    calibrated (mean 0, sd 1) across shapes.

    Returns a z-score: > 0 = rows ``lag`` apart are more alike than a random
    reordering (ordered structure); ~ 0 = indistinguishable; < 0 = anti-ordered.
    Deterministic -> BIT-IDENTICAL on numpy and torch.  Backend-agnostic; O(N^2 F)
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

@dataclass
class ModeFootprint:
    """The localization FINGERPRINT of one resolved screen mode: the fill fraction
    of its ordered-axis (``phi_T``) and feature-axis (``phi_F``) singular vectors and
    their product ``etendue`` -- the phase-space AREA the mode occupies.  This is the
    etendue of the axis reads (section 3) resolved PER MODE, and it reads the shape
    the scalar noise floor cannot: a broadband transient (``phi_F`` ~ 1, ``phi_T``
    small), narrowband/persistent structure such as RFI (``phi_F`` small, ``phi_T`` ~
    1), or a compact blob (both small).  All three can share one singular value."""
    index:   int      # mode rank on the screen (0 = leading)
    sigma:   float    # the mode's singular value
    phi_T:   float    # fill of the ordered-axis (left) singular vector, in (0,1]
    phi_F:   float    # fill of the feature-axis (right) singular vector, in (0,1]
    etendue: float    # phi_T * phi_F: the mode's phase-space footprint area


def _vector_fill(xp, v) -> float:
    """Fill fraction of a (unit) vector: ``2^H(|v|^2) / len(v)`` in (0,1] -- 1 when
    the mode is spread uniformly over the axis, -> 1/len when it localizes on a
    single coordinate.  The entropic fill of section 3 applied to one mode vector."""
    n = int(v.shape[0])
    if n == 0:
        return 0.0
    return 2.0 ** shannon_bits(xp.abs(v) ** 2) / n


def footprints(U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
               k_signal: int) -> list[ModeFootprint]:
    """Per-mode LOCALIZATION read: for each of the ``k_signal`` modes standing above
    the noise floor, the fill fractions of its left (ordered) and right (feature)
    singular vectors and their product (the per-mode etendue).  Reads the SHAPE the
    scalar floor is blind to -- distinguishing a broadband signal from a localized
    (narrowband or compact) blob at the same singular value.  Deterministic; pure
    entropy of the SVD modes.  Backend-agnostic (numpy or torch)."""
    xp = _env.ns(S)
    out: list[ModeFootprint] = []
    for k in range(int(k_signal)):
        pt = _vector_fill(xp, U[:, k])
        pf = _vector_fill(xp, Vt[k])
        out.append(ModeFootprint(index=k, sigma=float(S[k]),
                                 phi_T=pt, phi_F=pf, etendue=pt * pf))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SVD embedding + waterfall reconstruction (the projection's structure)
# ══════════════════════════════════════════════════════════════════════════════

def embed(screen: np.ndarray,
          svd: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
          noise_floor_override: float | None = None,
          *, far: float = 0.05, null=None, seed: int = 0,
          ) -> tuple[np.ndarray | None, np.ndarray | None, int, int, float]:
    """Project the screen into a continuous SVD embedding.

    Two distinct K quantities:
      K        -- SVD truncation dimension; default = floor(sqrt(N * F_eff)) (the
                  geometric mean of the screen dimensions).  A geometric
                  stability criterion, not a Shannon-derived scale; any
                  K in [1, min(N, F_eff)] is valid.
      K_signal -- count of singular values above the noise floor; always derived
                  from the data.  K_signal <= K.  When K_signal == 0 no structured
                  signal is detected and the embedding is None.

    Returns (embeddings (N, K), vocabulary (K, F_eff), K, K_signal, H_screen),
    or (None, None, 0, 0, 0.0) when K_signal == 0.  H_screen is the Shannon
    entropy (bits) of the signal-mode power weights.
    """
    xp = _env.ns(screen)
    N, F_eff = int(screen.shape[0]), int(screen.shape[1])
    if svd is None:
        U, S, Vt = xp.linalg.svd(screen, full_matrices=False)
    else:
        U, S, Vt = svd
    floor = (noise_floor_override if noise_floor_override is not None
             else noise_floor(screen, far=far, null=null, s=S, seed=seed))
    signal_mask = S > floor
    K_signal = int(_env.sum_ax(xp, signal_mask))
    if K_signal == 0:
        return None, None, 0, 0, 0.0
    S_signal = S[signal_mask]
    H_screen = shannon_bits(S_signal ** 2)               # entropy of the signal-mode power weights
    K = max(1, min(N, F_eff, math.isqrt(N * F_eff)))
    # Split the singular values symmetrically so neither factor has unit-norm
    # rows by construction: embeddings @ vocabulary = U diag(S) Vt.
    sqrtS = xp.sqrt(S[:K])
    embeddings = U[:, :K] * sqrtS
    vocabulary = sqrtS[:, None] * Vt[:K]
    return embeddings, vocabulary, K, K_signal, H_screen


# ══════════════════════════════════════════════════════════════════════════════
# The full read -- as a dataclass (read) and as an object (Screen)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScreenRead:
    """The full screen read as a plain record (output of ``read``)."""
    delta_T:     float   # ordered window width (entropy-matched scale)
    delta_F:     float   # feature bin width (entropy-matched scale)
    coherence:   float   # ordered-axis coherence z-score (>~2 => significant structure)
    K_signal:    int     # SVD modes above the noise floor (the resolved sector)
    H_screen:    float   # Shannon entropy (bits) of the signal-mode weights
    sigma_top:   float   # top singular value of the screen
    noise_floor: float   # the singular-value noise floor


class Screen:
    """The PROJECTION of a signal onto its entropy-matched screen.

    Construct from a 2-D array ``W`` (T, F): axis-0 the ORDERED axis, axis-1 the
    FEATURE axis.  On construction the signal's own entropy geometry sets the
    scale, the waterfall is folded onto the screen, and the screen's SVD structure
    is read.  Read-only: there is no write-back -- a denoised view is a FILTER of
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
    embeddings, vocabulary    : continuous SVD embedding (None if K_signal == 0)
    K                         : embedding dimension (geometric-mean heuristic)
    """

    def __init__(self, W, mask=None, *, far: float = 0.05, null=None, seed: int = 0):
        xp = _env.ns(W)
        if len(getattr(W, "shape", ())) != 2:
            raise ValueError(f"Screen expects a 2-D array (T, F); got {getattr(W, 'shape', None)}")
        nan = ~xp.isfinite(xp.abs(W))
        if mask is not None or bool(nan.any()):
            # masked / gapped -> numpy path (robust normalize); rare on the GPU stream
            W, xp = _env.to_numpy(W), np
            nan = ~np.isfinite(np.abs(W))
            if nan.any():
                mask = (mask | nan) if mask is not None else nan
        self.W = W
        self.mask = mask
        self.T, self.F = int(W.shape[0]), int(W.shape[1])

        # Fully-dead rows/cols (every cell missing) carry no information: IGNORE them
        # (drop) rather than impute zeros -- zeros inject a false "flat" block of
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
        self._U = self._Vt = None      # full SVD basis -- LAZY (heavy; only embeddings / footprints)
        self._coh = None               # coherence -- LAZY (a separate lag-1 pass)
        self._emb = None               # (embeddings, vocabulary, K) -- LAZY
        # LIGHTWEIGHT signal decision: only the singular VALUES drive the noise floor and
        # K_signal, so use ``svdvals`` (no U / Vt). The full SVD basis, the embeddings, and
        # the coherence are HEAVY and DEFERRED to first access -- a streaming monitor that
        # reads K_signal / has_signal never pays for them (the high-speed capture path).
        self.S = _env.svdvals(xp, self.screen)
        self.noise_floor = noise_floor(self.screen, far=far, null=null, s=self.S, seed=seed)
        self.sigma_top = float(self.S[0]) if int(self.S.shape[0]) else 0.0
        sig = self.S[self.S > self.noise_floor]
        self.K_signal = int(sig.shape[0])
        self.H_screen = shannon_bits(sig ** 2) if int(sig.shape[0]) else 0.0

    # ── heavy outputs: LAZY (first access only; the monitoring path never touches them) ──
    def _basis(self):
        """Full SVD basis (U, Vt) -- computed once, on demand (embeddings / footprints only)."""
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
        monitor read. For an SVD-free gate on a RAW frame (before building a Screen), use the
        module-level `probe_signal(W, ...)`."""
        return self.sigma_top > self.noise_floor

    @property
    def coherence(self) -> float:
        """Ordered-axis coherence z-score (lag-1 permutation null). LAZY -- a separate pass, not
        part of the K_signal signal decision."""
        if self._coh is None:
            self._coh = coherence(self.screen, lag=1)
        return self._coh

    def _embedding(self):
        if self._emb is None:
            U, Vt = self._basis()
            e, v, K, _, _ = embed(self.screen, svd=(U, self.S, Vt),
                                  noise_floor_override=self.noise_floor)
            self._emb = (e, v, K)
        return self._emb

    @property
    def embeddings(self):
        """Ordered-axis embedding: the resolved left modes scaled by their singular values; lazy."""
        return self._embedding()[0]

    @property
    def vocabulary(self):
        """Feature-axis basis of the resolved right modes; lazy."""
        return self._embedding()[1]

    @property
    def K(self) -> int:
        """Number of resolved modes above the noise floor (``K_signal``); lazy."""
        return self._embedding()[2]

    @property
    def N(self) -> int:
        """Ordered-axis length of the screen (rows)."""
        return self.screen.shape[0]

    @property
    def F_eff(self) -> int:
        """Folded feature-axis length of the screen (columns)."""
        return self.screen.shape[1]

    @property
    def footprints(self) -> list[ModeFootprint]:
        """Per-mode localization fingerprints for the ``K_signal`` resolved modes:
        each mode's ordered/feature fill fractions and per-mode etendue (see
        ``screen.footprints``).  Empty when ``K_signal == 0``.  Reads the SHAPE of
        each mode above the floor -- broadband signal vs a localized (narrowband or
        compact) blob -- that ``K_signal`` alone cannot distinguish."""
        return footprints(self.U, self.S, self.Vt, self.K_signal)

    @property
    def significance(self) -> ModeSignificance:
        """Per-mode evidence against the noise null (see ``screen.mode_significance``):
        the standardized Tracy-Widom deviate and tail probability ``p_k`` of every
        singular value.  With the default ``mp`` provider, ``K_signal == #(p_k < far)``
        (exact for a non-tabulated ``far`` inverted from the same survival function; at a
        tabulated ``far`` it can differ by one within the Chiani approximation gap -- see
        ``mode_significance``) -- the read exposes the evidence, the false-alarm level is the caller's.
        (The p-values are always the analytic ``mp`` evidence; under a different ``null``
        provider ``K_signal`` follows that provider instead, so the identity is specific
        to the ``mp`` default.)"""
        return mode_significance(self.screen, self.S)

    def read(self) -> ScreenRead:
        """The full read as a plain :class:`ScreenRead` record."""
        return ScreenRead(delta_T=self.delta_T, delta_F=self.delta_F,
                          coherence=self.coherence, K_signal=self.K_signal,
                          H_screen=self.H_screen, sigma_top=self.sigma_top,
                          noise_floor=self.noise_floor)

    def tensor(self, d: int | None = None, *, rank: tuple | None = None) -> dict:
        """Delay-embedded Tucker (HOSVD) of this signal at NATIVE resolution -- the
        within-window fine structure the averaged screen SVD loses.  ``d`` is the
        delay-window width.  See tensor.tensor_embed."""
        from .tensor import tensor_read    # deferred import (tensor path is optional)
        return tensor_read(self.W, self.mask, d, rank=rank)

    def aperture(self):
        """The companion :class:`aperture.Aperture` for the same signal -- the
        OPTICS of this projection."""
        from .aperture import Aperture       # deferred import (avoids a cycle)
        return Aperture(self.W, self.mask)

    def __repr__(self) -> str:
        return (f"Screen(shape={self.W.shape}, N={self.N}, F_eff={self.F_eff}, "
                f"K_signal={self.K_signal}, coherence={self.coherence:.2f})")


def read(W: np.ndarray, mask: np.ndarray | None = None, *, far: float = 0.05,
         null=None, seed: int = 0) -> ScreenRead:
    """Read a signal onto the screen and return the full :class:`ScreenRead`:
    geometry -> normalize -> project -> coherence -> noise-floor modes.  Fully
    parameter-free by default: the entropy-matched fold and the derived ``mp`` noise floor
    carry no fitted constant; ``far`` is the floor's significance level (default 5%).
    ``null`` is a null-provider callback selecting the noise floor (``None`` = derived
    default ``mp``; see :func:`noise_floor` and :mod:`null_providers`).  Deterministic (a
    resampling provider is deterministic per ``seed``).  (= ``Screen(W, ...).read()``.)"""
    return Screen(W, mask=mask, far=far, null=null, seed=seed).read()


def _sigma_top_upper(screen) -> float:
    """A rigorous, SVD-FREE upper bound on the top singular value of ``screen``:
    ``sqrt(||screen||_1 * ||screen||_inf)`` (max column abs-sum times max row abs-sum) -- the
    spectral norm is bounded by the geometric mean of the 1- and inf- operator norms.  O(NF)."""
    xp = _env.ns(screen)
    A = xp.abs(screen)
    col = float(A.sum(axis=0).max()) if int(screen.shape[1]) else 0.0
    row = float(A.sum(axis=1).max()) if int(screen.shape[0]) else 0.0
    return math.sqrt(col * row)


def probe_signal(W, mask=None, *, far: float = 0.05, null=None, seed: int = 0) -> bool:
    """SVD-FREE signal gate for high-speed / streaming capture.  Builds the entropy-matched fold
    and the noise floor (both O(NF); no SVD for the derived / reference nulls), then tests a
    RIGOROUS upper bound on the screen's top singular value against the floor.  Returns True iff
    signal MAY be present -- so a monitor can decide whether to build the full :class:`Screen`
    (and pay the SVD / embedding / coherence) WITHOUT one.  CONSERVATIVE: never False when
    ``Screen(W, ...).K_signal > 0`` (the bound only over-estimates the true top singular value),
    so gating on it cannot drop a real detection.  A masked / gapped frame returns True (defer to
    the full Screen).  ``null`` scores the floor as in :func:`noise_floor`; a sampled provider
    that needs the spectrum computes one internally (no saving for those)."""
    xp = _env.ns(W)
    if mask is not None or not bool(xp.all(xp.isfinite(xp.abs(W)))):
        return True                                    # masked / gapped -> defer to the full Screen
    geom = geometry(W, mask)
    screen = project(normalize(W, mask), geom["delta_T"], geom["delta_F"])
    floor = noise_floor(screen, far=far, null=null, s=None, seed=seed)
    return _sigma_top_upper(screen) > floor


# ══════════════════════════════════════════════════════════════════════════════
# Batched MONITOR path -- the throughput lever for ENSEMBLES (numpy / CPU)
# ══════════════════════════════════════════════════════════════════════════════
# The per-frame Python loop (wrapper plane reduction, ensemble traces) pays call + object
# overhead per frame and vectorizes nothing.  ``read_batch`` folds + svd-values + floors a
# STACK of same-shape frames in one vectorized pass -- BIT-IDENTICAL to reading each frame
# with ``Screen`` (the fold is per-column, the per-channel medians are per-frame, so batching
# only changes the loop nesting, never a float).  It is the small-F ensemble lever the GPU
# cannot provide (cuSOLVER has no occupancy on many tiny SVDs).  numpy only: frames that are
# masked / non-finite / complex / a different shape fall back to the per-frame ``Screen``.

@dataclass
class BatchRead:
    """One frame's lightweight monitor read from :func:`read_batch` -- the same
    ``K_signal`` / ``sigma_top`` / ``noise_floor`` / singular values ``S`` a per-frame
    :class:`Screen` produces (bit-identical)."""
    K_signal:    int
    sigma_top:   float
    noise_floor: float
    S:           np.ndarray


def _fold_target_batch(stack: np.ndarray) -> np.ndarray:
    """Per-frame feature-fold target ``F_eff`` (== ``geometry``'s ``n_F``) for a real,
    finite ``(B, N, F)`` stack -- ``entropy.geometry``'s ``H_F`` + fold guard, batched."""
    B, N, F = stack.shape
    P = _env.asnum(np.abs(stack)); P = P * P                       # |W|^2 (compute precision)
    marg = P.sum(axis=1)                                           # (B, F) feature power marginal
    tot = marg.sum(axis=1)                                         # (B,)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = marg / np.where(tot[:, None] > 1e-30, tot[:, None], 1.0)
        H = -np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1.0)), 0.0).sum(axis=1)   # shannon_bits
    log2F = float(np.log2(F)); ln2 = math.log(2.0)
    H_F = np.where(tot > 1e-30, H, log2F)                         # P_total<=0 -> maximal entropy
    band_F = min((F - 1) / (2.0 * max(1, N) * ln2), 0.5 * log2F)
    F_eff = np.full(B, F, dtype=int)
    fold = H_F < log2F - band_F                                   # else no fold (F_eff = F)
    if bool(fold.any()):
        n_real = np.minimum(float(F), 2.0 ** H_F[fold])
        F_eff[fold] = np.maximum(1, np.minimum(F, np.round(n_real).astype(int)))
    return F_eff


def _normalize_batch(stack: np.ndarray) -> np.ndarray:
    """Per-channel robust MAD whiten of a real, finite ``(B, N, F)`` stack -- ``entropy.
    normalize``'s clean path over axis 1, reusing ``_shrink_mad`` per frame (bit-identical)."""
    eps = 1e-12
    data = _env.asnum(stack)
    med = np.median(data, axis=1, keepdims=True)                 # (B,1,F) per-channel median over N
    centered = data - med
    mad = np.median(np.abs(centered), axis=1) * MAD_SCALE         # (B, F)
    B, F = mad.shape
    mad_eff = np.empty_like(mad)
    for b in range(B):                                           # per-frame shrink (cheap scalars)
        m = mad[b]; pos = m > eps
        typical = float(np.median(m[pos])) if bool(pos.any()) else 1.0
        mad_eff[b] = _shrink_mad(np, m, pos, typical, int(data.shape[1]), eps)
    safe = mad_eff > eps                                         # (B, F)
    div = np.where(safe, mad_eff, 1.0)[:, None, :]
    return np.where(safe[:, None, :], centered / div, 0.0)


def _project_batch(data: np.ndarray, F_eff: int) -> np.ndarray:
    """Fold a whitened real ``(B, N, F)`` stack to ``(B, N, F_eff)`` -- ``project`` over the
    feature axis (axis 2); the ordered axis is never folded (delta_T=1)."""
    out = _fold_axis(data, F_eff, axis=2)
    out = np.where(np.isfinite(np.abs(out)), out, np.zeros_like(out))
    return _env.as_compute(np, out)


def read_batch(frames, *, far: float = 0.05, null=None, seed: int = 0) -> list[BatchRead]:
    """Read a batch of same-shape frames onto their screens in ONE vectorized pass and return
    a per-frame :class:`BatchRead` (``K_signal`` / ``sigma_top`` / ``noise_floor`` / ``S``),
    **bit-identical** to ``[Screen(f).read()-equivalent for f in frames]`` but amortizing the
    per-frame fold + object overhead (the ensemble throughput lever at small ``F``).  Frames
    are grouped by their feature-fold target so each group folds to one width and svd-values in
    one ``linalg.svd`` call.  ``null`` scores the floor as in :func:`noise_floor` (``None`` =
    derived ``mp``, batched; any other provider is applied per frame -- still one fold/SVD pass).
    A frame that is masked / non-finite / complex, or a differing shape, falls back to a
    per-frame :class:`Screen`.  numpy / CPU (the GPU is slower on many tiny SVDs; for a single
    LARGE frame use ``Screen`` with ``set_precision(32)`` on a torch-cuda tensor)."""
    frames = [np.asarray(f) for f in frames]
    out: list = [None] * len(frames)
    shape0 = next((f.shape for f in frames if f.ndim == 2 and not np.iscomplexobj(f)), None)
    batchable = []
    if shape0 is not None:                                       # same-shape real frames: one
        same = [i for i, f in enumerate(frames)                  # stacked, VECTORIZED finiteness check
                if f.ndim == 2 and not np.iscomplexobj(f) and f.shape == shape0]
        st = np.stack([frames[i] for i in same])
        finite = np.isfinite(st).reshape(len(same), -1).all(axis=1)
        batchable = [same[k] for k in range(len(same)) if bool(finite[k])]
    bset = set(batchable)
    for i, f in enumerate(frames):                              # everything else -> per-frame Screen
        if i not in bset:
            sc = Screen(f, far=far, null=null, seed=seed)
            out[i] = BatchRead(sc.K_signal, sc.sigma_top, float(sc.noise_floor), sc.S)
    if batchable:
        idx = np.array(batchable)
        stack = np.stack([frames[i] for i in batchable])          # (B, N, F)
        N, F = shape0
        F_eff_all = _fold_target_batch(stack)
        data = _normalize_batch(stack)                                        # (B, N, F)
        for Fe in np.unique(F_eff_all):                                       # group by fold width
            sel = np.where(F_eff_all == Fe)[0]
            screen = _project_batch(data[sel], int(Fe))                       # (Bg, N, Fe)
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
    batch -- the same edge :func:`noise_floor` computes per frame, vectorized over the batch."""
    q = _tw1_quantile(far)
    mu, sig_J = _johnstone(N, F_eff)
    c_F = (1.0 - 2.0 / (9.0 * max(F_eff, 1))) ** 3
    dof = max(N - 1, 1) / N
    row_energy = (np.abs(screen) ** 2).sum(axis=2)                            # (B, N)
    sigma2 = np.median(row_energy, axis=1) / (F_eff * c_F * dof) + 1e-30      # (B,)
    return np.sqrt(sigma2 * (mu + q * sig_J))
