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
  noise_floor(S)           -- the singular-value noise floor (default: the exact permutation test).
  mode_significance(S)     -- closed-form per-mode evidence (TW deviate + p-value): #(p<far) is mp's count.
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
                      macheps, feature_axis_is_continuous, _carried)
# The noise floor is a caller-suppliable null provider (a FloorContext -> float callback);
# null_providers ships the providers (the exact permutation default, mp, robust) + the plumbing.  The shared
# primitives are re-exported here under their private names (``_tw1_sf`` etc.) so a caller
# that imports them from ``projection`` directly keeps working unchanged.
from .beam import Beam
from .null_providers import (                          # noqa: F401 (re-exported)
    johnstone as _johnstone, tw1_quantile as _tw1_quantile,
    tw1_sf as _tw1_sf, tw_sf as _tw_sf, noise_sigma2 as _noise_sigma2, apply_floor as _apply_floor,
    noise_sigma2_from_spectrum as _noise_sigma2_from_spectrum,
    debias_denominator as _debias_denominator, screen_floor_sq as _screen_floor_sq, mp as _mp,
    _select_provider, fewest_draws as _fewest_draws,
    top_spectrum_value as _top_spectrum_value, _scored, _screen_scale, _deviate, _roundoff_deviate,
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
    xp = _env.ns(data)
    out = xp.where(xp.isfinite(xp.abs(data)), data, xp.zeros_like(data))  # a missing cell at its centre
    out = _fold_axis(out, N, axis=0)
    out = _fold_ortho(xp, out, F_eff, axis=1)
    return _env.as_compute(xp, out)   # pin the screen to the set precision (float64 default => no-op)


def _groups(n_in: int, n_out: int) -> np.ndarray:
    """The boundaries of the partition fold of ``n_in`` channels onto ``n_out`` cells: cell ``j``
    holds the whole channels ``b[j] .. b[j+1] - 1``, ``b[j] = floor(j n_in / n_out)``, so every
    cell is a contiguous run and the runs differ in length by at most one."""
    return (np.arange(int(n_out) + 1) * int(n_in)) // int(n_out)


def fold_variance(n_in: int, n_out: int) -> float:
    """The mean per-cell variance :func:`project`'s ordered-axis fold leaves i.i.d. unit cells at:
    ``tr(R^T R) / n_out`` for the area-mean ``R`` when the axis coarsens, 1 when it is kept or
    refined (a hold repeats cells).  The feature axis's partition fold always leaves 1."""
    n_in, n_out = int(n_in), int(n_out)
    if n_out >= n_in:
        return 1.0
    tr, step = 0.0, max(1, (1 << 22) // max(n_in, 1))
    for i0 in range(0, n_in, step):                     # rows of R in blocks, never all of it
        i1 = min(n_in, i0 + step)
        E = np.zeros((i1 - i0, n_in))
        E[np.arange(i1 - i0), np.arange(i0, i1)] = 1.0
        tr += float(np.sum(np.asarray(downsample(E, n_out, 1), float) ** 2))
    return tr / n_out


def _fold_ortho(xp, A, F_eff: int, axis: int):
    """Fold the last axis of ``A`` onto ``F_eff`` cells by PARTITION: each cell is a contiguous run
    of whole channels (:func:`_groups`), summed and divided by the square root of its length.

    The fold's columns are orthonormal by construction -- the runs are disjoint -- so independent
    unit-variance channels fold to independent unit-variance cells, exactly, which is the premise
    the floor reads.  An area mean does not have it: cells of unequal size carry unequal noise,
    and a cell that takes a fraction of a channel shares it with its neighbour, so the folded
    columns are correlated (a record whose channels merely sat at different levels folded, and
    then read structure in half of pure-noise records).  A channel is a discrete unit, so a cell
    takes whole ones.  O(T F), no factorisation.  ``axis`` is the last axis of ``A``."""
    n_in = int(A.shape[axis])
    if F_eff >= n_in:
        return A if F_eff == n_in else upsample(A, F_eff, axis)
    b = _groups(n_in, F_eff)
    size = np.diff(b).astype(float)
    if _env.is_torch(xp):
        idx = xp.as_tensor(np.repeat(np.arange(F_eff), np.diff(b)), device=A.device)
        shape = list(A.shape); shape[axis] = F_eff
        out = xp.zeros(shape, dtype=A.dtype, device=A.device).index_add_(axis, idx, A)
        return out / xp.as_tensor(np.sqrt(size), dtype=out.real.dtype, device=A.device)
    out = np.add.reduceat(np.asarray(A), b[:-1], axis=axis)
    return out / np.sqrt(size).astype(np.asarray(out).real.dtype, copy=False)


def _screen_resample(W, mask, data, fold_far: float | None = 0.05):
    """The null draw of a read's screen (``FloorContext.resample``).  One permutation per channel
    moves its measured values among its own measured cells -- the missing ones stay where they
    are -- and is applied alike to the live record ``W`` and to its whitening ``data``; the draw's
    fold is then decided on the shuffled record, exactly as the observed fold was decided on
    ``W`` (``geometry`` at ``fold_far``; ``None`` for a read that does not fold), and the shuffled
    whitening is folded at it.  So every step the observed screen went through, the fold's
    decision included, is taken again on the draw, and under the null of channels that are
    independent sequences of exchangeable values -- of any law, with any pattern of missing
    cells -- the observed screen is one of the draws.

    The fold's decision must be redrawn and not carried: whether the feature axis is continuous
    is read from how neighbouring channels move together, which the shuffle destroys, so a fold
    carried from the observed record into the draws conditions the test on structure the draws
    do not share (it ran at 0.085 given a fold, on lognormal noise).  And the draw is taken on the
    channels, before the fold, because a shuffle of the finished screen keeps the energy a fold
    cell gathers from channels that move together and is blind to them."""
    Wr = np.asarray(_env.to_numpy(W))
    Z = np.asarray(_env.to_numpy(data))
    M = None if mask is None else np.asarray(_env.to_numpy(mask), bool)
    miss = ~np.isfinite(np.abs(Z)) if M is None else (M | ~np.isfinite(np.abs(Z)))
    T, F = Z.shape
    # The fold's two conditions, split by what the shuffle moves.  Concentration reads the column
    # power marginal and the measured extents -- a channel's values and its missing cells stay
    # its own under the shuffle -- so its verdict and the width it folds to are the observed
    # record's, read once.  Continuity reads how neighbouring channels move together, which the
    # shuffle destroys, so it is read again on every draw, and only when concentration has put a
    # fold on the table (as ``fold_width`` itself orders them).
    fold_to = None
    if fold_far is not None:
        have = ~miss
        F_eff = max(1, int(np.count_nonzero(have.any(axis=0))))
        T_eff = max(1, int(np.count_nonzero(have.any(axis=1))))
        H_F = geometry(Wr, M, far=fold_far)["H_F"]
        n_F, d_F = fold_width(H_F, T, F, None, M, far=fold_far, F_eff=F_eff, T_eff=T_eff)
        if (n_F, d_F) != (F, 1.0):
            fold_to = d_F
    # Channel-major: each channel's values are one contiguous row, so a shuffle is a gather along
    # rows (about twice as fast as along strided columns).
    ZT = np.ascontiguousarray(Z.T)
    # the continuity test is read on the record at the power of two of its peak, as ``geometry``
    # reads it (a shuffle moves no value, so the peak and the carry are every draw's)
    WT = np.ascontiguousarray(_carried(np, Wr, M).T) if fold_to is not None else None
    if miss.any():
        missT = np.ascontiguousarray(miss.T)
        at = np.argsort(missT, axis=1, kind="stable")        # measured cells first, in time order

        def index(rng):
            key = np.where(missT, 2.0, rng.random(missT.shape))   # the same cells, in random order
            g = np.empty(missT.shape, dtype=np.int32 if T < 2 ** 31 else np.intp)
            np.put_along_axis(g, at, np.argsort(key, axis=1), axis=1)
            return g
    else:
        cols = np.broadcast_to(np.arange(T, dtype=np.int32 if T < 2 ** 31 else np.intp), (F, T))

        def index(rng):
            return rng.permuted(cols, axis=1)

    if fold_to is None and not miss.any():
        # nothing to fold and nothing missing: the screen is the whitened record itself (``project``
        # at unit deltas only pins the precision), so the channels are permuted directly -- one
        # pass, no index to build and gather through
        def draw(rng):
            Zs = rng.permuted(ZT, axis=1).T
            return Zs if fold_far is None else _env.as_compute(np, Zs)
        return draw

    def draw(rng):
        g = index(rng)
        Zs = np.take_along_axis(ZT, g, axis=1).T
        if fold_to is None:
            return Zs if fold_far is None else project(Zs, 1.0, 1.0)
        Ws = np.take_along_axis(WT, g, axis=1).T
        d = fold_to if feature_axis_is_continuous(Ws, M, far=fold_far) else 1.0
        return project(Zs, 1.0, d)
    return draw


def noise_floor(screen: np.ndarray, *, far: float = 0.05, null=None,
                s: np.ndarray | None = None, seed: int = 0, resample=None) -> float:
    """Singular-value noise floor: the level above which a singular value is
    structure, not noise -- the scalar a null provider returns for this screen.
    ``K_signal = #(S > floor)``.

    ``null`` is a null-provider callback ``FloorContext -> float`` (see
    :mod:`null_providers`); ``None`` uses the default, the exact permutation test: each channel
    shuffled in time, scored at the exact Monte Carlo rank over the fewest draws the level
    allows, so the false-alarm rate is ``far`` for any law of the noise.  ``resample`` is the
    read's own null draw of its screen (:class:`null_providers.FloorContext`); without it the
    screen given is shuffled as the record.  Pass ``null_providers.mp`` for the closed-form
    Johnstone / Tracy-Widom edge ``sqrt(sigma^2*(mu + q*sigma_J))``, ``null_providers.robust``,
    or your own provider (a local reference / physics null) to score the floor differently -- it
    is evaluated on this screen, so under a per-window / streaming read it is local, not global.
    ``s`` is the precomputed singular spectrum (computed when a provider needs it and it is not
    given); resampling providers are deterministic per ``seed``.  Backend-agnostic."""
    if len(screen.shape) != 2 or int(screen.shape[0]) * int(screen.shape[1]) == 0:
        return float("inf")
    N, F = int(screen.shape[0]), live_columns(screen)
    if N <= 0 or F <= 0:
        return float("inf")
    return _apply_floor(null, spectrum=s, data=screen, shape=(N, F),
                        far=far, kind="projection", seed=seed, resample=resample)


def live_columns(screen) -> int:
    """The number of feature channels the floor's null is sized by: columns that are not
    identically zero.

    Not ``screen.shape[1]``: the floor is the edge of an N x F noise ensemble, so F decides
    where it sits, and a column of exact zeros is not a channel of that ensemble. Two routes
    produce them, both upstream in this module: :func:`project` mean-imputes an all-missing
    screen cell to ``0``, and :func:`normalize` returns a channel it could not scale (no spread) as zeros. Either way the column carries no observation.

    Counting them biases the floor in two opposing directions at once, which is worse than
    either alone because the errors do not announce themselves by cancelling: the de-biased
    per-cell variance divides the screen's energy by the cell count N*F, so dead columns add
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
    standardized Tracy-Widom deviate and a p-value for every singular value.  The p-value is the
    evidence of the floor it belongs to: the Tracy-Widom tail (:func:`mode_significance`, the
    ``mp`` floor, whose count is ``#(pvalue < far)`` at any ``far``), or the exact Monte Carlo
    p-value of a permutation floor (``Projection.significance``, whose count is
    ``#(pvalue <= far)`` at the level its draws were sized for)."""
    deviate: np.ndarray   # g_k = (s_k^2/sigma^2 - mu)/sigma_J, the standardized TW deviate
    pvalue:  np.ndarray   # the tail probability of s_k under the floor's null


def mode_significance(screen: np.ndarray, s: np.ndarray | None = None) -> ModeSignificance:
    """Per-mode significance of the screen's singular spectrum against the derived noise
    null, free of any threshold: for each singular value ``s_k`` the standardized
    Tracy-Widom deviate ``g_k = (s_k^2/sigma^2 - mu)/sigma_J`` and its tail probability
    ``p_k = P(TW > g_k)`` -- TW1 for a real screen, TW2 (with the complex Johnstone centring) for a
    complex one, both exact to float precision.  This is the evidence the noise floor thresholds:
    the resolved count ``K_signal`` equals ``#(p_k < far)``, so the false-alarm level ``far`` is
    applied by the reader, not baked into the read; the floor's quantile and these p-values come
    from the same law, so the identity holds at every ``far``.  Deterministic; numpy-only."""
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
    cx = _env.is_complex_obj(screen)
    mu, sig_J = _johnstone(N, F, complex_=cx)
    sigma2 = _noise_sigma2(xp, screen, N, F, s=sv)
    if not (sigma2 > 0.0):
        # A screen with no energy has no noise ensemble to stand above, so no mode is evidence.
        # This is the ONE consumer that divides by sigma^2, and handling the degenerate case HERE
        # is what lets `noise_sigma2` stay an exact variance.  It previously carried an additive
        # `1e-30` for this division alone, which made the reported variance a constant rather than
        # a measurement once the screen fell near 1e-15 (inflated 2.79x at 1e-15, 1.8e+06x at
        # 1e-18) -- an absolute number standing in for a dimensioned one.
        z = np.zeros_like(sv)
        return ModeSignificance(deviate=z, pvalue=np.ones_like(sv))
    g = (sv ** 2 / sigma2 - mu) / sig_J
    p = np.asarray(_tw_sf(g, complex_=cx), dtype=float)
    return ModeSignificance(deviate=g, pvalue=p)


def _coherence_z(S1: float, S2: float, U: float, Asum: float, N: int, lag: int, ref) -> float:
    """The exact permutation z-score from the graph moments of ``R = G^2`` (see :func:`coherence`).
    ``ref`` is an array in the arithmetic the moments were formed in, for its epsilon."""
    xp = _env.ns(ref)
    M = N - lag
    Dp = N * (N - 1)
    mu = S1 / Dp                                    # exact null mean (Theorem 5.2)
    mu2 = S2 / Dp
    mu_sq = mu * mu
    E_share = (U - S2) / (N * (N - 1) * (N - 2))
    E_disj = (S1 * S1 - 4.0 * U + 2.0 * S2) / (N * (N - 1) * (N - 2) * (N - 3))
    n_share = 2 * max(0, N - 2 * lag)              # ordered pairs of terms sharing one index
    n_disj = M * (M - 1) - n_share
    var_A = (M * (mu2 - mu_sq)
             + n_share * (E_share - mu_sq)
             + n_disj * (E_disj - mu_sq)) / (M * M)
    # The guard is on CANCELLATION, not on magnitude, so it must be RELATIVE.  `var_A` is
    # assembled by subtracting terms of size `mu^2`, so its floating-point resolution is that
    # magnitude times the arithmetic's own epsilon -- the same `scale * shape * macheps` idiom
    # the rest of the library uses.  An ABSOLUTE floor cannot work here: `R` is a squared inner
    # product, so it scales as the FOURTH power of the screen and `var_A` as the EIGHTH.  With
    # the previous `1e-24`, one factor of ten in the caller's units took this read from
    # z = 18.82 to exactly 0.0 on the identical signal -- structure silently reported as none.
    tol = max(mu_sq, mu2) * float(N * N) * macheps(xp, ref)
    if var_A <= tol:
        return 0.0
    A = Asum / float(M)                                        # the lag-th superdiagonal mean
    return (A - mu) / math.sqrt(var_A)


def _coherence_feature_side(Y: np.ndarray, lag: int) -> float:
    """:func:`coherence` from the real ``(N, F')`` record ``Y``, never forming the ``N x N`` Gram.

    With ``G = Y Y^T`` and ``C = Y^T Y``: ``sum_ab G_ab^2 = ||C||_F^2``; the row sums
    ``sum_b G_ab^2 = y_a^T C y_a``; ``sum_ab G_ab^4 = ||Z^T Z||_F^2`` with ``Z``'s rows the outer
    products ``vec(y_a y_a^T)`` (accumulated in row blocks no larger than the record); and the lag
    diagonal ``sum_i G_{i,i+lag}^2`` directly.  O(N F'^4) time, O(F'^4) memory."""
    N, Fp = Y.shape
    q = np.einsum("ij,ij->i", Y, Y)                  # ||y_a||^2 = G_aa
    peak = float(q.max())
    if not peak > 0.0:
        return 0.0
    Y = Y / math.sqrt(peak)                          # exact in the value (scale-invariant z)
    q = q / peak
    d = q * q                                        # R_aa = G_aa^2
    C = Y.T @ Y
    S1 = float(np.sum(C * C) - d.sum())
    rowsum = np.einsum("ij,jk,ik->i", Y, C, Y) - d
    U = float(np.sum(rowsum * rowsum))
    blk = max(1, N // Fp)
    D = np.zeros((Fp * Fp, Fp * Fp))
    for s0 in range(0, N, blk):
        Yb = Y[s0:s0 + blk]
        Z = (Yb[:, :, None] * Yb[:, None, :]).reshape(Yb.shape[0], Fp * Fp)
        D += Z.T @ Z
    S2 = float(np.sum(D * D) - np.sum(d * d))
    g = np.einsum("ij,ij->i", Y[:-lag], Y[lag:])
    return _coherence_z(S1, S2, U, float(np.sum(g * g)), N, lag, np.asarray(C))


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
    Deterministic.  Backend-agnostic.  The moments are sums over the ``N x N`` row Gram; on numpy
    they are taken on whichever side is cheaper -- the row Gram, O(N^2 F), or the feature side,
    O(N F^4) with no ``N x N`` array (:func:`_coherence_feature_side`) -- exactly either way.
    """
    xp = _env.ns(screen)
    N = int(screen.shape[0])
    if N < 2 * lag + 2 or lag < 1 or N < 4:
        return 0.0
    if not _env.is_torch(xp):
        # The moments are sums over the N x N row Gram, and each has an exact form on the feature
        # side (G is the real Gram of the real embedding Y): the row Gram costs N^2 F', the feature
        # side N F'^4, so the cheaper is taken -- both exact, the value is the same either way.
        Y = np.asarray(_env.to_numpy(screen))
        Y = np.concatenate([Y.real, Y.imag], axis=1) if np.iscomplexobj(Y) else Y.astype(float)
        Fp = int(Y.shape[1])
        if Fp ** 3 < N:
            return _coherence_feature_side(Y, lag)
    G = xp.real(screen @ xp.conj(screen).T)        # (N, N) Hermitian row Gram
    # `R` is the SQUARE of `G` and the moments below take squares of ITS row sums, so the
    # arithmetic reaches the eighth power of the screen and overflowed to `nan` above a screen
    # magnitude of ~1e38.  The z-score is scale-invariant by construction (numerator and
    # standard deviation carry the same power), so dividing `G` by its own peak first is exact
    # in the value and removes the cliff.
    gmax = float(_env.to_numpy(xp.max(xp.abs(G))))
    if gmax > 0.0:
        G = G / gmax
    R = G ** 2                                     # squared real inner products (symmetric)
    d = xp.diagonal(R)                             # main diagonal = ||row||^4
    S1 = float(_env.sum_ax(xp, R)) - float(_env.sum_ax(xp, d))            # sum over off-diag pairs
    S2 = float(_env.sum_ax(xp, R * R)) - float(_env.sum_ax(xp, d * d))    # sum of squares, off-diag
    rowsum = _env.sum_ax(xp, R, 1) - d             # per-row sum excluding the diagonal (u_a)
    U = float(_env.sum_ax(xp, rowsum * rowsum))    # sum of squared off-diagonal row sums
    Asum = float(_env.sum_ax(xp, xp.diagonal(R, lag)))       # the lag-th superdiagonal
    return _coherence_z(S1, S2, U, Asum, N, lag, R)


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
    centre, scale             : the whitening map on the SCREEN's feature grid -- a screen-side
                                array ``x`` is read back into the caller's units as
                                ``x * scale + centre`` (see ``_screen_units``)
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

        # A channel the whitening cannot give a scale (scale 0: every measured value is the same --
        # it never moved) carries nothing to the screen, which
        # reads the centred record -- yet its level would still steer the entropy fold and dilute
        # the columns it is folded into, and the floor would count it as a unit-noise dimension.
        # With two constant channels beside eight of noise the read claimed structure in 52% of
        # noise-only records (96% with eight).  So it leaves the screen the way a dead channel does:
        # out of the geometry, the fold and the floor.  It was measured, though, so ``_flat_cols``
        # records it apart from the dead ones, and a filter keeps it at its centre.
        # Taking a flat channel out can leave a row whose only measured cells were on it; that row
        # is then dead, and dropping it changes the measured cells the other channels' statistics
        # are read over.  The set is taken to its closure (it only shrinks, so at most F passes).
        self._flat_cols = None
        stats = normalize(W, mask, return_stats=True)
        flat = np.asarray(_env.to_numpy(stats[2])) == 0
        while flat.any() and not flat.all():
            W, xp = _env.to_numpy(W), np
            lr_full = np.ones(self.T, bool) if self._live_rows is None else self._live_rows.copy()
            lc_full = np.ones(self.F, bool) if self._live_cols is None else self._live_cols.copy()
            flat_full = np.zeros(self.F, bool) if self._flat_cols is None else self._flat_cols.copy()
            flat_full[np.flatnonzero(lc_full)[flat]] = True
            self._flat_cols = flat_full
            lc_full = lc_full & ~flat_full
            W = W[:, ~flat]
            if mask is not None:
                mask = mask[:, ~flat]
                dead = np.all(mask, axis=1)
                if dead.any():
                    lr_full[np.flatnonzero(lr_full)[dead]] = False
                    W, mask = W[~dead], mask[~dead]
                if not mask.any():
                    mask = None
            self._live_rows, self._live_cols = lr_full, lc_full
            stats = normalize(W, mask, return_stats=True)          # whiten what remains
            flat = np.asarray(_env.to_numpy(stats[2])) == 0

        geom = geometry(W, mask, far=far)             # the fold gate at the read's own level
        self.H_T = geom["H_T"]
        self.H_F = geom["H_F"]
        self.n_T = geom["n_T"]
        self.n_F = geom["n_F"]
        self.delta_T = geom["delta_T"]
        self.delta_F = geom["delta_F"]
        self._geom = geom

        data, centre, scale = normalize(W, mask, return_stats=True) if stats is None else stats
        # Every whitened column has sum |z|^2 = T, so |z| <= sqrt(T) and the screen's squares stay
        # far inside the float range at any record's level and in any working precision.
        self.screen = project(data, self.delta_T, self.delta_F)
        self.centre, self.scale = self._screen_units(centre, scale)

        self.null = null
        self._xp = xp; self._far = float(far); self._seed = int(seed)
        self._U = self._Vt = None      # full SVD basis -- lazy (heavy; only beam / footprints)
        self._coh = None               # coherence -- lazy (a separate lag-1 pass)
        # Lightweight signal decision: only the singular values drive the noise floor and
        # K_signal, so use ``svdvals`` (no U / Vt). The full SVD basis, the beam, and
        # the coherence are heavy and deferred to first access -- a streaming monitor that
        # reads K_signal / has_signal never pays for them (the high-speed capture path).
        S = _env.svdvals(xp, self.screen)
        self._draw = _screen_resample(W, mask, data, far)
        self._perm_sig = None
        floor = noise_floor(self.screen, far=far, null=null, s=S, seed=seed,
                            resample=self._draw)
        sig = S[S > floor]
        self.K_signal = int(sig.shape[0])
        self.H_screen = shannon_bits(sig ** 2) if int(sig.shape[0]) else 0.0
        self.S = S
        self.noise_floor = floor
        self.sigma_top = float(self.S[0]) if int(self.S.shape[0]) else 0.0

    def _screen_units(self, centre, scale):
        """Carry the whitening map onto the screen's feature grid: ``(centre, scale)``.

        ``normalize`` reports one ``(centre, scale)`` per INPUT channel; the screen's feature
        axis is those channels folded to ``F_eff``.  A screen-side array is therefore read back
        into the caller's units by this pair and not by the input-channel one, so the fold is
        applied to the stats with the same :func:`_fold_axis` applied to the data -- one fold,
        not two conventions.  Only the F fold matters: the whitening is per column, so folding
        the ordered axis moves no channel's units (it changes the row count, which the shape
        already says).

        The map is the channel's own in two of the fold's three regimes, both of which leave a
        screen column standing for exactly ONE whitened channel: ``F_eff == F`` returns the
        input unchanged, and ``F_eff > F`` is a nearest-block hold, which holding the stats the
        same way inverts.  When the axis COARSENS, screen column ``j`` is ``sqrt(n_j)`` times the
        mean of ``(W[:, i] - c_i) / s_i`` over its run of ``n_j`` channels, and the map is the run's
        common scale over ``sqrt(n_j)``.  That
        is the right map rather than a fallback, for two reasons that hold together:

          * It is EXACT for whatever is common across the group.  If the group's whitened
            channels share a component ``w``, then ``mean_i(s_i w + c_i) = w * mean(s) +
            mean(c)`` -- and a component common across channels is precisely what a resolved
            mode is, so it is exactly the content ``clean`` carries.  The error lives in what
            differs within a group, which is the noise the filter has already attenuated.
          * The fold merges only ADJACENT channels, and it coarsens only when the feature
            marginal is smooth -- which is the same statement as neighbouring channels being
            alike.  Measured across a gain ramp swept from 1x to 60x over the band, the worst
            WITHIN-group scale ratio stayed at ~1.3 while the across-band ratio reached 60.
            The regime that would strain this map is the regime that does not fold."""
        F_eff = int(self.screen.shape[1])
        n_in = int(centre.shape[0])
        if F_eff == n_in:
            return centre, scale
        if F_eff > n_in:
            fold = lambda v: _fold_axis(v.reshape(1, -1), F_eff, axis=1).reshape(-1)
            return fold(centre), fold(scale)
        # a coarsened column j is sum_i z_i / sqrt(n_j) over its run: the run's mean centre, and its
        # mean scale over sqrt(n_j), read a component common to the run back into the caller's units
        xp = _env.ns(centre)
        rt = np.sqrt(np.diff(_groups(n_in, F_eff)).astype(float))
        if _env.is_torch(xp):
            rt = xp.as_tensor(rt, dtype=scale.dtype, device=scale.device)
        mean = lambda v: _fold_ortho(xp, v.reshape(1, -1), F_eff, axis=1).reshape(-1) / rt
        return mean(centre), mean(scale) / rt

    def refloor(self, null):
        """This same projection, floored by a different ``null`` provider.

        ``null`` reaches exactly one line of ``__init__`` -- ``noise_floor(..., s=self.S)`` -- and
        that line is handed the spectrum rather than the data. So the provider chooses WHERE the
        floor sits in an already-computed spectrum; it does not change the decomposition, the
        screen, or the normalization that produced either. Rebuilding a whole ``Projection`` to
        change it recomputes an identical ``svdvals``.

        Measured 2026-08-27 on a 64-patch sweep at ``patch=256``: ``sweep(null="local")`` built
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
        other._perm_sig = None
        floor = noise_floor(self.screen, far=self._far, null=null, s=self.S, seed=self._seed,
                            resample=self._draw)
        sig = self.S[self.S > floor]
        other.K_signal = int(sig.shape[0])
        other.H_screen = shannon_bits(sig ** 2) if int(sig.shape[0]) else 0.0
        other.noise_floor = floor
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
        monitor read. For the same gate on a raw frame without building a Projection, use the
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
        """Per-mode evidence against the noise null of the floor this projection used, with
        ``K_signal == #(p_k <= far)``.

        Under an exact permutation floor (the default, or any ``permutation()``, directly or
        through a mapping) ``p_k`` is the exact Monte Carlo p-value of ``s_k`` against the same
        surrogate draws the floor was read from, ``(1 + #{draws >= s_k}) / (n + 1)``, each draw
        and the observed read carrying its round-off bound as the floor does; it resolves steps
        of ``1 / (n + 1)``, so the identity holds at the level the draws were sized for (the
        smallest attainable p is that level).  Under ``null=mp``, and under any other provider,
        it is the closed-form Tracy-Widom evidence of :func:`mode_significance` (and the identity
        is ``mp``'s, with ``<``).  ``deviate`` is the Tracy-Widom deviate in every case: where
        each value sits against the light-tailed edge."""
        closed = mode_significance(self.screen, self.S)
        spec = getattr(_select_provider(self.null, "projection", self.screen), "exact_permutation", None)
        if spec is None:
            return closed
        return self._c_sig(closed, spec)

    def _c_sig(self, closed: ModeSignificance, spec) -> ModeSignificance:
        if getattr(self, "_perm_sig", None) is None:
            draws, far = spec
            far = self._far if far is None else far
            n = _fewest_draws(far) if draws is None else int(draws)
            rng = np.random.default_rng(self._seed)            # the floor's own draws, in order
            scr = np.asarray(_env.to_numpy(self.screen))
            scale = _screen_scale(scr)
            top = _top_spectrum_value(scr, "projection")
            obs = 0.0 if scale is None else _roundoff_deviate(scr, top, scale)
            tops = np.sort([_scored(self._draw(rng), "projection") for _ in range(n)]) + obs
            sv = np.asarray(_env.to_numpy(self.S), dtype=float)
            g = np.array([_deviate(float(v), scale) for v in sv])     # each s_k as the floor ranks it
            ge = n - np.searchsorted(tops, g, side="left")       # draws at or above each s_k
            self._perm_sig = ModeSignificance(deviate=closed.deviate, pvalue=(1.0 + ge) / (n + 1.0))
        return self._perm_sig

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
    parameter-free by default: the entropy-matched fold and the exact permutation floor carry
    no fitted constant; ``far`` is the floor's significance level (default 5%).  ``null`` is a
    null-provider callback selecting the noise floor (``None`` = the exact permutation test; see
    :func:`noise_floor` and :mod:`null_providers`).  Deterministic (a resampling provider is
    deterministic per ``seed``).  (= ``Projection(W, ...).read()``.)"""
    return Projection(W, mask=mask, far=far, null=null, seed=seed).read()


def probe_signal(W, mask=None, *, far: float = 0.05, null=None, seed: int = 0) -> bool:
    """Signal gate for high-speed / streaming capture: the fold, the screen's singular values and
    the noise floor, and nothing else -- no basis, embedding or coherence.  Returns True iff the
    top singular value clears the floor, so a monitor can decide whether to build the full
    :class:`Projection` without one.  The floor and the count read the spectrum, so the gate costs
    the singular values; what it saves is everything after them.  A masked / gapped frame returns True (defer to the full Projection).  ``null``
    scores the floor as in :func:`noise_floor`."""
    xp = _env.ns(W)
    if mask is not None or not bool(xp.all(xp.isfinite(xp.abs(W)))):
        return True                                    # masked / gapped -> defer to the full Projection
    from .entropy import whiten_stats
    flat = np.asarray(_env.to_numpy(whiten_stats(xp, W)[1])) == 0      # as Projection leaves them out
    if flat.any() and not flat.all():
        W = W[:, xp.as_tensor(~flat, device=W.device) if _env.is_torch(xp) else ~flat]
    geom = geometry(W, mask, far=far)
    data = normalize(W, mask)
    screen = project(data, geom["delta_T"], geom["delta_F"])
    S = _env.svdvals(xp, screen)
    floor = noise_floor(screen, far=far, null=null, s=S, seed=seed,
                        resample=_screen_resample(W, mask, data, far))
    return bool(int(S.shape[0]) and float(S[0]) > floor)


# ══════════════════════════════════════════════════════════════════════════════
# Batched monitor path -- the throughput lever for ensembles (numpy / CPU)
# ══════════════════════════════════════════════════════════════════════════════
# The per-frame Python loop (wrapper plane reduction, ensemble traces) pays call + object
# overhead per frame and vectorizes nothing.  ``read_batch`` folds + svd-values + floors a
# stack of same-shape frames in one vectorized pass -- bit-identical to reading each frame
# with ``Projection`` (the fold is per-column, the per-channel stats are per-frame, so batching
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
    from .entropy import _carried
    stack = _env.stack(xp, [_carried(xp, stack[b]) for b in range(B)], 0)   # as geometry carries
    P = _env.asnum(xp.abs(stack)); P = P * P                       # |W|^2 (compute precision)
    marg = _env.sum_ax(xp, P, 1)                                   # (B, F) feature power marginal
    tot = _env.sum_ax(xp, marg, 1)                                 # (B,) on xp
    # Batched shannon_bits over the feature marginal (no per-frame loop -- a per-frame call would
    # serialise B tiny reductions on the GPU).  Same formula as ``entropy.shannon_bits``:
    # H = -sum p log2 p, p = marg / sum(marg), with the p>0 guard and [1e-12,1] clip in the log.
    # EXACT tests, matching `entropy.shannon_bits`: the weights are non-negative, so `> 0` is
    # "no weight at all", not "less weight than some number".  This replaces an absolute `1e-30`
    # on `tot` (a POWER SUM, so dimensioned) and a `1e-12` clip on the probabilities.  Both are
    # reachable in principle -- `tot` falls below 1e-30 for a screen scaled near 1e-16 -- but NO
    # input has been found where either changed a returned `F_eff`, so this is agreement with the
    # per-frame path, which this function's own docstring promises, and not a fixed defect.
    safe_tot = xp.where(tot > 0, tot, xp.ones_like(tot))       # empty rows are discarded below
    p = marg / safe_tot[:, None]
    plog = xp.where(p > 0, p * xp.log2(xp.where(p > 0, p, xp.ones_like(p))), xp.zeros_like(p))
    H = -_env.sum_ax(xp, plog, 1)                                  # (B,) on xp
    log2F = float(np.log2(F))
    tot_np = np.asarray(_env.to_numpy(tot))
    H_F = np.where(tot_np > 0.0, np.asarray(_env.to_numpy(H)), log2F)   # P_total<=0 -> maximal entropy
    # The same `fold_width` the per-frame `geometry` calls -- including its continuity gate --
    # so the batched read stays bit-identical to reading each frame with `Projection`.  Only the
    # H_F reduction is batched; the decision itself is per frame because continuity is.
    F_eff = np.empty(B, dtype=int)
    for b in range(B):
        F_eff[b] = fold_width(float(H_F[b]), N, F, stack[b], far=far)[0]
    return F_eff


def normalize_batch(xp, stack):
    """Whiten a real, finite ``(B, N, F)`` stack -- ``entropy.normalize``'s clean path over axis 1,
    batched: each frame's channels centred on their mean and scaled by their RMS about it, through
    the same core (:func:`entropy._whiten_core`), so it is bit-identical on numpy to
    the per-frame read."""
    from .entropy import _whiten_core
    data = _env.asnum(stack)
    xs, cs, ss, _ = _whiten_core(xp, data, axis=1)
    safe = ss > 0.0
    zero = _env.zeros(xp, tuple(int(v) for v in data.shape),
                      ref=(data if _env.is_torch(xp) else None))
    return xp.where(safe, (xs - cs) / xp.where(safe, ss, xp.ones_like(ss)), zero)


def project_batch(xp, data, F_eff: int):
    """Fold a whitened real ``(B, N, F)`` stack to ``(B, N, F_eff)`` -- ``project`` over the
    feature axis (axis 2); the ordered axis is never folded (delta_T=1).  Backend-agnostic."""
    out = xp.where(xp.isfinite(xp.abs(data)), data, xp.zeros_like(data))
    out = _fold_ortho(xp, out, F_eff, axis=2)
    return _env.as_compute(xp, out)


def read_batch(frames, *, far: float = 0.05, null=None, seed: int = 0) -> list[BatchRead]:
    """Read a batch of same-shape frames onto their screens in one vectorized pass and return
    a per-frame :class:`BatchRead` (``K_signal`` / ``sigma_top`` / ``noise_floor`` / ``S``),
    **bit-identical** to ``[Projection(f).read()-equivalent for f in frames]`` but amortizing the
    per-frame fold + object overhead (the ensemble throughput lever at small ``F``).  Frames
    are grouped by their feature-fold target so each group folds to one width and svd-values in
    one ``linalg.svd`` call.  ``null`` scores the floor as in :func:`noise_floor` (``None`` = the
    exact permutation test, per frame, with each frame's own draws; ``null_providers.mp`` = the
    closed-form edge, batched; any other provider is applied per frame -- still one fold/SVD pass).
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
        # a frame with a channel the whitening could not scale leaves that channel out of its
        # screen (Projection._flat_cols): read it per frame, exactly as Projection reads it
        zero = np.all(data == 0, axis=1)                                      # (B, F)
        solo = zero.any(axis=1) & ~zero.all(axis=1)
        for k in np.flatnonzero(solo):
            sc = Projection(frames[int(idx[k])], far=far, null=null, seed=seed)
            out[int(idx[k])] = BatchRead(sc.K_signal, sc.sigma_top, float(sc.noise_floor), sc.S)
        for Fe in np.unique(F_eff_all):                                       # group by fold width
            sel = np.where((F_eff_all == Fe) & ~solo)[0]
            if not sel.size:
                continue
            screen = project_batch(np, data[sel], int(Fe))                   # (Bg, N, Fe)
            S = np.linalg.svd(screen, compute_uv=False)                       # (Bg, min(N,Fe))
            if null is _mp:                                                   # batched mp floor
                floors = _mp_floor_batch(screen, S, N, int(Fe), far)
            else:                                    # each frame's own floor, as Projection's
                floors = np.array([noise_floor(screen[j], far=far, null=null, s=S[j], seed=seed,
                                               resample=_screen_resample(frames[int(idx[k])],
                                                                         None, data[k], far))
                                   for j, k in enumerate(sel)])
            for j, k in enumerate(sel):
                s = S[j]; fl = float(floors[j])
                K = int((s > fl).sum())
                out[int(idx[k])] = BatchRead(K, float(s[0]) if s.shape[0] else 0.0, fl, s)
    return out


def _mp_floor_batch(screen: np.ndarray, S: np.ndarray, N: int, F_eff: int, far: float) -> np.ndarray:
    """The derived ``mp`` screen floor ``sqrt(sigma^2 * (mu + q*sigma_J))`` for a ``(B, N, F_eff)``
    batch -- the same edge :func:`noise_floor` computes per frame, vectorized over the batch.
    Shares the de-biasing denominator and the Johnstone edge with the per-frame ``mp`` provider
    (via ``null_providers``) so the batch and per-frame floors cannot drift."""
    sigma2 = _noise_sigma2_from_spectrum(np.asarray(S, dtype=np.float64) ** 2, N, F_eff)
    return np.sqrt(_screen_floor_sq(sigma2, N, F_eff, far))
