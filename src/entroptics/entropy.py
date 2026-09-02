"""
entropy.py -- Entropy geometry: the matched scale read from a signal's own
Shannon entropy, plus the fold (fractional resample) and normalize.

This is the entropy side of Entroptics.  The optics side is reads.py / aperture.py;
the projection side is projection.py.  Standalone: numpy only.

Axis convention for every 2-D input W (shape (T, F)):
  * axis-0 (rows, "T") is the ordered / evolution axis   -> subscript _T
  * axis-1 (cols, "F") is the feature / channel axis      -> subscript _F
"time"/"frequency" are roles, not literal physics -- any 2-D array with one
ordered axis works (spectrogram, waterfall, embedding stack, market panel, image).

Per-axis geometry symbols (a = T or F):
  H_a      Shannon entropy (bits) of that axis' power marginal
  n_a      effective mode count = round(2^H_a)
  delta_a  matched cell scale  = len_a / 2^H_a   (delta_T window width, delta_F bin width)
"""
from __future__ import annotations

import logging as _logging

import numpy as np

from . import environment as _env

try:
    from scipy.stats import norm as _scipy_norm
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    _scipy_norm = None  # type: ignore[assignment]

# MAD_SCALE: median-absolute-deviation -> Gaussian sigma.  Exact 1/Phi^{-1}(0.75).
#
# The fallback carries the same number to full float64 precision, so a scipy-less install reads
# the same rank as a scipy-having one.  The constant propagates through the per-channel whitening
# into `noise_sigma2`, into the floor, and out through `k`, so a 5-dp abbreviation (~1.5e-7
# relative) moves results.
#
# The scipy branch is the derivation: if scipy ever disagrees with the literal, that is a finding to act on.
_MAD_SCALE_EXACT: float = 1.482602218505602   # 1/Phi^{-1}(0.75), float64-exact

if _SCIPY_AVAILABLE:
    MAD_SCALE: float = float(1.0 / _scipy_norm.ppf(0.75))
else:  # pragma: no cover - exercised only on a scipy-less install
    _logging.getLogger(__name__).warning(
        "entroptics: scipy is unavailable, so MAD_SCALE is taken from its exact float64 "
        "literal (%r). The value is "
        "identical; this is logged because the derivation was not re-run, not because the "
        "number changed.", _MAD_SCALE_EXACT,
    )
    MAD_SCALE: float = _MAD_SCALE_EXACT

# MAD_LOGVAR: the asymptotic sampling variance of log(MAD-hat) from N Gaussian samples is
# ~ MAD_LOGVAR / N.  This is the analytic influence-function variance 1/(16 f(D)^2 D^2) =
# 1.36046, evaluated at D = Phi^{-1}(3/4) = 0.67449 with f the standard-normal density
# (equivalently CV(MAD/sigma) = sqrt(1.36046)/sqrt(N) = 1.166/sqrt(N)).  Derived like
# MAD_SCALE, not fitted; sets the shrinkage of noisy small-N per-channel scales toward the
# pooled one (see normalize / _shrink_mad).
MAD_LOGVAR: float = 1.36046


def shannon_bits(weights, axis=None):
    """Shannon entropy (bits) of a non-negative weight array -- the ONE definition
    used across Entroptics (geometry marginals, screen mode weights, decay/axis
    spectra).  Normalises internally.  Backend-agnostic (numpy or torch).
    H(w) = -sum p log2 p, p = w / sum w.

    ``axis=None`` (the default) reduces the whole array and returns a Python float: the
    behaviour every existing caller relies on, unchanged.

    Give ``axis`` to reduce along that axis instead and get one entropy per remaining index,
    as an array.  It is the SAME definition applied N times -- not a new read -- and it exists
    because a caller scoring a stack of candidates otherwise pays N round trips through Python
    for arithmetic that vectorises exactly.  Measured on a 121-candidate blind scan: 258 ms of
    scalar calls against 93 ms for the same numbers in one call per candidate.  A slice with no
    weight reports 0.0, the same convention the scalar path uses."""
    xp = _env.ns(weights)
    if axis is None:
        s = float(_env.sum_ax(xp, weights))
        if s <= 0.0:                      # exact: the weights are non-negative, so this is
            return 0.0                    # "no weight at all", not "less weight than some number"
        p = weights / s
        return -float(_env.sum_ax(
            # log2 is taken on the branch `p > 0` only; substituting 1.0 in the discarded branch
            # keeps log2 finite there without altering any p the sum actually uses.  A floor on
            # p would change H for a frame whose weights run below it.
            xp, xp.where(p > 0, p * xp.log2(xp.where(p > 0, p, xp.ones_like(p))), xp.zeros_like(p))))
    # Axis form: the same two branches, kept elementwise.  The normaliser is guarded rather
    # than tested, because one empty slice must not decide the result for the others.
    tot = _env.sum_ax(xp, weights, axis, keep=True)
    p = weights / xp.where(tot > 0, tot, xp.ones_like(tot))
    terms = xp.where(p > 0, p * xp.log2(xp.where(p > 0, p, xp.ones_like(p))), xp.zeros_like(p))
    H = -_env.sum_ax(xp, terms, axis)
    live = _env.sum_ax(xp, weights, axis)
    return xp.where(live > 0, H, xp.zeros_like(H))


def surprisal_bits(observed, total):
    """Self-information (surprisal) in bits: I(x) = -log2(p), p = observed / total.

    The per-event half of `shannon_bits` -- entropy is its expectation,
    H = sum p * I(x) -- so both live here, share the base, and share the clip.
    One definition on purpose: a second `-log2` written at a call site is how two
    quantities that must agree stop agreeing, silently, while both keep returning
    plausible bit-counts.

    How much does observing x narrow the field?  Something present in nearly every
    observation carries almost nothing (p -> 1, I -> 0); something rare carries a
    lot.  That is the entire content of the measure, and it is what lets an
    observation count rank a symbol without any hand-written list of which symbols
    are supposed to matter.

    Returns None when no probability can be formed -- total <= 0, observed <= 0,
    observed > total, or a non-numeric argument.  NOT 0.0: zero bits is a real
    reading (the thing is everywhere and tells you nothing), and an unmeasurable
    count must never be reported as one that was measured.
    """
    try:
        obs = float(observed)
        tot = float(total)
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(obs) and np.isfinite(tot)):
        return None
    if tot <= 0.0 or obs <= 0.0 or obs > tot:
        return None
    p = obs / tot                           # in (0, 1] already: obs > 0, tot > 0, obs <= tot
    return float(-np.log2(p)) + 0.0         # + 0.0 normalises the -0.0 that p == 1 produces


def joint_power(W_x, W_y, mask_x=None, mask_y=None):
    """The (F1, F2) joint power table of two co-registered frames -- `J[i, j]` is the power that
    channel `i` of the first frame and channel `j` of the second put on the screen together,
    accumulated over the shared ordered axis:

        J = P1^T P2,      P = |W|^2 over the cells that carry a measurement

    This is 1948 section 12's `p(i, j)`, "the probability of the joint occurrence of `i` for the
    first and `j` for the second", read on power, the intensity of each cell.

    Which joint this is: expanding the sum, `p(i,j) = sum_t p(t) p(i|t) p(j|t)` -- the two frames
    are conditionally independent given the ordered index, and the shared ordering is the only
    channel through which they communicate.  This is exactly what co-registration asserts: the
    axis is the correspondence, and there is no other pairing between an `F1`-alphabet and an
    `F2`-alphabet to appeal to.

    Lesne (MSCS 2014, section 2.3, eq. 7) states the standard result that `I(X;Y) = H(X)` when
    `X = Y` -- true of one random variable observed twice, where the joint sits on the diagonal.
    That does not hold here: handing the same frame in twice draws the two channel indices
    independently from each row's own power profile, so a frame whose channels fire together
    correctly reads as telling little about itself.  The pairing that saturates is a per-row
    deterministic one -- each row's power in a single channel, coupled by a permutation -- and
    that pairing is the ceiling control in the tests.

    Co-registration is the precondition, and it is enforced.  Two channels each
    ordered by its own private axis superpose into noise, and a joint read over them is a
    measurement of the misalignment.  The ordered lengths must match
    and a mismatch raises: degrading to a truncation or a resample would publish a number about a
    frame nobody handed in.

    A cell absent in either frame is absent from the joint.  Nonfinite and masked cells carry no
    power into the product, on the same rule :func:`geometry` applies per axis: a cell that was not
    measured is not a cell carrying zero power.  The feature counts may differ freely -- `F1` and
    `F2` are separate alphabets and nothing here requires them to be the same one."""
    xp = _env.ns(W_x)
    A, B = _env.asnum(xp.abs(W_x)), _env.asnum(xp.abs(W_y))
    if len(A.shape) != 2 or len(B.shape) != 2:
        raise ValueError("joint_power expects two 2-D (ordered, feature) frames; "
                         f"got shapes {tuple(A.shape)} and {tuple(B.shape)}")
    if int(A.shape[0]) != int(B.shape[0]):
        raise ValueError(
            "the two frames are NOT co-registered: ordered lengths %d and %d. A joint read needs "
            "ONE shared ordering axis -- align them before reading."
            % (int(A.shape[0]), int(B.shape[0])))
    A = xp.where(xp.isfinite(A), A, xp.zeros_like(A))
    B = xp.where(xp.isfinite(B), B, xp.zeros_like(B))
    if mask_x is not None:
        A = xp.where(mask_x, xp.zeros_like(A), A)
    if mask_y is not None:
        B = xp.where(mask_y, xp.zeros_like(B), B)
    return (A * A).T @ (B * B)


def joint_entropies(W_x, W_y, mask_x=None, mask_y=None) -> dict:
    """H(X), H(Y) and H(X, Y) of two co-registered frames, all read off one joint table.

    The shared computation is the point.  Every quantity below is a difference of these three,
    and computing them separately -- one marginal here, another there, the joint somewhere else --
    makes the identities that define them hold only to float noise, and only while three code paths
    agree about normalisation, about the clip, and about which cells were absent.  Taking all three
    from a single `J` makes them hold exactly and makes disagreement impossible:

        I_XY   = H(X) + H(Y) - H(X,Y)        the mutual information
        H_Y(X) = H(X,Y) - H(Y)               the equivocation
        I_XY   = H(X) - H_Y(X)               1948 section 12's rate, R

    Returns `{"H_X", "H_Y", "H_XY", "I_XY", "H_X_given_Y", "H_Y_given_X"}` -- the three entropies and the
    three differences, so a caller that wants two of them pays for one table."""
    J = joint_power(W_x, W_y, mask_x, mask_y)
    xp = _env.ns(J)
    H_XY = shannon_bits(J)
    H_X = shannon_bits(_env.sum_ax(xp, J, 1))     # row marginal: the first frame's channels
    H_Y = shannon_bits(_env.sum_ax(xp, J, 0))     # column marginal: the second frame's
    return {
        "H_X": H_X, "H_Y": H_Y, "H_XY": H_XY,
        "I_XY": H_X + H_Y - H_XY,
        "H_X_given_Y": H_XY - H_Y,                   # uncertainty about X once Y is known
        "H_Y_given_X": H_XY - H_X,
    }



# Naming:
#
#   · the mutual information is `I_XY`, never a bare `I`. `I` already means incident energy in the
#     conservation identity this instrument publishes -- `||I||^2 = ||A||^2 + ||T||^2` -- and two
#     unrelated quantities under one letter, inside one instrument, is the defect the three
#     separately-named "coherence" reads exist to warn about.
#   · the arguments carry the same subscript as the keys, so `W_x` produces `H_X` and no caller has
#     to hold a mapping in their head. Each layer keeps its own noun for a frame -- entroptics `W`,
#     prism `frame`, the aperture `rows` -- and only the subscript is shared, because the noun is
#     that layer's vocabulary and the subscript is Shannon's.
def geometry(W: np.ndarray, mask: np.ndarray | None = None, *, far: float = 0.05) -> dict:
    """Read the natural matched scale of a 2-D observation W (T x F) from its own
    Shannon entropy.

    H_T (ordered) and H_F (feature) are the Shannon entropies of the global power
    marginals of P = |W|^2:

        p_t[t] = sum_f P[t,f] / sum P        (ordered marginal)
        p_f[f] = sum_t P[t,f] / sum P        (feature marginal)

    Power-weighting lets bright (on-signal) rows dominate regardless of what
    fraction of the capture they occupy.

    Returns
    -------
    dict with per-axis keys (a = T, F):
        H_T, H_F          : marginal Shannon entropies (bits)
        n_T, n_F          : effective mode counts = round(2^H_a)
        delta_T, delta_F  : matched cell scales.  delta_T := 1.0 always (the ordered axis
                            is never folded, see fold policy below); only delta_F carries the
                            entropy ratio len_F / 2^H_F (float >= 1, the feature bin width).

    delta_F is the real (un-floored) matched-scale ratio -- one parameter-free scale derived
    from the signal's own Shannon entropy.  The feature fold (normalize/project)
    resamples fractionally to round(2^H_F) cells, or takes the exact-integer reshape fast
    path when the ratio is whole.

    Fold policy: only the feature axis folds; the ordered axis is kept at native resolution.
    The ordered reads -- coherence (adjacent-row similarity), the decay/OTF (lag structure),
    the exact rates (the ordered trajectory) -- all require native ordered spacing, and
    folding the ordered axis would blend adjacent rows into spurious correlation.  For the
    feature axis, a near-uniform marginal (structureless noise) sits below its max log2(F)
    only by a finite-sample deficit, so the fold snaps to none (delta_F = 1.0) whenever H_F lies
    within a band of log2(F).  The band is the conservative Miller-Madow uniform-null bias
    (F-1)/(2 T ln2), capped at log2(F)/2: deliberately large (neither noise nor a low-rank
    signal's delocalised marginal folds -- folding either would blur the SVD modes) but capped
    so it never exceeds log2(F) and disables the fold vacuously for wide-short data.  The cap
    means the guard always folds once power concentrates below sqrt(F) effective channels.
    Closed-form; the only choice is the sqrt(F) concentration floor.  See
    research/validation/miller_madow_check.py.

    Read BEFORE the whitening, and why that is forced.  `projection.read` runs
    `geometry -> normalize -> project`: this read is taken on the RAW frame, and only then is
    the frame whitened per channel and folded.  The order is not incidental, because the two
    steps want opposite things -- `geometry` measures how unevenly the raw amplitudes are
    spread across the channels, and `normalize` divides each channel by its OWN robust scale,
    whose whole job is to remove that unevenness.  Composed the other way, a channel carrying
    only noise is lifted to the amplitude of one carrying the signal and the concentration the
    fold exists to find is gone: measured on a planted line of known width, `n_F` tracks the
    line width in 9/9 cases read on the raw frame and returns `n_F = F` in 9/9 read after
    whitening, the on-line channels holding 88.2% of the power before and 9.6% after
    (research/validation/exp16_scale_before_whitening.py).

    PRECONDITION -- the feature channels must be commensurate.  Because this read is taken on
    raw amplitudes, `H_F` / `n_F` / `delta_F` only mean "how many channels are in play" when
    the channels are already in the same units.  A frame of mixed units reads its units: eight
    share prices in dollars, where one ticker's number has four digits and another's has two,
    give `2^H_F = 1.5 of 8` -- a true statement about the numbers and a false one about the
    market.  This cannot be detected from one frame, since scaling a column by c is
    indistinguishable from that channel being c times louder, so it is declared by the caller
    like which axis is the ordered one, and it is a property of the INPUT rather than a
    parameter of the read.  It is also not something to fix by pre-standardising each channel:
    that is precisely the whiten-first order measured above.  For a panel whose columns are
    incommensurate levels, hand in per-row innovations (differences) rather than levels -- the
    same quantity in every cell -- and let `normalize` do the scaling.
    """
    xp = _env.ns(W)                      # numpy or torch -- ONE code path (GPU when fed a tensor)
    T, F = int(W.shape[0]), int(W.shape[1])

    P = _env.asnum(xp.abs(W))           # |W| as float (real, complex, negative all fine)
    have = xp.isfinite(P)               # present cells -- reused for the extents below
    P = xp.where(have, P, xp.zeros_like(P))             # NaN/Inf -> 0 (handles gaps)
    if mask is not None:
        P = xp.where(mask, xp.zeros_like(P), P)
        have = have & (mask == False)   # noqa: E712 -- `~mask` is not backend-portable
    P = P * P                           # weight peaks, suppress noise floor

    # ── an unmeasured cell is not a cell carrying zero power ──────────────────────────────────
    # Zeroing above is right for the power sum: an absent cell contributes nothing, and
    # `0 log 0 = 0` leaves the entropy itself untouched.  But `T, F` off `W.shape` would size
    # every quantity the entropy is compared against, and the entropy only ever appears in a
    # comparison:
    #
    #     H_F = log2(F)                  the no-signal maximum
    #     H_F >= log2(F) - fold_band     the concentration test
    #     fold_band(T, F) = min((F-1)/(2 T ln2), log2(F)/2)   the capped Miller-Madow band
    #     delta_F = F / 2^H_F            the matched cell scale
    #
    # so widening the axis with cells nothing was observed in raises the bar the signal must
    # clear, without adding any signal.  The extent used below is instead the number of cells
    # carrying a measurement.  A row or column with no finite, unmasked cell is absent, not
    # empty -- and absence is not an observation of zero.  Both extents floor at 1: an entirely
    # unmeasured frame has one degenerate cell per axis, log2(1) = 0 bits, the honest reading
    # (no channels resolved).  log2(F) would assert spread that was never observed.
    #
    # Absent means nonfinite or masked.  An all-zero-but-finite column is not absent: zero is a
    # real observation of no power there, it belongs in the extent, and is rightly held against
    # a signal that failed to reach it, even though the two look identical downstream.
    #
    # Reduced on the backend, with only the two small boolean vectors crossing to the host --
    # `geometry` is called by every read, so materialising the whole frame here would put a
    # device-to-host copy of it on the hot path.
    F_eff = max(1, min(F, int(np.count_nonzero(np.asarray(_env.to_numpy(xp.any(have, 0)))))))
    T_eff = max(1, min(T, int(np.count_nonzero(np.asarray(_env.to_numpy(xp.any(have, 1)))))))

    P_total = float(_env.sum_ax(xp, P))
    if P_total <= 0:                     # no signal -> maximal entropy (fully spread; no fold)
        H_F = float(np.log2(F_eff))
        H_T = float(np.log2(T_eff))
    else:
        H_F = shannon_bits(_env.sum_ax(xp, P, 0))   # entropy of the feature power marginal
        H_T = shannon_bits(_env.sum_ax(xp, P, 1))   # entropy of the ordered power marginal

    # The fold decision lives in `fold_width` -- one place, shared with the batched monitor.
    # It is handed the measured extents, for the reason above: its band, its ceiling and its
    # matched scale are all read against log2(F), and absent columns must not inflate any of them.
    n_F, delta_F = fold_width(H_F, T, F, W, mask, far=far, F_eff=F_eff, T_eff=T_eff)
    # The ordered axis is kept at native resolution (never folded): the ordered reads --
    # coherence (adjacent-row similarity, section 5), the decay/OTF (lag structure, section
    # 4), and the exact rates (the ordered trajectory, section 9) -- all require native
    # ordered spacing; folding it would blend adjacent rows into spurious correlation
    # (corrupting the coherence read) and blur the SVD modes.  Only the feature axis folds.
    n_T, delta_T = T, 1.0

    return {
        "H_T": H_T, "n_T": n_T, "delta_T": delta_T,   # ordered axis
        "H_F": H_F, "n_F": n_F, "delta_F": delta_F,   # feature axis
    }


def feature_adjacency(W, mask=None) -> float:
    """Continuity of the feature axis: the adjacency z-score of neighbouring feature channels
    against the exact permutation null over channels (``projection.coherence`` read across the
    transpose -- deterministic, closed form, no RNG).

    ``> 0`` neighbouring channels are more alike than a random reordering of channels, so the
    axis is continuous and adjacency carries meaning (a frequency axis, a spatial axis).
    ``~ 0`` the channels are exchangeable -- a nominal axis, where "next to" means nothing.

    This is what licenses a fold.  Folding area-means adjacent cells, which preserves
    information only where the signal varies continuously across them; averaging two unrelated
    channels destroys both.  Continuity and sparsity are independent -- a narrow line on a
    frequency axis is sparse and folds perfectly well, while unordered channels are nominal and
    cannot be folded however concentrated their power is."""
    from .projection import coherence            # deferred: projection imports this module
    Wn = np.asarray(_env.to_numpy(W))
    if Wn.ndim != 2:
        return 0.0
    have = np.isfinite(np.abs(Wn))
    if mask is not None:
        have &= ~np.asarray(_env.to_numpy(mask), bool)

    # An absent channel is dropped, not zero-filled.  Here the difference is not a threshold
    # subtlety -- it manufactures the very thing being measured.  This read asks whether neighbouring
    # channels resemble each other; filling every unmeasured channel with the same constant 0.0 makes
    # each pair of them perfectly alike, so a run of dead channels reads as a smooth, continuous
    # stretch of axis and can license a fold across a region where nothing was observed at all.  A
    # channel with no measurement is not part of the axis for this question, so it is removed before
    # the null is taken (which also keeps the permutation null over the channels that exist).
    live = have.any(axis=0)
    if int(np.count_nonzero(live)) < 4:
        return 0.0                            # too few measured channels for the null -> no adjacency
    Wn = Wn[:, live]
    # Scattered gaps inside a live channel stay zero-filled: the channel is real and its level is
    # measured elsewhere, so 0 is the resting value.
    Wn = np.where(have[:, live], Wn, 0.0)
    return float(coherence(np.real(Wn).T.copy(), lag=1))


def feature_axis_is_continuous(W, mask=None, *, far: float = 0.05) -> bool:
    """Is the feature axis continuous enough for a fold to preserve information?  The
    one-sided decision on :func:`feature_adjacency` at the reader's level ``far`` (the z-score
    has an exact permutation null, so the level is the only input)."""
    from .null_providers import _norm_isf     # deferred: paid only when the continuity test runs
    return bool(feature_adjacency(W, mask) > _norm_isf(float(far)))


def _digamma(x: float) -> float:
    """psi(x), x > 0, numpy-only: recurrence to 6 then the standard asymptotic series."""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + float(np.log(x)) - 0.5 / x + f * (
        -1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0 + f * (-1.0 / 132.0)))))


def _trigamma(x: float) -> float:
    """psi'(x), x > 0, numpy-only."""
    r = 0.0
    while x < 6.0:
        r += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r + (1.0 / x) * (1.0 + 0.5 / x + f * (
        1.0 / 6.0 - f * (1.0 / 30.0 - f * (1.0 / 42.0 - f / 30.0))))


def _tail_multiplier(far: float) -> float:
    """The number of standard deviations that bounds an upper tail at ``far``, for ANY
    distribution with a finite variance.

    Cantelli's inequality: ``P(X - mu >= k sigma) <= 1 / (1 + k^2)``, so setting the right side
    to ``far`` gives ``k = sqrt(1/far - 1)`` exactly.  A normal quantile would be tighter --
    1.645 against 4.359 at ``far = 0.05`` -- but it would be tighter by ASSUMING normality of the
    deficit, which is not established, and reaching it without scipy means carrying a rational
    approximation whose coefficients are numerically fitted.  A derived bound that is
    conservative is the right trade for a guard: the cost of being too wide is a fold not taken,
    and the cost of being too narrow is a fold that should not have been.
    """
    return float(np.sqrt(1.0 / float(np.clip(far, 1e-9, 0.5)) - 1.0))


def dirichlet_entropy_moments(K: int, a: float) -> tuple[float, float]:
    """``(mean, sd)`` of the plug-in entropy in BITS for a symmetric ``Dirichlet(a)`` over ``K``
    components -- the exact Wolpert-Wolf moments, not an expansion.

    Verified against 20,000-draw simulation from 64x64 to 32x16384: mean to 6 significant
    figures, sd to within Monte-Carlo error at every shape.
    """
    K = max(2, int(K))
    Ka = K * a
    ln2 = float(np.log(2.0))
    mean = _digamma(Ka + 1.0) - _digamma(a + 1.0)
    t1 = (a + 1.0) / (Ka + 1.0) * ((_digamma(a + 2.0) - _digamma(Ka + 2.0)) ** 2
                                   + _trigamma(a + 2.0) - _trigamma(Ka + 2.0))
    t2 = a * (K - 1.0) / (Ka + 1.0) * ((_digamma(a + 1.0) - _digamma(Ka + 2.0)) ** 2
                                       - _trigamma(Ka + 2.0))
    return mean / ln2, float(np.sqrt(max(t1 + t2 - mean * mean, 0.0))) / ln2


def fold_band(T: int, F: int, *, far: float = 0.05) -> float:
    """The band on ``H_F`` below ``log2(F)`` within which the feature axis is NOT folded.

    A fold has to clear two bars, and each is derived from a null the instrument already carries.

    **Significance -- is the concentration real?**  Under an iid Gaussian null the feature
    marginal is a symmetric ``Dirichlet(a)`` over ``F`` components, ``a = T/2`` for real cells and
    ``a = T`` for complex (a real square is ``chi^2_1``, a complex modulus-square is ``Exp(1)``);
    ``T/2`` is taken because it carries exactly twice the deficit spread and so is conservative
    for either input.  The deficit ``D = log2 F - H_F`` then has exact moments, and the bar is
    ``E[D] + k sd(D)`` with ``k`` from Cantelli's inequality -- a distribution-free bound, so a
    pure-noise record clears it with probability at most ``far`` without assuming the deficit is
    normal.

    **Sufficiency -- is the fold worth making?**  A fold that changes the width by a fraction of a
    percent gains nothing and perturbs the floor that depends on the shape.  The floor of the
    projection sits at the Marchenko-Pastur edge ``mu = (sqrt N + sqrt F)^2`` with a Tracy-Widom
    margin ``q_far * varsigma_J``, so a width change earns its place only when it moves that edge
    by more than the margin::

        |dmu/dF| dF > q_far varsigma_J   <=>   dF > q_far sqrt(F) (1/sqrt(N) + 1/sqrt(F))^(1/3)

    which as a band on ``H_F`` is ``-log2(1 - dF/F)``.

    The band is the larger of the two: a fold must be both real and worth making.  The first term
    governs wide-short frames, where the null deficit is large; the second governs square and tall
    ones, where it is not.

    **This replaces a capped Miller-Madow band** -- ``T`` times the *mean* deficit, capped at
    ``(1/2) log2 F``.  Both numbers were consequences of summarising a tail by a mean and then
    stopping the summary running past the entropy range.  Neither term here can reach ``log2 F``,
    so no cap can bind, and neither carries a constant that is not derived from a stated null.
    """
    Fi = max(2, int(F))
    Ti = max(1, int(T))
    lgF = float(np.log2(Fi))
    far = float(np.clip(far, 1e-9, 0.5))

    mean_H, sd_H = dirichlet_entropy_moments(Fi, Ti / 2.0)
    significance = lgF - mean_H + _tail_multiplier(far) * sd_H

    from .null_providers import tw1_quantile
    dF = float(tw1_quantile(far)) * np.sqrt(Fi) * (1.0 / np.sqrt(Ti) + 1.0 / np.sqrt(Fi)) ** (1.0 / 3.0)
    dF = float(min(dF, Fi - 1.0))
    sufficiency = -float(np.log2(1.0 - dF / Fi))

    return max(significance, sufficiency)


def fold_width(H_F: float, T: int, F: int, W=None, mask=None, *, far: float = 0.05,
               F_eff: int | None = None, T_eff: int | None = None):
    """THE fold decision, in one place: ``(n_F, delta_F)`` for a feature axis of ``F`` channels
    whose power marginal has entropy ``H_F``, read from ``T`` samples.

    Two extents, and they are not the same number.  ``F``/``T`` are the array's shape -- the
    coordinates ``n_F`` is returned in, because ``n_F`` is a resample target that a caller applies
    to the real array.  ``F_eff``/``T_eff`` are the measured extents: how many channels/samples
    actually carry a finite, unmasked cell.  Every threshold below is read against the measured
    extent, because a channel nothing was ever observed in must not raise the bar the signal has
    to clear (see the note in :func:`geometry`): a sparse ``(76, 2048)`` frame with only 250
    channels actually measured resolves against ``F_eff = 250``, not the nominal ``F = 2048``.
    They default to the array shape, so a caller with no gaps is unaffected.

    A fold needs both conditions and they are independent:
      concentration  ``H_F < log2(F) - fold_band`` -- there is something to fold at all, and
                     ``2^{H_F}`` says how far.
      continuity     :func:`feature_axis_is_continuous` -- adjacency along the feature axis
                     means something, so area-meaning neighbouring cells preserves the signal.
                     Sparse-and-nominal (a few active but unrelated channels) is concentrated
                     exactly like sparse-and-continuous and must not fold: averaging unrelated
                     channels destroys both.  ``W=None`` skips the continuity test (callers that
                     have already established it).
    The continuity test runs only after concentration has put a fold on the table, so the
    common noise / delocalised / streaming path never pays for it."""
    Fe = int(F if F_eff is None else max(1, min(int(F), int(F_eff))))
    Te = int(T if T_eff is None else max(1, min(int(T), int(T_eff))))
    if H_F >= float(np.log2(Fe)) - fold_band(Te, Fe, far=far):
        return F, 1.0                     # no fold -> the array's own width, untouched
    if W is not None and not feature_axis_is_continuous(W, mask, far=far):
        return F, 1.0
    # The fold can never resolve more cells than were measured: 2^H_F is capped by Fe, not by F.
    n_real = min(float(Fe), 2.0 ** H_F)
    # `n_F` is a resample target, so it is clamped into the array's coordinates.  `delta_F` is an
    # information ratio -- measured channels per resolved cell -- so both of its sides live on the
    # measured axis.  Mixing them (F / n_real) made the matched scale grow in proportion to how many
    # absent channels happened to be appended, which is a property of the padding, not the signal.
    # With no gaps Fe == F and this is the original expression exactly.
    return max(1, min(F, int(round(n_real)))), max(1.0, Fe / n_real)


def downsample(A: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Coarsen ``A`` along ``axis`` to ``n_out`` cells (the scale > 1 regime,
    n_out <= n_in): area-weighted mean resample.  Exact block-average when the
    factor is an integer (byte-identical); area-weighted (cumsum-interpolated)
    otherwise.  Level-preserving (the per-cell value, not the summed energy).
    Real- and complex-safe; vectorised.  ``n_out == n_in`` returns ``A`` unchanged.
    Backend-agnostic (numpy or torch)."""
    xp = _env.ns(A)
    A = _env.movedim(xp, A, axis, 0)
    n_in = int(A.shape[0])
    if n_out == n_in or n_in == 0:
        return _env.movedim(xp, A, 0, axis)
    cplx = xp.is_complex(A) if _env.is_torch(xp) else np.iscomplexobj(A)
    sh = tuple(int(s) for s in A.shape[1:])
    flat = _env.asnum(A.reshape(n_in, -1), complex=cplx)
    ref = flat if _env.is_torch(xp) else None
    z = _env.zeros(xp, (1, int(flat.shape[1])), complex=cplx, ref=ref)
    cs = _env.cat0(xp, [z, _env.cumsum0(xp, flat)])
    edges = _env.linspace(xp, 0.0, float(n_in), n_out + 1, ref=ref)
    lo = _env.cliprange(xp, _env.floor_int(xp, edges), 0, n_in)
    fr = (edges - lo)[:, None]
    cs_lo = cs[lo]
    interp = cs_lo + fr * (cs[_env.cliprange(xp, lo + 1, 0, n_in)] - cs_lo)
    binned = (interp[1:] - interp[:-1]) / (edges[1:] - edges[:-1])[:, None]
    out = binned.reshape((n_out,) + sh)
    return _env.movedim(xp, out if cplx else xp.real(out), 0, axis)


def upsample(A: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Refine ``A`` along ``axis`` to ``n_out`` cells (the scale < 1 regime,
    n_out >= n_in): nearest-block hold resample (== ``np.repeat`` for an integer
    factor, so the integer-matched inverse fold is unchanged).  Level-preserving.
    ``n_out == n_in`` returns ``A`` unchanged.  Backend-agnostic (numpy or torch)."""
    xp = _env.ns(A)
    A = _env.movedim(xp, A, axis, 0)
    n_in = int(A.shape[0])
    if n_out == n_in or n_in == 0:
        return _env.movedim(xp, A, 0, axis)
    idx = _env.cliprange(xp, (_env.arange_int(xp, n_out, ref=A) * n_in) // n_out, 0, n_in - 1)
    return _env.movedim(xp, A[idx], 0, axis)


def live_view(W, mask: np.ndarray | None = None):
    """The read-side analogue of ``Projection``'s ignore-missing: drop fully-dead rows/cols
    (every cell missing or masked) and fill any remaining scattered missing cell with
    its column mean, returning a clean, NaN-free array so the correlation / SVD reads
    never see a gap.  No missing data -> returns ``W`` unchanged (backend preserved);
    otherwise returns a numpy array."""
    xp = _env.ns(W)
    bad = ~xp.isfinite(xp.abs(W))
    if mask is not None:
        bad = bad | mask
    if not bool(bad.any()):
        return W
    Wn = np.asarray(_env.to_numpy(W))
    b = np.asarray(_env.to_numpy(bad), dtype=bool)
    if Wn.ndim == 2:
        live_r, live_c = ~b.all(axis=1), ~b.all(axis=0)
        if live_r.any() and live_c.any() and not (live_r.all() and live_c.all()):
            Wn, b = Wn[np.ix_(live_r, live_c)], b[np.ix_(live_r, live_c)]
    if b.any():                                  # scattered gaps -> column mean (0 after centring)
        # Counted: a column with nothing in it has a count of 0, which is
        # a number this can test, not an empty slice to be warned about and then repaired.
        seen = (~b).sum(axis=0)
        total = np.where(b, 0.0, Wn).sum(axis=0)
        col = np.where(seen > 0, total / np.maximum(seen, 1), 0.0)
        Wn = np.where(b, col[None, :] if Wn.ndim == 2 else col, Wn)
    return Wn


def macheps(xp, ref) -> float:
    """The working dtype's machine epsilon -- the smallest relative difference the arithmetic can
    actually represent (2.2e-16 at float64, 1.2e-07 at float32).  Read off the array, so it follows
    the backend and the compute precision instead of being asserted."""
    dt = ref.dtype if _env.is_torch(xp) else np.asarray(ref).dtype
    if _env.is_torch(xp):
        import torch
        return float(torch.finfo(dt).eps)
    return float(np.finfo(dt if dt.kind in "fc" else _env.rdtype(np)).eps)


def resolution_floor(xp, typical, ref):
    """The scale below which a channel's spread cannot be told from the round-off in computing it.

    A MAD carries the record's units, so no absolute number can test one: the same signal in volts
    and in microvolts is the same signal, and a fixed cut would call every channel of the second
    one dead. The test is relative to the frame's OWN pooled scale ``typical``, taken at the
    resolution the working dtype actually has. Below ``typical * eps`` a channel's variation is
    smaller than the arithmetic that produced it, so there is no scale there to whiten by.

    Derived on both sides -- the pooled MAD is measured from the frame, the epsilon is read off the
    array -- so it moves with the data and with the backend, and nothing here is picked."""
    return typical * macheps(xp, ref)


def mad_stats(xp, data, *, complex_median: bool = False):
    """The per-channel whiten stats of a ``(N, F)`` frame: ``(centre, scale, centred)``.

    The robust centre (per-channel median) and the James-Stein-shrunk MAD scale, in one place
    because the same pair is wanted from two vantage points and the arithmetic must not differ
    between them: :func:`normalize` divides a frame by them on the spot, while
    ``batch._frozen_whiten`` freezes them off a warmup block so a growing stream keeps a stable
    scale (the median/MAD of a growing axis is not incrementally updatable).

    ``complex_median`` takes the median of the real and imaginary parts separately, which is what
    a complex frame's centre is.  Backend-agnostic (numpy or torch)."""
    if complex_median:
        med = _env.median0(xp, xp.real(data)) + 1j * _env.median0(xp, xp.imag(data))
    else:
        med = _env.median0(xp, data)
    centred = data - med[None, :]
    mad = _env.median0(xp, xp.abs(centred)) * MAD_SCALE
    # `mad > 0` is exact: a MAD is a median of magnitudes, so it is >= 0 and is 0 only when the
    # channel never moved off its own median.  The pooled scale of the channels that DID move then
    # sets the floor for the rest -- see resolution_floor.
    spread = mad > 0
    typical = float(_env.median1d(xp, mad[spread])) if bool(spread.any()) else 0.0
    floor = resolution_floor(xp, typical, mad)
    pos = mad > floor
    return med, _shrink_mad(xp, mad, pos, typical, int(data.shape[0])), centred


def _shrink_mad(xp, mad, pos, typical: float, N: int):
    """James-Stein shrinkage of the per-channel MAD toward the pooled scale ``typical``.

    The per-channel MAD from ``N`` rows has log-sampling-variance ``V_samp ~
    MAD_LOGVAR/N``.  Shrink each channel's log-MAD toward ``log(typical)`` by the
    data-derived weight ``w = max(0, 1 - V_samp/V_obs)`` (``V_obs`` = observed
    cross-channel variance of the log-MADs, James-Stein / empirical Bayes).  When the
    channels are homoscedastic (``V_obs ~ V_samp``, e.g. iid noise with few rows)
    ``w -> 0`` and every channel gets the same pooled scale, so a noisy small-N MAD
    cannot disperse the whitened screen; when they genuinely differ (``V_obs >>
    V_samp``) ``w -> 1`` and full per-channel whitening is preserved, each channel
    equalised to unit noise.  Parameter-free.  Backend-agnostic (numpy or torch)."""
    if typical <= 0.0:
        return mad
    # take the log only where there is a scale to take it of; an unresolvable channel is carried
    # through as exactly 0.  A substituted value would be propagated by the shrinkage.
    one = xp.ones_like(mad)
    lm = xp.log(xp.where(pos, mad, one))
    lm0 = float(np.log(typical))
    if int(pos.sum()) > 1:
        V_obs = float(_env.std0(xp, lm[pos])) ** 2
        w = 0.0 if V_obs <= 0.0 else max(0.0, min(1.0, 1.0 - (MAD_LOGVAR / max(N, 1)) / V_obs))
    else:
        w = 0.0
    return xp.where(pos, xp.exp(lm0 + w * (lm - lm0)), xp.zeros_like(mad))


def normalize(W: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-channel robust (MAD) whitening at native resolution -- give each feature
    channel a common, unit noise scale so the screen's noise floor is a clean iid
    reference.  This is normalization only; the entropy-matched rescaling of both
    axes is done together by ``projection.project`` (feature and ordered in one call).

    Each channel's median is subtracted and it is divided by MAD * MAD_SCALE
    (robust sigma).  Because a per-channel MAD from few rows is noisy, each channel's
    scale is shrunk toward the pooled cross-channel scale by a data-derived weight
    (``_shrink_mad``): homoscedastic noise pools to one stable scale so a small-N MAD
    cannot disperse the screen and inflate the floor, while genuinely different
    channels are each equalised to unit noise.  No analyst-chosen floor.  Masked /
    non-finite cells are excluded from the statistics and marked missing (NaN), so ``project``'s fold averages only valid cells (a zero
    would drag the average toward the noise mean).

    Returns a (T, F) array (same shape as W) of whitened channels; masked cells NaN.
    """
    xp = _env.ns(W)
    is_complex = xp.is_complex(W) if _env.is_torch(xp) else np.iscomplexobj(W)
    data = _env.asnum(W, complex=is_complex)
    bad = ~xp.isfinite(xp.abs(data))
    if mask is not None:
        bad = bad | mask
    if bool(bad.any()):
        # robust masked / non-finite path (np.ma, numpy) -- the rare batch-with-gaps case.
        out = _normalize_masked_np(_env.to_numpy(W), _env.to_numpy(bad))
        if _env.is_torch(xp):
            import torch
            return torch.as_tensor(out, device=W.device)
        return out

    # clean path -- backend-agnostic (numpy on CPU, torch on its device).
    _, mad_eff, centered = mad_stats(xp, data, complex_median=is_complex)
    safe = mad_eff > 0.0          # mad_stats already zeroed anything below the resolution floor
    zero = _env.zeros(xp, tuple(int(s) for s in data.shape), complex=is_complex,
                     ref=(data if _env.is_torch(xp) else None))
    return xp.where(safe[None, :], centered / xp.where(safe[None, :], mad_eff[None, :], 1.0), zero)


def _normalize_masked_np(W, bad):
    """Robust per-channel MAD whitening in numpy with a bad-cell mask (np.ma);
    masked / non-finite cells -> NaN.  The rare batch-with-gaps path (see normalize)."""
    is_complex = np.iscomplexobj(W)
    data = W.astype(np.complex128 if is_complex else np.float64).copy()
    mask = np.asarray(bad, bool)
    data[mask] = 0.0
    data_ma = np.ma.array(data, mask=mask)
    if is_complex:
        med = (np.ma.median(data_ma.real, axis=0, keepdims=True).filled(0.0)
               + 1j * np.ma.median(data_ma.imag, axis=0, keepdims=True).filled(0.0))
    else:
        med = np.ma.median(data_ma, axis=0, keepdims=True).filled(0.0)
    centered = data - med
    centered_ma = np.ma.array(centered, mask=mask)
    mad_raw = (np.ma.median(np.abs(centered_ma), axis=0, keepdims=True).filled(0.0)) * MAD_SCALE
    spread = mad_raw > 0                      # exact -- see mad_stats
    typical_mad = float(np.median(mad_raw[spread])) if np.any(spread) else 0.0
    posm = mad_raw > resolution_floor(np, typical_mad, mad_raw)
    # James-Stein shrink toward the pooled scale (see _shrink_mad), with a per-channel
    # sampling variance: channel j with n_j valid cells has V_samp = MAD_LOGVAR/n_j, so a
    # heavily-gapped channel (noisier MAD, fewer valid cells) is shrunk harder toward the
    # pooled scale -- unlike a single row count, which would under-shrink gapped channels.
    n_valid = np.maximum((~mask).sum(axis=0, keepdims=True).astype(float), 1.0)   # (1, F)
    if typical_mad > 0.0 and int(np.count_nonzero(posm)) > 1:
        lm = np.log(np.where(posm, mad_raw, 1.0)); lm0 = float(np.log(typical_mad))
        V_obs = float(np.var(lm[posm]))
        w = (np.zeros_like(mad_raw) if V_obs <= 0.0
             else np.clip(1.0 - (MAD_LOGVAR / n_valid) / V_obs, 0.0, 1.0))
        mad_eff = np.where(posm, np.exp(lm0 + w * (lm - lm0)), 0.0)
    else:
        mad_eff = np.maximum(mad_raw, typical_mad)
    safe = mad_eff > 0.0
    out = np.where(safe, centered / np.where(safe, mad_eff, 1.0), 0.0)
    out[mask] = np.nan
    return out
