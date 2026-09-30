"""
entropy.py -- Entropy geometry: the matched scale read from a signal's own
Shannon entropy, plus normalize (the fold itself is ``projection.project``'s partition of whole channels).

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

import math as _math

from functools import lru_cache

import numpy as np

from . import environment as _env

# MAD_SCALE: median-absolute-deviation -> Gaussian sigma.  Exact 1/Phi^{-1}(0.75).
#
# The constant propagates through the per-channel whitening into `noise_sigma2`, into the floor,
# and out through `k`, so a 5-dp abbreviation (~1.5e-7 relative) moves results.  It is written to
# full float64 precision for that reason.
#
# **The derivation is a test, not an import.**  This module used to do `from scipy.stats import
# norm` at module level and compute `1/norm.ppf(0.75)`, falling back to the literal below when
# scipy was absent.  The two are bit-identical, so the import recomputed a number the file already
# contained -- and it pulled `scipy.stats` into every process that touches entroptics, for a single
# scalar.
#
# It was also not the check it was documented as: it USED scipy's value when scipy was present
# rather than comparing the two, so a disagreement would have been adopted in silence.  The
# comparison it was meant to be now lives in `src/tests/test_mad_scale_derivation.py`, which fails
# if scipy ever disagrees with this literal.
MAD_SCALE: float = 1.482602218505602   # 1/Phi^{-1}(0.75), float64-exact; derived in the tests

# MAD_SCALE_C: the same map for a COMPLEX Gaussian.  MAD_SCALE above is 1/Phi^{-1}(0.75), the
# REAL Gaussian's median-|x| -> sigma constant, and it is wrong by 23% on complex data.  For
# z with E|z|^2 = 1 (real and imaginary parts each variance 1/2), |z|^2 ~ Exp(1), so
#     median|z| = sqrt(median of Exp(1)) = sqrt(ln 2) = 0.8325546...
# and the constant that recovers sigma from median|z| is 1/sqrt(ln 2).  Measured on 400k draws:
# median|z| * MAD_SCALE = 1.2325 (target 1.0); median|z| * MAD_SCALE_C = 0.9985.  Derived, not
# calibrated -- the same standard as MAD_SCALE, and checked against it in the tests.
MAD_SCALE_C: float = 1.0 / _math.sqrt(_math.log(2.0))   # 1/sqrt(ln 2), derived here



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
def _carried(xp, W, mask=None):
    """``W`` at the power of two of its largest measured magnitude, in one memory layout.

    The geometry squares the frame, so a frame at 1e155 (1e19 in float32) would overflow and one
    at 1e-300 underflow; every quantity it reads is a ratio of those squares (an entropy of a
    normalised marginal, a correlation), so carrying the frame by ``2^-e`` changes none of them --
    a product by a power of two is exact in binary floating point.  The layout is fixed too: a sum
    runs in a different order over a column-major array, and a read must not depend on how its
    input was sliced."""
    W = _env.asnum(W)
    if mask is not None:                     # a masked cell is not read: it must not set the carry
        W = xp.where(xp.as_tensor(mask, device=W.device) if _env.is_torch(xp) else np.asarray(mask, bool),
                     xp.zeros_like(W), W)
    if _env.is_torch(xp):
        W = W.contiguous()
        A = xp.abs(W)
        ok = xp.isfinite(A) if mask is None else (xp.isfinite(A) & (mask == False))  # noqa: E712
        peak = float(_env.to_numpy(xp.max(xp.where(ok, A, xp.zeros_like(A))))) if A.numel() else 0.0
    else:
        W = np.ascontiguousarray(W)
        A = np.abs(W)
        ok = np.isfinite(A) if mask is None else (np.isfinite(A) & ~np.asarray(mask, bool))
        peak = float(np.max(np.where(ok, A, 0.0))) if A.size else 0.0
    if not (peak > 0.0) or not np.isfinite(peak):
        return W
    return _pow2(xp, W, np.asarray(-np.frexp(peak)[1]))


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
    folds to round(2^H_F) cells, each a run of whole channels (``projection._fold_ortho``), equal
    runs when the ratio is whole.

    Fold policy: only the feature axis folds; the ordered axis is kept at native resolution.
    The ordered reads -- coherence (adjacent-row similarity), the decay/OTF (lag structure),
    the exact rates (the ordered trajectory) -- all require native ordered spacing, and
    folding the ordered axis would blend adjacent rows into spurious correlation.  For the
    feature axis, a near-uniform marginal (structureless noise) sits below its max log2(F)
    only by a finite-sample deficit, so the fold snaps to none (delta_F = 1.0) whenever H_F lies
    within the band of `fold_band`.  The band is the larger of a Cantelli bound on the exact
    Dirichlet null deficit and the width change that moves the Marchenko-Pastur edge past its
    Tracy-Widom margin: a fold must be both real and worth making.  Neither term can reach
    log2(F), so the band stays inside the entropy range without a cap, and neither carries a
    constant that is not derived from a stated null.  Closed-form, fixed by the axis lengths
    alone.

    Read BEFORE the whitening, and why that is forced.  `projection.read` runs
    `geometry -> normalize -> project`: this read is taken on the RAW frame, and only then is
    the frame whitened per channel and folded.  The order is not incidental, because the two
    steps want opposite things -- `geometry` measures how unevenly the raw amplitudes are
    spread across the channels, and `normalize` divides each channel by its OWN RMS,
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
    W = _carried(xp, W, mask)           # one layout, at the power of two of its peak: exact, and
                                        # every read below is a ratio, so nothing it returns moves

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
    #     fold_band(T, F)                the null-deficit band
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

    This is what licenses a fold.  Folding merges adjacent cells, which preserves
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


@lru_cache(maxsize=None)
def _bernoulli_even() -> tuple:
    """The even Bernoulli numbers B_2, B_4, ... as exact rationals (Akiyama-Tanigawa), as far as
    the digamma series ever needs them: its terms fall until k ~ pi x, so up to where B_2k / 2k
    first exceeds the reciprocal of eps at the recurrence's end."""
    from fractions import Fraction
    out, a = [], []
    n = 0
    while True:
        a.append(Fraction(1, n + 1))
        for j in range(n, 0, -1):
            a[j - 1] = j * (a[j - 1] - a[j])
        if n >= 2 and n % 2 == 0:
            out.append(a[0])
            if abs(float(a[0])) / n > 1.0 / float(np.finfo(float).eps):
                return tuple(out)
        n += 1


# The digamma asymptotic series' smallest term is ~ exp(-2 pi x): from here up it is below round-off.
_PSI_X = -_math.log(float(np.finfo(float).eps)) / (2.0 * _math.pi)


def _digamma(x: float) -> float:
    """psi(x), x > 0, to float precision: the recurrence psi(x) = psi(x + 1) - 1/x up to where the
    asymptotic series reaches round-off, then that series summed until its terms do."""
    r = 0.0
    while x < _PSI_X:
        r -= 1.0 / x
        x += 1.0
    s = _math.log(x) - 0.5 / x
    x2 = x * x
    xp = x2
    for k, B in enumerate(_bernoulli_even(), start=1):
        t = float(B) / (2 * k * xp)
        s -= t
        if abs(t) <= float(np.finfo(float).eps) * abs(s):
            break
        xp *= x2
    return r + s


def _trigamma(x: float) -> float:
    """psi'(x), x > 0, to float precision: psi'(x) = psi'(x + 1) + 1/x^2, then the asymptotic
    series 1/x + 1/(2x^2) + sum B_2k / x^(2k+1), summed until its terms reach round-off."""
    r = 0.0
    while x < _PSI_X:
        r += 1.0 / (x * x)
        x += 1.0
    s = 1.0 / x + 0.5 / (x * x)
    x2 = x * x
    xp = x2 * x
    for B in _bernoulli_even():
        t = float(B) / xp
        s += t
        if abs(t) <= float(np.finfo(float).eps) * abs(s):
            break
        xp *= x2
    return r + s


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
    far = float(far)
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    return float(np.sqrt(1.0 / far - 1.0))


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

    The band is the larger of the two: a fold must be both real and worth making.  Which one
    binds is set by the ROW COUNT, not by the aspect ratio: the significance term governs frames
    with very few rows and the sufficiency term governs everything else.  Measured crossover --
    the smallest ``T`` at which sufficiency overtakes significance -- is ``T = 8`` at ``F = 8``,
    11 at 16, 14 at 32, 19 at 64, 26 at 128, 36 at 256 and 51 at 512, i.e. roughly
    ``T ~ 2-3 sqrt(F)``.  So ``(16, 64)`` sits in the significance regime while ``(32, 64)``
    already sits in the sufficiency one, though both are wide and short.

    Neither term carries a constant that is not derived from a stated null.  The sufficiency
    term CAN reach ``log2 F`` -- when the required width change exceeds ``F - 1``, no achievable
    fold clears the margin and every fold is refused, which is the intended reading and not a
    cap chosen to keep it in range.
    """
    Fi = max(2, int(F))
    Ti = max(1, int(T))
    lgF = float(np.log2(Fi))
    far = float(far)
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")

    mean_H, sd_H = dirichlet_entropy_moments(Fi, Ti / 2.0)
    significance = lgF - mean_H + _tail_multiplier(far) * sd_H

    from .null_providers import tw1_quantile
    dF = float(tw1_quantile(far)) * np.sqrt(Fi) * (1.0 / np.sqrt(Ti) + 1.0 / np.sqrt(Fi)) ** (1.0 / 3.0)
    # `q_TW1` is the (1 - far) quantile of a law centred near -1.21, so it turns NEGATIVE above
    # far ~ 0.168, and with it `dF` and `sufficiency`.  Clamping at zero keeps the intermediate
    # quantity meaning what its name says -- a width change -- and changes NO returned value:
    # `significance` is strictly positive, so `max` already selected it in all 320 of the 576
    # (T, F, far) cells where the unclamped term went negative, to 0.0e+00.  Hygiene, not a fix.
    #
    # The UPPER clamp binds too, and that is the correct reading rather than a defect: when the
    # required width change reaches `F - 1`, this term lands on exactly `log2 F`, the largest
    # deficit attainable, so no record can clear it.  At that level no achievable fold moves the
    # Marchenko-Pastur edge past its own Tracy-Widom margin, so every fold is refused.  It is why
    # a tight `far` on a narrow feature axis folds nothing: `F = 4` refuses all folds at
    # `far <= 0.01`, `F = 8` at `far = 0.001`.
    dF = float(np.clip(dF, 0.0, Fi - 1.0))
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
                     means something, so merging neighbouring cells preserves the signal.
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
    return live_view_and_gaps(W, mask)[0]


def live_view_and_gaps(W, mask: np.ndarray | None = None):
    """:func:`live_view`, and the gaps it filled: ``(view, gaps)``, ``gaps`` a boolean array aligned
    with ``view`` marking the cells that were missing (``None`` when none were).  A read that draws
    a null from the view holds these cells in place: they are fixed structure, not samples."""
    xp = _env.ns(W)
    bad = ~xp.isfinite(xp.abs(W))
    if mask is not None:
        bad = bad | mask
    if not bool(bad.any()):
        return W, None
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
    return Wn, (b if b.any() else None)


def macheps(xp, ref) -> float:
    """The working dtype's machine epsilon -- the smallest relative difference the arithmetic can
    actually represent (2.2e-16 at float64, 1.2e-07 at float32).  Read off the array, so it follows
    the backend and the compute precision instead of being asserted."""
    dt = ref.dtype if _env.is_torch(xp) else np.asarray(ref).dtype
    if _env.is_torch(xp):
        import torch
        return float(torch.finfo(dt).eps)
    return float(np.finfo(dt if dt.kind in "fc" else _env.rdtype(np)).eps)


def _pow2(xp, a, k):
    """``a * 2^k`` exactly, ``k`` an integer numpy array broadcastable against ``a``.  Applied as two
    factors of half the exponent each, so neither factor leaves the float range for any exponent
    the dtype has; a product by a power of two is exact in binary floating point."""
    k = np.asarray(k, dtype=np.int64)
    h1 = k // 2
    h2 = k - h1
    rd = a.real.dtype
    if _env.is_torch(xp):
        f1 = xp.as_tensor(np.ldexp(1.0, h1), dtype=rd, device=a.device)
        f2 = xp.as_tensor(np.ldexp(1.0, h2), dtype=rd, device=a.device)
    else:
        f1 = np.ldexp(1.0, h1).astype(rd)
        f2 = np.ldexp(1.0, h2).astype(rd)
    return (a * f1) * f2


def _whiten_core(xp, data, bad=None, axis=0):
    """The whitening statistics in carried units: ``(xs, cs, ss, e)`` with ``xs = data * 2^-e``,
    centre ``cs * 2^e`` and scale ``ss * 2^e`` (see :func:`whiten_stats`), ``e`` the per-channel
    exponent of the channel's peak magnitude (a numpy integer array, ``axis`` kept).

    Each channel is carried by the power of two of its own peak, which is exact, so the sums below
    neither overflow nor underflow for any level the dtype holds, and ``(data - c) / s`` is
    ``(xs - cs) / ss`` bit for bit.  ``axis`` is the ordered axis (0 for a ``(T, F)`` frame, 1 for
    a ``(B, T, F)`` stack); the sums along it run in one fixed order whatever the caller's memory
    layout -- a pairwise sum and a running sum differ in the last bits, and a read must not depend
    on how its input was sliced.  ``bad`` (numpy, the frame's shape) leaves cells out of both
    statistics; only the ``(T, F)`` numpy path takes it."""
    if _env.is_torch(xp):
        data = data.contiguous()
        peak = xp.amax(xp.abs(data), dim=axis, keepdim=True)
        e = np.frexp(np.asarray(_env.to_numpy(peak), dtype=np.float64))[1]
    else:
        data = np.ascontiguousarray(data)
        mag = np.abs(data) if bad is None else np.where(bad, 0.0, np.abs(data))
        peak = np.max(mag, axis=axis, keepdims=True) if data.shape[axis] else \
            np.zeros(tuple(1 if a == axis else n for a, n in enumerate(data.shape)))
        e = np.frexp(np.asarray(peak, dtype=np.float64))[1]
    xs = _pow2(xp, data, -e)
    if bad is None:
        first = xs[:1] if axis == 0 else xs[:, :1]
        if _env.is_torch(xp):
            cs = first + (xs - first).mean(dim=axis, keepdim=True)
            moved = xp.amax(xp.abs(xs - first), dim=axis, keepdim=True) > 0
            m = (xp.abs(xs - cs) ** 2).mean(dim=axis, keepdim=True)
        else:
            cs = first + (xs - first).mean(axis=axis, keepdims=True)
            moved = np.max(np.abs(xs - first), axis=axis, keepdims=True) > 0
            m = (np.abs(xs - cs) ** 2).mean(axis=axis, keepdims=True)
        # a channel that never moved is its own value exactly (its computed mean need not be)
        cs = xp.where(moved, cs, first)
        return xs, cs, xp.where(moved, xp.sqrt(m), xp.zeros_like(m)), e
    B = np.asarray(bad, bool)
    n = (~B).sum(axis=0, keepdims=True)                             # measured cells per channel
    X0 = np.where(B, 0.0, xs)
    idx = np.argmax(~B, axis=0)[None, :]                             # first measured cell
    first = np.take_along_axis(X0, idx, axis=0)
    cs = np.where(n > 0, first + np.where(B, 0.0, X0 - first).sum(axis=0, keepdims=True)
                  / np.maximum(n, 1), 0.0)
    moved = np.max(np.where(B, 0.0, np.abs(X0 - first)), axis=0, keepdims=True) > 0
    m = np.where(B, 0.0, np.abs(X0 - cs) ** 2).sum(axis=0, keepdims=True) / np.maximum(n, 1)
    cs = np.where(moved | (n == 0), cs, first)
    return xs, cs, np.where(moved, np.sqrt(m), 0.0), e


def whiten_stats(xp, data, bad=None):
    """The per-channel whitening of a ``(T, F)`` frame: ``(centre, scale)``, each ``(F,)``.

    The screen's floor is the Tracy-Widom edge of a matrix whose cells are i.i.d. with one variance,
    so the whitening has one job under the null -- each channel i.i.d. in time, the channels
    independent, each at its own level and of any marginal: centre every column exactly, and bring
    every column to one scale.

      centre  the channel's MEAN over its measured cells.  The exact column centring: any other
              centre leaves a per-channel offset, and on skewed noise (exponential, lognormal,
              counts) those offsets are a rank-one mode that grows with the record.
      scale   the channel's RMS about that mean.  Every whitened column then has the same norm,
              and the screen's Gram is a sample correlation matrix, whose largest eigenvalue
              follows the Tracy-Widom law for i.i.d. entries of any marginal with finite fourth
              moment (Bao, Pan & Zhou 2012; Pillai & Yin 2012).  The scale is blind to time order,
              which is what keeps the null's directions exchangeable: a scale read from the order
              (successive differences) weights a column by its own direction and hands the null a
              mode along the smooth ones.  The cost is that a channel's own signal is part of its
              scale, so a signal carried by a few channels is read against their total variance.

    A channel whose measured values are all equal has no scale (``scale = 0``) and its centre is
    that value exactly: it never moved.  No number stands between "moved" and "never moved", so the
    rule carries no units.  Masked / non-finite cells are left out of both statistics.  Backend-agnostic on a
    clean frame; a frame with missing cells takes the numpy path.  Computed in each channel's own
    carried units (:func:`_whiten_core`), so any level the dtype holds is read."""
    data = _env.asnum(data)
    nonfinite = ~xp.isfinite(xp.abs(data))
    bad = nonfinite if bad is None else (nonfinite | bad)
    if bool(_env.to_numpy(bad).any()):
        X = np.asarray(_env.to_numpy(data))
        xs, cs, ss, e = _whiten_core(np, X, np.asarray(_env.to_numpy(bad), bool))
        c, s = _pow2(np, cs, e)[0], _pow2(np, ss, e)[0]
        if _env.is_torch(xp):
            c = xp.as_tensor(c, device=data.device)
            s = xp.as_tensor(s, device=data.device)
        return c, s
    xs, cs, ss, e = _whiten_core(xp, data)
    return _pow2(xp, cs, e)[0], _pow2(xp, ss, e)[0]


def normalize(W: np.ndarray, mask: np.ndarray | None = None, *, return_stats: bool = False):
    """Whiten each feature channel onto one noise scale -- :func:`whiten_stats`'s mean centre and
    RMS scale -- so the screen's floor reads a clean i.i.d. reference.  This is
    normalization only; the entropy-matched fold is done by ``projection.project``.

    Returns a ``(T, F)`` array (same shape as ``W``) of whitened channels; masked / non-finite cells
    NaN, so the fold never reads them.  ``return_stats`` also returns the map that was applied,
    ``(whitened, centre, scale)``, each of ``centre`` and ``scale`` an ``(F,)`` vector, so that

        W[:, j] ~= whitened[:, j] * scale[j] + centre[j]

    recovers the caller's units.  A channel that never moved comes back with ``scale = 0`` and
    whitened cells of 0: its every value is its centre."""
    xp = _env.ns(W)
    is_complex = xp.is_complex(W) if _env.is_torch(xp) else np.iscomplexobj(W)
    data = _env.asnum(W, complex=is_complex)
    bad = ~xp.isfinite(xp.abs(data))
    if mask is not None:
        bad = bad | mask
    if bool(bad.any()):
        Xn = np.asarray(_env.to_numpy(W)).astype(np.complex128 if is_complex else np.float64)
        Bn = np.asarray(_env.to_numpy(bad), bool)
        xs, cs, ss, e = _whiten_core(np, Xn, Bn)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(ss > 0, (xs - cs) / np.where(ss > 0, ss, 1.0), 0.0)
        out = np.where(Bn, np.nan, out)
        c, s = _pow2(np, cs, e)[0], _pow2(np, ss, e)[0]
        if _env.is_torch(xp):
            import torch
            out = torch.as_tensor(out, device=W.device)
            c = torch.as_tensor(c, device=W.device)
            s = torch.as_tensor(s, device=W.device)
        return (out, c, s) if return_stats else out
    xs, cs, ss, e = _whiten_core(xp, data)
    safe = ss > 0.0
    zero = _env.zeros(xp, tuple(int(v) for v in data.shape), complex=is_complex,
                      ref=(data if _env.is_torch(xp) else None))
    out = xp.where(safe, (xs - cs) / xp.where(safe, ss, xp.ones_like(ss)), zero)
    if not return_stats:
        return out
    return out, _pow2(xp, cs, e)[0], _pow2(xp, ss, e)[0]
