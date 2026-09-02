"""
proximity.py -- a magnitude-carrying, width-free spectral digest of a frame, and a probe on it.

This asks, for every direction a frame resolves, *how far it stands from where the frame's own
noise law says it should* -- and returns the whole profile, so that two frames can be compared
without either of them first being collapsed to a count.

## Why not the count

A count is a threshold, and a threshold flips. Measured on 60 BeIR frames, a 5% row edit moves
a Tracy-Widom resolved-mode count on 30% of them, and across the whole corpus the count takes
only 4 distinct values -- a digest with four states cannot separate sixty records. The fix is
not a better threshold; it is to stop thresholding. Every mode is kept and reported by a
continuous deviation: nothing is in or out, so nothing can flip.

## Every number here is a function of the frame

There is no width, no rank, no rate, no tolerance, no grid and no learned constant in this
module. In order:

    v_j        = mean_i (A - med(A))_ij^2      the centred per-channel second moment
    F_eff      = (sum v)^2 / sum v^2           the bulk's effective width          f(v)
    sigma^2    = median_i ||A_i||^2 / (F_eff * c_Feff * dof)                       f(A, N, F_eff)
    mu         = (sqrt(N-1) + sqrt(F_eff))^2   the Marchenko-Pastur centring       f(N, F_eff)
    edge       = sqrt(sigma^2 * mu)                                                f(sigma^2, mu)
    s_MP(k)    = sqrt(F_eff * sigma^2 * x_k),  MP_b(x_k) = 1 - k/N       f(N, F_eff, sigma^2)
    dev_k      = (s_k - s_MP(k)) / sqrt(N)                                         f(s, s_MP, N)

Each is derived below at the function that computes it. Nothing was swept.

## Why a deviation and not a clipped excess

The read this module shipped first was `max(0, s_k - edge)/sqrt(N)` -- every mode's excess above
the single point where the bulk ends. It is parameter-free and it is continuous, but it throws
away every mode at or below the edge, and on lexical material that is almost the entire frame:
measured, the clipped excess had support on 1-3 of 64 modes on ppmi and recalled 0.183 under a
10% row deletion.

The obvious repairs all fail the same way. `s - e*tanh(s/e)`, `s^2/(s+e)`, softplus at the edge,
softplus at the Tracy-Widom width: four parameter-free transitions that disagree by up to 0.80
recall on a single cell of the table. Nothing selects between them except a score, which makes
*the choice of function* the swept constant.

`s_k - s_MP(k)` has no such choice in it. It does not soften the edge; it stops comparing every
mode to the same point. The Marchenko-Pastur law already predicts where the k-th largest
singular value of a pure-noise frame of this shape and this variance sits, so mode `k` is
compared to *its own* prediction. Above the bulk the prediction is bounded by the edge and
converges to it as `k/N -> 0` (`s_MP(0+) == bulk_edge` exactly -- see :func:`mp_spectrum`), so
the leading modes are read against essentially the same reference the excess used; below it the
prediction tracks the bulk, and a sub-edge mode reads its deviation from the noise it is made of.

Measured on a 60-frame corpus, recall@1, against a clipped-excess read:

    embed_normed          1e-8   1e-4   1e-2   swap5  swap10 swap25  drop5 drop10   mean
      clipped excess     1.000  1.000  1.000   0.867  0.533  0.200   0.950  0.633  0.689
      MP deviation       1.000  1.000  1.000   0.950  0.750  0.400   0.983  0.917  0.780
    ppmi
      clipped excess     1.000  0.967  0.633   0.517  0.367  0.200   0.533  0.183  0.493
      MP deviation       1.000  1.000  1.000   0.933  0.833  0.483   1.000  0.767  0.781

It wins or ties every cell on both representations and at a second draw of every perturbation
seed. The cost is density: the deviation is dense where a clipped excess is sparse, so stored
records run to full width
(:func:`common_prefix`), and the probe prunes less (:class:`SpectrumProbe`).

## The edge, and how it relates to `null_providers.mp`

`null_providers.mp` is `sqrt(sigma^2 * (mu + q(far) * sigma_J))`: the same `mu`, plus a
finite-size fluctuation allowance whose size is set by a chosen Tracy-Widom false-alarm rate
`far`. :func:`bulk_edge` is that expression with the allowance removed, which is the
Marchenko-Pastur bulk edge itself: for an i.i.d. matrix the bulk's upper support point is
`sigma * (sqrt(N) + sqrt(F))`, a function of the de-biased per-cell variance and the aspect
ratio and of nothing else. No rate appears because no hypothesis is being tested -- the bulk
edge is where the noise spectrum *ends*, not where it becomes improbable.

The two are therefore not rivals and one is not a re-tuning of the other:

    mp(ctx)  =  bulk_edge(M) * sqrt(1 + q(far) * sigma_J / mu)

and `sigma_J / mu` vanishes as `N^{-2/3}`, so the floor converges to the bulk edge from above.
`null_providers.johnstone` is reused directly below -- one definition of the finite-size
centring/scaling, not a second copy of it.

## Read on the centred frame, not a normalised one

Per-vector or per-channel normalisation destroys exactly the magnitude this module is built to
keep -- see `aperture.py` / `null_providers.py`'s own robust-scale (MAD) whitening, which this
module deliberately does not apply. Centring is kept, because it is not a normalisation and not
optional: Marchenko-Pastur is the law of a *zero-mean* i.i.d. bulk, and an uncentred frame
carries a rank-1 mean direction that is not noise and that inflates both the spectrum and the
variance estimate. Subtracting the per-channel median is scale-covariant (`med(c*A) = c*med(A)`),
so magnitude survives it.

## What this is not

Not a recall path, not a ranking, not a retrieval policy. This is a capability -- a digest a
store can hold and compare, not a representation it can invert (see :func:`mp_deviation`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .null_providers import johnstone, debias_denominator

#: The instrument that produced a proximity read. A different reference is a different
#: instrument: a caller handed a bare array must still be able to tell what the numbers were
#: stated against, and records digested against the bulk edge are not comparable with records
#: digested against the per-mode prediction. Re-digest, do not mix.
ENGINE_ID_PROXIMITY = "entroptics.mp.dev"

__all__ = [
    "ENGINE_ID_PROXIMITY", "SpectrumProbe", "bulk_edge", "centre", "common_prefix",
    "effective_width", "mp_deviation", "mp_spectrum", "spectral_distance",
]


def _as_matrix(M) -> np.ndarray:
    """Materialise a 2-D float64 matrix. Scattered non-finite cells are filled with the column
    mean (0 after centring); NaN propagated into an SVD, where it poisons every
    mode at once."""
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"the input must be 2-D; got shape {A.shape}")
    if A.size == 0:
        raise ValueError("the input must be non-empty")
    if not np.all(np.isfinite(A)):
        A = np.where(np.isfinite(A), A, np.nan)
        col = np.nanmean(A, axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        A = np.where(np.isfinite(A), A, col)
    return A


# ═══════════════════════════════════════════════════════════════════════════
# The frame, centred
# ═══════════════════════════════════════════════════════════════════════════

def centre(M) -> np.ndarray:
    """The frame with each channel's median subtracted. Scale-covariant, so magnitude survives.

    The median, because a handful of bright rows is the signal being looked for,
    and letting them set the centre is how a detector talks itself out of its own findings."""
    A = _as_matrix(M)
    return A - np.median(A, axis=0)[None, :]


# ═══════════════════════════════════════════════════════════════════════════
# The bulk's effective width -- a continuous quantity where a live-channel count is discrete
# ═══════════════════════════════════════════════════════════════════════════

def effective_width(A: np.ndarray) -> float:
    """`(tr S)^2 / tr(S^2)` over the centred per-channel second moments -- the bulk's width.

    Derived, the continuous generalisation of a live-channel count: for a bulk with population
    covariance `S = diag(v_1..v_F)`, reducing to the isotropic bulk with the same first two
    spectral moments gives width `(tr S)^2 / tr(S^2)` and per-cell variance `tr(S^2) / tr(S)`.
    On the two-valued case a channel count was written for -- `v_j` equal to some `v` on `p`
    channels and 0 on the rest -- this returns exactly `p`, so it is that count's answer,
    extended off the two-valued case in the one way the generalised Marchenko-Pastur law
    permits, and smooth in the entries where a count is not.

    `v_j` is a SECOND MOMENT because that is what `S_jj` is; a robust (MAD-based) surrogate for
    it is only valid under Gaussianity, which sparse lexical material violates outright."""
    v = np.mean(np.asarray(A, dtype=np.float64) ** 2, axis=0)
    t1 = float(np.sum(v))
    t2 = float(np.sum(v ** 2))
    if not (t1 > 0.0 and t2 > 0.0):
        return 1.0          # a frame with no energy has no bulk
    return (t1 * t1) / t2


def _noise_sigma2_at(A: np.ndarray, N: int, F: float) -> float:
    """`null_providers.noise_sigma2` at a real-valued width. The MEDIAN row energy.

    The median, not any mean: a few bright rows are the signal, and a mean lets them raise the
    floor above themselves."""
    row_energy = np.sum(np.abs(A) ** 2, axis=1)
    return float(np.median(row_energy)) / debias_denominator(N, F) + 1e-30


def bulk_edge(M) -> float:
    """The Marchenko-Pastur bulk edge of a frame, in singular-value units. No false-alarm rate.

    `edge = sqrt(sigma^2 * mu)`, `mu = (sqrt(N-1) + sqrt(F_eff))^2`. For an i.i.d. matrix the
    noise spectrum's support ends at `sigma * (sqrt(N) + sqrt(F))`; every input to that is a
    property of the frame (its de-biased per-cell variance and its aspect ratio) and none of
    it is a policy choice. Contrast `null_providers.mp`, which adds a *distributional* allowance
    for how far the largest eigenvalue fluctuates past the support point, and is therefore
    necessarily stated at some false-alarm level.

    `N-1` because the frame is centred, the same reason `johnstone` is called
    with it here.

    Not vestigial: it is the `k -> 0` limit of :func:`mp_spectrum`, exactly, and that identity
    is what fixes the row count the prediction is stated at. See the derivation there.

    Scale-covariant: `bulk_edge(c*M) == |c| * bulk_edge(M)`."""
    A = centre(M)
    N = int(A.shape[0])
    F_eff = effective_width(A)
    mu, _sigma_J = johnstone(N, F_eff)      # width-agnostic; no int() in it
    return math.sqrt(_noise_sigma2_at(A, N, F_eff) * mu)


# ═══════════════════════════════════════════════════════════════════════════
# The Marchenko-Pastur prediction -- closed form, inverted by root-find, no grid
# ═══════════════════════════════════════════════════════════════════════════

def _mp_cdf(x, b: float) -> np.ndarray:
    """Mass of the continuous part of `MP_b` at or below `x`. Closed form, elementary.

    The density is `p(t) = sqrt((h - t)(t - l)) / (2 pi b t)` on `[l, h]`, `l,h = (1 -+ sqrt b)^2`.
    Write `S = l + h = 2(1+b)`, `P = l*h = (1-b)^2`, `Q(t) = -t^2 + S t - P`. Then
    `sqrt(Q)/t = -t/sqrt(Q) + S/sqrt(Q) - P/(t sqrt(Q))` and each piece is a standard integral,
    giving the antiderivative

        Phi(t) = sqrt(Q) + (1+b) arcsin((t - (1+b)) / (2 sqrt b))
                         - |1-b| arcsin(((1+b) t - (1-b)^2) / (2 sqrt b t)).

    The sign of the last term is pinned by the boundary -- it is the only
    sign for which the total continuous mass comes out `min(1, 1/b)`, the fraction of `AA^T`'s
    eigenvalues that are not structurally zero."""
    rb = math.sqrt(b)
    lo, hi = (1.0 - rb) ** 2, (1.0 + rb) ** 2
    t = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    q = np.sqrt(np.clip((hi - t) * (t - lo), 0.0, None))
    a1 = np.clip((t - (1.0 + b)) / (2.0 * rb), -1.0, 1.0)
    den = 2.0 * rb * t
    ok = den > 0.0
    a2 = np.clip(np.where(ok, ((1.0 + b) * t - (1.0 - b) ** 2) / np.where(ok, den, 1.0), -1.0),
                 -1.0, 1.0)
    phi = q + (1.0 + b) * np.arcsin(a1) - abs(1.0 - b) * np.arcsin(a2)
    base = (abs(1.0 - b) - (1.0 + b)) * (math.pi / 2.0)
    return (phi - base) / (2.0 * math.pi * b)


def _mp_quantiles(N: int, rows: int, F: float) -> np.ndarray:
    """`x_k`, `k = 1..N`, largest first: the `MP_b` quantile carrying upper-tail mass `k/N`.

    `b = rows / F`. The eigenvalues of `A A^T` for an i.i.d. `rows x F` frame of per-cell
    variance `sigma^2` are `F sigma^2` times `MP_b`, and the k-th largest of `N` draws sits at
    upper-tail mass `k/N`. For `b > 1` only `1/b` of the eigenvalues are nonzero; the rest are
    structurally zero and the targets that fall past the continuous mass are returned as
    exactly 0.

    The stopping rule is the machine, not a tolerance: bisection halves `[l, h]` until the
    midpoint is no longer strictly between the endpoints -- a property of the number format,
    not an iteration count or an `atol`."""
    b = rows / F
    rb = math.sqrt(b)
    lo, hi = (1.0 - rb) ** 2, (1.0 + rb) ** 2
    target = min(1.0, 1.0 / b) - np.arange(1, N + 1) / N
    a = np.full(N, lo, dtype=np.float64)
    c = np.full(N, hi, dtype=np.float64)
    while True:
        m = (a + c) / 2.0
        live = (m > a) & (m < c)        # a bracket wider than one representable step
        if not bool(np.any(live)):
            break
        below = _mp_cdf(m, b) < target
        a = np.where(live & below, m, a)
        c = np.where(live & ~below, m, c)
    return np.where(target < 0.0, 0.0, a)


def mp_spectrum(M) -> np.ndarray:
    """`s_MP(k)`: where the bulk law says the k-th singular value of THIS frame should sit.

    `s_MP(k) = sqrt(F_eff * sigma^2 * x_k)`, descending, length `min(N, D)`. Every input is the
    frame's own -- its shape, its effective width and its de-biased per-cell variance -- so this
    is a prediction the frame makes about itself, not a reference imported from anywhere.

    The row count is forced, not chosen: the law is stated at `rows = N - 1` because the frame
    is centred (the same reason `johnstone` uses it), and at that row count the top of the
    predicted support is exactly :func:`bulk_edge` -- the prediction and the edge are the same
    law read at two places, and only one row count makes them agree where they overlap.

    Bounded above by `bulk_edge`, rising to it as `k/N -> 0`; it does not reach it at `k = 1`,
    because the largest of `N` draws sits at upper-tail mass `1/N`, not at the support endpoint."""
    A = centre(M)
    N = int(A.shape[0])
    F_eff = effective_width(A)
    sigma2 = _noise_sigma2_at(A, N, F_eff)
    x = _mp_quantiles(N, max(N - 1, 1), F_eff)[:min(N, int(A.shape[1]))]
    return np.sqrt(np.clip(F_eff * sigma2 * x, 0.0, None))


# ═══════════════════════════════════════════════════════════════════════════
# The read
# ═══════════════════════════════════════════════════════════════════════════

def mp_deviation(M) -> np.ndarray:
    """The read: `(s_k - s_MP(k)) / sqrt(N)`, in descending-mode order. Length `min(N, D)`.

    Every mode, none dropped and none clipped -- see :func:`common_prefix` for how two records
    of different length are compared.

    The division by `sqrt(N)` is a frame normalisation, not a per-vector one: `N` is the
    frame's row count, a property of the record as a whole, so dividing by it cannot discard
    anything that distinguishes one record's magnitude from another's. It is forced: a coherent direction of per-cell strength `s` spread over `N` rows has
    `s_k ~ s*sqrt(N*D)`, so `s_k / sqrt(N)` is the one power of `N` that makes a mode's reading
    independent of how many rows happen to carry it. What is forbidden -- dividing each record
    by its *own* norm -- is not done: `c*M` gives `c*dev`, and the read is not scale-invariant.

    Nothing flips: the read is a difference of two continuous functions of the frame, with no
    `max`, no clip and no membership anywhere in it.

    One-way, by dimension counting, not by hardness: `U` and `Vt` are computed and discarded;
    only the deviation is returned. An `N x D` frame has `N*D` free parameters and the read
    keeps `min(N, D)` numbers, so the preimage of any read is a manifold of dimension
    `N*D - min(N, D)`, and the frame is not recoverable by any procedure because the
    information is not present in the output.

    What that does NOT guarantee: it is not confidentiality (the read is designed to preserve
    proximity, so two similar frames give similar digests -- that is the feature); it is not
    resistance to verification (a candidate frame's read is computable and comparable); it says
    nothing under side information (`U`/`Vt` are discarded by this caller, not destroyed).

    Deterministic: no RNG is consulted, and repeated reads of one frame are bit-identical."""
    A = centre(M)
    N = int(A.shape[0])
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except Exception as exc:            # LinAlgError (non-convergent SVD) and friends
        raise ValueError(
            f"proximity read failed on a {A.shape} frame ({type(exc).__name__}: {exc})"
        ) from exc
    return (s - mp_spectrum(M)) / math.sqrt(N)


# ═══════════════════════════════════════════════════════════════════════════
# Comparing two records -- no width, no relative denominator, and no invented entries
# ═══════════════════════════════════════════════════════════════════════════

def common_prefix(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Both reads cut to the shorter one's length. The modes only one record has are DROPPED,
    never zero-padded.

    `s_MP(k)` is the quantile at upper-tail mass `k/N`; for `k > N` that mass exceeds 1 and the
    quantile does not exist. The read is defined on `{1..N}` and nowhere else, so a record
    simply has no value at a mode it does not have -- not zero, not the prediction, not
    anything. Writing a 0 there would assert a measurement that was never taken.

    Measured (on the corpus this module was derived against), zero-padding versus the common
    prefix under a 5% row deletion: recall@1 0.717 against 0.983; under 10%, 0.500 against
    0.917. Comparisons involving a short record are made on fewer modes, so a record cannot be
    found "far" merely for being short; records must be persisted at full length, since the
    dense read has no trailing-zero truncation to rely on in storage."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    return a[:n], b[:n]


def spectral_distance(x, y) -> float:
    """Euclidean distance between two reads on their common prefix.

    Plain L2, and deliberately ABSOLUTE. A relative or normalised distance divides by the
    records' own magnitudes and so throws away exactly the magnitude the whole construction was
    arranged to keep.

    L2: the deviation is in per-row cell-magnitude units, so its squares
    add across modes the way energy does. And L2's component bound `|x_j - y_j| <= ||x - y||`
    makes :class:`SpectrumProbe` exact."""
    a, b = common_prefix(x, y)
    return float(np.linalg.norm(a - b))


# ═══════════════════════════════════════════════════════════════════════════
# The probe -- retrieval without scanning everything
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _Hit:
    """One probe result: the record's index in the gallery and its full spectral distance."""
    index: int
    distance: float


class SpectrumProbe:
    """A range / k-nearest probe keyed on one component of the read. Exact, radius derived.

    The bound is universal, so the key needs no justification beyond being a component: for
    any two vectors and ANY index `j`, `|x_j - y_j| <= ||x - y||`. Component 0 is used because
    it is the first entry and so requires no selection.

    The radius is not a parameter: every record within full distance `R` of a query has its key
    within `R` of the query's key, so scanning the key interval `[x_0 - R, x_0 + R]` drops
    nothing -- it *is* the answer radius the caller asked for, and at that value the filter is
    lossless.

    For `nearest`, where the caller supplies no radius at all, the same bound gives an exact
    stopping rule: walk outward from the query's key position and stop as soon as the key gap
    exceeds the k-th best full distance found so far, since nothing further out can beat it.

    Both operations return the same answers a full scan would; the probe changes only how much
    work is done, never what is found. Deterministic -- `numpy.searchsorted` and a stable
    argsort, no RNG."""

    def __init__(self, spectra: Iterable[Sequence[float]]) -> None:
        self._gallery = [np.asarray(s, dtype=np.float64).ravel() for s in spectra]
        if not self._gallery:
            self._keys = np.zeros(0, dtype=np.float64)
            self._order = np.zeros(0, dtype=np.intp)
            return
        keys = np.array([float(s[0]) if s.size else 0.0 for s in self._gallery])
        self._order = np.argsort(keys, kind="stable")
        self._keys = keys[self._order]

    def __len__(self) -> int:
        return len(self._gallery)

    def within(self, query, radius: float) -> list[_Hit]:
        """Every record within `radius` of `query`. Identical to a full scan, by the bound above."""
        q = np.asarray(query, dtype=np.float64).ravel()
        if not self._gallery:
            return []
        if not (radius >= 0.0):
            raise ValueError(f"radius must be non-negative; got {radius!r}")
        q0 = float(q[0]) if q.size else 0.0
        lo = int(np.searchsorted(self._keys, q0 - radius, side="left"))
        hi = int(np.searchsorted(self._keys, q0 + radius, side="right"))
        out = []
        for pos in range(lo, hi):
            gid = int(self._order[pos])
            d = spectral_distance(q, self._gallery[gid])
            if d <= radius:
                out.append(_Hit(index=gid, distance=d))
        out.sort(key=lambda h: (h.distance, h.index))
        return out

    def nearest(self, query, k: int = 1) -> list[_Hit]:
        """The `k` nearest records. Exact, with no radius supplied and none invented."""
        q = np.asarray(query, dtype=np.float64).ravel()
        if not self._gallery or k <= 0:
            return []
        q0 = float(q[0]) if q.size else 0.0
        pos = int(np.searchsorted(self._keys, q0))
        left, right = pos - 1, pos
        best: list[_Hit] = []
        while True:
            gaps = []
            if left >= 0:
                gaps.append((abs(q0 - float(self._keys[left])), 0, left))
            if right < self._keys.size:
                gaps.append((abs(float(self._keys[right]) - q0), 1, right))
            if not gaps:
                break
            gap, side, idx = min(gaps)
            if len(best) >= k and gap > best[-1].distance:
                break
            gid = int(self._order[idx])
            best.append(_Hit(index=gid, distance=spectral_distance(q, self._gallery[gid])))
            best.sort(key=lambda h: (h.distance, h.index))
            del best[k:]
            if side == 0:
                left -= 1
            else:
                right += 1
        return best
