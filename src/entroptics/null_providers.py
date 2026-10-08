"""
null_providers.py -- the noise floor as a caller-suppliable, local null provider.

A *null provider* is a callback ``FloorContext -> float``: given one screen it returns
that screen's noise floor (a scalar in the units of its spectrum).  It is evaluated
locally, on every screen -- so under a per-plane / per-window / streaming read it is
recomputed on each local screen, never once globally, because a single global floor
self-contaminates: a loud region inflates the estimate and buries a quiet-region signal.

Noise is a substrate- and region-specific variable; no library can enumerate every
profile.  So every noise-vs-signal cutoff the library makes -- ``K_signal`` (projection),
``resolved_modes`` (correlation), and the resolved dimension of every downstream read --
routes through the provider: the caller is involved in every cutoff, by construction.

The false-alarm level (alpha, ``far``) travels with the null, not beside it: the cutoff is
one decision, so the provider owns both the threshold and the alpha it is drawn at
(``ctx.far`` is the read's target; a provider may honour it or pin its own).  Because a
provider is stateful and updated per frame in the dynamical, with the whole local/global
optics history at hand, it can sharpen alpha as a run progresses -- toward 99.999% and
beyond -- and adapt, even predictively, to a drifting noise level in any local or global
region.  The derived edge serves an arbitrary, arbitrarily-sharp alpha (the TW1 quantile is
inverted from the survival function, no table); a sampled provider sharpens alpha
empirically as its long-term surrogate sample grows.

So the read parameter is not a strategy name; it is a provider callback, and the library only

  1. ships a limited set of providers here -- the exact ``permutation`` test, the closed-form
     edge ``mp``, the library default that chooses between them per read (``switched``) and
     ``robust`` -- nothing fitted; and
  2. offers the plumbing to build a sampled null (``top_spectrum_value``,
     ``shuffle_in_time``, ``floor_from_null_sampler``).

The library does not calibrate the caller's null for them.  A confined-vacuum reference,
a phase-randomised surrogate, a physics null -- the caller writes it with the plumbing
(or from scratch) and passes it in; the library never needs to know what the null is.

The default is :func:`default_provider`, wherever the read holds its samples -- a screen, a
correlation read's centred samples, and the windows the pools hold (``ResolvedScreen`` and
``ResolvedScreenBatch``, ``SpectralAccumulator``, ``Dynamics``), whose rows expire as the aperture
forgets: :func:`switched`, the closed form where no single row of the data can carry a noise
eigenvalue over it at the read's level, the exact permutation test where one can.  The caller
plugs any method -- or its own callback -- via ``null=``:

  (1) analytic edge -- i.i.d. light-tailed noise, no reference:
      mp                                 finite-size Johnstone / Tracy-Widom edge; the noise
                                         level is estimated from the data, so nothing is supplied.
                                         Alone its level is not ``far``: on a sample correlation
                                         it sits below on light tails and above on heavy right
                                         tails, where single rows carry the top eigenvalue.  The
                                         default takes it only where no row can
                                         (``closed_form_holds``).
  (2) robust fence -- heavy-tailed spectrum, no reference:
      robust                             Tukey upper fence ``Q3 + 1.5*(Q3 - Q1)`` of the spectrum
                                         (a heuristic outlier fence, not a calibrated null).
  (3) empirical reference -- calibrate on a signal-free window (the exact rank):
      reference_null(planes | values)    the exact rank over a quiet window's realisations, each
                                         read at the width of the screen it thresholds; with n of
                                         them it claims nothing below far = 1/(n+1).
      ReferenceNull(..., forgetting=)    the stateful form; ``forgetting<1`` holds only the
                                         effective count of recent values, (1+f)/(1-f), to track
                                         drift.
      self_calibrating_null(noise, ...)  a ``reference_null`` calibrated locally on a region's
                                         own off-pulse noise -- self-contained, region-dynamic.
  (4) sampled / distribution-free -- no model, resample the data:
      permutation()                      the exact Monte Carlo floor of a per-channel time
                                         shuffle, at the fewest draws the level allows: the
                                         default at the projection cut point, at level ``far``
                                         for any law of the noise.
      floor_from_null_sampler(surrogate) turn any surrogate into a provider (block bootstrap,
                                         phase randomisation, a physics null, ...).
      shuffle_in_time, top_spectrum_value   the example surrogate + the scoring building block.
  (5) weighted aggregation -- the screen is a weighted mean of an ensemble:
      weighted_effective(stack, w)       CLOSED FORM.  A weighted mean has per-cell variance
                                         Var(X)/effective_n with Kish's count, so the aggregate
                                         is an ordinary screen at a noise level the weights
                                         dictate, and the derived edge applies unchanged once
                                         the noise is read at the effective count rather than
                                         the row count.  No draws, no seed, no quantile.  (A
                                         bootstrap version was written first and removed: it
                                         resampled the very ensemble whose unrepresentativeness
                                         it was meant to price, so it inherited the blindness.)

A different provider per cut point.  Each cut point (``KINDS``: ``"projection"`` = K_signal,
``"spectral"`` = single-screen resolved_modes, ``"bulk"`` = the pooled SpectralAccumulator)
is a separate decision and can take its own provider: pass ``null=by_kind(projection=P,
spectral=Q, bulk=R)`` (or the same ``{kind: provider}`` dict) to any read or to
``Aperture(null=...)``, where it routes the screen and spectral floors apart; an unset cut
point falls back to the default.

A provider may be a plain function or a stateful object with ``__call__(ctx) -> float``
and an optional ``update(frame)``: the streaming aperture calls ``update`` per frame, so
a provider can maintain an online local null that tracks a non-stationary stream -- it
runs in the dynamical, alongside the DMD operator.

Numpy-only; backend-agnostic where marked; a resampling provider is deterministic per the
seed carried in the ``FloorContext``.
"""
from __future__ import annotations

import math
from fractions import Fraction
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import tracy_widom as _tw

from . import environment as _env
from .entropy import MAD_SCALE

# The distinct cut points a provider can be keyed to (``ctx.kind``).  Each is a separate
# noise-vs-signal decision, so each can take its own provider (see ``by_kind`` / a mapping):
#   "projection" -- the screen singular-value floor (K_signal)        [projection.noise_floor]
#   "spectral"   -- the single-screen correlation-eigenvalue floor     [reads.spectral_optics]
#   "bulk"       -- the pooled ensemble correlation floor              [reads.SpectralAccumulator]
# A new cut point adds a new kind here and a new key.
KINDS = ("projection", "spectral", "bulk")


# The ensemble a screen belongs to is read off its dtype: `_env.is_complex_obj`.  A
# covariance-only context carries no data, so its caller states the ensemble (`FloorContext.complex_`,
# `apply_floor(..., complex_=)`).


# ══════════════════════════════════════════════════════════════════════════════
# Tracy-Widom_1 (GOE / real matrices): universal edge quantiles + survival function
# ══════════════════════════════════════════════════════════════════════════════

# The edge laws are evaluated exactly: :mod:`entroptics.tracy_widom` computes TW1 (real, GOE) and
# TW2 (complex, GUE) to float precision from their Fredholm determinants.  There is no table and no
# approximation, so no level is privileged and every level -- 0.05 or 1e-12 -- gets the same law.
# A complex matrix's largest eigenvalue follows TW2, not TW1; the ensemble is read off the data.


def _reg_gamma_upper(a: float, x: float) -> float:
    """Regularized upper incomplete gamma ``Q(a, x) = Gamma(a, x) / Gamma(a)`` for ``x >= 0``, to
    float precision: the series for ``x < a + 1``, Lentz's continued fraction otherwise, each summed
    until its next term is below round-off of the sum; ``log Gamma`` from ``math.lgamma``.  The
    continued fraction's zero guard is the float format's smallest normal number."""
    if math.isnan(a) or math.isnan(x):
        return math.nan
    if x <= 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    eps = float(np.finfo(float).eps)
    lead = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        ap, term = a, 1.0 / a
        total = term
        while abs(term) > eps * abs(total):
            ap += 1.0
            term *= x / ap
            total += term
        return 1.0 - total * math.exp(lead)
    tiny = float(np.finfo(float).tiny)
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    i = 0
    while True:
        i += 1
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = d if abs(d) >= tiny else tiny
        c = b + an / c
        c = c if abs(c) >= tiny else tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= eps:
            return math.exp(lead) * h


def tw1_quantile(far: float) -> float:
    """The TW1 upper quantile ``q`` with ``P(TW1 <= q) = 1 - far``, exact to float precision at
    any ``far`` in (0, 1)."""
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    return _tw.quantile(float(far), 1)


def tw1_sf(g):
    """P(TW1 > g): the Tracy-Widom_1 upper-tail probability, exact to float precision (relative
    precision in the upper tail); elementwise over an array."""
    return _tw.survival(g, 1)


def tw2_sf(g):
    """P(TW2 > g): the Tracy-Widom_2 (GUE) upper-tail probability -- the law a COMPLEX matrix's
    largest eigenvalue follows -- exact to float precision; elementwise over an array."""
    return _tw.survival(g, 2)


def tw2_quantile(far: float) -> float:
    """The TW2 upper quantile ``q`` with ``P(TW2 <= q) = 1 - far`` -- the complex sibling of
    :func:`tw1_quantile`, exact to float precision at any ``far`` in (0, 1)."""
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    return _tw.quantile(float(far), 2)


def tw_sf(g, *, complex_: bool = False):
    """P(TW > g) for the ensemble the data belongs to: TW1 for a real matrix, TW2 for a complex one."""
    return tw2_sf(g) if complex_ else tw1_sf(g)


def tw_quantile(far: float, *, complex_: bool = False) -> float:
    """The edge quantile for the ensemble the data belongs to: TW1 (GOE) for a real matrix,
    TW2 (GUE) for a complex one.  Dispatching on the data's own dtype is the point -- the
    ensemble is a fact about the data, not a caller preference."""
    return tw2_quantile(far) if complex_ else tw1_quantile(far)


# ══════════════════════════════════════════════════════════════════════════════
# Finite-size Johnstone edge + de-biased per-cell noise variance (shared primitives)
# ══════════════════════════════════════════════════════════════════════════════

def johnstone(N: int, F: int, *, complex_: bool = False) -> tuple[float, float]:
    """Johnstone (2001) centering ``mu`` and scaling ``sigma`` for the largest
    eigenvalue (top singular value squared) of an N x F Gaussian matrix, so that
    ``(lambda_max - mu)/sigma -> Tracy-Widom``.  The derived finite-size edge (no fitted
    coefficient).  Called with (N, F) for the screen, (T, N) for the correlation floor.

    ``complex_`` selects the ensemble's own adjustment.  Chiani (2014) eq. (33)-(34) writes the
    centring as ``(sqrt(N + a1) + sqrt(F + a2))^2`` with ``a1 = a2 = -1/2`` for the real Wishart
    and ``a1 = a2 = 0`` for the complex one; the real branch keeps Johnstone's own ``N - 1``,
    which is what every real read here has always used and must not move."""
    nn = math.sqrt(max(N, 1)) if complex_ else math.sqrt(max(N - 1, 1))
    ff = math.sqrt(max(F, 1))
    a = nn + ff
    return a * a, a * (1.0 / nn + 1.0 / ff) ** (1.0 / 3.0)


def debias_denominator(N: int, F: float, *, complex_: bool = False) -> float:
    """The de-biasing denominator ``F * c_F * dof`` that turns a median row energy into the
    per-cell noise variance ``sigma^2``, for :func:`proximity.mp_spectrum`'s prediction (the
    screen's floor reads its null's exact variance instead, :func:`noise_sigma2_from_spectrum`):

      c_F        -- the sample median of ||row||^2 estimates the distribution median of a
                    chi^2_F (= F*c_F), not the mean F (Wilson-Hilferty, a small-F bias);
      (N-1)/N    -- the per-channel median centring deflates the row energy; the mean-
                    centring dof is applied as a conservative correction (slightly over-
                    corrects at small N, raising the floor in the safe direction).

    ``F`` is real-valued, not integer: ``proximity`` reads an EFFECTIVE width whose whole claim
    is that it has no discrete steps, and truncating it here would put one back.  Bit-identical
    to the integer form at every integer width, so no caller moves."""
    Ff = float(F)
    # A REAL row energy is sigma^2 * chi^2_F, median ~ F(1 - 2/(9F))^3 (Wilson-Hilferty).  A
    # COMPLEX row energy is a sum of F i.i.d. Exp(sigma^2), i.e. sigma^2 * chi^2_{2F}/2, whose
    # median is F(1 - 1/(9F))^3 -- the same correction at twice the degrees of freedom.
    c_F = (1.0 - (1.0 if complex_ else 2.0) / (9.0 * Ff)) ** 3     # Ff >= 1: a participation width
    dof = max(int(N) - 1, 1) / int(N)
    return Ff * c_F * dof


def screen_floor_sq(sigma2, N: int, F: int, far: float, *, complex_: bool = False):
    """The screen noise floor squared (in variance / eigenvalue units): ``sigma^2 * (mu +
    q*sigma_J)`` with the finite-size Johnstone centring/scaling and the TW1 quantile at ``far``.
    ``sigma2`` may be a scalar (one screen) or an array (per-frame over a batch); the return has
    its shape.  Take ``sqrt`` for the singular-value floor.  One definition shared by the
    per-frame ``mp`` provider, the numpy batch floor, and the batched resolved read."""
    mu, sig_J = johnstone(int(N), int(F), complex_=complex_)
    q = tw_quantile(far, complex_=complex_)
    return sigma2 * (mu + q * sig_J)


def noise_sigma2_from_spectrum(s2, N: int, F: int):
    """The per-cell variance the ``mp`` floor builds on, from an ``N x F`` screen's squared
    singular values ``s2`` (``(..., r)``): its mean cell energy with the degree of freedom each
    column's mean took, ``sum s^2 / (N F) * N / (N - 1)``.  Returns ``(...)``.

    The whitening gives every column the same energy (:func:`entropy.whiten_stats`), so the
    screen's Gram is a sample correlation matrix and this is the variance of its null -- channels
    independent -- exactly, for noise of any marginal: ``N / (N - 1)`` on an unfolded screen of
    live columns, and whatever the fold and the flat channels leave otherwise.  It is not an
    estimate of a noise level under a signal.  A signal takes its share of each channel's energy,
    and the noise is left with the rest, so a mode is read against what independent channels of
    the same energies would produce -- the correlation read's own question.  (A variance estimated
    to be robust to the signal instead -- a median -- finds the noise left at a different level in
    each channel, by that channel's signal share, and reads the loudest as modes: 6-12 modes on a
    rank-2 record.)  ``F`` is the LIVE width: a dead column is not a cell of the ensemble."""
    s2 = np.asarray(s2, dtype=np.float64)
    N, F = int(N), int(F)
    if N < 1 or F < 1:
        return np.zeros(s2.shape[:-1]) if s2.ndim > 1 else 0.0
    out = np.clip(s2, 0.0, None).sum(axis=-1) / (N * F) * N / max(N - 1, 1)
    return out if s2.ndim > 1 else float(out)


def noise_sigma2(xp, screen, N: int, F: int, *, complex_: bool = False, s=None) -> float:
    """The per-cell variance the ``mp`` floor builds on (:func:`noise_sigma2_from_spectrum`), read
    from the screen's singular values ``s`` (computed when not given).  Shared by the
    per-frame floor, the per-mode significance and the batched floors, so they cannot drift.
    ``F`` is the LIVE width (:func:`projection.live_columns`): a dead column is not a cell of the
    ensemble.  ``complex_`` is kept for the callers' signature; the law is the same."""
    if int(N) <= 0 or not (F > 0):
        return 0.0
    if s is None:
        s = _env.svdvals(xp, screen)
    s = np.asarray(_env.to_numpy(s), dtype=np.float64)
    return noise_sigma2_from_spectrum(s ** 2, N, F)


# ══════════════════════════════════════════════════════════════════════════════
# The null-provider contract
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FloorContext:
    """Everything a null provider may need for one screen, so a provider is a pure
    ``FloorContext -> float`` returning a scalar in the units of ``spectrum``.

    ``kind`` is the cut point (see ``KINDS``); it fixes the units and how a surrogate is
    scored, and it is the key a per-cut-point mapping / ``by_kind`` dispatches on:
      "projection" -- ``spectrum`` are singular values; ``data`` is the (N, F) screen; a
                    surrogate is scored by its top singular value.
      "spectral"   -- ``spectrum`` are eigenvalues of a unit-diagonal correlation matrix;
                    ``data`` is the (T, N) centred samples; a surrogate is scored by the top
                    eigenvalue of its correlation matrix (one screen).
      "bulk"       -- as "spectral" but a pooled floor (``SpectralAccumulator``, the
                    ``Dynamics`` feature spectrum): same correlation units, a separate key so it
                    takes its own provider; ``data`` is the pool's held samples (its window), whose
                    Gram is the pooled covariance, and ``resample`` the pool's own null draw of them.
    ``data`` is ``None`` only where no rows are held (a ``Dynamics`` resumed from a state that
    carried none): then only the closed-form providers (``mp`` / ``robust`` / a ``reference_null``)
    apply, and a resampling provider raises."""
    spectrum: np.ndarray | None            # descending singular values (screen) or corr eigenvalues
    data:     np.ndarray | None            # the matrix behind the spectrum, or None (covariance-only)
    shape:    tuple                        # (N, F) screen; (T, N) correlation ("spectral"/"bulk")
    far:      float                        # false-alarm level -- the provider owns the cutoff (below)
    kind:     str                          # one of KINDS: "projection" | "spectral" | "bulk"
    rng:      "np.random.Generator"        # seeded generator for any resampling (determinism)
    complex_: bool = False                 # the data's ensemble when ``data`` is None (a covariance)
    resample: "Callable[[np.random.Generator], np.ndarray] | None" = None
    dof: "int | None" = None               # correlation kinds: the covariance's degrees of freedom
    gram: "np.ndarray | None" = None       # data^H data, when the read already formed it (T >= F)
    # ``resample(rng)`` is one surrogate SCREEN drawn by the read that made ``data``: its whitened
    # record with each channel's measured values shuffled among that channel's measured cells, the
    # mask held, then folded as the observed screen was.  A read with the record offers it; with
    # missing cells it must, because the missingness is fixed structure: shuffling the finished screen would scatter the
    # zeros standing in for the missing cells, which is not a draw from the null.
    # ``dof`` is the degrees of freedom of a correlation kind's covariance when they are not the
    # rows less one: a covariance pooled over planes each centred on its own mean has the rows less
    # one per plane.  ``None``: the rows less one, one centring.

    # ``far`` is the false-alarm level (alpha) delivered with the null, because the
    # noise-vs-signal cutoff is one decision, not two: the provider owns it.  ``ctx.far``
    # is the read's target; a provider may honour it, pin its own, or -- being stateful and
    # updated per frame in the dynamical -- sharpen it as its long-term local/global sample
    # grows (toward 99.999%+), even predictively as the noise level drifts.


# NullProvider = Callable[[FloorContext], float]   (optionally a stateful object with
# __call__(ctx)->float and update(frame); the streaming aperture calls update per frame).


# ══════════════════════════════════════════════════════════════════════════════
# The closed-form providers (pure functions of the local screen)
# ══════════════════════════════════════════════════════════════════════════════

def mp(ctx: FloorContext) -> float:
    """The closed-form provider: the finite-size Johnstone / Tracy-Widom edge; the default's branch
    where no single row can carry a noise eigenvalue over it (:func:`closed_form_holds`), and the
    floor where no rows are held.  Projection:
    ``sqrt(sigma^2 * (mu + q*sigma_J))`` with the null's per-cell variance
    (:func:`noise_sigma2_from_spectrum`).  Correlation
    floor: ``(mu + q*sigma_J)/T`` in correlation units.  Parameter-free; only ``far``."""
    # The ensemble is a fact about the data, so it is read off the data and never asked for: a
    # complex screen's largest eigenvalue follows TW2 (GUE), with the complex Wishart centring
    # and the complex row-energy de-bias.  Real input takes exactly the path it always did.
    cx = ctx.complex_ or _env.is_complex_obj(ctx.data)
    q = tw_quantile(ctx.far, complex_=cx)
    if ctx.kind == "projection":
        N, F = int(ctx.shape[0]), int(ctx.shape[1])
        xp = _env.ns(ctx.data)
        s2 = noise_sigma2(xp, ctx.data, N, F, complex_=cx, s=ctx.spectrum)
        return math.sqrt(screen_floor_sq(s2, N, F, ctx.far, complex_=cx))
    # the closed form reads the rows as one centring's worth: a covariance with ``dof`` degrees of
    # freedom is the edge of ``dof + 1`` rows (a pool of planes centred one by one has fewer than
    # its rows, and its edge is that of the smaller sample)
    T = int(ctx.shape[0]) if ctx.dof is None else int(ctx.dof) + 1
    N = int(ctx.shape[1])
    mu, sig_J = johnstone(T, N, complex_=cx)
    return (mu + q * sig_J) / T


def robust(ctx: FloorContext) -> float:
    """The deterministic Tukey upper fence ``Q3 + 1.5*(Q3 - Q1)`` of the spectrum -- a
    heuristic outlier fence for heavy-tailed spectra (not a calibrated null)."""
    base = ctx.spectrum if ctx.spectrum is not None else ctx.data
    xp = _env.ns(base)
    sv = ctx.spectrum if ctx.spectrum is not None else xp.linalg.svd(ctx.data, compute_uv=False)
    q1 = float(xp.quantile(sv, 0.25)); q3 = float(xp.quantile(sv, 0.75))
    return q3 + 1.5 * (q3 - q1)


DEFAULT: Callable[[FloorContext], float] = mp   # the closed-form floor: the default without samples


# ══════════════════════════════════════════════════════════════════════════════
# Build-your-own-null plumbing (the caller supplies the null; the library the quantile)
# ══════════════════════════════════════════════════════════════════════════════

def top_spectrum_value(X: np.ndarray, kind: str) -> float:
    """Score a surrogate ``X`` in the same units the floor thresholds: the top singular
    value (``kind="projection"``) or the top unit-diagonal correlation eigenvalue (any other
    kind -- ``"spectral"`` / ``"bulk"``).  The building block for a sampled null provider.
    Numpy (the occasional calibrated read).

    The correlation branch is read off the column-scaled frame.  Scaling column ``j`` by ``1/d_j``
    gives ``Y`` with ``Y^H Y == R`` exactly, so ``R``'s top eigenvalue is the top eigenvalue of
    ``Y``'s smaller Gram, ``min(T, F)`` square -- the projection branch's own read, at a fraction of
    an SVD of the ``T x F`` frame (0.15 ms against 0.70 ms at 2000 x 32), and within the round-off
    :func:`_roundoff` prices, which is a Gram's.

    This stays pure numpy: ``scipy.linalg.eigh(subset_by_index=...)`` also returns just the top
    eigenvalue, bit-identically, but scipy is an optional extra here (``pyproject``: core is
    ``numpy>=2.0`` alone), and this is a hot path -- a core read should not take on an optional
    dependency for it.
    """
    X = np.asarray(X)
    if kind not in KINDS:
        # Named, not defaulted: the two branches score different quantities, so a kind that is
        # not a cut point must raise, never fall through to the correlation branch and
        # calibrate a reference on the wrong statistic.
        raise ValueError(f"top_spectrum_value: unknown kind {kind!r}; expected one of {KINDS}. "
                         f"The cut point formerly called 'screen' is now 'projection'.")
    if kind == "projection":
        # the top singular value is the root of the smaller Gram's top eigenvalue: to round-off,
        # at a fraction of an SVD's cost (the Gram is min(N, F) square)
        G = X.conj().T @ X if X.shape[0] >= X.shape[1] else X @ X.conj().T
        return float(np.sqrt(max(np.linalg.eigvalsh(G)[-1], 0.0))) if G.size else 0.0
    # d_j is column j's 2-norm -- i.e. sqrt(Cov_jj) -- so the covariance never has to be formed.
    # RELATIVE floor: column energies scale as the square of the data (see reads.spectral_optics)
    if X.size == 0:
        return 0.0                                     # no live column: nothing to correlate
    return _correlation_top(X, np.real(np.sum(X.conj() * X, axis=0)))


def _correlation_top(X: np.ndarray, e: np.ndarray) -> float:
    """:func:`top_spectrum_value` at a correlation cut point, from the column energies ``e`` already
    read off ``X`` -- so a caller that also prices the round-off reads them once."""
    top = float(e.max()) if e.size else 0.0
    d = np.sqrt(np.clip(e, top * np.finfo(float).eps, None))
    d = np.where(d == 0, 1.0, d)
    Y = X / d
    G = Y.conj().T @ Y if Y.shape[0] >= Y.shape[1] else Y @ Y.conj().T
    return float(max(np.linalg.eigvalsh(G)[-1], 0.0))


def shuffle_in_time(X: np.ndarray, rng) -> np.ndarray:
    """An example surrogate: shuffle each column independently along the ordered axis
    (rows), destroying ordered and cross-channel structure while preserving every channel's
    own marginal.  The surrogate behind ``permutation``.

    ``Generator.permuted`` draws an independent uniform permutation of every column (a
    Fisher-Yates shuffle each, no sort).  Column marginals are exactly preserved, since a
    permutation moves values but never alters them.  A fixed seed reproduces the draw; a
    different shuffling scheme would draw a different sample of the same null.
    """
    return rng.permuted(np.asarray(X), axis=0)


def floor_from_null_sampler(surrogate: Callable[[np.ndarray, "np.random.Generator"], np.ndarray],
                            *, draws: int | None = None, far: float | None = None,
                            ) -> Callable[[FloorContext], float]:
    """Turn any surrogate into a null provider: the floor is the ``ceil((1 - far)(draws + 1))``-th
    smallest top spectrum value over ``draws`` draws of ``surrogate(data, rng)``.  The library owns
    the order statistic; the caller owns the null mechanism -- shuffle, block bootstrap, phase
    randomisation, a draw from a signal-free reference, a physics surrogate.  The returned
    provider is a local, per-screen callback like any other.

    That order statistic makes it an exact Monte Carlo test: when the observed record is
    exchangeable with its surrogates, it lands in any of the ``draws + 1`` ranks with equal
    chance, so it exceeds the floor with probability at most ``far`` at every ``draws`` -- no
    interpolated quantile, no large-sample approximation.  ``draws`` sets power, not level; below
    ``1/far - 1`` no rank is extreme enough and the floor is infinite (nothing can be claimed);
    ``None`` takes exactly that many (:func:`fewest_draws`).

    ``far`` couples the false-alarm level into the provider: ``None`` uses the caller's
    ``ctx.far`` (the read's target), a value pins the provider's own level."""
    def _provider(ctx: FloorContext) -> float:
        if ctx.data is None:
            raise ValueError("a sampled null provider needs the raw samples; not available "
                             "from a covariance-only accumulator")
        f = ctx.far if far is None else far
        n = fewest_draws(f) if draws is None else int(draws)
        # The observed exceeds the k-th of n at n + 1 - k of the n + 1 equally likely ranks.
        X = np.asarray(_env.to_numpy(ctx.data))
        if surrogate is shuffle_in_time and ctx.resample is not None:
            score = _exact_rank_score(lambda: _scored(ctx.resample(ctx.rng), ctx.kind), n, f)
            return _observed_floor(score, X, ctx.kind)
        Xs = X
        if surrogate is shuffle_in_time:
            # a column that is identically zero shuffles to itself and carries no part of any
            # spectrum; leaving it out makes the draws independent of dead columns, not just their law
            Xs = X[:, np.any(X != 0, axis=0)]
        score = _exact_rank_score(lambda: _scored(surrogate(Xs, ctx.rng), ctx.kind), n, f)
        return _observed_floor(score, X, ctx.kind)
    _provider.__name__ = f"{getattr(surrogate, '__name__', 'sampled')}_null"
    return _provider


def fewest_draws(far: float) -> int:
    """The fewest surrogate draws with which an exact Monte Carlo test can claim anything at
    ``far``: ``n`` such that the observed record, one of ``n + 1`` equally likely ranks, can sit
    alone above every draw with probability at most ``far`` -- ``ceil(1/far) - 1`` (19 at 0.05).
    Measured on planted signals, 199 draws found no more than 19 did."""
    return max(1, math.ceil(1 / Fraction(far)) - 1)


def _roundoff(X: np.ndarray, kind: str, value: float, e=None) -> float:
    """A bound on the round-off in :func:`top_spectrum_value`'s ``value`` for ``X``.  The top value
    is the root (``"projection"``) or the value (correlation kinds) of the largest eigenvalue of a
    Gram of inner size ``m`` and order ``p``; forming it moves that eigenvalue by at most
    ``m eps ||X||_F^2`` and the symmetric eigensolver by ``p eps ||X||_F^2`` (Higham), so
    ``|d lambda| <= (m + p) eps ||X||_F^2`` -- a column-scaled frame for the correlation kinds.

    A comparison finer than this is not evidence: when the statistic is exactly invariant under
    the surrogate (one live channel: its norm is what a time shuffle keeps), the observed value
    and every draw differ only in the order the sums were taken, and without the margin the
    observed exceeds a draw by an ulp about as often as not."""
    X = np.asarray(X)
    if X.size == 0:
        return 0.0
    m, p = max(X.shape), min(X.shape)
    e = np.real(np.sum(X.conj() * X, axis=0)) if e is None else e
    if kind != "projection":
        e = (e > 0).astype(float)                         # unit-diagonal: each live column is 1
    # the arithmetic the Gram was formed in: a float32 screen rounds 5e8 times coarser than float64
    eps = float(np.finfo(X.dtype).eps) if np.issubdtype(X.dtype, np.inexact) else float(np.finfo(float).eps)
    dlam = (m + p) * eps * float(np.sum(e))
    if kind != "projection":
        return dlam
    return min(dlam / value, math.sqrt(dlam)) if value > 0 else math.sqrt(dlam)


def _screen_scale(X: np.ndarray, e=None):
    """The closed-form null's scale for a screen ``X`` at its own shape: the per-cell variance
    (:func:`noise_sigma2_from_spectrum`, over the live width) and the Johnstone centring and
    fluctuation scale.  ``None`` for a screen with no energy.  ``e``: the column energies, when
    the caller has them."""
    X = np.asarray(X)
    N = int(X.shape[0])
    e = np.real(np.sum(X.conj() * X, axis=0)) if e is None else e
    F = max(int(np.count_nonzero(e)), 1)
    s2 = noise_sigma2_from_spectrum(np.array([float(e.sum())]), N, F)
    if not (s2 > 0.0):
        return None
    mu, sig_J = johnstone(N, F, complex_=bool(np.iscomplexobj(X)))
    return float(s2), mu, sig_J


def _deviate(value: float, scale) -> float:
    """A projection top value ``s`` as the standardized deviate ``(s^2/sigma^2 - mu)/sigma_J`` of
    its own screen's null (``-inf`` for a screen with no energy)."""
    if scale is None:
        return -math.inf
    s2, mu, sig_J = scale
    return (value * value / s2 - mu) / sig_J


def _scored(X: np.ndarray, kind: str) -> float:
    """What a surrogate is ranked by, raised by its round-off bound.  Correlation kinds: the top
    eigenvalue, in correlation units.  Projection: the top value as its screen's standardized
    deviate, because a draw's screen need not have the observed one's width -- its fold is decided
    on the draw -- and a singular value is only comparable to another at the same width; at the
    same width the deviate is the singular value's own order, so nothing else moves."""
    X = np.asarray(X)
    if kind != "projection":
        if kind not in KINDS or X.size == 0:
            v = top_spectrum_value(X, kind)
            return v + _roundoff(X, kind, v)
        e = np.real(np.sum(X.conj() * X, axis=0))            # one pass, shared by both reads
        v = _correlation_top(X, e)
        return v + _roundoff(X, kind, v, e)
    v = top_spectrum_value(X, kind)
    e = np.real(np.sum(X.conj() * X, axis=0))                # one pass, shared by both reads
    scale = _screen_scale(X, e)
    g = _deviate(v, scale)
    return g + (0.0 if scale is None else _roundoff_deviate(X, v, scale, e))


def _roundoff_deviate(X: np.ndarray, value: float, scale, e=None) -> float:
    """:func:`_roundoff` carried into deviate units: ``d(s^2) / (sigma^2 sigma_J)``."""
    ds = _roundoff(X, "projection", value, e)
    s2, _, sig_J = scale
    return ((value + ds) ** 2 - value * value) / (s2 * sig_J)


def _observed_floor(score: float, X: np.ndarray, kind: str) -> float:
    """A floor found in scoring units, stated in the observed screen's spectrum units, raised by
    the observed read's own round-off."""
    X = np.asarray(X)
    if kind != "projection":
        if kind not in KINDS or X.size == 0:
            return score + _roundoff(X, kind, top_spectrum_value(X, kind))
        e = np.real(np.sum(X.conj() * X, axis=0))            # one pass, shared by both reads
        return score + _roundoff(X, kind, _correlation_top(X, e), e)
    v = top_spectrum_value(X, kind)
    scale = _screen_scale(X)
    if scale is None or not math.isfinite(score):
        return float("inf") if (scale is None or score > 0) else 0.0
    s2, mu, sig_J = scale
    g = score + _roundoff_deviate(X, v, scale)
    return math.sqrt(max(s2 * (mu + g * sig_J), 0.0))


def _exact_rank_score(draw, n: int, far: float) -> float:
    """The ``n + 1 - floor(far (n + 1))``-th smallest of ``n`` scored draws (exact rationals on
    ``far``): the floor of an exact Monte Carlo test, in scoring units, which the observed must
    clear by more than either side's arithmetic can move it.  Infinite when no rank is rare
    enough."""
    k = n + 1 - math.floor(Fraction(far) * (n + 1))
    if k > n:
        return float("inf")
    tops = np.fromiter((draw() for _ in range(n)), float, count=n)
    return float(np.partition(tops, k - 1)[k - 1])


def permutation(*, draws: int | None = None, far: float | None = None) -> Callable[[FloorContext], float]:
    """The exact permutation floor: each channel shuffled in time (destroying ordered and
    cross-channel structure, keeping every marginal), scored at the exact Monte Carlo rank
    (:func:`floor_from_null_sampler`).  Distribution-free: the false-alarm rate is at most ``far``
    whatever the noise's law.  ``draws=None`` takes :func:`fewest_draws` at the level in force.
    Deterministic per the seed carried in the ``FloorContext``.  ``far=None`` uses the read's
    ``ctx.far``; a value pins the provider's own."""
    provider = floor_from_null_sampler(shuffle_in_time, draws=draws, far=far)
    # marks the provider as this exact test, with its operating point, so a read can report the
    # evidence the floor was drawn from (``Projection.significance``) whichever instance it is
    provider.exact_permutation = (draws, far)
    return provider


def weighted_effective(stack, weights) -> Callable[["FloorContext"], float]:
    """The floor for a weighted aggregation, DERIVED -- no resampling, no draws, no quantile.

    Bootstrapping the ensemble would price the aggregate by resampling it.  That is a
    fit, and worse, it is a fit on the very sample whose unrepresentativeness is the thing
    being priced: if the ensemble missed the configurations that carry the answer, every
    replicate misses them too, and the floor inherits the blindness it was built to measure.

    This one is closed form.  A weighted mean of M samples has per-cell variance

        Var(C) = Var(X) * sum w^2 / (sum w)^2 = Var(X) / effective_n

    with `effective_n` Kish's count -- a derived property of the weights alone, already in the
    library as `reads.carriage(...).effective_n`.  So the aggregate IS an ordinary screen, at a
    noise level the weights dictate, and `mp`'s own derived edge applies to it unchanged once
    the noise is read at the effective count rather than the row count.

    Nothing here is drawn, chosen or calibrated: the per-cell dispersion is measured across the
    ensemble (a measurement, not a fit), divided by a derived count, and handed to the same
    `screen_floor_sq` every other read goes through.
    """
    stack = np.asarray(_env.to_numpy(stack))
    w = np.asarray(_env.to_numpy(weights), dtype=float).ravel()
    if stack.ndim != 3:
        raise ValueError(f"stack must be (M, T, F); got shape {stack.shape}")
    if w.shape[0] != stack.shape[0]:
        raise ValueError(f"weights has {w.shape[0]} entries for {stack.shape[0]} samples")
    sw, sw2 = float(w.sum()), float(np.sum(w ** 2))
    if sw == 0.0:
        raise ValueError("weights sum to zero: the aggregation is undefined, so is its floor")
    eff = sw * sw / sw2                                   # Kish, exactly reads.carriage's
    aggregate = np.tensordot(w, stack, axes=(0, 0)) / sw

    def _provider(ctx: "FloorContext") -> float:
        from .entropy import whiten_stats
        from .projection import fold_variance, _fold_ortho
        N, F = int(ctx.shape[0]), int(ctx.shape[1])
        _, scale = whiten_stats(np, aggregate)
        scale = np.asarray(scale)
        live = (scale > 0) & np.isfinite(scale)                       # the channels the read keeps
        agg, stk, scale = aggregate[:, live], stack[:, :, live], scale[live]
        T, Fin = agg.shape
        dev = (stk - agg[None]) / scale[None, None, :]                # screen units

        # Each row's MEAN per-cell variance -- not a robust centre, which reports the quiet cells
        # when the loud ones set the top singular value.
        per_row = np.mean(dev ** 2, axis=(0, 2))                      # (T,)

        # ...and one variance only describes the screen if the rows SHARE one.  How much
        # row-to-row spread mere sampling produces is not taken from an asymptotic formula --
        # the first version used sqrt(2/(M*Fin)) and fired on homoscedastic ensembles, because
        # the aggregate is a SIGNED reweighting and its effective degrees of freedom are not
        # M*Fin.  It is measured from the ensemble instead: split the samples in half, and the
        # disagreement between the halves' per-row variances IS the sampling spread, with no
        # formula and nothing to tune.  A screen whose rows differ by more than that is refused
        # rather than priced -- the floor would be a single number for a thing that has none,
        # and returning it quietly is how a read comes back confidently wrong.
        M = stack.shape[0]
        h = M // 2
        if h >= 2:
            a = np.mean(dev[:h] ** 2, axis=(0, 2))
            b = np.mean(dev[h:2 * h] ** 2, axis=(0, 2))
            sampling = float(np.std(a - b) / math.sqrt(2.0))      # per-half -> per-row scale
            between = float(np.std(per_row))
            if between > sampling * _norm_isf(ctx.far / max(T, 1)):
                raise ValueError(
                    "weighted_effective: the ensemble's per-sample noise is not homoscedastic "
                    f"in the screen's units -- row-to-row spread {between:.3g} against the "
                    f"{sampling:.3g} this ensemble's own split-half disagreement allows. A "
                    "single floor does not describe this screen; whiten the ordered axis "
                    "first, or use a provider that does not assume one noise level.")

        # The columns need not share one variance: the aggregate's channel scale carries its own
        # signal, so its noise lands at a different level in each column, and the edge of such a
        # matrix is set by the second moment of the column variances as well as the first.  The
        # isotropic matrix with the same first two moments has per-cell variance
        # sum v^2 / sum v at width (sum v)^2 / sum v^2 (``proximity.effective_width``), which is
        # the one-variance edge exactly when the columns are equal.  The column variances are
        # carried onto the screen by its folds: the feature axis orthonormally (column k of the
        # screen holds sum_i v_i Q_ik^2), the ordered axis by its own averaging factor.
        v_in = np.mean(np.abs(dev) ** 2, axis=(0, 1))                   # (Fin,)
        if F != Fin:
            Q = np.asarray(_fold_ortho(np, np.eye(Fin), F, axis=1))
            v = (v_in[:, None] * Q ** 2).sum(axis=0)
        else:
            v = v_in
        # A sample's deviation from the weighted mean it is part of has expected square
        # Var * (1 - 2/M + 1/eff) (its own weight enters the mean), so the per-sample variance is
        # the measured one over that factor, and the aggregate's is the per-sample one over eff.
        M_s = int(stack.shape[0])
        v = v / (1.0 - 2.0 / M_s + 1.0 / eff) * fold_variance(T, N) / eff
        sv, sv2 = float(np.sum(v)), float(np.sum(v ** 2))
        if not (sv > 0.0):
            return 0.0
        sigma2, width = sv2 / sv, sv * sv / sv2
        cx = ctx.complex_ or _env.is_complex_obj(ctx.data)
        mu, sig_J = johnstone(N, width, complex_=cx)
        edge = sigma2 * (mu + tw_quantile(ctx.far, complex_=cx) * sig_J)
        return math.sqrt(edge) if ctx.kind == "projection" else edge

    _provider.__name__ = "weighted_effective_null"
    _provider.effective_n = eff
    return _provider


# ── reference-calibrated Gaussian null (deterministic, O(1), analytically sharp) ──

def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF: ``z`` with ``P(Z <= z) = p``, to float precision
    (:func:`_norm_isf`)."""
    return -_norm_isf(p)


def _norm_isf(p: float) -> float:
    """Inverse survival: ``z`` with ``P(Z > z) = p``, to float precision.  Newton's method on the
    log-survival ``log Q(z) = log(erfc(z / sqrt 2) / 2)``, which is concave (the normal law is
    log-concave), so from a start right of the root the steps close on it monotonically; the
    start ``sqrt(-2 log p)`` is right of it because ``Q(z) <= exp(-z^2 / 2) / 2``.  The steps stop
    when one is inside the round-off of ``z``.  No table and no fitted rational."""
    p = float(p)
    if not (p > 0.0):
        return math.inf
    if not (p < 1.0):
        return -math.inf
    if p > 0.5:
        return -_norm_isf(1.0 - p)
    target = math.log(p)
    z = math.sqrt(-2.0 * target)
    eps = float(np.finfo(float).eps)
    while True:
        Q = 0.5 * math.erfc(z / math.sqrt(2.0))
        slope = -math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi) / Q     # d log Q / dz
        z_new = z - (math.log(Q) - target) / slope
        if abs(z_new - z) <= 2.0 * eps * max(1.0, abs(z_new)):
            return z_new
        z = z_new


def _check_shape(shape, ctx: "FloorContext", who: str):
    """Refuse a screen of a different shape from the one a reference was calibrated at: a top
    singular value is an absolute level, and its null moves with the screen's (N, F)."""
    if shape is not None and tuple(int(v) for v in ctx.shape) != tuple(int(v) for v in shape):
        raise ValueError(
            f"{who} was calibrated on {tuple(shape)} screens and is applied to a "
            f"{tuple(int(v) for v in ctx.shape)} one; its floor is a top singular value at the "
            f"calibration shape. Calibrate a reference per screen shape.")


def _reference_planes(reference):
    """``reference`` as a list of 2-D planes, or ``None`` when it is a set of top values."""
    if isinstance(reference, np.ndarray):
        return list(reference) if reference.ndim == 3 else None
    items = list(reference)
    if items and all(np.ndim(x) == 2 for x in items):
        return [np.asarray(x) for x in items]
    return None


def _plane_top(plane, kind: str, width: int) -> float:
    """A signal-free plane read the way the screen it calibrates was read -- whitened and folded to
    ``width`` feature cells at the projection cut point, centred at the correlation ones -- and
    scored by its top spectrum value."""
    from .entropy import normalize                   # deferred: entropy is imported by this module's users
    from .projection import project
    X = np.asarray(plane)
    if kind != "projection":
        Xc = X - np.nanmean(X, axis=0, keepdims=True)
        return top_spectrum_value(np.where(np.isfinite(Xc), Xc, 0.0), kind)
    Z = np.asarray(_env.to_numpy(normalize(X)))
    return top_spectrum_value(np.asarray(_env.to_numpy(project(Z, 1.0, X.shape[1] / width))), kind)


def _rank_of(values, far: float) -> float:
    """The exact rank floor over ``values``: the ``n + 1 - floor(far (n + 1))``-th smallest of the
    ``n`` values (infinite when no rank is rare enough)."""
    v = np.sort(np.asarray(values, dtype=float).ravel())
    n = int(v.size)
    k = n + 1 - math.floor(Fraction(far) * (n + 1))
    return float("inf") if k > n or n == 0 else float(v[k - 1])


def reference_null(reference_top_values, *, far: float | None = None,
                   shape=None) -> Callable[[FloorContext], float]:
    """A null calibrated on a signal-free reference: the floor is the exact rank over the
    reference's top-mode values -- the ``n + 1 - floor(far (n + 1))``-th smallest of its ``n``
    realisations -- so a screen exchangeable with the reference exceeds it with probability at most
    ``far``, whatever the law of the reference's noise.  This is the "prewhiten from a signal-free
    window" null of [E] Def 8.2, the correct floor when you have a quiet window or a vacuum
    ensemble.  With ``n`` realisations it can claim nothing below ``far = 1 / (n + 1)`` (the floor
    is infinite there): resolving a rarer level needs more of them, not a model of their tail.
    The level is taken over the reference's draw as well as the screen's: one fixed reference of
    ``n`` realisations thresholds at a rate that scatters about ``far`` (0 to 0.23 over sets of 39
    at ``far = 0.05``, 0.043 on average), and the scatter narrows as ``n`` grows.
    (A normal model of the top values, ``mean + z(far) std``, under-covers: the top value's law is
    right-skewed, and on 8 x 8 planes from 40 realisations it claimed structure in 10% of reference
    records at ``far = 0.05``.)  ``far=None`` uses ``ctx.far``; a value pins the level.

    The reference is either its signal-free PLANES (a ``(M, T, F)`` array or a sequence of 2-D
    planes), or their top values.  Planes are the form that follows the read: each screen's fold is
    decided on its own record, so two planes of one shape can be read at different widths, and a
    top singular value is only comparable to another at the same width.  Given planes, the provider
    reads every reference plane at the width of the screen it thresholds -- whitened and folded to
    that screen's feature cells, as the screen was -- once per width, and ``shape`` is the ``(T, F)``
    of the PLANES, what the caller controls: a screen of another length is refused.

    Given top values, they were read at one width, and ``shape`` is the ``(N, F)`` of those screens
    (after any fold): a screen of any other shape is refused.  Either way the reference has to be
    read by the same library version as the screens it thresholds, since the screen's units follow
    its whitening."""
    planes = _reference_planes(reference_top_values)
    if planes is None:
        sv = np.asarray(reference_top_values, dtype=float).ravel()

        def _provider(ctx: FloorContext) -> float:
            _check_shape(shape, ctx, "reference_null")
            return _rank_of(sv, ctx.far if far is None else far)
        _provider.__name__ = "reference_null"
        _provider.center = float(sv.mean()); _provider.scale = float(sv.std())
        _provider.shape = shape
        return _provider

    T0, F0 = (int(v) for v in (shape if shape is not None else planes[0].shape))
    if any(tuple(p.shape) != (T0, F0) for p in planes):
        raise ValueError("reference_null: the reference planes must share one shape")
    by_width: dict = {}

    def _provider(ctx: FloorContext) -> float:
        N, width = int(ctx.shape[0]), int(ctx.shape[1])
        if ctx.kind == "projection" and N != T0:
            raise ValueError(
                f"reference_null was calibrated on planes of {T0} rows and is applied to a screen "
                f"of {N}; calibrate a reference per record length.")
        key = (ctx.kind, width)
        if key not in by_width:
            by_width[key] = np.array([_plane_top(p, ctx.kind, width) for p in planes])
        return _rank_of(by_width[key], ctx.far if far is None else far)
    _provider.__name__ = "reference_null"
    _provider.shape = (T0, F0)
    _provider.by_width = by_width
    return _provider


class ReferenceNull:
    """Stateful reference null (see :func:`reference_null`): signal-free reference top-mode values
    arrive by ``push(*values)`` from a separate calibration stream, and the floor is the exact rank
    over the values it holds, so a screen exchangeable with them exceeds it with probability at
    most ``far``, whatever their law.  It has no ``update`` hook, so the streaming aperture does not
    calibrate it on the data it is thresholding.  ``far=None`` uses ``ctx.far``; with ``n`` values
    held it can claim nothing below ``far = 1 / (n + 1)``.

    ``forgetting`` (in (0, 1], default 1.0 = perfect memory) gives the reference a fading memory
    for a drifting noise level, as an aperture sweeps across regions: it holds the most recent
    ``ceil((1 + forgetting) / (1 - forgetting))`` values -- the effective count (Kish) of the
    exponential weights ``forgetting ** age`` it stands for -- and forgets the rest.  Memory is
    that count, or every value pushed at ``forgetting == 1``.  ``shape``: as in
    :func:`reference_null`, the calibration screens' ``(N, F)``; a screen of another shape is
    refused."""

    def __init__(self, reference_top_values=None, *, far: float | None = None,
                 forgetting: float = 1.0, shape=None):
        if not (0.0 < forgetting <= 1.0):
            raise ValueError(f"forgetting must be in (0, 1]; got {forgetting}")
        from collections import deque
        self._far = far
        self.shape = None if shape is None else tuple(int(v) for v in shape)
        self._forget = float(forgetting)
        lam = Fraction(forgetting)                    # exact rationals on the value given
        keep = None if forgetting == 1.0 else math.ceil((1 + lam) / (1 - lam))
        self._values = deque(maxlen=keep)
        if reference_top_values is not None:
            self.push(*np.asarray(reference_top_values, dtype=float).ravel())

    def push(self, *values) -> "ReferenceNull":
        """Add signal-free reference top-mode values; with ``forgetting < 1`` the oldest beyond the
        effective count are forgotten."""
        self._values.extend(float(v) for v in values)
        return self

    @property
    def center(self) -> float:
        return float(np.mean(self._values)) if self._values else 0.0

    @property
    def scale(self) -> float:
        return float(np.std(self._values)) if self._values else 0.0   # population std

    @property
    def n_reference(self) -> int:
        return len(self._values)

    def __call__(self, ctx: FloorContext) -> float:
        _check_shape(self.shape, ctx, "ReferenceNull")
        return _rank_of(np.fromiter(self._values, float), ctx.far if self._far is None else self._far)


def self_calibrating_null(noise, kind: str = "projection", *, block_rows: int,
                          stride: int | None = None, far: float | None = None):
    """A :func:`reference_null` calibrated locally on a region's own signal-free noise -- the self-
    contained, region-dynamic form.  ``noise`` is a signal-free window from the same region as the
    screen being thresholded (e.g. the off-pulse rows of an aperture patch: the burst is localized on
    the ordered axis, so its complement carries the region's real RFI / bandpass with no signal).

    The window is cut into blocks of ``block_rows`` rows (the row count the signal screen has -- for
    the no-fold aperture, equal to its N), each scored by ``top_spectrum_value`` in the floor's own
    units; reference_null is calibrated on those top-mode values: the exact rank over the real
    local noise, no i.i.d. assumption, and region-dynamic (rebuild per sweep position).  The calibration slice is signal-free and separate from the screen, so this does
    not self-contaminate -- the same guarantee ``ReferenceNull`` gives by carrying no ``update`` hook.

    Needs at least ``1/far - 1`` blocks for any rank to be rare enough at ``far``.  The blocks are scored raw -- valid when the
    screen is not folded (the aperture's full-resolution regime, screen == window); fold the noise
    the same way first if the screen folds."""
    X = np.asarray(_env.to_numpy(noise))
    T = int(X.shape[0]); br = int(block_rows); st = int(stride or br)
    tops = [top_spectrum_value(X[i:i + br], kind) for i in range(0, T - br + 1, st)]
    if len(tops) < 2:
        raise ValueError(f"self_calibrating_null needs >= 2 signal-free blocks of {br} rows; the "
                         f"noise window has {T} rows -> {len(tops)}. Widen the window or lower block_rows.")
    return reference_null(np.asarray(tops, dtype=float), far=far)


# ══════════════════════════════════════════════════════════════════════════════
# Per-cut-point selection: a different provider for screen vs spectral vs bulk
# ══════════════════════════════════════════════════════════════════════════════

def by_kind(**providers) -> Callable[[FloorContext], float]:
    """Compose per-cut-point providers into one provider that dispatches on ``ctx.kind``:
    ``by_kind(projection=P, spectral=Q, bulk=R)`` sends the screen floor (``K_signal``), the
    single-screen correlation floor (``resolved_modes``), and the pooled ensemble floor
    (``SpectralAccumulator``) to P, Q, R respectively.  A cut point left unset falls back to
    :func:`default_provider`.  Keys must be in :data:`KINDS`.

    Equivalent to passing the same ``{kind: provider}`` mapping as ``null=`` -- both a
    ``by_kind(...)`` callable and a bare ``dict`` are accepted anywhere a provider is (a
    read, or ``Aperture(null=...)`` where they route the screen and spectral floors apart)."""
    bad = set(providers) - set(KINDS)
    if bad:
        raise ValueError(f"by_kind keys must be in {KINDS}; got unknown {sorted(bad)}")
    def _provider(ctx: FloorContext) -> float:
        return float(providers.get(ctx.kind, default_provider(ctx.kind, ctx.data))(ctx))
    _provider.__name__ = "by_kind"
    _provider.providers = dict(providers)             # resolved per cut point by _select_provider
    return _provider


# ══════════════════════════════════════════════════════════════════════════════
# Apply a provider to one screen (the single call site every floor shares)
# ══════════════════════════════════════════════════════════════════════════════

_EXACT_PERMUTATION = permutation()


def _score_frame(X, kind: str) -> np.ndarray:
    """``X`` in the frame whose Gram the floor reads: the screen itself at the projection cut point,
    unit-norm columns (a correlation frame) at the correlation ones."""
    X = np.asarray(_env.to_numpy(X))
    if kind == "projection" or X.size == 0:
        return X
    e = np.real(np.sum(X.conj() * X, axis=0))
    top = float(e.max()) if e.size else 0.0
    d = np.sqrt(np.clip(e, top * np.finfo(float).eps, None))
    return X / np.where(d == 0, 1.0, d)


def row_influence(X, kind: str) -> float:
    """``delta``: the most any single row of ``X``, as it stands, lifts the top eigenvalue the floor
    thresholds, in that eigenvalue's units -- the squared singular value at the projection cut
    point, the unit-diagonal correlation eigenvalue at the correlation ones.

    The Gram is a sum of one rank-one term per row, ``G = sum_t y_t y_t^H``.  Deleting row ``t``
    leaves ``lambda_1(G - y_t y_t^H) >= v^H (G - y_t y_t^H) v = lambda_1 - |v . y_t|^2`` for the top
    eigenvector ``v`` (Courant-Fischer), so no row carries more of ``lambda_1`` than
    ``max_t |v . y_t|^2`` -- an exact bound, read off one eigensolve of the smaller Gram.  On a
    light tail every row carries about ``lambda_1 / T``; on a heavy right tail one row carries a
    third to a half of it (the coincidence of two channels' extreme cells)."""
    X = np.asarray(_env.to_numpy(X))
    return _row_influence(X, kind, _column_scale(X, kind) if X.ndim == 2 and X.size else None)


def _row_influence(X: np.ndarray, kind: str, d, G=None) -> float:
    """:func:`row_influence` with the column scale ``d`` already read, and the Gram ``X^H X`` when
    the read formed it."""
    if X.ndim != 2 or X.size == 0:
        return 0.0
    if X.shape[0] >= X.shape[1]:
        # the scaled frame's Gram is the record's, scaled: Y = X / d gives Y^H Y = G / (d d^T) and
        # Y v = X (v / d), so the T x F frame is never formed
        G = X.conj().T @ X if G is None else G
        w, V = np.linalg.eigh(G if d is None else G / np.outer(d, d))
        proj = X @ (V[:, -1] if d is None else V[:, -1] / d)
    else:
        Y = _score_frame(X, kind)
        w, U = np.linalg.eigh(Y @ Y.conj().T)
        proj = U[:, -1] * math.sqrt(max(float(w[-1]), 0.0))
    return float(np.max(np.abs(proj) ** 2)) if proj.size else 0.0


def _column_scale(X: np.ndarray, kind: str, G=None):
    """``d``: what :func:`_score_frame` divides each column by -- its 2-norm at the correlation cut
    points, ``None`` at the projection one.  ``G``: the Gram ``X^H X``, whose diagonal is the
    column energies, when the read formed it."""
    if kind == "projection":
        return None
    if G is not None:
        e = np.real(np.diag(G)).copy()
    else:
        e = np.real(np.sum(X.conj() * X, axis=0)) if np.iscomplexobj(X) else np.einsum("ij,ij->j", X, X)
    top = float(e.max()) if e.size else 0.0
    d = np.sqrt(np.clip(e, top * np.finfo(float).eps, None))
    return np.where(d == 0, 1.0, d)


def coincidence_influence(X, kind: str, far: float) -> float:
    """``delta_null(far)``: the largest lift of the top eigenvalue that a coincidence of two
    channels' cells in one row reaches under the null with probability more than ``far``.

    Under the permutation null each channel's values are placed in the rows independently and
    uniformly, so a given cell of channel ``j`` and a given cell of channel ``k`` share a row with
    probability exactly ``1 / T``; when they do, the row adds ``y_tj y_t'k`` to the Gram's
    ``(j, k)`` entry, which lifts the top eigenvalue by up to ``|y_tj| |y_t'k|`` (the top eigenvalue
    of the two-channel off-diagonal block).  The probability that any of the ``N(I)`` cell pairs
    with ``|y_tj| |y_t'k| >= I`` shares a row is at most ``N(I) / T`` (union bound), so a lift above
    the closed-form room happens with probability at most ``far`` exactly when fewer than
    ``k = floor(far T) + 1`` cross-channel cell pairs reach it: the bound is the ``k``-th largest
    cross-channel product of absolute cell values (0 when fewer than ``k`` pairs exist).  It reads
    the potential arrangement, not the one observed, so a heavy tail whose extreme cells happen not
    to share a row in this record is still priced; a light tail's largest products sit inside the
    collective fluctuation the closed form already holds.  Exact, by counting: the ``k``-th largest product is
    the least ``t`` with fewer than ``k`` products above it (:func:`_cross_above`), found by
    bisection over the floats; no product is formed beyond the cells that can exceed ``t``."""
    X = np.asarray(_env.to_numpy(X))
    if X.ndim != 2 or X.size == 0 or X.shape[1] < 2:
        return 0.0
    d = _column_scale(X, kind)
    k = int(math.floor(Fraction(far) * int(X.shape[0]))) + 1
    if _cross_above(X, d, 0.0) < k:
        return 0.0
    m = _largest_cell(X, d)
    top = m * m
    lo, hi = 0, int(np.float64(top).view(np.int64))         # non-negative floats order as their bits
    while lo < hi:                                          # the least t with fewer than k above it
        mid = (lo + hi) // 2
        if _cross_above(X, d, float(np.int64(mid).view(np.float64))) < k:
            hi = mid
        else:
            lo = mid + 1
    return float(np.int64(lo).view(np.float64))


def _pairs_above(v: np.ndarray, t: float) -> int:
    """Ordered pairs ``(a, b)`` of ``v`` (an element with itself included) whose float product
    ``v_a * v_b`` exceeds ``t``: per value, the first partner past ``t / v_a`` -- moved until the
    products themselves agree, so the count is of the products as computed, not of the quotient."""
    if v.size == 0:
        return 0
    u, c = np.unique(v, return_counts=True)                 # ascending, non-negative
    above = np.concatenate([np.cumsum(c[::-1])[::-1], [0]])  # partners at index >= i
    n = u.size
    pos = u > 0
    i = np.full(n, n, dtype=np.intp)
    i[pos] = np.searchsorted(u, t / u[pos], side="right")
    while True:                                             # the quotient is off by an ulp at most
        dn = pos & (i > 0) & (u * u[np.maximum(i - 1, 0)] > t)
        up = pos & (i < n) & (u * u[np.minimum(i, n - 1)] <= t)
        if not (dn.any() or up.any()):
            break
        i = i - dn + up
    return int(np.sum(c * above[i]))


def _largest_cell(X: np.ndarray, d) -> float:
    """The largest ``|x_tj| / d_j`` (``d`` ``None``: unscaled), read without an absolute copy."""
    a = np.abs(X).max(axis=0) if np.iscomplexobj(X) else np.maximum(X.max(axis=0), -X.min(axis=0))
    return float((a if d is None else a / d).max())


def _cross_above(X: np.ndarray, d, t: float) -> int:
    """Pairs of cells of ``|X| / d`` (``(T, F)``; ``d`` per column, ``None`` for none) in different
    channels whose product exceeds ``t``, each pair once.  Only a cell above ``t`` over the largest
    cell can be in one, so only those are scaled and counted."""
    T, F = int(X.shape[0]), int(X.shape[1])
    if t < 0.0:
        return T * T * F * (F - 1) // 2
    # first a bound read in two contiguous passes: no cell of column j exceeds A / d_j for A the
    # largest cell of the whole record, so no cross-channel product exceeds A^2 over the two
    # smallest scales.  Where that is within t the count is 0 without a column-wise pass.
    A = float(np.abs(X).max()) if np.iscomplexobj(X) else max(float(X.max()), -float(X.min()))
    if A == 0.0:
        return 0
    dd = np.ones(2) if d is None else np.partition(np.asarray(d, float), 1)[:2]
    if (A / dd[0]) * (A / dd[1]) <= t:
        return 0
    m = _largest_cell(X, d)
    if m == 0.0 or m * m <= t:                              # no product can exceed t
        return 0
    cut = np.nextafter(np.nextafter(t / m, 0.0), 0.0)       # below every partner a product can need
    lo = (np.full(F, cut) if d is None else cut * d) * (1.0 - 4.0 * np.finfo(float).eps)
    if np.iscomplexobj(X):
        rows, cols = np.nonzero(np.abs(X) > lo)
    else:
        rows, cols = np.nonzero((X > lo) | (X < -lo))
    v = np.abs(X[rows, cols]) if d is None else np.abs(X[rows, cols]) / d[cols]
    within = sum(_pairs_above(v[cols == j], t) for j in np.unique(cols))
    return (_pairs_above(v, t) - within) // 2


def closed_form_margin(ctx: "FloorContext") -> float:
    """How far the closed-form floor at ``ctx.far`` stands above the null edge's centre, in the
    eigenvalue units :func:`row_influence` reads: ``q(far) * sigma^2 * sigma_J`` at the projection
    cut point, ``q(far) * sigma_J / T`` at the correlation ones (``T`` the degrees of freedom plus
    one, as :func:`mp` reads them).  Negative where the level sits below the null's centre."""
    cx = ctx.complex_ or _env.is_complex_obj(ctx.data)
    q = tw_quantile(ctx.far, complex_=cx)
    if ctx.kind == "projection":
        N, F = int(ctx.shape[0]), int(ctx.shape[1])
        xp = _env.ns(ctx.data)
        s2 = noise_sigma2(xp, ctx.data, N, F, complex_=cx, s=ctx.spectrum)
        _, sig_J = johnstone(N, F, complex_=cx)
        return q * s2 * sig_J
    T = int(ctx.shape[0]) if ctx.dof is None else int(ctx.dof) + 1
    _, sig_J = johnstone(T, int(ctx.shape[1]), complex_=cx)
    return q * sig_J / T


def closed_form_holds(ctx: "FloorContext") -> bool:
    """Whether the closed-form floor answers this read at its level.  Two conditions, each read off
    the data:

    * at the screen's cut points (``"projection"``, ``"spectral"``), the level is finer than the
      record resolves, ``far * n < 1`` for ``n`` rows read; at the operator's (``"bulk"``: a window,
      a pool) there is no such condition, below.  At a level
      the record resolves the exact test is the read: its draws, ``ceil(1/far) - 1``, number fewer
      than the record's own rows, and it is the more powerful floor -- the closed form is the edge of
      a sample covariance, and a sample correlation's top eigenvalue fluctuates less at finite size,
      so on light tails the closed form sits above the exact floor (Gaussian 1024 x 64: 1% false
      alarms at 5%, a narrowband line found 0.17 against 0.35).  Below ``1/n`` the exact test would
      draw more surrogates than the record has rows, and the closed form answers where it can:
    * no row lifts a noise eigenvalue from the null edge's centre over the closed-form floor --
      neither a row as it stands (:func:`row_influence`) nor a coincidence of two channels' cells
      that the null makes with probability more than ``far`` (:func:`coincidence_influence`):
      ``closed_form_margin(ctx) >= max(row_influence, coincidence_influence)``.

    The Tracy-Widom law is the fluctuation of a top eigenvalue that many rows make together; its
    quantile at ``far`` is how far that collective fluctuation reaches.  A heavy right tail adds what
    the law does not hold -- single rows where channels' extreme cells meet -- and the closed form
    over-reads exactly when such a row can carry a noise eigenvalue past the floor at the read's
    level.  Where none can, the closed form is the floor; where one can, the floor is the exact
    permutation test, whose level holds whatever the rows are.  The level a record switches at
    (:func:`closed_form_far`) is its own and never above ``1/n``: on a light tail it is ``1/n`` (the
    exact test at every level the record resolves, the closed form at no draws below it), on a heavy
    tail far smaller.

    The operator's window takes the closed form at every level where no row can cross it.  Its read
    is a long record more often than not -- a stream's window, a block handed in whole -- and there
    the exact test's draws are each a pass over every row: 19 of them cost an exact read of a million
    rows thirteen to fifteen times an FFT pipeline's instructions on 16 channels.  The closed form is
    one pass, its level holds wherever the row bounds do, and the exact test still answers wherever
    a row could carry a noise eigenvalue over it.  What it gives up is the exact test's extra power
    on a light tail near the edge, where the closed form sits a little above it.

    The coincidence bound is decided by a count, not a selection: the ``k``-th largest cross-channel
    product is within the room exactly when fewer than ``k`` products exceed it, and only a cell
    above the room over the largest cell can be in such a product (:func:`_cross_above`)."""
    if ctx.data is None:
        return True
    X = np.asarray(_env.to_numpy(ctx.data))
    if ctx.kind != "bulk" and Fraction(ctx.far) * int(X.shape[0]) >= 1:
        return False                       # a screen that resolves the level: the exact test is the read
    room = closed_form_margin(ctx)
    G = ctx.gram if (ctx.gram is not None and X.ndim == 2 and X.shape[0] >= X.shape[1]) else None
    d = _column_scale(X, ctx.kind, G) if X.ndim == 2 and X.size else None
    if not room >= _row_influence(X, ctx.kind, d, G):
        return False
    if X.ndim != 2 or X.size == 0 or X.shape[1] < 2:
        return True
    k = int(math.floor(Fraction(ctx.far) * int(X.shape[0]))) + 1
    return _cross_above(X, d, room) < k                     # the k-th largest product is within the room


def closed_form_far(ctx: "FloorContext") -> float:
    """``far*``: the supremum of the levels at which :func:`closed_form_holds` takes the closed form
    for this read (1 without data).  The coincidence bound is constant on each ``[(k-1)/T, k/T)``,
    and the room grows as the level falls, so on each such interval the closed form holds up to
    ``P(TW > max(delta, delta_k) / scale)``."""
    if ctx.data is None:
        return 1.0
    cx = ctx.complex_ or _env.is_complex_obj(ctx.data)
    unit = closed_form_margin(FloorContext(spectrum=ctx.spectrum, data=ctx.data, shape=ctx.shape,
                                           far=_tw_unit_far(cx), kind=ctx.kind, rng=ctx.rng,
                                           complex_=ctx.complex_, dof=ctx.dof))
    if not (unit > 0.0):
        return 0.0
    d0 = row_influence(ctx.data, ctx.kind)
    T = int(np.asarray(_env.to_numpy(ctx.data)).shape[0])
    best = 0.0
    for k in range(1, T + 1):
        lo, hi = (k - 1) / T, k / T
        sf = float(tw_sf(max(d0, coincidence_influence(ctx.data, ctx.kind, lo)) / unit, complex_=cx))
        if sf < lo:
            break                                     # the room cannot clear this interval or any above
        best = max(best, min(sf, hi))
    return min(best, 1.0 / max(T, 1))               # and below the record's own resolution


def _tw_unit_far(cx: bool) -> float:
    """The level whose Tracy-Widom quantile is 1, so a margin at it is the margin per unit quantile."""
    return float(tw_sf(1.0, complex_=cx))


def switched(ctx: "FloorContext") -> float:
    """The library's floor where the read holds its samples: the closed form where it holds at the
    read's level (:func:`closed_form_holds`), the exact permutation test where it does not."""
    return float(mp(ctx) if closed_form_holds(ctx) else _EXACT_PERMUTATION(ctx))


switched.exact_permutation = (None, None)     # the exact branch's operating point (the read's own)
switched.closed_form_holds = closed_form_holds


def default_provider(kind: str, data=None):
    """The floor used when the caller names none.  Wherever the read holds its samples -- the
    screen, the centred samples of a correlation read, a pool's held rows -- it is :func:`switched`:
    the exact :func:`permutation` test, whose level is ``far`` whatever the noise's law, unless no
    single row of the data can carry a noise eigenvalue over the closed-form floor at that level,
    where the closed form (:func:`mp`) answers at no draws.  The closed form alone over-reads heavy
    right tails (on lognormal noise it claimed structure in 36% of records at ``far = 0.05``).
    Without samples the default is :func:`mp`: a covariance carries no rows to shuffle or weigh."""
    return switched if data is not None else DEFAULT


def _held_shuffle(X, miss):
    """A time shuffle of each column of ``X`` (``(T, F)``) among its own measured cells, the cells
    ``miss`` marks staying where they are: the null draw of a record with gaps, where the gaps are
    fixed structure and not part of the null.  Returns ``draw(rng) -> (T, F)``."""
    X = np.asarray(_env.to_numpy(X))
    XT = np.ascontiguousarray(X.T)
    missT = np.ascontiguousarray(np.asarray(miss, bool).T)
    at = np.argsort(missT, axis=1, kind="stable")            # measured cells first, in time order

    def draw(rng):
        key = np.where(missT, 2.0, rng.random(missT.shape))   # the same cells, in random order
        g = np.empty(missT.shape, dtype=np.intp)
        np.put_along_axis(g, at, np.argsort(key, axis=1), axis=1)
        return np.take_along_axis(XT, g, axis=1).T
    return draw


def _select_provider(null, kind: str, data=None):
    """Resolve ``null`` to the provider for this cut point: ``None`` -> :func:`default_provider`;
    a ``{kind: provider}`` mapping -> its entry for ``kind`` (missing -> default); otherwise
    the callable itself.  So one ``null=`` can carry a distinct provider per cut point."""
    if null is None:
        return default_provider(kind, data)
    if not isinstance(null, Mapping) and isinstance(getattr(null, "providers", None), Mapping):
        null = null.providers                         # a by_kind(...) composition: its entries
    provider = (null.get(kind) or default_provider(kind, data)) if isinstance(null, Mapping) else null
    if not callable(provider):
        raise TypeError(
            "null must be a null-provider callback FloorContext -> float, a {kind: provider} "
            f"mapping, or None for the library default; got {provider!r}")
    return provider


def apply_floor(null=None, *, spectrum, data, shape, far: float, kind: str, seed: int = 0,
                complex_: bool = False, resample=None, dof: int | None = None) -> float:
    """Evaluate the null provider for cut point ``kind`` on one screen and return its scalar
    floor.  ``null`` is a provider callback (``FloorContext -> float``), a ``{kind: provider}``
    mapping (a different provider per cut point), or ``None`` for the library default
    (:func:`default_provider`).  A fresh generator seeded by ``seed`` is placed on the context
    so any resampling provider is deterministic per ``seed`` and per (local) screen.  ``complex_``
    names a complex ensemble when only its covariance is passed (``data=None``); with ``data`` the
    ensemble is read off its dtype.  ``resample`` is the read's own surrogate draw (see
    :class:`FloorContext`), used by the time shuffle in place of shuffling ``data``.  ``dof``: a
    correlation kind's degrees of freedom when they are not the rows less one."""
    provider = _select_provider(null, kind, data)
    ctx = FloorContext(spectrum=spectrum, data=data, shape=tuple(shape),
                       far=float(far), kind=kind, rng=np.random.default_rng(seed),
                       complex_=bool(complex_), resample=resample,
                       dof=None if dof is None else int(dof))
    return float(provider(ctx))


def _scored_stack(X: np.ndarray, kind: str) -> np.ndarray:
    """:func:`_scored` of every surrogate in a stack ``X`` ``(B, T, F)`` of one shape and memory
    order, bit for bit: the same reductions, products and eigensolves, taken over the stack in one
    call each."""
    X = np.asarray(X)
    B, T, F = (int(v) for v in X.shape)
    e = np.real(np.sum(X.conj() * X, axis=1))                       # (B, F) column energies
    m, p = max(T, F), min(T, F)
    eps = float(np.finfo(X.dtype).eps) if np.issubdtype(X.dtype, np.inexact) else float(np.finfo(float).eps)
    if kind != "projection":
        top = e.max(axis=1, keepdims=True)
        d = np.sqrt(np.clip(e, top * np.finfo(float).eps, None))
        d = np.where(d == 0, 1.0, d)
        Y = X / d[:, None, :]
        Yh = Y.conj().transpose(0, 2, 1)
        v = np.maximum(np.linalg.eigvalsh(Yh @ Y if T >= F else Y @ Yh)[:, -1], 0.0)
        return v + (m + p) * eps * np.sum((e > 0).astype(float), axis=1)
    Xh = X.conj().transpose(0, 2, 1)
    G = Xh @ X if T >= F else X @ Xh
    v = np.sqrt(np.maximum(np.linalg.eigvalsh(G)[:, -1], 0.0))
    out = np.empty(B)
    cx = bool(np.iscomplexobj(X))
    for b in range(B):                     # the scale and its round-off: scalars per surrogate
        Fb = max(int(np.count_nonzero(e[b])), 1)
        esum = float(np.sum(e[b]))
        s2 = noise_sigma2_from_spectrum(np.array([esum]), T, Fb)
        vb = float(v[b])
        if not (s2 > 0.0):
            out[b] = -math.inf
            continue
        mu, sig_J = johnstone(T, Fb, complex_=cx)
        dlam = (m + p) * eps * esum
        ds = min(dlam / vb, math.sqrt(dlam)) if vb > 0 else math.sqrt(dlam)
        out[b] = (vb * vb / s2 - mu) / sig_J + ((vb + ds) ** 2 - vb * vb) / (s2 * sig_J)
    return out


def _layout(a: np.ndarray) -> str:
    """The memory order a surrogate was drawn in: ``"C"``, ``"F"``, or ``""`` for neither."""
    return "C" if a.flags.c_contiguous else ("F" if a.flags.f_contiguous else "")


def _stacked(arrays, layout: str):
    """Stack same-shape surrogates of one memory order in that order, so every reduction and
    product over the stack runs as it runs on one."""
    if layout == "F":
        return np.stack([a.T for a in arrays]).transpose(0, 2, 1)
    return np.stack(arrays)


def _draws_of(ctx: FloorContext):
    """The exact permutation test's surrogate draw for ``ctx``, as :func:`floor_from_null_sampler`
    takes it: the read's own resample where it offers one, else each live channel shuffled in time."""
    if ctx.resample is not None:
        return ctx.resample
    X = np.asarray(_env.to_numpy(ctx.data))
    Xs = X[:, np.any(X != 0, axis=0)]
    return lambda rng: shuffle_in_time(Xs, rng)


def apply_floors(null=None, *, items, far: float, kind: str, seed: int = 0) -> np.ndarray:
    """:func:`apply_floor` over many screens at one level, cut point and seed, bit for bit: ``items``
    is a sequence of ``dict`` holding each screen's ``spectrum``, ``data``, ``shape`` and, where it
    has them, ``complex_``, ``resample`` and ``dof``.  Returns ``(len(items),)``.

    Every screen the exact permutation test answers draws its surrogates exactly as it would alone,
    from its own generator seeded by ``seed``; the screens' ``j``-th draws are then scored together,
    one stacked eigensolve per draw for the screens that share a shape and memory order.  The calls
    around the eigensolves are paid once per draw, not once per screen; the eigensolves and the
    shuffles themselves are not reduced.  It holds one draw per screen at a time, as much again as
    the screens it reads.  Any other provider is applied per screen."""
    out = np.empty(len(items))
    exact = []
    for i, it in enumerate(items):
        data = it.get("data")
        provider = _select_provider(null, kind, data)
        ctx = FloorContext(spectrum=it.get("spectrum"), data=data, shape=tuple(it["shape"]),
                           far=float(far), kind=kind, rng=np.random.default_rng(seed),
                           complex_=bool(it.get("complex_", False)), resample=it.get("resample"),
                           dof=None if it.get("dof") is None else int(it["dof"]), gram=it.get("gram"))
        if data is not None and (provider is _EXACT_PERMUTATION
                                 or (provider is switched and not closed_form_holds(ctx))):
            exact.append((i, ctx))
        elif provider is switched:
            out[i] = float(mp(ctx))
        else:
            out[i] = float(provider(ctx))
    if len(exact) < 2:                     # nothing to score together: the provider itself
        for i, ctx in exact:
            out[i] = float(_EXACT_PERMUTATION(ctx))
        return out
    f = float(far)
    n = fewest_draws(f)
    k = n + 1 - math.floor(Fraction(f) * (n + 1))
    scores = np.full((len(exact), n), math.inf)
    if k <= n:
        draws = [_draws_of(ctx) for _, ctx in exact]
        for j in range(n):
            S = [np.asarray(draw(ctx.rng)) for draw, (_, ctx) in zip(draws, exact)]
            groups: dict = {}
            for a, X in enumerate(S):
                groups.setdefault((X.shape, X.dtype.str, _layout(X)), []).append(a)
            for (shape, _, layout), idx in groups.items():
                if layout and len(idx) > 1 and 0 not in shape:
                    scores[idx, j] = _scored_stack(_stacked([S[a] for a in idx], layout), kind)
                else:                              # alone, strided or empty: scored as drawn
                    scores[idx, j] = [_scored(S[a], kind) for a in idx]
    for a, (i, ctx) in enumerate(exact):
        score = float("inf") if k > n else float(np.partition(scores[a], k - 1)[k - 1])
        out[i] = _observed_floor(score, np.asarray(_env.to_numpy(ctx.data)), kind)
    return out
