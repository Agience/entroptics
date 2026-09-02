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

  1. ships a limited set of derived (closed-form) default providers here -- ``mp`` (default)
     and ``robust`` -- each a pure function of the local screen, nothing fitted; and
  2. offers the plumbing to build a sampled null (``top_spectrum_value``,
     ``shuffle_in_time``, ``floor_from_null_sampler``).

The library does not calibrate the caller's null for them.  A confined-vacuum reference,
a phase-randomised surrogate, a physics null -- the caller writes it with the plumbing
(or from scratch) and passes it in; the library never needs to know what the null is.

The library ships four null methods; ``mp`` is the default (self-contained, no knob), and the
caller plugs any method -- or its own callback -- via ``null=``:

  (1) analytic edge -- i.i.d. noise, no reference (the default):
      mp                                 finite-size Johnstone / Tracy-Widom edge; the noise
                                         level is estimated from the data, so nothing is supplied.
  (2) robust fence -- heavy-tailed spectrum, no reference:
      robust                             Tukey upper fence ``Q3 + 1.5*(Q3 - Q1)`` of the spectrum
                                         (a heuristic outlier fence, not a calibrated null).
  (3) empirical reference -- calibrate on a signal-free window (closed-form Gaussian):
      reference_null(top_values)         O(1) floor ``center + z(far)*scale`` from a quiet
                                         window; sharpens analytically to any far (no draws).
      ReferenceNull(..., forgetting=)    the stateful (Welford) form; ``forgetting<1`` fades old
                                         samples (~1/(1-forgetting) effective) to track drift.
      self_calibrating_null(noise, ...)  a ``reference_null`` calibrated locally on a region's
                                         own off-pulse noise -- self-contained, region-dynamic.
  (4) sampled / distribution-free -- no model, resample the data:
      permutation()                      the (1-far) quantile of a per-channel time-shuffle
                                         surrogate -- the correct floor for non-Gaussian data.
      floor_from_null_sampler(surrogate) turn any surrogate into a provider (block bootstrap,
                                         phase randomisation, a physics null, ...).
      shuffle_in_time, top_spectrum_value   the example surrogate + the scoring building block.

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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import environment as _env

# The distinct cut points a provider can be keyed to (``ctx.kind``).  Each is a separate
# noise-vs-signal decision, so each can take its own provider (see ``by_kind`` / a mapping):
#   "projection" -- the screen singular-value floor (K_signal)        [projection.noise_floor]
#   "spectral"   -- the single-screen correlation-eigenvalue floor     [reads.spectral_optics]
#   "bulk"       -- the pooled ensemble correlation floor              [reads.SpectralAccumulator]
# A new cut point adds a new kind here and a new key.
KINDS = ("projection", "spectral", "bulk")


# ══════════════════════════════════════════════════════════════════════════════
# Tracy-Widom_1 (GOE / real matrices): universal edge quantiles + survival function
# ══════════════════════════════════════════════════════════════════════════════

# TW1 upper quantiles: P(TW1 <= q) = 1 - far.  Universal constants of the TW1
# distribution (Bejan 2005; Chiani 2014), fixed once for a target false-alarm rate.
# This is the only number in the "mp" provider, and it is derived, not calibrated.
_TW1_UPPER_Q: dict[float, float] = {0.10: 0.4501, 0.05: 0.9793, 0.025: 1.3675, 0.01: 2.0234}


def tw1_quantile(far: float) -> float:
    """The TW1 upper quantile ``q`` with ``P(TW1 <= q) = 1 - far``.  Exact tabulated values
    for the standard levels; for any other ``far`` it is obtained by inverting the Chiani
    survival function ``tw1_sf`` -- so a caller sharpening ``far`` toward 1e-5 and beyond
    (a 99.999% threshold) still gets a derived edge, no table lookup required."""
    if far in _TW1_UPPER_Q:
        return _TW1_UPPER_Q[far]
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    return _tw1_quantile_invert(far)


# TW1 survival function via Chiani (2014)'s Gamma approximation, moment-matched to
# TW1's mean/variance/skewness (max CDF error ~7e-3).  The TW1 CDF has no closed form;
# this lets the per-mode significance read report p_k = P(TW1 > g_k) with no scipy
# dependency and deterministically.  Reports evidence, never sets a floor.
_TW1_G_K = 46.44580     # Gamma shape    = 4 / skew^2
_TW1_G_TH = 0.1861300   # Gamma scale    = sqrt(var) * skew / 2
_TW1_G_LOC = -9.848007  # Gamma location = mean - shape * scale
_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def _gammaln(x: float) -> float:
    """log Gamma(x) for x > 0 via the Lanczos approximation (g = 7)."""
    x -= 1.0
    a = _LANCZOS[0]
    t = x + 7.5
    for i in range(1, 9):
        a += _LANCZOS[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _reg_gamma_upper(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = Gamma(a, x) / Gamma(a), x >= 0
    (Numerical Recipes: series for x < a+1, continued fraction otherwise)."""
    if x <= 0.0:
        return 1.0
    gln = _gammaln(a)
    if x < a + 1.0:
        ap = a; s = 1.0 / a; d = s
        for _ in range(400):
            ap += 1.0; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    tiny = 1e-300
    b = x + 1.0 - a; c = 1.0 / tiny; d = 1.0 / b; h = d
    for i in range(1, 400):
        an = -i * (i - a); b += 2.0
        d = an * d + b
        if abs(d) < tiny: d = tiny
        c = b + an / c
        if abs(c) < tiny: c = tiny
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def tw1_sf(g: float) -> float:
    """P(TW1 > g): the Tracy-Widom_1 upper-tail probability (Chiani Gamma approximation)."""
    return _reg_gamma_upper(_TW1_G_K, (g - _TW1_G_LOC) / _TW1_G_TH)


def _tw1_quantile_invert(far: float) -> float:
    """Invert the (monotone-decreasing) survival function: the ``q`` with ``tw1_sf(q) = far``,
    by bisection.  Lets the derived edge serve an arbitrary, arbitrarily-sharp ``far``."""
    lo, hi = -10.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tw1_sf(mid) > far:          # tail too heavy -> need a larger quantile
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ══════════════════════════════════════════════════════════════════════════════
# Finite-size Johnstone edge + de-biased per-cell noise variance (shared primitives)
# ══════════════════════════════════════════════════════════════════════════════

def johnstone(N: int, F: int) -> tuple[float, float]:
    """Johnstone (2001) centering ``mu`` and scaling ``sigma`` for the largest
    eigenvalue (top singular value squared) of an N x F Gaussian matrix, so that
    ``(lambda_max - mu)/sigma -> Tracy-Widom_1``.  The derived finite-size edge (no fitted
    coefficient).  Called with (N, F) for the screen, (T, N) for the correlation floor."""
    nn = math.sqrt(max(N - 1, 1)); ff = math.sqrt(max(F, 1))
    a = nn + ff
    return a * a, a * (1.0 / nn + 1.0 / ff) ** (1.0 / 3.0)


def debias_denominator(N: int, F: float) -> float:
    """The de-biasing denominator ``F * c_F * dof`` that turns a median row energy into the
    per-cell noise variance ``sigma^2``.  Split out so every screen-floor call site (per-frame
    ``noise_sigma2`` / ``mp``, the numpy batch ``projection._mp_floor_batch``, and the batched
    resolved read) shares one definition and cannot drift:

      c_F        -- the sample median of ||row||^2 estimates the distribution median of a
                    chi^2_F (= F*c_F), not the mean F (Wilson-Hilferty, a small-F bias);
      (N-1)/N    -- the per-channel median centring deflates the row energy; the mean-
                    centring dof is applied as a conservative correction (slightly over-
                    corrects at small N, raising the floor in the safe direction).

    ``F`` is real-valued, not integer: ``proximity`` reads an EFFECTIVE width whose whole claim
    is that it has no discrete steps, and truncating it here would put one back.  Bit-identical
    to the integer form at every integer width, so no caller moves."""
    Ff = float(F)
    c_F = (1.0 - 2.0 / (9.0 * max(Ff, 1e-30))) ** 3
    dof = max(int(N) - 1, 1) / int(N)
    return Ff * c_F * dof


def screen_floor_sq(sigma2, N: int, F: int, far: float):
    """The screen noise floor squared (in variance / eigenvalue units): ``sigma^2 * (mu +
    q*sigma_J)`` with the finite-size Johnstone centring/scaling and the TW1 quantile at ``far``.
    ``sigma2`` may be a scalar (one screen) or an array (per-frame over a batch); the return has
    its shape.  Take ``sqrt`` for the singular-value floor.  One definition shared by the
    per-frame ``mp`` provider, the numpy batch floor, and the batched resolved read."""
    mu, sig_J = johnstone(int(N), int(F))
    q = tw1_quantile(far)
    return sigma2 * (mu + q * sig_J)


def noise_sigma2(xp, screen, N: int, F: int) -> float:
    """The de-biased robust per-cell noise variance the ``mp`` / ``bulk`` providers build
    on: the median row energy over F divided by :func:`debias_denominator` (the chi^2 median
    ``c_F`` and the centring dof ``(N-1)/N``).  Shared with the per-mode significance so they
    agree."""
    return float(_env.median1d(xp, _env.sum_ax(xp, xp.abs(screen) ** 2, 1))) / debias_denominator(N, F) + 1e-30


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
      "bulk"       -- as "spectral" but the pooled ensemble floor (``SpectralAccumulator``):
                    same correlation units, a separate key so it takes its own provider.
    ``data`` may be ``None`` when only a pooled covariance is available (``bulk`` via a
    ``SpectralAccumulator``): then only the closed-form providers (``mp`` / ``robust`` /
    a ``reference_null``) apply, and a resampling provider raises."""
    spectrum: np.ndarray | None            # descending singular values (screen) or corr eigenvalues
    data:     np.ndarray | None            # the matrix behind the spectrum, or None (covariance-only)
    shape:    tuple                        # (N, F) screen; (T, N) correlation ("spectral"/"bulk")
    far:      float                        # false-alarm level -- the provider owns the cutoff (below)
    kind:     str                          # one of KINDS: "projection" | "spectral" | "bulk"
    rng:      "np.random.Generator"        # seeded generator for any resampling (determinism)

    # ``far`` is the false-alarm level (alpha) delivered with the null, because the
    # noise-vs-signal cutoff is one decision, not two: the provider owns it.  ``ctx.far``
    # is the read's target; a provider may honour it, pin its own, or -- being stateful and
    # updated per frame in the dynamical -- sharpen it as its long-term local/global sample
    # grows (toward 99.999%+), even predictively as the noise level drifts.


# NullProvider = Callable[[FloorContext], float]   (optionally a stateful object with
# __call__(ctx)->float and update(frame); the streaming aperture calls update per frame).


# ══════════════════════════════════════════════════════════════════════════════
# The derived default providers (closed-form; pure functions of the local screen)
# ══════════════════════════════════════════════════════════════════════════════

def mp(ctx: FloorContext) -> float:
    """The default provider: the finite-size Johnstone / Tracy-Widom edge.  Projection:
    ``sqrt(sigma^2 * (mu + q*sigma_J))`` with the de-biased per-cell variance.  Correlation
    floor: ``(mu + q*sigma_J)/T`` in correlation units.  Parameter-free; only ``far``."""
    q = tw1_quantile(ctx.far)
    if ctx.kind == "projection":
        N, F = int(ctx.shape[0]), int(ctx.shape[1])
        xp = _env.ns(ctx.data)
        return math.sqrt(screen_floor_sq(noise_sigma2(xp, ctx.data, N, F), N, F, ctx.far))
    T, N = int(ctx.shape[0]), int(ctx.shape[1])
    mu, sig_J = johnstone(T, N)
    return (mu + q * sig_J) / T


def robust(ctx: FloorContext) -> float:
    """The deterministic Tukey upper fence ``Q3 + 1.5*(Q3 - Q1)`` of the spectrum -- a
    heuristic outlier fence for heavy-tailed spectra (not a calibrated null)."""
    base = ctx.spectrum if ctx.spectrum is not None else ctx.data
    xp = _env.ns(base)
    sv = ctx.spectrum if ctx.spectrum is not None else xp.linalg.svd(ctx.data, compute_uv=False)
    q1 = float(xp.quantile(sv, 0.25)); q3 = float(xp.quantile(sv, 0.75))
    return q3 + 1.5 * (q3 - q1)


DEFAULT: Callable[[FloorContext], float] = mp   # the library's derived default null provider


# ══════════════════════════════════════════════════════════════════════════════
# Build-your-own-null plumbing (the caller supplies the null; the library the quantile)
# ══════════════════════════════════════════════════════════════════════════════

def top_spectrum_value(X: np.ndarray, kind: str) -> float:
    """Score a surrogate ``X`` in the same units the floor thresholds: the top singular
    value (``kind="projection"``) or the top unit-diagonal correlation eigenvalue (any other
    kind -- ``"spectral"`` / ``"bulk"``).  The building block for a sampled null provider.
    Numpy (the occasional calibrated read).

    The correlation branch is read off the column-scaled frame, not an ``FxF`` matrix.  Scaling
    column ``j`` by ``1/d_j`` gives ``Y`` with ``Y^H Y == R`` exactly, so ``R``'s top eigenvalue is
    ``sigma_max(Y)**2`` -- and ``sigma_max`` comes off the ``TxF`` frame directly, without ever
    forming the ``FxF`` covariance to eigendecompose for one eigenvalue.  Agrees with the direct
    eigendecomposition to round-off (not a different quantity), and needs no shape assumption --
    checked at ``T<F`` and ``T>F`` (64x256, 300x256, 64x16, 400x64) -- because it never forms the
    Gram either way.

    This stays pure numpy: ``scipy.linalg.eigh(subset_by_index=...)`` also returns just the top
    eigenvalue, bit-identically, but scipy is an optional extra here (``pyproject``: core is
    ``numpy>=2.0`` alone), and this is a hot path -- a core read should not take on an optional
    dependency for it.
    """
    X = np.asarray(X)
    if kind not in KINDS:
        # Named, not defaulted: the two branches score DIFFERENT quantities, so a kind that is
        # not a cut point must raise, never fall through to the correlation branch and
        # calibrate a reference on the wrong statistic.  ("screen" was this kind before 0.2.1.)
        raise ValueError(f"top_spectrum_value: unknown kind {kind!r}; expected one of {KINDS}. "
                         f"The cut point formerly called 'screen' is now 'projection'.")
    if kind == "projection":
        return float(np.linalg.svd(X, compute_uv=False)[0])
    # d_j is column j's 2-norm -- i.e. sqrt(Cov_jj) -- so the covariance never has to be formed.
    d = np.sqrt(np.clip(np.real(np.sum(X.conj() * X, axis=0)), 1e-30, None))
    return float(np.linalg.svd(X / d, compute_uv=False)[0] ** 2)


def shuffle_in_time(X: np.ndarray, rng) -> np.ndarray:
    """An example surrogate: shuffle each column independently along the ordered axis
    (rows), destroying ordered and cross-channel structure while preserving every channel's
    own marginal.  The surrogate behind ``permutation``.

    One ``argsort`` draws all ``F`` independent permutations at once: independent uniforms per
    (row, column), and an ``argsort`` down each column is that column's random permutation --
    i.i.d. uniforms sorted give a uniform random permutation, so this is exact, not an
    approximation.  Column marginals are exactly preserved, since a permutation moves values but
    never alters them.

    It consumes the generator differently from a per-channel loop, so a fixed seed does not
    reproduce a per-channel loop's draw.  That is a difference of sample, not of distribution: a
    sampled null is an estimate, and two valid estimators of the same null disagree at
    O(sd/sqrt(draws)).  A floor is never pinned to a literal value across such a change --
    ``draws`` is what bounds its resolution (see ``floor_from_null_sampler``).
    """
    X = np.asarray(X)
    n, F = int(X.shape[0]), int(X.shape[1])
    # Independent uniforms per (row, column); argsort down each column is that column's permutation.
    idx = np.argsort(rng.random((n, F)), axis=0)
    return np.take_along_axis(X, idx, axis=0)


def floor_from_null_sampler(surrogate: Callable[[np.ndarray, "np.random.Generator"], np.ndarray],
                            *, draws: int = 200, far: float | None = None,
                            ) -> Callable[[FloorContext], float]:
    """Turn any surrogate into a null provider: ``floor = (1 - far)`` quantile of the top
    spectrum value over ``draws`` draws of ``surrogate(data, rng)``.  The library owns the
    quantile; the caller owns the null mechanism -- shuffle, block bootstrap, phase
    randomisation, a draw from a signal-free reference, a physics surrogate.  The returned
    provider is a local, per-screen callback like any other.

    ``far`` couples the false-alarm level into the provider: ``None`` uses the caller's
    ``ctx.far`` (the read's target), a value pins the provider's own level -- and since it
    is empirical, ``draws`` bounds how sharp it can be (resolving a level ``far`` needs
    ``draws >> 1/far``).  A stateful provider that accumulates surrogates over a run can
    therefore sharpen its level as its long-term sample grows."""
    def _provider(ctx: FloorContext) -> float:
        if ctx.data is None:
            raise ValueError("a sampled null provider needs the raw samples; not available "
                             "from a covariance-only accumulator")
        f = ctx.far if far is None else far
        X = np.asarray(_env.to_numpy(ctx.data))
        tops = np.empty(int(draws))
        for b in range(int(draws)):
            tops[b] = top_spectrum_value(surrogate(X, ctx.rng), ctx.kind)
        return float(np.quantile(tops, 1.0 - f))
    _provider.__name__ = f"{getattr(surrogate, '__name__', 'sampled')}_null"
    return _provider


def permutation(*, draws: int = 200, far: float | None = None) -> Callable[[FloorContext], float]:
    """One built-in example of a caller-style sampled null:
    ``floor_from_null_sampler(shuffle_in_time, draws=draws, far=far)`` -- the distribution-
    free permutation floor (destroys cross-channel structure, keeps each marginal).
    Deterministic per the seed carried in the ``FloorContext``.  ``far=None`` uses the
    read's ``ctx.far``; a value pins the provider's own level.  Use ``null=permutation()``
    (or write your own the same way)."""
    return floor_from_null_sampler(shuffle_in_time, draws=draws, far=far)


# ── reference-calibrated Gaussian null (deterministic, O(1), analytically sharp) ──

def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam rational approximation): O(1), deterministic,
    |abs error| < 1.2e-9.  Returns z with P(Z <= z) = p."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return ((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                / (((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)))
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -((((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5])
                 / (((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)))
    q = p - 0.5; r = q * q
    return ((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q
            / ((((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)))


def _norm_isf(p: float) -> float:
    """Inverse survival: ``z`` with ``P(Z > z) = p`` (= ``-_norm_ppf(p)``)."""
    return -_norm_ppf(p)


def reference_null(reference_top_values, *, far: float | None = None) -> Callable[[FloorContext], float]:
    """A deterministic O(1) null calibrated on a signal-free reference: the floor is
    ``center + z(far)*scale`` with ``center, scale`` the mean and std of the reference's
    top-mode values (``top_spectrum_value`` of each signal-free realisation) and ``z(far)``
    the standard-normal inverse survival function.  This is the "prewhiten from a signal-free
    window" null of [E] Def 8.2 in closed form: no stored samples, no resampling, and it
    sharpens analytically to any ``far`` (the normal quantile inverts to 1e-5 and beyond --
    no ``draws >> 1/far`` requirement, unlike a sampled ``permutation`` floor).  It generalises
    ``mp``: the same closed-form edge, its noise model calibrated on the caller's reference
    instead of an i.i.d.-Gaussian bulk -- the correct floor when you have a quiet window / vacuum
    ensemble.  ``far=None`` uses ``ctx.far``; a value pins the level."""
    sv = np.asarray(reference_top_values, dtype=float).ravel()
    center = float(sv.mean()); scale = float(sv.std() + 1e-30)
    def _provider(ctx: FloorContext) -> float:
        return center + _norm_isf(ctx.far if far is None else far) * scale
    _provider.__name__ = "reference_null"
    _provider.center = center; _provider.scale = scale
    return _provider


class ReferenceNull:
    """Stateful, O(1) sharpening reference null (see :func:`reference_null`): maintains the
    running mean/variance of a signal-free reference's top-mode values by Welford's method
    (O(1) per sample, O(1) memory) and floors at ``mean + z(far)*std``, sharpening analytically
    to any ``far``.  Feed reference top-values with ``push(*values)`` from a separate
    calibration stream, not the signal: it has no ``update`` hook, so the streaming aperture
    does not calibrate it on the data it is thresholding.  ``far=None`` uses ``ctx.far``.

    ``forgetting`` (in (0, 1], default 1.0 = perfect memory) gives the reference a fading memory:
    each push decays the accumulated weight by ``forgetting`` first, so the null tracks the local,
    drifting noise as an aperture sweeps across regions (the effective sample is ~1/(1-forgetting)
    recent values).  ``forgetting < 1`` is the region-dynamic mode -- calibrate off nearby signal-
    free (low-coherence) patches and the far ones fade out."""

    def __init__(self, reference_top_values=None, *, far: float | None = None,
                 forgetting: float = 1.0):
        if not (0.0 < forgetting <= 1.0):
            raise ValueError(f"forgetting must be in (0, 1]; got {forgetting}")
        self._far = far
        self._forget = float(forgetting)
        self._n = 0.0                       # float: effective (possibly faded) sample weight
        self._mean = 0.0
        self._M2 = 0.0
        if reference_top_values is not None:
            self.push(*np.asarray(reference_top_values, dtype=float).ravel())

    def push(self, *values) -> "ReferenceNull":
        """Add signal-free reference top-mode values (fading-memory Welford, O(1) each).  With
        ``forgetting == 1`` this is exact Welford; with ``forgetting < 1`` older values decay."""
        f = self._forget
        for v in values:
            x = float(v)
            self._n = f * self._n + 1.0
            d = x - self._mean
            self._mean += d / self._n
            self._M2 = f * self._M2 + d * (x - self._mean)
        return self

    @property
    def center(self) -> float:
        return self._mean

    @property
    def scale(self) -> float:
        return math.sqrt(self._M2 / max(self._n, 1.0)) + 1e-30    # population std (matches reference_null)

    @property
    def n_reference(self) -> float:
        return self._n

    def __call__(self, ctx: FloorContext) -> float:
        return self.center + _norm_isf(ctx.far if self._far is None else self._far) * self.scale


def self_calibrating_null(noise, kind: str = "projection", *, block_rows: int,
                          stride: int | None = None, far: float | None = None):
    """A :func:`reference_null` calibrated locally on a region's own signal-free noise -- the self-
    contained, region-dynamic form.  ``noise`` is a signal-free window from the same region as the
    screen being thresholded (e.g. the off-pulse rows of an aperture patch: the burst is localized on
    the ordered axis, so its complement carries the region's real RFI / bandpass with no signal).

    The window is cut into blocks of ``block_rows`` rows (the row count the signal screen has -- for
    the no-fold aperture, equal to its N), each scored by ``top_spectrum_value`` in the floor's own
    units; reference_null is calibrated on those top-mode values.  Machine-precision (closed-form
    Gaussian from the real local noise, no i.i.d. assumption) and region-dynamic (rebuild per sweep
    position).  The calibration slice is signal-free and separate from the screen, so this does
    not self-contaminate -- the same guarantee ``ReferenceNull`` gives by carrying no ``update`` hook.

    Needs >= 2 blocks (>= ~6 for a stable scale).  The blocks are scored raw -- valid when the
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
    the derived default (``mp``).  Keys must be in :data:`KINDS`.

    Equivalent to passing the same ``{kind: provider}`` mapping as ``null=`` -- both a
    ``by_kind(...)`` callable and a bare ``dict`` are accepted anywhere a provider is (a
    read, or ``Aperture(null=...)`` where they route the screen and spectral floors apart)."""
    bad = set(providers) - set(KINDS)
    if bad:
        raise ValueError(f"by_kind keys must be in {KINDS}; got unknown {sorted(bad)}")
    def _provider(ctx: FloorContext) -> float:
        return float(providers.get(ctx.kind, DEFAULT)(ctx))
    _provider.__name__ = "by_kind"
    return _provider


# ══════════════════════════════════════════════════════════════════════════════
# Apply a provider to one screen (the single call site every floor shares)
# ══════════════════════════════════════════════════════════════════════════════

def _select_provider(null, kind: str):
    """Resolve ``null`` to the provider for this cut point: ``None`` -> derived default; a
    ``{kind: provider}`` mapping -> its entry for ``kind`` (missing -> default); otherwise
    the callable itself.  So one ``null=`` can carry a distinct provider per cut point."""
    if null is None:
        return DEFAULT
    provider = null.get(kind, DEFAULT) if isinstance(null, Mapping) else null
    if not callable(provider):
        raise TypeError(
            "null must be a null-provider callback FloorContext -> float, a {kind: provider} "
            f"mapping, or None for the derived default 'mp'; got {provider!r}")
    return provider


def apply_floor(null=None, *, spectrum, data, shape, far: float, kind: str, seed: int = 0) -> float:
    """Evaluate the null provider for cut point ``kind`` on one screen and return its scalar
    floor.  ``null`` is a provider callback (``FloorContext -> float``), a ``{kind: provider}``
    mapping (a different provider per cut point), or ``None`` for the derived default
    (:data:`DEFAULT` = ``mp``).  A fresh generator seeded by ``seed`` is placed on the context
    so any resampling provider is deterministic per ``seed`` and per (local) screen."""
    provider = _select_provider(null, kind)
    ctx = FloorContext(spectrum=spectrum, data=data, shape=tuple(shape),
                       far=float(far), kind=kind, rng=np.random.default_rng(seed))
    return float(provider(ctx))
