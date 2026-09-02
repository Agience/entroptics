"""
reads.py -- the Entroptics OPTICS reads (read-only measurements about a structure).

An aperture's resolution is fixed by the signal's own Shannon entropy.
Every read here is an optical / wave quantity tied to an established theorem, and
applies to any 2-D array W (T, F): axis-0 the ORDERED axis (_T), axis-1 the
FEATURE axis (_F).

Per-axis reads (a = T or F), so each axis is visible on its own:
  H_a       Shannon entropy (bits) of the axis marginal
  n_a       effective mode count = round(2^H_a)
  delta_a   matched cell scale (window / bin width)
  phi_a     fill fraction of the axis correlation spectrum (2^{H_sv}/len)
  sigma_a   dominant-mode singular value of the axis correlation (sqrt of lambda1)
  axis_spectrum(W, axis)  the per-axis correlation eigenspectrum phi_a/sigma_a derive from
  axis_read(W, axis)      bundles the five into an AxisRead.

Combined screen-area reads:
  phi(M)             screen fill fraction (2^{H_sv}/N over the whole block)
  magnification(M)   1/phi
  occupied_modes(w)  the RANK EDGE of an ordered spectrum: how many modes carry
                     power, read from the profile's own step and not from a floor
  etendue(W)         phi_F * phi_T           -- the conserved 2-D aperture area
  space_bandwidth(W) n_F * n_T               -- number of resolvable spots
  strehl(W)          lambda1/sum(lambda) of the ordered correlation (coherence)

Mode spectrum / propagation (from the correlation eigenspectrum):
  spectral_optics(W) contrast, top_share, resolved modes, noise floor, the
                     attenuation constant alpha, the phase constant beta, dispersion.
  attenuation_interval(W, band=...)   a certified interval for alpha.
  concentration_band(...)             the a-priori spectral-norm band from samples.
  concentration(rows)                 focus of a vector stack on its dominant axis.
  coupling(a, b)                      the SIGNED coupling between two sides of a screen
                                      (exact permutation null; the two-way screen's read).

Diffraction limit -- from the signal's OWN decay (no external input):
  decay(W)             the ordered-axis autocorrelation C(tau) = the OTF, as a
                       DIRECT lag average (coherent for signed/complex inputs,
                       incoherent |W|^2 for non-negative inputs).
  diffraction_limit(C) a_delta = 1/2^{H}, H = entropy of C^2 (entropy width, primary), plus the
                       classical Abbe integral length 1/xi (secondary).
  mercer_certificate(W) the model-free check: a_delta read the temporal way (decay
                       entropy) and the spectral way (stationary eigenspectrum) --
                       they must coincide (Mercer).
  rayleigh_shape_factor(C)  xi * a_delta (Rayleigh shape factor g; entropy-native, no transform).
  fresnel_number(W, w) near/far-field coordinate.
  shape_factor(W, C)   a_delta / phi_F (Rayleigh/Abbe).

  optics(W)            the full, fully-intrinsic optical read off one screen.

The chain (all standard theorems): Wiener-Khinchin (autocorrelation <-> power
spectrum) LICENSES the direct lag sum -> the Fourier-optics autocorrelation
theorem (OTF = pupil autocorrelation) -> Abbe/Rayleigh (resolution = 1/OTF-
bandwidth = 1/correlation-length).  The entropy width 1/2^{H} (H = entropy of C^2)
is the noise-robust estimator of that reciprocal length; by Mercer the temporal
(decay) and spectral (stationary operator eigenspectrum) reads coincide -- the
built-in certificate.

Re-exported for convenience: geometry, Projection, ProjectionRead, read.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from . import environment as _env
from .environment import ns as _ns
from .entropy import geometry, shannon_bits as _shannon, live_view, normalize, macheps
from .projection import Projection, ProjectionRead, read
from .null_providers import apply_floor as _apply_floor, _norm_isf


# ══════════════════════════════════════════════════════════════════════════════
# Combined scale: the screen fill fraction phi and its magnification 1/phi
# ══════════════════════════════════════════════════════════════════════════════

def phi(M, mask=None) -> float:
    """phi = 2^{H_sv}/N in (0,1]: the fraction of ACTIVE SVD modes of a screen
    block M (the Shannon entropy of its singular spectrum) -- the aperture's fill
    fraction, BOUNDED by construction.  phi -> 1 fully disordered (all modes
    active); phi -> 0 fully coherent (one mode).

    Read on the CENTRED block, as every other spectral read here is.  A baseline is not
    structure: an uncentred spectrum puts a constant offset into a leading singular value, so
    adding a single global constant -- which says nothing about the signal at all -- moved this
    read by more than a factor of three.  Centring is idempotent, so a block that already arrived
    centred (a projected screen) is unchanged.

    A spectrum carrying NO power is not a fill of 0 modes reported as one: 2^H is 1 for an empty
    or all-zero spectrum, so dividing by n would return 1/n -- which by Lemma 3.2 is exactly the
    rank-1 reading, the most coherent an aperture can be. A frame nothing was measured in would
    then be indistinguishable from a perfect single mode. There is no fraction of active modes to
    report when no mode is active, so the read is NaN."""
    M = live_view(M, mask)                # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(M)
    M = _centred(xp, M)                   # THE library's one centring -- see the note above
    S = _env.svdvals(xp, _env.asnum(M))   # singular values (real), on M's backend (GPU if torch)
    return _fill_of(S ** 2, int(S.shape[0]))


@dataclass
class OccupiedModes:
    """Where an aperture's occupied modes end and its empty ones begin, with the evidence for
    it (see :func:`occupied_modes`).  Carries no threshold: ``margin`` says how sharp the edge
    is and the reader decides whether to believe it."""
    k:      int      # modes carrying power
    margin: float    # the edge's step over the largest other step; 1.0 means there is no edge
    step:   float    # the edge's own step, in bits (log2 of the ratio across it)


def occupied_modes(weights) -> OccupiedModes:
    """The RANK EDGE of an ordered spectrum: how many modes carry power, read from the profile
    own step rather than from a noise floor.

    This is the companion to the two counts already here, and it exists because neither reaches
    a FILLED aperture:

    * ``phi`` (and ``2^H``) gives the entropic effective mode count, which is EXACT for an
      occupied set at one level -- a spectrum of ``k`` equal modes and ``n-k`` zeros has
      ``2^H = k`` identically.  But entropy counts every mode that carries anything, so noise in
      the empty modes lifts it: measured on a screen of 10 occupied modes in 64, ``2^H`` reads
      10.00 clean, 10.65 at 20 dB, 13.27 at 12 dB and 20.95 at 6 dB.
    * ``resolved_dimension`` counts modes standing ABOVE a noise bulk, which needs a bulk to
      stand above.  Measured against a known occupancy at 30 dB it is exact to about a quarter
      of the aperture and then falls away -- 10 of 64 reads 10, 20 reads 20, 32 reads 15, 48
      reads 4, 60 reads 0 -- because an aperture that fills its own space leaves no empty
      subspace for a spike to clear.

    The edge does not care about either.  Occupied modes sit at one level and empty ones at
    another, so the boundary is the largest STEP in the ordered profile, and a step is invariant
    to how many modes lie on each side of it.  Measured on the same screens it is exact at every
    occupancy from 10 to 63 of 64 and at every level from clean to 6 dB.

    ``margin`` is the evidence: the edge's step divided by the largest step anywhere else.  A
    filled-then-empty spectrum gives a large margin; a smoothly decaying one gives ~1, which is
    the read saying there is no edge here rather than inventing one.  A perfectly flat spectrum
    has no step at all, and is reported as fully occupied.

    ``weights`` are non-negative: eigenvalues, squared singular values, or per-mode energies.
    Deterministic; numpy-only."""
    w = np.sort(np.abs(np.asarray(_env.to_numpy(weights), dtype=float).ravel()))[::-1]
    n = int(w.size)
    if n == 0 or not np.isfinite(w).all() or w[0] <= 0.0:
        return OccupiedModes(0, float("nan"), 0.0)
    if n == 1:
        return OccupiedModes(1, float("nan"), 0.0)
    lg = np.log2(np.maximum(w, np.finfo(float).tiny))
    steps = lg[:-1] - lg[1:]
    top = int(np.argmax(steps))
    big = float(steps[top])
    if big <= 0.0:                       # no step anywhere: every mode is as lit as every other
        return OccupiedModes(n, 1.0, 0.0)
    rest = np.delete(steps, top)
    second = float(rest.max()) if rest.size else 0.0
    margin = float("inf") if second <= 0.0 else big / second
    return OccupiedModes(top + 1, margin, big)


@dataclass
class LevelEdge:
    """Where an ordered profile separates into two populations, and how much of its spread
    that separation accounts for (see :func:`level_edge`).  Carries no threshold:
    ``separability`` says whether the split means anything and the reader decides."""
    k:            int      # readings in the leading population
    separability: float    # eta^2 = between/total variance, in [0, 1]; 0.0 = one population


def level_edge(weights) -> LevelEdge:
    """The LEVEL EDGE of an ordered profile: the split that best separates it into a high
    group and a low one, by maximum between-class variance.

    Companion to :func:`occupied_modes`, and the two read different things.
    ``occupied_modes`` finds the largest STEP in the log-profile -- a local feature, sharp
    when a spectrum is filled-then-empty and mute on a smooth decay.  This finds the best
    PARTITION by variance -- a global statistic, so one noisy adjacent pair does not move
    it, and it answers for a profile with no step at all.

    ``separability`` is the correlation ratio ``eta^2 = between / total``:

        0.0   no split explains anything -- one population (uniform, or a single level)
        ->1.0 the two groups are cleanly apart

    It measures how cleanly the profile falls into two groups, NOT how far apart they are:
    an exact two-level profile reads 1.000 whatever the gap, at a ratio of 9 and at a ratio
    of 1.001 alike, because two values explain all of the variance between them.  On real
    profiles, where both levels carry spread, it does fall with the gap -- 0.994 / 0.921 /
    0.702 / 0.673 at ratios 9 / 3 / 1.5 / 1.05 with noise at 0.3.  A caller who needs the
    SIZE of the break wants the ratio across it, which is a different number.

    It is the field this read exists for.  As a "is there structure here at all" test on a
    retrieval salience spectrum -- 50 structured pools against 50 structureless ones, the
    per-item signal power of an 80-candidate screen -- it reaches AUROC **0.998**, median
    0.909 against 0.645, with the two populations overlapping only at the edges (0.700
    against 0.726).  On the same draws ``occupied_modes.margin`` reaches 0.812 and the
    largest relative gap across the break 0.929.  A profile's own break POSITION and the
    question of whether it HAS one are not the same read, and this is the second one.

    ``k`` is a count: the leading ``k`` readings are the high group.  It is returned because
    the partition produces it, not because it is the better cut -- measured against a
    relative-gap lock on planted-relevance pools it is worse at every noise level but one
    (0.855 against 0.981 at three facets, 0.542 against 0.717 at four).  Use ``k`` when the
    partition itself is the question; use it as evidence otherwise.

    ``weights`` are non-negative and finite: eigenvalues, squared singular values, per-mode
    energies, or any salience.  A profile with nothing to separate reports ``k = n`` and
    ``separability = 0.0`` rather than being cut anyway.  Deterministic; numpy-only."""
    w = np.sort(np.abs(np.asarray(_env.to_numpy(weights), dtype=float).ravel()))[::-1]
    n = int(w.size)
    if n == 0:
        return LevelEdge(0, 0.0)
    if n == 1 or not np.isfinite(w).all():
        return LevelEdge(n, 0.0)
    mean = float(w.mean())
    var_total = float(((w - mean) ** 2).sum())
    if var_total <= 0.0:                 # every reading identical: nothing to separate
        return LevelEdge(n, 0.0)
    i = np.arange(1, n, dtype=float)                     # split after i readings
    run = np.cumsum(w)[:-1]                              # sum of the leading group
    m1 = run / i
    m2 = (w.sum() - run) / (n - i)
    between = i * (m1 - mean) ** 2 + (n - i) * (m2 - mean) ** 2
    best = int(np.argmax(between))
    return LevelEdge(best + 1, float(between[best] / var_total))


def duality_of(p: float) -> dict:
    """Both faces of the scale from an ALREADY-READ ``phi``: ``{phi, magnification,
    at_diffraction_limit}`` -- reciprocal, meeting at ``phi = magnification = 1``.

    The arithmetic in one place.  :func:`magnification` and :func:`scale_duality` read ``phi``
    off a frame; ``Aperture`` reads it off its own cached read and calls this with the number,
    so the reciprocal and the limit test cannot drift between the free function and the front
    door -- and the front door still pays for no second SVD."""
    mag = float(1.0 / p) if p > 0 else float("inf")
    # phi = 1 exactly at the critically-sampled limit, so the test is equality -- taken at the
    # resolution the arithmetic that produced phi actually has (an entropy of an SVD spectrum),
    # not at a fixed distance from 1.
    tol = macheps(np, np.asarray(p, float)) ** 0.5
    return dict(phi=float(p), magnification=mag,
                at_diffraction_limit=bool(abs(mag - 1.0) <= tol))


def magnification(M, mask=None) -> float:
    """delta = 1/phi = N/2^{H_sv} in [1, inf): the reciprocal of the fill fraction
    -- the magnification / oversampling face.  =1 is the critically-sampled
    diffraction limit; -> inf is maximal magnification (one coherent mode)."""
    return duality_of(phi(M, mask))["magnification"]


def scale_duality(M, mask=None) -> dict:
    """Both faces of the scale off ONE screen: {phi, magnification,
    at_diffraction_limit} -- reciprocal, meeting at phi = magnification = 1."""
    return duality_of(phi(M, mask))


# ══════════════════════════════════════════════════════════════════════════════
# Per-axis reads: phi_a, sigma_a, and the AxisRead bundle
# ══════════════════════════════════════════════════════════════════════════════

def _corr_evals(M: np.ndarray, mask=None) -> np.ndarray:
    """Correlation eigenvalues (descending, non-negative) of the COLUMNS of M as
    variables, using the ROWS of M as samples.  M shape (n_samples, n_vars).
    Complex-safe (Hermitian correlation).  Backend-agnostic (numpy or torch)."""
    M = live_view(M, mask)                # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(M)
    M = _env.asnum(M)
    Xc = M - _env.mean0(xp, M)
    C = Xc.conj().T @ Xc
    # A channel that was measured and never varied has zero variance AND exactly-zero covariance
    # with every other channel, so its row of C is exactly 0 and the correlation is 0/0.  Divide it
    # by 1: the quotient is 0 either way, which is the standard convention (a constant correlates
    # with nothing) and reaches it without putting a small number in the denominator.
    dg = xp.real(xp.diag(C))                  # a sum of squared magnitudes: exactly >= 0
    d = xp.where(dg > 0, xp.sqrt(dg), xp.ones_like(dg))
    C = C / xp.outer(d, d)
    ev = _env.clampmin(xp, xp.real(xp.linalg.eigvalsh(C)), 0.0)
    return _env.flip(xp, ev)              # descending


def _centred(xp, X):
    """The CONNECTED frame: ``X`` column-centred with non-finite cells zeroed.  THE library's
    one centring -- every correlation read here de-means exactly this way, so they cannot
    drift apart.  Unlike ``live_view`` it keeps every row: a read that pairs two frames
    row-by-row (``coupling``, a screen side) needs them to stay ROW-ALIGNED."""
    Xf = _env.asnum(X)
    diff = Xf - _env.nanmean0(xp, Xf)
    return xp.where(xp.isnan(diff), xp.zeros_like(diff), diff)


def _fill_of(ev, length: int) -> float:
    """``2^H(ev)/length`` -- THE fill fraction, formed once so the whole-screen and per-axis reads
    cannot disagree about what an empty spectrum means.  See :func:`phi`: a spectrum with no power
    has no active modes to be a fraction of, and reporting ``1/length`` for it would return the
    rank-1 reading (Lemma 3.2) for a frame that carries nothing."""
    xp = _ns(ev)
    if length <= 0 or int(ev.shape[0]) == 0 or not float(_env.sum_ax(xp, ev)) > 0.0:
        return float("nan")
    return float(2.0 ** _shannon(ev) / length)


def axis_spectrum(W, axis: int, mask=None):
    """Correlation eigenvalues (descending, non-negative) along ``axis`` (0 =
    ordered/T, 1 = feature/F) -- the per-axis correlation eigenspectrum every axis
    read is derived from.  Compute it ONCE and pass it to ``axis_read`` / ``strehl``
    to avoid recomputing the (up to O(len^3)) eigendecomposition.  Backend-agnostic.

    Formed on the CONNECTED screen -- ``_centred``, the library's one centring.  A per-channel
    static level is not temporal structure: it is the same value at every time, so it carries no
    order at all.  Left in, it reaches the ordered-axis correlation as one vector added to every
    time point's sample, i.e. a rank-1 mode, and the read calls that coherence.  On white noise
    plus a static bandpass the Strehl ratio read 0.99 -- a higher score than a real sinusoid --
    and did not move when the rows were shuffled.  The feature-axis read is unaffected: it centres
    over time already, so connecting first is idempotent there."""
    L = live_view(W, mask)               # what was measured, before anything is subtracted from it
    C = _centred(_ns(L), L)              # ... then the static per-channel level comes off
    return _corr_evals(C if axis == 1 else C.T)   # columns become the axis variables


def phi_T(W, mask=None) -> float:
    """phi_T: the ORDERED-axis fill fraction = 2^{H_sv}/T over the ordered-axis
    correlation spectrum (the "time" aperture).

    The divisor is the MEASURED extent, not the array's: a row carrying no finite, unmasked
    cell was never observed, and dividing by it would raise the bar the signal has to clear
    without adding any signal."""
    L = live_view(W, mask)
    return _fill_of(axis_spectrum(L, 0), int(L.shape[0]))


def phi_F(W, mask=None) -> float:
    """phi_F: the FEATURE-axis fill fraction = 2^{H_sv}/F over the feature-axis
    correlation spectrum.  HIGH = many active feature modes (disorder);
    LOW = few active modes (long-range coherence).

    The divisor is the MEASURED extent, not the array's: a channel carrying no finite,
    unmasked cell was never observed, and dividing by it would raise the bar the signal has
    to clear without adding any signal."""
    L = live_view(W, mask)
    return _fill_of(axis_spectrum(L, 1), int(L.shape[1]))


def sigma_T(W) -> float:
    """sigma_T: dominant-mode singular value of the ordered-axis correlation
    (sqrt of its top eigenvalue) -- the strength of the leading temporal mode."""
    ev = axis_spectrum(W, 0)
    return float(ev[0]) ** 0.5 if int(ev.shape[0]) else 0.0


def sigma_F(W) -> float:
    """sigma_F: dominant-mode singular value of the feature-axis correlation
    (sqrt of its top eigenvalue) -- the strength of the leading feature mode."""
    ev = axis_spectrum(W, 1)
    return float(ev[0]) ** 0.5 if int(ev.shape[0]) else 0.0


@dataclass
class AxisRead:
    """Everything about ONE axis of a screen (see axis_read())."""
    H:     float   # Shannon entropy (bits) of the axis power marginal
    n:     int     # matched-grid width for this axis: the RESAMPLE TARGET a caller folds to
                   # (PAPER Def 2.2 -- the ordered axis keeps native resolution, and an
                   # unlicensed feature fold keeps F), NOT the effective mode count 2^H
    delta: float   # matched cell scale = len / 2^H (window / bin width)
    phi:   float   # fill fraction of the axis correlation spectrum (2^{H_sv}/len)
    sigma: float   # dominant-mode singular value of the axis correlation


def axis_read(W, axis: int, mask=None, *, geom: dict | None = None, evals=None) -> AxisRead:
    """Bundle the per-axis reads (H, n, delta, phi, sigma) for ``axis``
    (0 = ordered/T, 1 = feature/F).  Pass a precomputed ``geom`` (from geometry())
    and/or ``evals`` (from axis_spectrum(W, axis)) to skip recomputing them."""
    g = geometry(W, mask) if geom is None else geom
    # the fill divides by the MEASURED extent: a row or column with no finite, unmasked cell
    # was never observed, and counting it raises the bar without adding any signal.
    L = live_view(W, mask)
    if axis == 0:
        H, n, delta, length = g["H_T"], g["n_T"], g["delta_T"], int(L.shape[0])
    else:
        H, n, delta, length = g["H_F"], g["n_F"], g["delta_F"], int(L.shape[1])
    ev = axis_spectrum(L, axis) if evals is None else evals
    return AxisRead(H=float(H), n=int(n), delta=float(delta),
                    phi=_fill_of(ev, length),
                    sigma=float(ev[0]) ** 0.5 if int(ev.shape[0]) else 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Combined screen-area reads
# ══════════════════════════════════════════════════════════════════════════════

def etendue(W, mask=None) -> float:
    """Optical ETENDUE = phi_F * phi_T: the joint 2-D aperture area (the
    phase-space volume the finite screen carries).  A conserved invariant of a
    lossless system (optics: A*Omega is conserved)."""
    return float(phi_F(W, mask) * phi_T(W, mask))


def strehl(W, mask=None, *, evals=None) -> float:
    """STREHL ratio (coherence): lambda1/sum(lambda) of the ordered-axis
    correlation -- peak-mode power vs total.  S -> 1 perfectly coherent (one
    dominant mode); S -> 1/T disordered (power spread across modes).  Pass
    ``evals`` (from axis_spectrum(W, 0)) to reuse an already-computed spectrum."""
    ev = axis_spectrum(W, 0, mask) if evals is None else evals
    xp = _ns(ev)
    tot = float(_env.sum_ax(xp, ev))
    return float(xp.max(ev)) / tot if tot > 0 else 0.0


def space_bandwidth(W, mask=None) -> int:
    """SPACE-BANDWIDTH PRODUCT = n_F * n_T, the degrees of freedom the screen CAN carry
    [Lukosz 1966] -- its capacity on the matched grid, finite <=> band-limited.

    A capacity, not a content: ``n_a`` is the matched-grid width (PAPER Def 2.2), so an
    unfolded screen reports T*F whatever is on it, and a single bright cell reports the same
    SBW as broadband noise.  What the screen actually FILLS is ``etendue * space_bandwidth``
    -- exactly 2^{H_T} * 2^{H_F} of the axis correlation spectra, which reads 1 for one mode
    and rises with the modes present."""
    g = geometry(W, mask)
    return int(g["n_F"]) * int(g["n_T"])


# ══════════════════════════════════════════════════════════════════════════════
# Mode spectrum / propagation constant (from the correlation eigenspectrum)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SpectralOptics:
    """The optics of the correlation eigenspectrum (see spectral_optics())."""
    contrast:       float        # lambda1 / noise_floor  -- peak mode above the floor (>1 => structure)
    top_share:      float        # lambda1 / sum(lambda)  -- dominant-mode power fraction (Strehl-like)
    resolved_modes: int          # eigenvalues above the noise floor
    noise_floor:    float        # finite-size Tracy-Widom (Johnstone) correlation edge (mp default)
    attenuation:    float        # log(lambda1/max(lambda2,floor)) -- attenuation constant alpha (Re gamma)
    phase:          float        # per-step phase advance of the dominant mode -- phase constant beta (Im gamma)
    dispersion:     float        # std of per-mode attenuation across resolved modes (spectral dispersion)
    resolved_power: float        # sum over modes above the floor of (lambda_k - edge) -- power standing above the noise sea
    dominance:      float        # (lambda1 - 1)/(F - 1) in [0,1], F = variables -- top-mode excess over the mean eigenvalue, normalized
    eigenvalues:    np.ndarray   # correlation eigenvalues, descending


def _spectral_from_cov(xp, Cov, T: int, N: int, *, null=None, far: float = 0.05,
                       data=None, seed: int = 0, kind: str = "spectral") -> SpectralOptics:
    """Assemble ``SpectralOptics`` from an (N, N) column-covariance ``Cov`` accumulated over
    ``T`` samples (rows) of ``N`` variables (columns).  Shared by ``spectral_optics`` (one
    screen, one covariance) and ``SpectralAccumulator`` (a covariance pooled over intact
    planes / an ensemble), so the two cannot drift.  ``null`` is the noise-floor provider
    (``None`` = the derived default ``mp``); ``data`` is the centred samples a resampling
    provider needs (``None`` for a covariance-only accumulator -> only closed-form providers)."""
    if int(N) < 2 or int(T) < 3:
        return SpectralOptics(contrast=0.0, top_share=0.0, resolved_modes=0,
                              noise_floor=float("inf"), attenuation=0.0, phase=0.0,
                              dispersion=0.0, resolved_power=0.0, dominance=0.0,
                              eigenvalues=np.zeros(int(N)))
    d = xp.sqrt(_env.clampmin(xp, xp.real(xp.diag(Cov)), 1e-30))
    Cmat = Cov / xp.outer(d, d)                          # exact correlation matrix (unit diagonal)
    evals, evecs = xp.linalg.eigh(Cmat)
    order = _env.argsort_desc(xp, xp.real(evals))
    ev = _env.clampmin(xp, xp.real(evals)[order], 0.0)
    v1 = evecs[:, int(order[0])]                         # dominant mode
    total = float(_env.sum_ax(xp, ev))
    # Upper noise edge for the eigenvalues of a unit-diagonal CORRELATION matrix (N
    # variables from T samples), from the null PROVIDER evaluated on THIS screen.  The
    # default ``mp`` is the finite-size Johnstone / Tracy-Widom edge -- the SAME derivation
    # as the screen floor, in correlation units (the Wishart edge / T; -> (1 + sqrt(N/T))^2
    # as T, N grow, plus the finite-size correction that keeps the LARGEST sample
    # eigenvalue's TW fluctuation from over-firing in the wide N > T regime).  A caller may
    # pass any provider (see null_providers); a resampling one uses ``data``.
    edge = _apply_floor(null, spectrum=ev, data=data, shape=(T, N),
                        far=far, kind=kind, seed=seed)
    lam1 = float(ev[0]) if int(ev.shape[0]) else 0.0
    lam2 = float(ev[1]) if int(ev.shape[0]) > 1 else 0.0
    ref = max(lam2, edge)
    attenuation = math.log(lam1 / ref) if (lam1 > ref > 0) else 0.0
    phase = (float(xp.angle(_env.sum_ax(xp, v1[1:] * xp.conj(v1[:-1]))))
             if int(v1.shape[0]) > 1 else 0.0)
    above = ev[ev > edge]
    # `edge > 0` guards the same degenerate floor `contrast` already guards: a caller-supplied
    # null provider may legitimately return 0 (resolve everything), and log(x/0) is not a read.
    dispersion = (float(_env.std0(xp, xp.log(above / edge)))
                  if (int(above.shape[0]) > 1 and edge > 0) else 0.0)
    resolved_power = float(_env.sum_ax(xp, above - edge)) if int(above.shape[0]) else 0.0
    dominance = min(1.0, max(0.0, (lam1 - 1.0) / (N - 1))) if N > 1 else 0.0
    return SpectralOptics(
        contrast=(lam1 / edge) if edge > 0 else 0.0,
        top_share=(lam1 / total) if total > 0 else 0.0,     # exact: a share needs only a positive total
        resolved_modes=int(_env.sum_ax(xp, ev > edge)),
        noise_floor=float(edge),
        attenuation=attenuation,
        phase=phase,
        dispersion=dispersion,
        resolved_power=resolved_power,
        dominance=dominance,
        eigenvalues=ev,
    )


def spectral_optics(data: np.ndarray, mask: np.ndarray | None = None,
                    *, null=None, far: float = 0.05, seed: int = 0) -> SpectralOptics:
    """Read the optics of a screen's correlation eigenspectrum.

    The dominant mode's separation from the noise sea, read as optical / wave
    quantities.  ``data`` is (T, N): T samples (rows) of N variables (columns); the
    correlation is taken over the columns.  Higher-dimensional fields must be reduced
    first (raise on >2-D -- see ``entroptics.fields.slabs`` / ``over_planes``).

    The propagation constant gamma = alpha + i*beta of the dominant mode:
      attenuation alpha = log(lambda1 / max(lambda2, floor)) -- the log-decay rate
                          of the dominant mode above the next mode (or the noise
                          floor). Grows as the mode isolates.
      phase       beta  = per-step phase advance of the dominant eigenvector -- the
                          carrier/oscillation frequency when the axis is ordered.
                          0 for real, non-oscillatory spectra.
      dispersion        = spread of the per-mode attenuation across the resolved
                          modes: how the propagation constant varies mode-to-mode.
      dominance         = (lambda1 - 1)/(N - 1) in [0, 1] -- the top mode's excess
                          over the mean eigenvalue (1 for a unit-diagonal correlation)
                          normalized by the maximum; the magnitude of the leading mode.

    ``null`` is the noise-floor NULL PROVIDER (a ``FloorContext -> float`` callback; the
    SAME contract the screen floor takes -- see :mod:`null_providers`), evaluated on THIS
    correlation spectrum.  ``None`` uses the derived default ``mp`` (the finite-size
    Johnstone / Tracy-Widom edge for the largest correlation eigenvalue at ``far``; the
    asymptotic Marchenko-Pastur bulk edge (1 + sqrt(N/T))^2 plus the finite-size
    correction, well calibrated for T >= ~80).  Pass ``null_providers.robust``,
    ``null_providers.permutation()`` (the distribution-free null -- the correct floor for
    correlated data where ``mp`` conflates bulk correlation with signal, deterministic per
    ``seed``), or your OWN provider (a local reference / physics null).  ``far`` is the
    significance level (5%)."""
    xp = _ns(data)
    nd = len(data.shape)
    if nd != 2:
        raise ValueError(
            f"spectral_optics expects a 2-D array (T, N); got {nd}-D. Reduce a "
            f"higher-D field with entroptics.fields.slabs / over_planes (keep the "
            f"plane intact) or fields.pool (flatten sites as samples) first.")
    data = live_view(data, mask)             # ignore fully-dead rows/cols; clean scattered gaps
    mask = None
    xp = _ns(data)
    empty = SpectralOptics(contrast=0.0, top_share=0.0, resolved_modes=0,
                           noise_floor=float("inf"), attenuation=0.0, phase=0.0,
                           dispersion=0.0, resolved_power=0.0, dominance=0.0,
                           eigenvalues=np.zeros(int(data.shape[1])))
    if int(data.shape[0]) < 3 or int(data.shape[1]) < 2:
        return empty
    T, N = int(data.shape[0]), int(data.shape[1])
    Xc = _centred(xp, data)                  # de-mean; invalid -> 0 (the one centring)
    Cov = Xc.conj().T @ Xc
    return _spectral_from_cov(xp, Cov, T, N, null=null, far=far, data=Xc, seed=seed)


def principal_directions(data: np.ndarray, mask: np.ndarray | None = None,
                         *, null=None, far: float = 0.05, seed: int = 0) -> np.ndarray:
    """The feature-space directions of the resolved correlation modes -- the ``(N, k)`` basis whose
    columns span the resolved sector, ordered by descending eigenvalue.

    The directional companion to :func:`spectral_optics`: that read reports how many modes stand
    above the noise floor (``resolved_modes``); this reports which directions they are, so a caller
    that wants the resolved subspace reads it from the instrument rather
    than re-deriving an SVD.  ``k`` is exactly ``spectral_optics(...).resolved_modes`` for the same
    arguments (same floor, same null), and column ``j`` pairs with the ``j``-th resolved eigenvalue.

    Read off the same unit-diagonal correlation matrix as ``spectral_optics`` -- **not** the
    entropy-folded monitor screen -- so the basis lives in the native feature space and is
    scale-invariant (per-channel scale cancels in the correlation), and a sparse carrier is never
    folded away.  ``data`` is (T, N).  Dead (all-invalid) columns are dropped first, exactly as the
    spectral read drops them, so the returned basis spans the live columns in their original order;
    an ``(N_live, k)`` array.  Returns an ``(N, 0)`` array when nothing resolves.

    A complex frame returns a complex basis.  The correlation is Hermitian
    (``Xc.conj().T @ Xc``), so ``eigh`` gives complex eigenvectors whose phase is part of the
    direction: discarding it does not project onto the resolved sector, it projects onto the
    sector's shadow, so the phase is kept.  A real frame returns a real basis
    -- ``eigh`` on a real symmetric matrix gives real eigenvectors, so that path takes the
    identical expression it always does.

    A complex frame's ``resolved_modes`` is not directly comparable to its real part's: one
    complex direction spans two real ones (a mode ``e^{i w t} b`` has real part spanning both
    ``Re b`` and ``Im b``), so the two counts describe the same structure in different bases. At
    matched per-cell variance the complex read resolves at least as many modes as the real one,
    because a complex cell carries two real degrees of freedom, so the correlation is better
    conditioned at the same row count."""
    xp = _ns(data)
    nd = len(data.shape)
    if nd != 2:
        raise ValueError(
            f"principal_directions expects a 2-D array (T, N); got {nd}-D. Reduce a higher-D "
            f"field with entroptics.fields.slabs / over_planes / pool first.")
    data = live_view(data, mask)                              # drop dead rows/cols, as spectral does
    N = int(data.shape[1]) if len(data.shape) == 2 else 0
    if int(data.shape[0]) < 3 or N < 2:
        return np.zeros((N, 0))
    # The COUNT and the FLOOR come from the spectral read itself, so k here cannot drift from
    # `resolved_modes` (one read, one floor, one null).
    k = int(spectral_optics(data, None, null=null, far=far, seed=seed).resolved_modes)
    if k <= 0:
        return np.zeros((N, 0))
    # The vectors come from the SAME correlation matrix `_spectral_from_cov` reads its eigenvalues
    # off (unit-diagonal correlation, per-channel scale cancelled), so the eigenvalue order the count
    # was taken in is the order the columns are returned in.
    Xc = _centred(xp, data)                                   # the one centring
    Cov = Xc.conj().T @ Xc
    d = xp.sqrt(_env.clampmin(xp, xp.real(xp.diag(Cov)), 1e-30))
    Cmat = Cov / xp.outer(d, d)                               # exact correlation matrix (unit diagonal)
    evals, evecs = xp.linalg.eigh(Cmat)
    order = _env.argsort_desc(xp, xp.real(evals))
    basis = evecs[:, order[:k]]                               # (N_live, k), descending eigenvalue
    # The PHASE is part of the direction when the frame is complex -- see the docstring. Gated on the
    # INPUT, not on `basis`'s dtype, so the real path takes the identical expression it always
    # did, instead of relying on eigh happening to return a real array.
    cplx = xp.is_complex(data) if _env.is_torch(xp) else np.iscomplexobj(data)
    return np.asarray(_env.asnum(basis if cplx else xp.real(basis)))


def spectral_batch(frames, *, null=None, far: float = 0.05, seed: int = 0) -> list:
    """Read :func:`spectral_optics` for a BATCH of same-shape 2-D frames in one pass -- the
    column de-mean and the ``(N, N)`` covariances are formed batched (``(B, T, N)`` ->
    ``(B, N, N)``), then EACH frame's eigenspectrum + floor + optics is assembled by the SAME
    :func:`_spectral_from_cov` the per-frame ``spectral_optics`` calls.  So the result is
    **bit-identical** to ``[spectral_optics(f, null=null, far=far, seed=seed) for f in frames]``,
    while amortizing the per-frame de-mean / covariance / call overhead (the ensemble lever for
    the correlation read on small planes).  A frame that is
    masked / non-finite / complex / a different shape / too small falls back to a per-frame
    ``spectral_optics``.  numpy / CPU."""
    frames = [np.asarray(f) for f in frames]
    out: list = [None] * len(frames)
    shape0 = next((f.shape for f in frames if f.ndim == 2 and not np.iscomplexobj(f)
                   and int(f.shape[0]) >= 3 and int(f.shape[1]) >= 2), None)
    batchable = []
    if shape0 is not None:
        same = [i for i, f in enumerate(frames)
                if f.ndim == 2 and not np.iscomplexobj(f) and f.shape == shape0]
        finite = np.isfinite(np.stack([frames[i] for i in same])).reshape(len(same), -1).all(axis=1)
        batchable = [same[k] for k in range(len(same)) if bool(finite[k])]
    bset = set(batchable)
    for i, f in enumerate(frames):                               # non-finite / complex / odd shape
        if i not in bset:
            out[i] = spectral_optics(f, null=null, far=far, seed=seed)
    if batchable:
        T, N = int(shape0[0]), int(shape0[1])
        st = _env.asnum(np.stack([frames[i] for i in batchable]))        # (B, T, N), compute precision
        Xc = st - st.mean(axis=1, keepdims=True)                         # de-mean (finite: no NaN path)
        Cov = np.matmul(Xc.conj().transpose(0, 2, 1), Xc)               # (B, N, N) column covariances
        for j, i in enumerate(batchable):
            out[i] = _spectral_from_cov(np, Cov[j], T, N, null=null, far=far, data=Xc[j], seed=seed)
    return out


@dataclass
class CertifiedInterval:
    """A certified interval for the attenuation constant alpha (see attenuation_interval())."""
    attenuation:    float   # the point read (= spectral_optics(...).attenuation)
    attenuation_lo: float   # certified lower bound given the band
    attenuation_hi: float   # certified upper bound
    band:           float   # the propagated spectral-norm band ||C_input - C_true||_2
    certified:      bool     # attenuation_lo > 0: a positive attenuation survives the band


def attenuation_interval(data: np.ndarray, mask: np.ndarray | None = None,
                         *, band: float, sg: "SpectralOptics | None" = None) -> CertifiedInterval:
    """Certified interval for the attenuation constant alpha given an input
    spectral-norm band ``band`` (an upper bound on ||C_input - C_true||_2).

    The estimator alpha = log(lambda1) - log(max(lambda2, floor)) is exact on the
    correlation matrix it is handed; the only error is the input matrix's
    deviation from the truth.  By Weyl, |lambda_k_read - lambda_k_true| <= band,
    and since alpha is increasing in lambda1 and decreasing in the reference, the
    band propagates exactly to [alpha_lo, alpha_hi].  A certified positive
    attenuation is alpha_lo > 0.  Pass a precomputed ``sg`` (spectral_optics read)
    to reuse it instead of recomputing.
    """
    sg = spectral_optics(data, mask) if sg is None else sg
    ev = np.asarray(_env.to_numpy(sg.eigenvalues))   # certify in numpy (occasional read)
    edge = float(sg.noise_floor)
    d = float(band)
    if ev.size < 2 or not math.isfinite(edge) or edge <= 0.0:
        return CertifiedInterval(sg.attenuation, 0.0, 0.0, d, False)
    lam1 = float(ev[0]); lam2 = float(ev[1])
    lo1 = max(lam1 - d, 1e-300)                     # smallest plausible top eigenvalue
    ref_hi = max(lam2 + d, edge)                    # largest plausible reference  -> smallest alpha
    ref_lo = max(lam2 - d, edge)                    # smallest plausible reference -> largest alpha
    alpha_lo = math.log(lo1 / ref_hi)
    alpha_hi = math.log((lam1 + d) / ref_lo)
    return CertifiedInterval(sg.attenuation, alpha_lo, alpha_hi, d, alpha_lo > 0.0)


@dataclass
class CertifiedCount:
    """A certified interval for the resolved-mode count (see resolved_dimension_interval())."""
    resolved_modes: int    # the point read (= spectral_optics(...).resolved_modes)
    resolved_lo:    int    # certified minimum count given the band
    resolved_hi:    int    # certified maximum count
    band:           float  # the propagated spectral-norm band ||C_input - C_true||_2


def resolved_dimension_interval(data: np.ndarray, mask: np.ndarray | None = None,
                                *, band: float,
                                sg: "SpectralOptics | None" = None) -> CertifiedCount:
    """Certified interval for the resolved-mode count ``K = #{eigenvalue > edge}`` given an
    input spectral-norm band ``band`` (an upper bound on ``||C_input - C_true||_2``).  The
    count analogue of ``attenuation_interval`` (Lemma 6.2): by Weyl
    ``|lambda_k_read - lambda_k_true| <= band``, and the Marchenko-Pastur edge is a fixed
    function of the shape ``(N, T)`` (it does not move with the data band), so

        K_lo = #{lambda_k - band > edge}   (modes CERTAINLY above the floor)
        K_hi = #{lambda_k + band > edge}   (modes POSSIBLY above the floor),

    and the true count lies in ``[K_lo, K_hi]``.  Pass a precomputed ``sg`` (spectral_optics
    read, e.g. the pooled read of ``SpectralAccumulator``) to reuse it.  A certified LOW
    resolved dimension is a small ``K_hi``; certifying it against a nats floor goes through
    the companion attenuation (the top singular value is ``1 - attenuation``)."""
    sg = spectral_optics(data, mask) if sg is None else sg
    ev = np.asarray(_env.to_numpy(sg.eigenvalues), dtype=float)
    edge = float(sg.noise_floor)
    d = float(band)
    if ev.size < 1 or not math.isfinite(edge):
        return CertifiedCount(int(sg.resolved_modes), 0, int(ev.size), d)
    k_lo = int(np.count_nonzero(ev - d > edge))
    k_hi = int(np.count_nonzero(ev + d > edge))
    return CertifiedCount(int(sg.resolved_modes), k_lo, k_hi, d)


def concentration_band(n_rows: int, n_cols: int, *, spec_norm: float = 1.0,
                       c_conc: float = 2.0) -> float:
    """A-priori spectral-norm band for the EMPIRICAL correlation matrix from
    ``n_rows`` iid samples of an ``n_cols``-dim vector.  Matrix concentration
    gives ||C_hat - C||_2 <= c_conc * ||C|| * (sqrt(N/T) + N/T) for T >= N
    (Vershynin, high probability).  Feed the result to
    ``attenuation_interval(..., band=...)`` to certify how many samples make the
    read tight."""
    T, N = int(n_rows), int(n_cols)
    if T <= 0 or N <= 0:
        return float("inf")
    r = math.sqrt(N / T) + (N / T)
    return c_conc * float(spec_norm) * r


class SpectralAccumulator:
    """Pool the feature correlation over intact 2-D planes and/or an ensemble into ONE
    spectrum.  The feature-side analogue of ``Dynamics.merge``: where the dynamical operator
    sums ``Pxx, Pyx`` over a stream, this sums the de-meaned column-covariance ``Xc^H Xc`` over
    planes, keeping each plane's within-plane correlation intact.  Feed it from
    ``fields.slabs`` (intact planes), never a bare flatten of the volume.

    The pooled sample count ``T`` grows with the ensemble, so the certified band
    (``concentration_band(T, F)``) tightens; ``spectral()`` then reads one ``SpectralOptics``
    on which ``attenuation_interval`` / ``resolved_dimension_interval`` certify.  This is the
    ensemble-level Aperture bound: a certificate over the whole configuration ensemble rather
    than a single sample.  Numpy accumulation (the occasional certified read, not the hot path).
    """

    def __init__(self, n_features: int, *, whiten: bool = False) -> None:
        self.F = int(n_features)     # number of variables (columns), fixed across planes
        self.T = 0                   # total pooled samples (rows) accumulated
        self._cov = None             # (F, F) running column-covariance; dtype follows the data
        self.whiten = bool(whiten)   # per-channel robust (MAD) whitening before accumulation

    def add(self, plane) -> "SpectralAccumulator":
        """Accumulate one intact 2-D plane ``(T_p samples, F features)``: de-meaned per plane
        (the connected read; with ``whiten=True`` each channel is first robust-normalised by
        the library's MAD whitening, the screen's per-channel scale removal), its
        column-covariance summed in."""
        X = np.asarray(_env.to_numpy(plane))
        if X.ndim != 2:
            raise ValueError(f"SpectralAccumulator.add expects a 2-D plane; got {X.ndim}-D")
        if int(X.shape[1]) != self.F:
            raise ValueError(f"plane has {int(X.shape[1])} features; accumulator holds {self.F}")
        if int(X.shape[0]) < 1:
            return self
        if self.whiten:
            X = np.asarray(_env.to_numpy(normalize(X)))
            X = np.where(np.isfinite(X), X, 0.0)
        Xc = X - X.mean(axis=0, keepdims=True)
        G = Xc.conj().T @ Xc
        self._cov = G if self._cov is None else self._cov + G
        self.T += int(X.shape[0])
        return self

    def merge(self, other: "SpectralAccumulator") -> "SpectralAccumulator":
        """Splice another accumulator in (sum the covariances and the sample counts)."""
        if int(other.F) != self.F:
            raise ValueError(f"feature mismatch: {self.F} vs {int(other.F)}")
        if other._cov is not None:
            self._cov = other._cov.copy() if self._cov is None else self._cov + other._cov
        self.T += int(other.T)
        return self

    def spectral(self, *, null=None) -> SpectralOptics:
        """One ``SpectralOptics`` read on the pooled covariance (large ``T`` -> tight band).
        This is the ``"bulk"`` cut point (the pooled ENSEMBLE floor); ``null`` is its
        provider (``None`` = derived default ``mp``, or the ``"bulk"`` entry of a
        ``{kind: provider}`` mapping).  Only closed-form providers (``mp`` / ``robust`` /
        a ``reference_null``) apply here -- a resampling provider needs the raw samples the
        accumulator does not retain."""
        cov = np.zeros((self.F, self.F)) if self._cov is None else self._cov
        return _spectral_from_cov(np, cov, self.T, self.F, null=null, kind="bulk")

    def band(self, **kw) -> float:
        """The Vershynin certified band for the pooled spectrum (``concentration_band``)."""
        return concentration_band(self.T, self.F, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# Concentration / focus of a vector stack (Fisher information)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Concentration:
    """The concentration / focus of a stack of row-vectors (see concentration())."""
    intensity:  float   # sigma1^2      -- total power on the dominant axis (EXTENSIVE)
    focus:      float   # sigma1^2 / M  -- dominant-axis power fraction in (0,1] (AXIAL)
    resultant:  float   # |mean row|    -- von Mises-Fisher sufficient statistic (DIRECTIONAL)
    n:          int      # M, number of rows (independent observations)
    dim:        int      # D, ambient dimension


def concentration(vectors: np.ndarray, mask=None, *, normalize: bool = True) -> Concentration:
    """Fisher-information CONCENTRATION of a stack of row-vectors -- how sharply a
    cloud of vectors FOCUSES on its dominant axis.

        intensity = sigma1^2       top squared singular value of the stack (EXTENSIVE)
        focus     = sigma1^2 / M   AXIAL concentration: the power fraction on the
                                   leading principal axis, in (0, 1] for unit rows
                                   (axis powers sum to 1).  Axial, so an antipodal
                                   cloud still reads ~1.
        resultant = |mean row|     DIRECTIONAL concentration: the von Mises-Fisher
                                   sufficient statistic (an antipodal cloud reads ~0).

    focus and resultant are DISTINCT measures (they coincide only for a one-sided,
    rank-1 cloud); focus sees an axis, resultant sees a direction.

    Parameters
    ----------
    vectors   : (M, D) array of row-vectors (real or complex).
    normalize : L2-normalise each row first (default True) so focus/resultant read
                on the unit sphere; pass False to weight rows by magnitude.
    """
    vectors = live_view(vectors, mask)    # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(vectors)
    nd = len(vectors.shape)
    if nd != 2 or int(vectors.shape[0]) < 1:
        return Concentration(intensity=0.0, focus=0.0, resultant=0.0, n=0,
                             dim=int(vectors.shape[1]) if nd == 2 else 0)
    M, D = int(vectors.shape[0]), int(vectors.shape[1])
    Xf = _env.asnum(vectors)
    if normalize:
        norms = xp.sqrt(_env.sum_ax(xp, xp.abs(Xf) ** 2, 1, keep=True))
        norms = xp.where(norms == 0, xp.ones_like(norms), norms)   # leave zero rows at the origin
        Xf = Xf / norms
    if D <= M:
        G = Xf.conj().T @ Xf
    else:
        G = Xf @ Xf.conj().T
    top = float(_env.clampmin(xp, xp.real(xp.linalg.eigvalsh(G)), 0.0)[-1])
    resultant = float(_env.vnorm(xp, _env.mean0(xp, Xf)))
    return Concentration(intensity=top, focus=top / M if M > 0 else 0.0,
                         resultant=resultant, n=int(M), dim=int(D))


# ══════════════════════════════════════════════════════════════════════════════
# Coupling between two sides of a screen (the signed, exact-permutation read)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Coupling:
    """The MEASURED coupling between two sides that meet on a shared basis (see
    :func:`coupling`).  ``sign``/``strength`` are the DECISION (zeroed below the level);
    ``z``/``phase``/``tightness`` are the EVIDENCE, always reported."""
    z:            float   # signed exact-permutation z-score of the alignment (nan: no shared basis)
    sign:         int     # +1 attract / -1 detract / 0 unresolved OR no sign exists on this pair
    strength:     float   # signed cosine of the two centred frames, in [-1,1]; 0 when unresolved
    phase:        float   # arg of the Hermitian alignment: the U(1) face of the sign (0/pi if real)
    tightness:    float   # top_share of the cross-covariance spectrum in (0,1]: tight vs loose
    resolved:     bool    # |z| clears the two-sided level `far`
    cutoff:       float   # the |z| that level requires (the standard-normal inverse survival)
    n:            int     # rows (ordered-axis samples) compared


def _real_embed(xp, X):
    """The real embedding ``C^D -> R^{2D}``, ``x -> [Re x | Im x]``.  Exactly
    ``Re<a,b>_C = <emb a, emb b>_R``, so the signed read and its permutation variance are ONE
    real code path for real and complex sides alike (no complex-only branch of the math)."""
    is_c = xp.is_complex(X) if _env.is_torch(xp) else np.iscomplexobj(X)
    return _env.cat1(xp, [xp.real(X), xp.imag(X)]) if is_c else X


def coupling(a, b, *, far: float = 0.05) -> Coupling:
    """The MEASURED, SIGNED coupling between two sides ``a`` and ``b`` of a screen --
    ``(T, D_a)`` and ``(T, D_b)`` frames sharing the ordered axis (row ``t`` of each is the
    same instant / the same place on the screen).  Never a constant passed in.

    The statistic is the Hermitian ALIGNMENT of the two connected frames

        S = sum_t <a~_t, b~_t>  =  <A~, B~>_F        (a~, b~ column-centred)

    standardised by its EXACT null of a uniformly random re-pairing of the two sides
    (permute one side's rows; each side keeps its OWN internal structure entirely -- this is
    the null for "are these two coupled", not "does either have structure"):

        E_pi[S] = 0   (both sides centred)
        Var_pi[Re S] = tr(C_a C_b) / (T - 1),    C_a = A^T A,  C_b = B^T B   (REAL-EMBEDDED)
        coupling z  = Re S / sqrt(Var_pi[Re S])

    The Grams are formed on the real embedding iota(x) = (Re x, Im x), never on the Hermitian
    A~^H A~: Re<a,b>_C == <iota a, iota b>_R, while the Hermitian form gives
    E|S|^2 = Var[Re S] + Var[Im S] -- the wrong normaliser off the real axis. For
    a~ = (1, -1), b~ = (i, -i) the true Var[Re S] is 0 and the Hermitian form reads 4.

    The variance is the Pitman-Hoeffding permutation second moment, closed form -- NO
    sampling and no RNG, the same discipline as ``projection.coherence``'s exact Cliff-Ord null
    (that read standardises rows against LAGGED rows; this one standardises side A against
    side B).  The two-sided level ``far`` is applied through the standard-normal inverse
    survival; the permutation MOMENTS are exact, the tail is the Pitman CLT limit, so at very
    small ``T`` the level is approximate (the first two moments are not).

    The sign exists only on a shared basis, so this read requires one and raises without it.
    ``+`` is co-resolution (attract), ``-`` is anti-resolution (detract), and both are
    properties of the two sides' coordinates being the same coordinates.  With ``D_a != D_b``
    there is no such pairing: the only basis-free statistic is ``||A~^H B~||_F^2``, which is
    non-negative by construction and has no exact null here -- so the read raises.

    ``phase`` is the same fact one level up: ``arg S``.  A real side's coupling is the real
    shadow of that phase (``0`` attract, ``pi`` detract); a side carrying a phase, not a
    polarity -- a U(1)-like lens -- puts its coupling in ``phase`` and legitimately has no
    sign at all.  Whether a coupling HAS a sign is a property of what the lens carries.

    ``tightness`` is the dominant-mode share of the cross-covariance spectrum of
    ``A~^H B~`` in ``(0, 1]``: ``-> 1`` the two sides are locked through ONE mode (tight,
    lawful); ``-> 0`` the shared variance is spread over many modes (loose, statistical,
    many mediators).  It is defined on every pair.

    Deterministic and backend-agnostic; ``O(T D^2 + D^3)``.
    """
    xp = _ns(a)
    if len(getattr(a, "shape", ())) != 2 or len(getattr(b, "shape", ())) != 2:
        raise ValueError("coupling expects two 2-D frames (T, D_a) and (T, D_b); got "
                         f"{getattr(a, 'shape', None)} and {getattr(b, 'shape', None)}")
    if int(a.shape[0]) != int(b.shape[0]):
        raise ValueError(
            f"the two sides must share the ORDERED axis: got T={int(a.shape[0])} and "
            f"T={int(b.shape[0])}. A side carried on its own ordered axis is CONTAINMENT, "
            f"a containment, and the row-to-row pairing this read measures needs one order.")
    T, Da, Db = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    cutoff = _norm_isf(float(far) / 2.0)                # two-sided: the sign may go either way
    if Da != Db:
        raise ValueError(
            f"coupling needs ONE shared basis: got D={Da} and D={Db}. A signed coupling is a "
            f"statement about the two sides' coordinates being the same coordinates; across "
            f"two bases the only basis-free statistic is non-negative and carries no exact "
            f"null, so there is no sign to report and none is invented.")
    empty = Coupling(z=0.0, sign=0, strength=0.0, phase=0.0, tightness=0.0, resolved=False,
                     cutoff=cutoff, n=T)
    if T < 3 or Da < 1 or Db < 1:
        return empty
    # A coordinate ONE side never observed is not a shared coordinate -- the same argument the
    # mismatched-basis refusal above makes.  Left in, it contributes nothing to the cross term but
    # still carries the OTHER side's norm, so the reported strength is divided by coordinates that
    # were never compared: with a third of one side's coordinates unobserved the strength read
    # about a tenth low.  (`z` and the sign are unaffected -- the statistic and its exact null move
    # together -- so this is the size of the effect, not the decision about it.)
    fa = np.asarray(_env.to_numpy(xp.isfinite(xp.abs(a)))).any(axis=0)
    fb = np.asarray(_env.to_numpy(xp.isfinite(xp.abs(b)))).any(axis=0)
    shared = fa & fb
    if not bool(shared.all()):
        if not bool(shared.any()):
            return empty                                   # no coordinate was measured on both
        idx = np.nonzero(shared)[0]
        a, b = a[:, idx], b[:, idx]
        Da = Db = int(idx.size)
    Ac, Bc = _centred(xp, a), _centred(xp, b)
    M = xp.conj(Ac).T @ Bc                              # (Da, Db) cross-covariance
    sv = _env.svdvals(xp, M)
    tot = float(_env.sum_ax(xp, sv ** 2))
    tightness = (float(sv[0]) ** 2 / tot) if tot > 0 else 0.0
    S = complex(_env.to_numpy(_env.sum_ax(xp, xp.conj(Ac) * Bc)))      # <A~, B~>_F
    Ar, Br = _real_embed(xp, Ac), _real_embed(xp, Bc)   # Re<.,.>_C == <.,.>_R on the embedding
    Ca, Cb = Ar.T @ Ar, Br.T @ Br                       # real, symmetric -> tr(Ca Cb) = sum(Ca*Cb)
    var = float(_env.sum_ax(xp, Ca * Cb)) / (T - 1)     # EXACT permutation variance of Re S
    nA, nB = float(_env.vnorm(xp, Ar)), float(_env.vnorm(xp, Br))
    z = (S.real / math.sqrt(var)) if var > 0 else 0.0
    resolved = bool(abs(z) > cutoff)
    strength = (S.real / (nA * nB)) if (nA > 0 and nB > 0) else 0.0    # signed cosine, |.| <= 1
    return Coupling(z=float(z),
                    sign=(0 if not resolved else (1 if S.real > 0 else (-1 if S.real < 0 else 0))),
                    strength=float(strength) if resolved else 0.0,
                    phase=float(math.atan2(S.imag, S.real)), tightness=float(tightness),
                    resolved=resolved, cutoff=float(cutoff), n=T)


# ══════════════════════════════════════════════════════════════════════════════
# Diffraction limit from a decay / correlation profile
# ══════════════════════════════════════════════════════════════════════════════

def decay(W, mask=None) -> np.ndarray:
    """The signal's OWN ordered-axis autocorrelation C(tau) -- its optical transfer
    function (OTF), by the Fourier-optics autocorrelation theorem (OTF = pupil
    autocorrelation).  Computed as the DIRECT lag average: C(tau) is the
    (1/T)-scaled tau-th diagonal sum of the ordered Gram G = Xc conj(Xc)^T (one
    matmul + one bincount, no Python loop -> backend-agnostic, on-GPU for torch).
    Wiener-Khinchin LICENSES this (C = F^{-1}{|F{s}|^2}); it is not a step this read takes.
    C(tau) is the autocovariance of ``W`` AS GIVEN -- one path, no inference about what the
    numbers mean.  Whether a record is a field or an intensity is a fact about the instrument that
    produced it, not a property of its samples: an intensity and an amplitude are both
    non-negative, and subtracting a baseline from an intensity makes it signed without changing
    any physics.  Reading the sign to choose between them made a pedestal of -0.001 move
    ``a_delta`` by a factor of 3.6.  For the INCOHERENT OTF of an amplitude record, square it and
    pass the intensity -- ``decay(W ** 2)`` -- which states the modelling step where the caller
    can see it.
    Returns C(tau), tau = 0..T-1: C(0) is the zero-lag power (the peak, = variance)
    and C decays as the ordered axis decorrelates.  Real- and complex-safe.  O(T^2 F)."""
    W = live_view(W, mask)                # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(W)
    is_c = xp.is_complex(W) if _env.is_torch(xp) else np.iscomplexobj(W)
    X = _env.asnum(W, complex=True) if is_c else _env.asnum(W)   # the record, as given
    T = int(X.shape[0])
    if T < 2:
        return _env.ones(xp, 1, ref=X)
    Xc = X - _env.mean0(xp, X)                               # connected (drop the disconnected mean)
    # A record that never varied leaves only the round-off of subtracting its own mean, and that
    # is not a decay -- squared into a Gram it would be read as one.  Same rule as
    # ``entropy.resolution_floor``: measured against the record's OWN scale, at the resolution the
    # arithmetic has, so it follows the data and the dtype.  The
    # mean is a sum of T terms, so the residual it leaves carries that sum's backward error, T*eps
    # [Higham 2002, Thm 3.5] -- the same bound ``Screen.certify`` measures a round trip against.
    span = float(_env.to_numpy(xp.max(xp.abs(X))))
    if float(_env.to_numpy(xp.max(xp.abs(Xc)))) <= span * T * macheps(xp, Xc):
        return _env.zeros(xp, T, ref=(X if _env.is_torch(xp) else None))
    # C(tau) = (1/T) sum_i G[i, i+tau], G = Xc conj(Xc)^T -- the biased autocovariance.
    G = xp.real(Xc.conj() @ Xc.T)                          # (T, T) real ordered Gram
    off = _env.arange_int(xp, T, ref=G)
    offset = off[None, :] - off[:, None]                   # (T, T): j - i (each entry's lag)
    sel = offset >= 0
    return xp.bincount(offset[sel], weights=G[sel], minlength=T) / T


def _integral_length(cn: np.ndarray) -> float:
    """xi = sum of the normalized decay over its first positive lobe (up to the
    first zero crossing) -- the classical integral correlation length."""
    below = np.where(cn <= 0.0)[0]
    end = int(below[0]) if below.size else len(cn)
    return float(np.sum(cn[:max(1, end)]))


@dataclass
class DiffractionLimit:
    """The diffraction limit a_delta from a decay profile (the temporal read)."""
    a_delta:      float   # ENTROPY-width limit 1/2^{H}, H = entropy of C^2 -- primary read
    xi:           float   # integral correlation length = sum of the normalized decay lobe
    a_delta_abbe: float   # INTEGRAL-length limit 1/xi -- the classical Abbe/Rayleigh read (secondary)
    H:            float   # Shannon entropy (bits) of the decay POWER profile C^2


@dataclass
class DecayScatter:
    """How much of a decay is the record's OWN sampling scatter, measured from the record.

    ``decay`` sums one biased autocovariance per feature channel, so the channels are independent
    replicates of the same decay and their spread IS the estimator's uncertainty -- no null, no
    model, nothing subtracted.  Compare the two shares:

      * ``noise_share`` far below ``tail_share``  -> the decay away from zero lag is structure the
        channels agree on, and ``a_delta`` can be read as the signal's.
      * ``noise_share`` close to ``tail_share``   -> that power is scatter the channels do NOT
        agree on.  It still widens the entropy, so ``a_delta`` reads a longer correlation than the
        record supports, and the cure is more channels.

    On an uncorrelated record the two coincide at every width (0.031 / 0.030 at F=16, 0.0022 /
    0.0020 at F=256): all of the tail is scatter.  On a record with a real correlation length they
    separate (0.78 / 0.0035 at F=256): the tail is structure."""
    noise_share: float   # sum_tau SE(tau)^2 / sum_tau C(tau)^2 -- what the channel scatter carries
    tail_share:  float   # sum_{tau>0} C(tau)^2 / sum_tau C(tau)^2 -- what sits away from zero lag
    channels:    int     # replicates the scatter was measured over (the live feature width)


def decay_scatter(W, mask=None) -> DecayScatter:
    """Measure the sampling scatter behind :func:`decay`, from the channels' own disagreement.

    ``decay`` is a sum over per-channel biased autocovariances, so this recomputes those channel
    terms (one FFT per channel -- cheaper than the lag sum ``decay`` itself takes) and reads their
    standard error.  ``sum_f C_f`` is ``decay(W)`` identically, which a test pins, so the two
    cannot drift apart.

    Returns shares of the decay's total power, both measured: see :class:`DecayScatter`."""
    L = live_view(W, mask)
    xp = _ns(L)
    is_c = xp.is_complex(L) if _env.is_torch(xp) else np.iscomplexobj(L)
    X = _env.asnum(L, complex=True) if is_c else _env.asnum(L)
    T = int(X.shape[0])
    F = int(X.shape[1]) if len(X.shape) > 1 else 1
    if T < 2 or F < 2:
        return DecayScatter(float("nan"), float("nan"), F)   # one channel has nothing to disagree
    Xc = X - _env.mean0(xp, X)
    n = 1
    while n < 2 * T:
        n <<= 1
    spec = xp.fft.fft(Xc, n=n, axis=0)                       # per channel, both backends
    Cf = xp.real(xp.fft.ifft(spec * xp.conj(spec), axis=0))[:T] / T    # (T, F) each channel's C
    C = _env.sum_ax(xp, Cf, 1) if hasattr(_env, "sum_ax") else Cf.sum(1)
    Cn = np.asarray(_env.to_numpy(C), float)
    Cfn = np.asarray(_env.to_numpy(Cf), float)
    total = float((Cn ** 2).sum())
    if not total > 0.0:
        return DecayScatter(float("nan"), float("nan"), F)
    # the channels are replicates of ONE decay: their standard error is the read's own uncertainty
    sem2 = Cfn.var(axis=1, ddof=1) * F                       # Var of the SUM = F * Var of a channel
    return DecayScatter(noise_share=float(sem2.sum() / total),
                        tail_share=float((Cn[1:] ** 2).sum() / total),
                        channels=F)


def diffraction_limit(profile) -> DiffractionLimit:
    """The diffraction limit a_delta from a 1-D decay ``profile`` C(tau), read from
    the ENTROPY WIDTH of the decay (see the module notes for the Wiener-Khinchin ->
    OTF -> Abbe chain):

        p(tau) = C(tau)^2 / sum C^2      (power-weighted distribution over lag)
        H      = -sum p log2 p           (Shannon entropy of the decay)
        a_delta = 1 / 2^H                (noise-robust; -> classical xi for exp decay)

    No exp(-a t) grid, no fitted rate, no peak search.  Also returns the classical
    Abbe secondary read a_delta_abbe = 1/xi, with xi = sum_tau C(tau)/C(0) over the
    positive lobe (the integral correlation length).  The independent SPECTRAL
    cross-check is ``mercer_certificate``.

    How many channels the read wants.  ``C`` is averaged over the feature axis, so the sampling
    noise left in its tail falls as 1/F.  That noise is spread across lags, it widens the entropy,
    and a wider entropy is a longer correlation -- so a narrow record OVERSTATES the correlation
    length, and does so by a lot.  On an uncorrelated record, where the answer is exactly 1, the
    read goes 0.43, 0.70, 0.90, 0.97, 0.99 at F = 4, 16, 64, 256, 1024; a correlated record
    converges the same way from the same side.  It is consistent -- more channels, closer -- and
    the residual is the entropy estimator's own small-sample bias, not a property of the signal.
    Nothing here removes it: what the noise contributes could only be subtracted by assuming what
    the decay would have been, which is a model, and this read reports the decay it measured."""
    c = np.asarray(_env.to_numpy(profile), float)
    if c.size < 2 or not np.any(c):
        return DiffractionLimit(0.0, 0.0, 0.0, 0.0)
    H = _shannon(c ** 2)                                   # entropy of the power decay (no tail fit)
    a_delta = 1.0 / (2.0 ** H)
    c0 = c[0] if c[0] != 0 else (float(np.abs(c).max()) or 1.0)
    xi = _integral_length(c / c0)
    a_delta_abbe = 1.0 / xi if xi > 0 else float("inf")
    return DiffractionLimit(a_delta=a_delta, xi=xi, a_delta_abbe=a_delta_abbe, H=H)


@dataclass
class MercerCertificate:
    """The model-free Mercer certificate: a_delta read TWO independent ways -- their
    ratio is fixed (the two widths coincide up to a constant shape factor, not 1:1; see
    mercer_certificate())."""
    a_delta_temporal: float   # 1/2^{H}, H = entropy of C^2 (the decay's entropy width)
    a_delta_spectral: float   # 2^{H(lambda)}/T from the stationary eigenspectrum {lambda_k}
    ratio:            float   # a_delta_spectral / a_delta_temporal (~const if stationary/clean)
    n_dof:            float   # 2^{H(lambda)} -- effective degrees of freedom (KL/PCA mode count)


def mercer_certificate(W, mask=None) -> MercerCertificate:
    """Certify the diffraction limit by MERCER: read a_delta the temporal way (the
    entropy width of the decay C) and the spectral way (the entropy width of the
    STATIONARY correlation operator's eigenspectrum {lambda_k}), and check their RATIO
    is constant.  Mercer's theorem makes {lambda_k} the spectrum of the kernel C(t-t'),
    so the two widths track each other up to a fixed shape factor (their ratio is
    constant, ~3-5 for AR(1), not 1) -- a model-free internal certificate: a stable
    ``ratio`` (not any particular value) validates the read; a departure says the
    signal isn't stationary/clean (not that a parameter was tuned).

    The stationary operator is the Toeplitz matrix built from the autocovariance C
    (NOT the rank-limited F-sample correlation used by phi_T).  O(T^2) memory +
    O(T^3) eig -- the validation, not the hot path."""
    c = np.asarray(_env.to_numpy(decay(W, mask)))
    T = c.size
    a_delta_temporal = (1.0 / (2.0 ** _shannon(c ** 2))) if (T >= 2 and np.any(c)) else 0.0
    if T < 2 or not np.any(c):
        return MercerCertificate(a_delta_temporal, 0.0, float("nan"), 0.0)
    idx = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    # The symmetric Toeplitz operator built from C.  LEMMA (Bochner / biased ACF):
    # because ``decay`` is the BIASED autocovariance (normalised by T, not T-tau) and
    # a sum over feature channels of per-channel biased ACFs, its Toeplitz matrix is
    # positive-SEMIdefinite -- so the clip below only removes float round-off at 0.
    lam = np.clip(np.linalg.eigvalsh(c[idx]), 0.0, None)   # stationary (Toeplitz) eigenspectrum
    if lam.sum() <= 0:
        return MercerCertificate(a_delta_temporal, 0.0, float("nan"), 0.0)
    n_dof = 2.0 ** _shannon(lam)                           # KL/PCA effective mode count
    a_delta_spectral = n_dof / T
    ratio = (a_delta_spectral / a_delta_temporal) if a_delta_temporal > 0 else float("nan")
    return MercerCertificate(a_delta_temporal, a_delta_spectral, ratio, n_dof)


def rayleigh_shape_factor(profile) -> float:
    """The Rayleigh SHAPE FACTOR g = xi * a_delta (the paper's "shape factor g", section 10)
    -- the integral correlation length times the entropy-width diffraction limit.  A scale-
    invariant, dimensionless shape functional (~O(1)) that reports the decay profile's SHAPE,
    from two SAME-domain (lag) length scales.  DISTINCT from ``shape_factor`` below, which is
    the Abbe resolution factor a_delta/phi_F.  It is not a Heisenberg-Gabor time-bandwidth
    product (which would need the Fourier-conjugate frequency width, a transform Entroptics
    does not run) and carries no uncertainty lower bound; it is the entropy-native shape read.
    ``profile``: a 1-D decay C(tau)."""
    dl = diffraction_limit(profile)
    return float(dl.xi * dl.a_delta)


def fresnel_number(W, window, mask=None) -> float:
    """FRESNEL number N_F ~ window * phi_T -- the near/far-field (UV/IR) coordinate.
    N_F >> 1 near field (geometric); N_F ~ 1 the focus; N_F << 1 far field
    (diffraction).  ``window`` is the probe extent in cells."""
    return float(window * phi_T(W, mask))


def shape_factor(W, profile, mask=None) -> float:
    """The Abbe RESOLUTION FACTOR c = a_delta / phi_F (Rayleigh / Abbe:
    resolution = factor / aperture) -- the screen's own "1.22", read per-signal, not a
    universal constant.  DISTINCT from ``rayleigh_shape_factor`` (the Rayleigh shape factor
    g = xi * a_delta of section 10); this is the a_delta-to-aperture ratio.  ``profile``: a
    1-D decay C(tau)."""
    pf = phi_F(W, mask)
    ad = diffraction_limit(np.asarray(profile, float)).a_delta
    return float(ad / pf) if pf > 0 else float("nan")


def assemble_optics(aT: AxisRead, aF: AxisRead, sp: SpectralOptics,
                    dl: DiffractionLimit, *, phi_val: float, strehl_val: float,
                    focus: float, intensity: float,
                    at_diffraction_limit: bool) -> dict:
    """THE canonical optics dict -- the single schema used by both ``optics(W)`` and
    ``Aperture.optics()`` (so the two can never drift).  Assembles precomputed
    sub-reads; every value is a plain Python scalar (backend-independent)."""
    mag = 1.0 / phi_val if phi_val > 0 else float("inf")
    return dict(
        # per-axis (ordered T, feature F)
        H_T=aT.H, n_T=aT.n, delta_T=aT.delta, phi_T=aT.phi, sigma_T=aT.sigma,
        H_F=aF.H, n_F=aF.n, delta_F=aF.delta, phi_F=aF.phi, sigma_F=aF.sigma,
        # combined screen area
        phi=phi_val, magnification=mag,
        etendue=aF.phi * aT.phi, space_bandwidth=int(aF.n * aT.n), strehl=strehl_val,
        # correlation spectrum / propagation
        contrast=sp.contrast, top_share=sp.top_share, resolved_modes=sp.resolved_modes,
        noise_floor=sp.noise_floor, attenuation=sp.attenuation, phase=sp.phase,
        dispersion=sp.dispersion, resolved_power=sp.resolved_power, dominance=sp.dominance,
        # concentration / focus
        focus=focus, intensity=intensity,
        # diffraction limit (entropy width + Abbe integral)
        a_delta=dl.a_delta, a_delta_abbe=dl.a_delta_abbe, correlation_length=dl.xi,
        decay_entropy=dl.H,
        rayleigh_shape_factor=dl.xi * dl.a_delta,
        shape_factor=(dl.a_delta / aF.phi if aF.phi > 0 else float("nan")),
        at_diffraction_limit=at_diffraction_limit,
    )


def optics(W, mask=None) -> dict:
    """The full, fully-INTRINSIC optical read off ONE screen: the per-axis reads
    (H/n/delta/phi/sigma for T and F), the combined screen-area values (phi,
    magnification, etendue, space_bandwidth, strehl), the correlation-spectrum
    optics (contrast / attenuation / phase / dispersion), the concentration/focus,
    and the diffraction limit derived from the signal's OWN decay (no external
    input).  Backend-agnostic.  The Mercer spectral certificate is the separate
    (heavier) ``mercer_certificate``."""
    if not hasattr(W, "shape"):
        W = np.asarray(W)                                # accept lists; arrays/tensors pass through
    g = geometry(W, mask)                                # one geometry read for both axes
    L = live_view(W, mask)                               # what was measured, once
    ev0, ev1 = axis_spectrum(L, 0), axis_spectrum(L, 1)  # one eigendecomposition per axis
    aT = axis_read(W, 0, mask, geom=g, evals=ev0)
    aF = axis_read(W, 1, mask, geom=g, evals=ev1)
    sp = spectral_optics(W, mask)
    dl = diffraction_limit(decay(W, mask))
    cn = concentration(W, mask)
    p = phi(W, mask)                                     # one whole-screen SVD
    mag = 1.0 / p if p > 0 else float("inf")
    return assemble_optics(
        aT, aF, sp, dl, phi_val=p, strehl_val=strehl(W, mask, evals=ev0),
        focus=cn.focus, intensity=cn.intensity,
        at_diffraction_limit=bool(abs(mag - 1.0) < 1e-9))


# ══════════════════════════════════════════════════════════════════════════════
# Structure vs observation window (the multi-scale aperture read)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScaleProfile:
    """Structure as a function of observation window (see scale_profile()) -- the zoom-by-density
    read.  Per-window arrays (aligned to ``windows``): ``K_signal`` (resolved-mode count),
    ``contrast`` (top singular value / noise floor, >1 => structure), ``coherence`` (ordered-axis
    z-score), ``a_delta`` (diffraction limit), ``phi_T`` (ordered-axis fill).  Scalars:
    ``resolved_window`` / ``dominant_window`` / ``transitions``."""
    windows:         np.ndarray   # ordered-axis window lengths swept, in CELLS (samples)
    K_signal:        np.ndarray   # resolved signal dimension per window
    contrast:        np.ndarray   # top singular value / noise floor per window (>1 => structure)
    coherence:       np.ndarray   # ordered-axis coherence z-score per window
    a_delta:         np.ndarray   # diffraction limit (entropy width) per window
    phi_T:           np.ndarray   # ordered-axis fill fraction per window
    resolved_window: int          # smallest window with K_signal >= 1, in cells (0 if none)
    dominant_window: int          # window of maximal ordered coherence, in cells (0 if none)
    transitions:    np.ndarray    # window lengths where K_signal changes from the previous window


def scale_profile(W, windows=None, *, mask=None, far: float = 0.05, null=None,
                  seed: int = 0) -> ScaleProfile:
    """STRUCTURE vs OBSERVATION WINDOW -- read the signal's structure as a function
    of how much of the ordered axis you look at (resolution vs aperture size).  Sweep
    TRAILING ordered-axis windows of increasing length; for each, read the screen and
    record K_signal, the ordered coherence z-score, the diffraction limit a_delta, and
    the ordered-axis fill phi_T.

    Domain-agnostic: ``windows`` are ORDERED-AXIS CELLS (samples), NOT seconds -- a
    caller maps cells to its own units.  ``resolved_window`` is the smallest window at
    which a signal mode first stands above the noise floor; ``dominant_window`` is the
    window of maximal ordered coherence (the scale at which structure is strongest);
    ``transitions`` are the windows where the resolved-mode count changes.  Deterministic
    and backend-agnostic.  ``windows`` defaults to a ~log-spaced sweep up to T.

    ``null`` / ``far`` are the reader's own floor and level, threaded to every windowed
    ``Projection``: ``K_signal`` counts modes against that floor and ``contrast`` divides
    by it, so a caller-supplied provider that did not reach here would be silently
    replaced by the derived ``mp`` default in both.  ``mask`` is windowed alongside ``W``."""
    if not hasattr(W, "shape"):
        W = np.asarray(W)
    if mask is not None and not hasattr(mask, "shape"):
        mask = np.asarray(mask)
    T = int(W.shape[0])
    if windows is None:
        lo = min(8, T)
        windows = np.unique(np.round(np.geomspace(lo, max(lo, T), 12)).astype(int))
    windows = np.asarray([int(w) for w in windows if 2 <= int(w) <= T], dtype=int)
    if windows.size == 0:
        z = np.zeros(0)
        return ScaleProfile(windows=windows, K_signal=z.astype(int), contrast=z, coherence=z,
                            a_delta=z, phi_T=z, resolved_window=0, dominant_window=0,
                            transitions=windows)
    Ks, cts, cohs, ads, pfs = [], [], [], [], []
    for w in windows:
        Wi = W[T - int(w):]                              # trailing window of length w
        Mi = None if mask is None else mask[T - int(w):]
        sc = Projection(Wi, mask=Mi, far=far, null=null, seed=seed)
        Ks.append(int(sc.K_signal))
        cts.append(float(sc.sigma_top / sc.noise_floor) if sc.noise_floor > 0 else 0.0)
        cohs.append(float(sc.coherence))
        ads.append(float(diffraction_limit(decay(Wi, Mi)).a_delta))
        pfs.append(float(phi_T(Wi, Mi)))
    Ks = np.asarray(Ks, int)
    cts = np.asarray(cts, float); cohs = np.asarray(cohs, float)
    ads = np.asarray(ads, float); pfs = np.asarray(pfs, float)
    resolved = windows[Ks >= 1]
    trans_mask = np.concatenate([[False], np.diff(Ks) != 0])
    return ScaleProfile(
        windows=windows, K_signal=Ks, contrast=cts, coherence=cohs, a_delta=ads, phi_T=pfs,
        resolved_window=int(resolved[0]) if resolved.size else 0,
        dominant_window=int(windows[int(np.argmax(cohs))]),
        transitions=windows[trans_mask])


__all__ = [
    # combined scale
    "phi", "magnification", "scale_duality",
    # per-axis
    "phi_T", "phi_F", "sigma_T", "sigma_F", "AxisRead", "axis_read", "axis_spectrum",
    # combined screen area
    "etendue", "strehl", "space_bandwidth",
    # spectrum / propagation
    "spectral_optics", "SpectralOptics",
    "attenuation_interval", "CertifiedInterval", "concentration_band",
    "resolved_dimension_interval", "CertifiedCount", "SpectralAccumulator",
    "concentration", "Concentration",
    # coupling between two sides of a screen (signed, exact-permutation null)
    "coupling", "Coupling",
    # decay (OTF) + diffraction limit + Mercer certificate
    "decay", "diffraction_limit", "DiffractionLimit",
    "mercer_certificate", "MercerCertificate",
    "rayleigh_shape_factor", "fresnel_number", "shape_factor",
    "optics", "assemble_optics",
    # structure vs observation window
    "scale_profile", "ScaleProfile",
    # re-exports
    "geometry", "Projection", "ProjectionRead", "read",
]
