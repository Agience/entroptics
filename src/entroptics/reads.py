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
  etendue(W)         phi_F * phi_T           -- the conserved 2-D aperture area
  space_bandwidth(W) n_F * n_T               -- number of resolvable spots
  strehl(W)          lambda1/sum(lambda) of the ordered correlation (coherence)

Mode spectrum / propagation (from the correlation eigenspectrum):
  spectral_optics(W) contrast, top_share, resolved modes, noise floor, the
                     attenuation constant alpha, the phase constant beta, dispersion.
  attenuation_interval(W, band=...)   a certified interval for alpha.
  concentration_band(...)             the a-priori spectral-norm band from samples.
  concentration(rows)                 focus of a vector stack on its dominant axis.

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

Re-exported for convenience: geometry, Screen, ScreenRead, read.
"""
import math
from dataclasses import dataclass

import numpy as np

from . import environment as _env
from .environment import ns as _ns
from .entropy import geometry, shannon_bits as _shannon, live_view
from .screen import Screen, ScreenRead, read
from .null_providers import apply_floor as _apply_floor


# ══════════════════════════════════════════════════════════════════════════════
# Combined scale: the screen fill fraction phi and its magnification 1/phi
# ══════════════════════════════════════════════════════════════════════════════

def phi(M) -> float:
    """phi = 2^{H_sv}/N in (0,1]: the fraction of ACTIVE SVD modes of a screen
    block M (the Shannon entropy of its singular spectrum) -- the aperture's fill
    fraction, BOUNDED by construction.  phi -> 1 fully disordered (all modes
    active); phi -> 0 fully coherent (one mode)."""
    M = live_view(M)                      # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(M)
    S = _env.svdvals(xp, _env.asnum(M))   # singular values (real), on M's backend (GPU if torch)
    n = int(S.shape[0])
    if n == 0:
        return 0.0
    return float(2.0 ** _shannon(S ** 2) / n)


def magnification(M) -> float:
    """delta = 1/phi = N/2^{H_sv} in [1, inf): the reciprocal of the fill fraction
    -- the magnification / oversampling face.  =1 is the critically-sampled
    diffraction limit; -> inf is maximal magnification (one coherent mode)."""
    p = phi(M)
    return float(1.0 / p) if p > 0 else float("inf")


def scale_duality(M) -> dict:
    """Both faces of the scale off ONE screen: {phi, magnification,
    at_diffraction_limit} -- reciprocal, meeting at phi = magnification = 1."""
    p = phi(M)
    mag = 1.0 / p if p > 0 else float("inf")
    return dict(phi=p, magnification=mag, at_diffraction_limit=bool(abs(mag - 1.0) < 1e-9))


# ══════════════════════════════════════════════════════════════════════════════
# Per-axis reads: phi_a, sigma_a, and the AxisRead bundle
# ══════════════════════════════════════════════════════════════════════════════

def _corr_evals(M: np.ndarray) -> np.ndarray:
    """Correlation eigenvalues (descending, non-negative) of the COLUMNS of M as
    variables, using the ROWS of M as samples.  M shape (n_samples, n_vars).
    Complex-safe (Hermitian correlation).  Backend-agnostic (numpy or torch)."""
    M = live_view(M)                      # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(M)
    M = _env.asnum(M)
    Xc = M - _env.mean0(xp, M)
    C = Xc.conj().T @ Xc
    d = xp.sqrt(_env.clampmin(xp, xp.real(xp.diag(C)), 1e-30))
    C = C / xp.outer(d, d)
    ev = _env.clampmin(xp, xp.real(xp.linalg.eigvalsh(C)), 0.0)
    return _env.flip(xp, ev)              # descending


def axis_spectrum(W, axis: int):
    """Correlation eigenvalues (descending, non-negative) along ``axis`` (0 =
    ordered/T, 1 = feature/F) -- the per-axis correlation eigenspectrum every axis
    read is derived from.  Compute it ONCE and pass it to ``axis_read`` / ``strehl``
    to avoid recomputing the (up to O(len^3)) eigendecomposition.  Backend-agnostic."""
    M = W if axis == 1 else W.T          # columns become the axis variables (.T ok on torch 2-D)
    return _corr_evals(M)


def phi_T(W) -> float:
    """phi_T: the ORDERED-axis fill fraction = 2^{H_sv}/T over the ordered-axis
    correlation spectrum (the "time" aperture)."""
    ev = axis_spectrum(W, 0)
    return float(2.0 ** _shannon(ev) / int(W.shape[0]))


def phi_F(W) -> float:
    """phi_F: the FEATURE-axis fill fraction = 2^{H_sv}/F over the feature-axis
    correlation spectrum.  HIGH = many active feature modes (disorder);
    LOW = few active modes (long-range coherence)."""
    ev = axis_spectrum(W, 1)
    return float(2.0 ** _shannon(ev) / int(W.shape[1]))


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
    n:     int     # effective mode count = round(2^H)
    delta: float   # matched cell scale = len / 2^H (window / bin width)
    phi:   float   # fill fraction of the axis correlation spectrum (2^{H_sv}/len)
    sigma: float   # dominant-mode singular value of the axis correlation


def axis_read(W, axis: int, *, geom: dict | None = None, evals=None) -> AxisRead:
    """Bundle the per-axis reads (H, n, delta, phi, sigma) for ``axis``
    (0 = ordered/T, 1 = feature/F).  Pass a precomputed ``geom`` (from geometry())
    and/or ``evals`` (from axis_spectrum(W, axis)) to skip recomputing them."""
    g = geometry(W) if geom is None else geom
    if axis == 0:
        H, n, delta, length = g["H_T"], g["n_T"], g["delta_T"], int(W.shape[0])
    else:
        H, n, delta, length = g["H_F"], g["n_F"], g["delta_F"], int(W.shape[1])
    ev = axis_spectrum(W, axis) if evals is None else evals
    return AxisRead(H=float(H), n=int(n), delta=float(delta),
                    phi=float(2.0 ** _shannon(ev) / length),
                    sigma=float(ev[0]) ** 0.5 if int(ev.shape[0]) else 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Combined screen-area reads
# ══════════════════════════════════════════════════════════════════════════════

def etendue(W) -> float:
    """Optical ETENDUE = phi_F * phi_T: the joint 2-D aperture area (the
    phase-space volume the finite screen carries).  A conserved invariant of a
    lossless system (optics: A*Omega is conserved)."""
    return float(phi_F(W) * phi_T(W))


def strehl(W, *, evals=None) -> float:
    """STREHL ratio (coherence): lambda1/sum(lambda) of the ordered-axis
    correlation -- peak-mode power vs total.  S -> 1 perfectly coherent (one
    dominant mode); S -> 1/T disordered (power spread across modes).  Pass
    ``evals`` (from axis_spectrum(W, 0)) to reuse an already-computed spectrum."""
    ev = axis_spectrum(W, 0) if evals is None else evals
    xp = _ns(ev)
    tot = float(_env.sum_ax(xp, ev))
    return float(xp.max(ev)) / tot if tot > 0 else 0.0


def space_bandwidth(W) -> int:
    """SPACE-BANDWIDTH PRODUCT = number of RESOLVABLE spots of a 2-D screen =
    n_F * n_T (effective mode count per axis, from geometry).  The screen's
    information capacity; finite <=> band-limited."""
    g = geometry(W)
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
    # UPPER NOISE EDGE for the eigenvalues of a unit-diagonal CORRELATION matrix (N
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
    dispersion = float(_env.std0(xp, xp.log(above / edge))) if int(above.shape[0]) > 1 else 0.0
    resolved_power = float(_env.sum_ax(xp, above - edge)) if int(above.shape[0]) else 0.0
    dominance = min(1.0, max(0.0, (lam1 - 1.0) / (N - 1))) if N > 1 else 0.0
    return SpectralOptics(
        contrast=(lam1 / edge) if edge > 0 else 0.0,
        top_share=(lam1 / total) if total > 1e-12 else 0.0,
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
    Xf = _env.asnum(data)
    if mask is not None:
        Xf = np.where(np.asarray(mask, bool), Xf, np.nan)
    mu = _env.nanmean0(xp, Xf)
    diff = Xf - mu
    Xc = xp.where(xp.isnan(diff), xp.zeros_like(diff), diff)   # de-mean; invalid -> 0
    Cov = Xc.conj().T @ Xc
    return _spectral_from_cov(xp, Cov, T, N, null=null, far=far, data=Xc, seed=seed)


def spectral_batch(frames, *, null=None, far: float = 0.05, seed: int = 0) -> list:
    """Read :func:`spectral_optics` for a BATCH of same-shape 2-D frames in one pass -- the
    column de-mean and the ``(N, N)`` covariances are formed batched (``(B, T, N)`` ->
    ``(B, N, N)``), then EACH frame's eigenspectrum + floor + optics is assembled by the SAME
    :func:`_spectral_from_cov` the per-frame ``spectral_optics`` calls.  So the result is
    **bit-identical** to ``[spectral_optics(f, null=null, far=far, seed=seed) for f in frames]``,
    while amortizing the per-frame de-mean / covariance / call overhead (the ensemble lever for
    the correlation read on small planes. ``contrast``).  A frame that is
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
    band propagates EXACTLY to [alpha_lo, alpha_hi].  A certified positive
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
            from .entropy import normalize                 # library per-channel MAD whitening
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


def concentration(vectors: np.ndarray, *, normalize: bool = True) -> Concentration:
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
    vectors = live_view(vectors)          # ignore fully-dead rows/cols; clean scattered gaps
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
# Diffraction limit from a decay / correlation profile
# ══════════════════════════════════════════════════════════════════════════════

def decay(W) -> np.ndarray:
    """The signal's OWN ordered-axis autocorrelation C(tau) -- its optical transfer
    function (OTF), by the Fourier-optics autocorrelation theorem (OTF = pupil
    autocorrelation).  Computed as the DIRECT lag average: C(tau) is the
    (1/T)-scaled tau-th diagonal sum of the ordered Gram G = Xc conj(Xc)^T (one
    matmul + one bincount, no Python loop -> backend-agnostic, on-GPU for torch).
    Wiener-Khinchin LICENSES this (C = F^{-1}{|F{s}|^2}); it is not a step we run.
      * complex or signed W -> COHERENT OTF: autocorrelate the (centered) field W.
      * non-negative W      -> INCOHERENT OTF: autocorrelate the intensity |W|^2.
    Returns C(tau), tau = 0..T-1: C(0) is the zero-lag power (the peak, = variance)
    and C decays as the ordered axis decorrelates.  Real- and complex-safe.  O(T^2 F)."""
    W = live_view(W)                      # ignore fully-dead rows/cols; clean scattered gaps
    xp = _ns(W)
    is_c = xp.is_complex(W) if _env.is_torch(xp) else np.iscomplexobj(W)
    if is_c:
        X = _env.asnum(W, complex=True)                     # coherent: complex field
    elif bool((W < 0).any()):
        X = _env.asnum(W)                                    # coherent: signed field
    else:
        X = _env.asnum(W) ** 2                               # incoherent: intensity |W|^2
    T = int(X.shape[0])
    if T < 2:
        return _env.ones(xp, 1, ref=X)
    Xc = X - _env.mean0(xp, X)                               # connected (drop the disconnected mean)
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
    cross-check is ``mercer_certificate``."""
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


def mercer_certificate(W) -> MercerCertificate:
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
    c = np.asarray(_env.to_numpy(decay(W)))
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


def fresnel_number(W, window) -> float:
    """FRESNEL number N_F ~ window * phi_T -- the near/far-field (UV/IR) coordinate.
    N_F >> 1 near field (geometric); N_F ~ 1 the focus; N_F << 1 far field
    (diffraction).  ``window`` is the probe extent in cells."""
    return float(window * phi_T(W))


def shape_factor(W, profile) -> float:
    """The Abbe RESOLUTION FACTOR c = a_delta / phi_F (Rayleigh / Abbe:
    resolution = factor / aperture) -- the screen's own "1.22", read per-signal, not a
    universal constant.  DISTINCT from ``rayleigh_shape_factor`` (the Rayleigh shape factor
    g = xi * a_delta of section 10); this is the a_delta-to-aperture ratio.  ``profile``: a
    1-D decay C(tau)."""
    pf = phi_F(W)
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


def optics(W) -> dict:
    """The full, fully-INTRINSIC optical read off ONE screen: the per-axis reads
    (H/n/delta/phi/sigma for T and F), the combined screen-area values (phi,
    magnification, etendue, space_bandwidth, strehl), the correlation-spectrum
    optics (contrast / attenuation / phase / dispersion), the concentration/focus,
    and the diffraction limit derived from the signal's OWN decay (no external
    input).  Backend-agnostic.  The Mercer spectral certificate is the separate
    (heavier) ``mercer_certificate``."""
    if not hasattr(W, "shape"):
        W = np.asarray(W)                                # accept lists; arrays/tensors pass through
    g = geometry(W)                                      # one geometry read for both axes
    ev0, ev1 = axis_spectrum(W, 0), axis_spectrum(W, 1)  # one eigendecomposition per axis
    aT = axis_read(W, 0, geom=g, evals=ev0)
    aF = axis_read(W, 1, geom=g, evals=ev1)
    sp = spectral_optics(W)
    dl = diffraction_limit(decay(W))
    cn = concentration(W)
    p = phi(W)                                           # one whole-screen SVD
    mag = 1.0 / p if p > 0 else float("inf")
    return assemble_optics(
        aT, aF, sp, dl, phi_val=p, strehl_val=strehl(W, evals=ev0),
        focus=cn.focus, intensity=cn.intensity,
        at_diffraction_limit=bool(abs(mag - 1.0) < 1e-9))


# ══════════════════════════════════════════════════════════════════════════════
# Structure vs observation window (the multi-scale aperture read)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScaleProfile:
    """Structure as a function of observation window (see scale_profile())."""
    windows:         np.ndarray   # ordered-axis window lengths swept, in CELLS (samples)
    K_signal:        np.ndarray   # resolved signal dimension per window
    coherence:       np.ndarray   # ordered-axis coherence z-score per window
    a_delta:         np.ndarray   # diffraction limit (entropy width) per window
    phi_T:           np.ndarray   # ordered-axis fill fraction per window
    resolved_window: int          # smallest window with K_signal >= 1, in cells (0 if none)
    dominant_window: int          # window of maximal ordered coherence, in cells (0 if none)
    transitions:    np.ndarray    # window lengths where K_signal changes from the previous window


def scale_profile(W, windows=None) -> ScaleProfile:
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
    and backend-agnostic.  ``windows`` defaults to a ~log-spaced sweep up to T."""
    if not hasattr(W, "shape"):
        W = np.asarray(W)
    T = int(W.shape[0])
    if windows is None:
        lo = min(8, T)
        windows = np.unique(np.round(np.geomspace(lo, max(lo, T), 12)).astype(int))
    windows = np.asarray([int(w) for w in windows if 2 <= int(w) <= T], dtype=int)
    if windows.size == 0:
        z = np.zeros(0)
        return ScaleProfile(windows=windows, K_signal=z.astype(int), coherence=z,
                            a_delta=z, phi_T=z, resolved_window=0, dominant_window=0,
                            transitions=windows)
    Ks, cohs, ads, pfs = [], [], [], []
    for w in windows:
        Wi = W[T - int(w):]                              # trailing window of length w
        sc = Screen(Wi)
        Ks.append(int(sc.K_signal))
        cohs.append(float(sc.coherence))
        ads.append(float(diffraction_limit(decay(Wi)).a_delta))
        pfs.append(float(phi_T(Wi)))
    Ks = np.asarray(Ks, int)
    cohs = np.asarray(cohs, float); ads = np.asarray(ads, float); pfs = np.asarray(pfs, float)
    resolved = windows[Ks >= 1]
    trans_mask = np.concatenate([[False], np.diff(Ks) != 0])
    return ScaleProfile(
        windows=windows, K_signal=Ks, coherence=cohs, a_delta=ads, phi_T=pfs,
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
    # decay (OTF) + diffraction limit + Mercer certificate
    "decay", "diffraction_limit", "DiffractionLimit",
    "mercer_certificate", "MercerCertificate",
    "rayleigh_shape_factor", "fresnel_number", "shape_factor",
    "optics", "assemble_optics",
    # structure vs observation window
    "scale_profile", "ScaleProfile",
    # re-exports
    "geometry", "Screen", "ScreenRead", "read",
]
