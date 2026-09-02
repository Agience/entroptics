"""cosmology.py -- the FRB propagation physics, driven by the physics-free entroptics core.

The interstellar/intergalactic medium reshapes a burst by a stack of known transforms, each with a
single free parameter that the entroptics reads fix by maximizing the resolved concentration:

    dispersion   tau(f) = K_DM * DM * f^-2    -> DM    from the leading-mode contrast
    Faraday      d(psi) = RM * lambda^2       -> RM    from the derotated polarized amplitude
    scattering   * exp(-t/tau_s), tau_s~f^-4  -> tau_s from the tail decay rate (per aperture band)
    instrument/RFI (persistent narrowband)    -> removed by the phi_F>phi_T geometric filter

    r = unwind(I, freqs, dt)                    # intensity: DM + scattering + persistent-RFI clean
    r = unwind(I, freqs, dt, Q=Q, U=U)          # + Faraday derotation, source-frame polarization
    r.source, r.dm, r.rm, r.tau_scatter, r.pol_fraction
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from entroptics import Screen, Aperture, sweep

K_DM = 4.148808e3        # dispersion constant, s * MHz^2 * cm^3 / pc
C_LIGHT = 299792458.0    # m/s (for Faraday lambda^2)


# ── dispersion: DM from the leading-mode contrast ─────────────────────────────────────────────
def _shifts(freqs: np.ndarray, dt: float, dm: float) -> np.ndarray:
    """Integer per-channel sample delays for a dispersion measure ``dm`` (pc/cm^3), referenced to
    the top of the band (the highest frequency, which is undelayed)."""
    freqs = np.asarray(freqs, dtype=float)
    fref = float(freqs.max())
    return np.round(K_DM * dm * (freqs ** -2 - fref ** -2) / dt).astype(np.int64)


def dedisperse(W: np.ndarray, freqs: np.ndarray, dt: float, dm: float) -> np.ndarray:
    """Remove a dispersion of ``dm`` from waterfall ``W`` (time, freq): advance each channel by its
    delay tau(f).  One vectorised circular gather -- no per-channel loop."""
    T, F = W.shape
    sh = _shifts(freqs, dt, dm)                                   # (F,)
    idx = (np.arange(T)[:, None] + sh[None, :]) % T              # (T, F)
    return W[idx, np.arange(F)[None, :]]


def _whiten(W: np.ndarray) -> np.ndarray:
    """Per-channel robust whiten (subtract median, divide by MAD) -- the essence of the library
    normalize, cheap enough to run inside the sweep."""
    m = np.median(W, axis=0, keepdims=True)
    s = np.median(np.abs(W - m), axis=0, keepdims=True) * 1.4826
    return (W - m) / np.where(s > 0, s, 1.0)


def sigma_top(W: np.ndarray, iters: int = 40, tol: float = 1e-6) -> float:
    """Top singular value of the whitened waterfall -- the CONCENTRATION objective.  Power iteration
    on the operator, never forming the Gram or the full SVD: each step is O(T*F)."""
    Wn = _whiten(W)
    T = Wn.shape[0]
    v = np.ones(T) / np.sqrt(T)
    s_prev = 0.0
    for _ in range(iters):
        w = Wn.T @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            return 0.0
        w /= nw
        v = Wn @ w
        s = np.linalg.norm(v)
        if s == 0:
            return 0.0
        v /= s
        if abs(s - s_prev) <= tol * s:
            break
        s_prev = s
    return float(s)


def find_dm(W: np.ndarray, freqs: np.ndarray, dt: float, *,
            dm_lo: float = -3.0, dm_hi: float = 3.0, n_coarse: int = 25):
    """The DM offset (pc/cm^3) that maximises the entroptics concentration objective ``sigma_top``.

    A coarse grid brackets the peak, then a bounded Brent refinement finds it to sub-grid precision.
    Returns ``(dm, objective, n_evals)``."""
    grid = np.linspace(dm_lo, dm_hi, n_coarse)
    obj = np.array([sigma_top(dedisperse(W, freqs, dt, d)) for d in grid])
    i = int(np.argmax(obj))
    a = grid[max(0, i - 1)]
    b = grid[min(n_coarse - 1, i + 1)]
    n_eval = [n_coarse]

    def neg(d):
        n_eval[0] += 1
        return -sigma_top(dedisperse(W, freqs, dt, float(d)))

    if b > a:
        res = minimize_scalar(neg, bounds=(a, b), method="bounded",
                              options={"xatol": (dm_hi - dm_lo) / n_coarse / 40})
        dm, best = float(res.x), -float(res.fun)
        if obj[i] >= best:                       # keep the grid point if Brent didn't improve
            dm, best = float(grid[i]), float(obj[i])
    else:
        dm, best = float(grid[i]), float(obj[i])
    return dm, best, n_eval[0]


# ── Faraday rotation: the SAME inversion, on the polarization vector ───────────────────────────
def _lambda2(freqs: np.ndarray) -> np.ndarray:
    """Wavelength-squared (m^2) for channel centre frequencies in MHz."""
    return (C_LIGHT / (np.asarray(freqs, dtype=float) * 1e6)) ** 2


def polarized_amplitude(Qspec: np.ndarray, Uspec: np.ndarray, freqs: np.ndarray, rm: float) -> float:
    """The coherently-derotated linear-polarization amplitude |sum_f (Q+iU) exp(-2i*RM*lambda^2)| --
    the Faraday analog of the dispersion contrast: the polarized signal is maximally CONCENTRATED
    (adds in phase across the band) at the true RM.  ``Qspec``/``Uspec`` are the on-pulse Stokes
    spectra per channel."""
    l2 = _lambda2(freqs)
    P = np.asarray(Qspec, dtype=float) + 1j * np.asarray(Uspec, dtype=float)
    return float(abs(np.sum(P * np.exp(-2j * rm * (l2 - l2.mean())))))


def find_rm(Qspec: np.ndarray, Uspec: np.ndarray, freqs: np.ndarray, *,
            rm_lo: float = -1.5e5, rm_hi: float = 1.5e5, n_coarse: int = 1501):
    """The RM (rad/m^2) that MAXIMIZES the derotated polarized amplitude -- Faraday's ``find_dm``.
    Coarse Faraday-depth grid brackets the peak, bounded Brent refines it.  Returns
    ``(rm, amplitude, n_evals)``."""
    grid = np.linspace(rm_lo, rm_hi, n_coarse)
    obj = np.array([polarized_amplitude(Qspec, Uspec, freqs, r) for r in grid])
    i = int(np.argmax(obj))
    a, b = grid[max(0, i - 1)], grid[min(n_coarse - 1, i + 1)]
    n_eval = [n_coarse]

    def neg(r):
        n_eval[0] += 1
        return -polarized_amplitude(Qspec, Uspec, freqs, float(r))

    if b > a:
        res = minimize_scalar(neg, bounds=(a, b), method="bounded",
                              options={"xatol": (rm_hi - rm_lo) / n_coarse / 40})
        rm, best = float(res.x), -float(res.fun)
        if obj[i] >= best:
            rm, best = float(grid[i]), float(obj[i])
    else:
        rm, best = float(grid[i]), float(obj[i])
    return rm, best, n_eval[0]


def derotate(Q: np.ndarray, U: np.ndarray, freqs: np.ndarray, rm: float):
    """Remove a Faraday rotation of ``rm`` from Stokes Q, U (rotate each channel by -2*RM*lambda^2).
    Broadcasts a per-channel rotation over any leading (time) axis.  Returns ``(Q', U')``."""
    th = 2.0 * rm * (_lambda2(freqs) - _lambda2(freqs).mean())
    c, s = np.cos(th), np.sin(th)
    return Q * c + U * s, -Q * s + U * c


# ── scattering: f^-4 separation over the coherence-gated aperture sweep ────────────────────────
def scattering(bands, f_ref: float = 600.0, dt: float | None = None, use: str = "tau_decay"):
    """Separate scattering from intrinsic broadening using the per-band decay times tau(f).

    Each band (a coherent patch from ``entroptics.sweep``) carries a tail decay ``tau_decay`` in
    samples and a centre frequency ``f_center`` (mapped from its column span by the caller).  The
    measured tau(f) adds in quadrature: tau(f)^2 = tau_int^2 + (tau_ref * (f/f_ref)^-4)^2 -- a
    frequency-independent intrinsic term and an f^-4 scattering tail.  Least squares in
    [1, (f/f_ref)^-8] recovers the intrinsic term and the scattering ``tau_ref``.  All times are in
    SAMPLES; they come back in samples UNLESS ``dt`` is supplied (the only place physical time enters).
    The log-log ``slope`` is dimensionless (-> -4 pure scattering, -> 0 pure intrinsic)."""
    fc = np.array([b.get("f_center", np.nan) for b in bands], float)
    tau = np.array([b.get(use, np.nan) for b in bands], float)
    good = np.isfinite(tau) & (tau > 0) & np.isfinite(fc) & (fc > 0)
    fc, tau = fc[good], tau[good]
    if len(fc) < 3:
        return None
    slope = float(np.polyfit(np.log(fc), np.log(tau), 1)[0])
    A = np.vstack([np.ones_like(fc), (fc / f_ref) ** -8.0]).T
    coef, *_ = np.linalg.lstsq(A, tau ** 2, rcond=None)
    tau_int = float(np.sqrt(max(coef[0], 0.0)))
    tau_ref = float(np.sqrt(max(coef[1], 0.0)))
    scale = 1.0 if dt is None else float(dt)                  # physical units are opt-in
    return {"tau_scatter": tau_ref * scale, "tau_intrinsic": tau_int * scale, "slope": slope,
            "read": use, "units": "samples" if dt is None else "dt-units",
            "n_bands": int(len(fc)), "f_ref": f_ref}


# ── unwind: one call from a raw burst to its SOURCE FRAME ──────────────────────────────────────
@dataclass
class UnwindResult:
    """The source-frame burst and the channel parameters read back out of it."""
    source:        np.ndarray        # (T, F) dedispersed + derotated + RFI-cleaned intensity
    dm:            float             # dispersion offset removed (pc/cm^3, relative to the input)
    rm:            float | None      # Faraday RM removed (rad/m^2); None if no polarization given
    tau_scatter:   float | None      # scattering timescale (samples, or dt-units if scattering_dt set)
    tau_intrinsic: float | None      # intrinsic width separated from scattering (same units)
    pol_fraction:  float | None      # source-frame linear polarization L/I; None if no Q,U
    contrast:      float             # leading-mode contrast of the cleaned source frame
    coherence:     float             # ordered-axis coherence z of the source frame
    K_signal:      int               # resolved modes above the derived floor
    n_bands:       int               # coherent aperture bands used for the scattering fit
    scan:          dict              # the raw reads {dm_evals, rm_evals, scattering, ...}


def _on_off(prof: np.ndarray, half: int = 3, gap: int = 30, wide: int = 60):
    """On-pulse slice (peak +/- half) and an off-pulse boolean mask, from a 1-D profile."""
    pk = int(np.argmax(prof - np.median(prof)))
    on = slice(max(0, pk - half), pk + half + 1)
    off = np.ones(prof.size, bool)
    off[max(0, pk - wide):pk + wide + 1] = False
    if off.sum() < gap:                                    # tiny window -> fall back to the ends
        off = np.ones(prof.size, bool); off[on] = False
    return pk, on, off


STAGES = ("dispersion", "faraday", "scattering", "rfi")


def unwind(I: np.ndarray, freqs: np.ndarray, dt: float, *, Q: np.ndarray | None = None,
           U: np.ndarray | None = None, dm_search=(-3.0, 3.0), rm_search=(-1.5e5, 1.5e5),
           far: float = 0.05, patch: int = 1024, coherence: float = 3.0,
           scattering_dt: float | None = None, null="local", skip=()) -> UnwindResult:
    """Unwind a burst's propagation channel back toward the source frame.

    ``I`` is the intensity waterfall (time, freq); ``freqs`` the channel centres in MHz; ``dt`` the
    sample time in seconds (for the dispersion shift).  ``Q``, ``U`` (same shape as ``I``) enable
    Faraday derotation.  Reads run through the entroptics core; the four inverse transforms are
    applied here.  Times are in SAMPLES unless ``scattering_dt`` scales them.

    ``skip`` -- an iterable of stage names to BYPASS, for data a transform has already been applied
    to (e.g. CHIME baseband voltage arrives coherently dedispersed, so ``skip={"dispersion"}`` stops
    ``unwind`` re-dedispersing it).  Any of :data:`STAGES` = ``"dispersion" / "faraday" /
    "scattering" / "rfi"``.  A skipped dispersion holds DM = 0; a skipped rfi returns the projected
    image unfiltered.  Returns :class:`UnwindResult`."""
    skip = set(skip)
    unknown = skip - set(STAGES)
    if unknown:
        raise ValueError(f"unknown unwind stage(s) to skip: {sorted(unknown)}; valid: {STAGES}")
    I = np.nan_to_num(np.asarray(I, dtype=float))          # masked/dead cells -> 0 (feed the raw file in)

    # 1. DISPERSION -> DM from the leading-mode contrast, then dedisperse (unless already applied).
    if "dispersion" in skip:
        dm, dm_ev, Id = 0.0, 0, I
    else:
        dm, _, dm_ev = find_dm(I, freqs, dt, dm_lo=dm_search[0], dm_hi=dm_search[1])
        Id = dedisperse(I, freqs, dt, dm)
    prof = np.nansum(Id, 1)
    pk, on, off = _on_off(prof)

    # 2. FARADAY -> RM from the derotated polarized amplitude, then derotate (if Q, U given).
    rm = pol_fraction = None
    rm_ev = 0
    if Q is not None and U is not None and "faraday" not in skip:
        Qd = np.asarray(Q, float) if "dispersion" in skip else dedisperse(np.asarray(Q, float), freqs, dt, dm)
        Ud = np.asarray(U, float) if "dispersion" in skip else dedisperse(np.asarray(U, float), freqs, dt, dm)
        Qsp = Qd[on].mean(0) - Qd[off].mean(0)             # on-pulse Stokes spectra (off-pulse removed)
        Usp = Ud[on].mean(0) - Ud[off].mean(0)
        rm, _, rm_ev = find_rm(Qsp, Usp, freqs, rm_lo=rm_search[0], rm_hi=rm_search[1])
        Qr, Ur = derotate(Qd, Ud, freqs, rm)
        Isp = Id[on].mean(0) - Id[off].mean(0)
        Qb = Qr[on].mean(0) - Qr[off].mean(0)
        Ub = Ur[on].mean(0) - Ur[off].mean(0)
        denom = float(np.sum(Isp))
        pol_fraction = float(abs(np.sum(Qb + 1j * Ub)) / denom) if denom > 0 else float("nan")

    # 3. SCATTERING -> tau_s from the tail decay rate, per aperture band (coherence-gated sweep).
    if "scattering" in skip:
        bands, sc = [], None
    else:
        bands = sweep(Id, patch=patch, coherence=coherence, far=far, null=null)
        for b in bands:                                    # map each band's column span to a centre freq
            f0, f1 = b["span"]
            b["f_center"] = float(np.nanmean(freqs[f0:f1]))
        sc = scattering(bands, dt=scattering_dt) if len(bands) >= 3 else None

    # 4. INSTRUMENT/RFI -> the source-frame image (project onto the resolved, burst-like modes).
    if "rfi" in skip:
        s = Screen(Id, far=far)
        clean = Id
        info = {"contrast": float(s.sigma_top / s.noise_floor) if s.noise_floor > 0 else 0.0,
                "coherence": float(s.coherence), "K_signal": int(s.K_signal)}
    else:
        clean, info = Aperture(Id, window=None).extract(far=far)   # full-frame filter, through the front door

    return UnwindResult(
        source=clean, dm=float(dm), rm=rm,
        tau_scatter=(sc["tau_scatter"] if sc else None),
        tau_intrinsic=(sc["tau_intrinsic"] if sc else None),
        pol_fraction=pol_fraction,
        contrast=float(info["contrast"]), coherence=float(info["coherence"]),
        K_signal=int(info["K_signal"]), n_bands=len(bands),
        scan={"dm_evals": dm_ev, "rm_evals": rm_ev, "on_pulse": pk, "scattering": sc},
    )
