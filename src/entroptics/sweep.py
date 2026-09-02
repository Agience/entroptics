"""sweep.py -- the entroptics APERTURE swept over an unbounded ordered x feature field.

Entroptics reads a FINITE aperture, not an infinite field.  Forcing the full width through one
decomposition is both wrong (a faint spread signal is noise-dominated across the whole width) and
unnecessary.  Instead: set the aperture to a fixed CAPACITY (a patch that keeps each read cheap),
then SWEEP it where there is COHERENCE -- the coherence read gates out noise-only patches, and the
full read runs only on signal-bearing ones.

Coherence is the gate (not contrast): on a spread, band-limited signal it separates signal patches
(coherence high, flat across amplitude) from persistent contaminants (~ 0.8) and noise (~ 0) cleanly
and AMPLITUDE-INDEPENDENTLY, whereas contrast fades with the signal's strength.  (Contrast is then a
per-patch concentration read; phi_F > phi_T flags any persistent mode that slips through.)

TIME IS FLUID: every width the aperture reads is in ABSTRACT window samples (dimensionless).  The
aperture has no physical clock and no physical frequency -- ``sweep`` returns each coherent band's
column ``span`` (index range) and its reads; the caller maps spans to physical units if it wants
them.

    bands = sweep(W)        # per-coherent-band reads: span (columns), width & tau_decay (samples)
"""
from __future__ import annotations

import numpy as np

from .screen import Screen
from .entropy import shannon_bits
from .null_providers import reference_null


def _entropy_width(prof: np.ndarray, pk: int, half: int = 28) -> float:
    """Entropy width (participation length) of the on-pulse profile, in SAMPLES (dimensionless):
    2^H of the on-pulse power -- the aperture's own window units."""
    seg = np.clip(prof[max(0, pk - half):pk + half], 0.0, None)
    s = seg.sum()
    if s <= 0:
        return float("nan")
    return float(2 ** shannon_bits(seg / s))


def _tail_decay(prof: np.ndarray, pk: int, span: int = 80, thr_frac: float = 0.12) -> float:
    """The exponential decay timescale of the post-peak TAIL, in SAMPLES.  A pure tail is
    exp(-t/tau), so a log-linear fit of the post-peak profile above a fraction ``thr_frac`` of the
    peak gives slope = -1/tau.  This reads the tail for what it is -- a decay rate (the entroptics
    ``attenuation`` -log|mu| of an exponential) -- not a width proxy, so it recovers a power-law
    tail far better than 2^H (which has a participation-length floor)."""
    tail = prof[pk:pk + span]
    if tail.size < 4 or tail[0] <= 0:
        return float("nan")
    m = tail > thr_frac * tail[0]
    if m.sum() < 4:
        return float("nan")
    tt = np.arange(tail.size)[m]
    slope = float(np.polyfit(tt, np.log(np.clip(tail[m], 1e-6, None)), 1)[0])
    return (-1.0 / slope) if slope < 0 else float("nan")


def sweep(W: np.ndarray, *, patch: int = 1024, step: int | None = None,
          coherence: float = 3.0, null="local", far: float = 0.05, local_window: int = 3):
    """Sweep a fixed ``patch``-column aperture across the feature axis of ``W`` (T, F); the entroptics
    COHERENCE of each patch is the gate (signal >> contaminant/noise, amplitude-independent).  Each
    read is a bounded thin ``Screen`` of (T, patch) -- no full-width decomposition.  Returns per-band
    dicts with the column ``span`` and the on-pulse ``width`` / ``tau_decay`` in SAMPLES.

    NULL POLICY -- ``null`` chooses how each coherent patch's floor is calibrated:
      * a PROVIDER (callable / dict) or ``None`` -> GLOBAL: that provider (or the ``mp`` default)
        thresholds every patch.  The regime for HOMOGENEOUS noise -- a globally-injected level or a
        pinned caller reference (the sensitive, single-global-null case).
      * ``"local"`` (default) -> REGION-DYNAMIC: each coherent patch is thresholded by a
        ``reference_null`` calibrated on the top-mode values of the NEAREST signal-free (low-
        coherence) patches within +/- ``local_window`` steps -- machine-precision, no i.i.d.
        assumption, and it tracks the noise drifting across the band.  Falls back to the default
        where too few nearby noise patches exist."""
    F = W.shape[1]
    step = step or patch
    scan = []
    for f0 in range(0, F, step):
        f1 = min(F, f0 + patch)
        sub = np.nan_to_num(W[:, f0:f1])
        sc = Screen(sub, far=far)                             # cheap thin read: coherence + top SV
        scan.append({"f0": f0, "f1": f1, "sub": sub, "coh": float(sc.coherence),
                     "top": float(sc.sigma_top), "sc": sc})
    bands = []
    for i, p in enumerate(scan):
        if p["coh"] < coherence:                              # COHERENCE GATE -> noise/contaminant, skip
            continue
        if null == "local":
            near = [q["top"] for q in scan if q["coh"] < coherence
                    and abs(q["f0"] - p["f0"]) <= local_window * step]
            prov = reference_null(np.asarray(near), far=far) if len(near) >= 2 else None
            sc = Screen(p["sub"], far=far, null=prov) if prov is not None else p["sc"]
        else:                                                 # GLOBAL caller-set provider (None -> mp)
            sc = Screen(p["sub"], far=far, null=null) if null is not None else p["sc"]
        prof = np.nansum(p["sub"], 1)
        prof = prof - np.median(prof)
        pk = int(np.argmax(prof))
        fp = sc.footprints
        lead = fp[0] if fp else None
        bands.append({
            "span": (int(p["f0"]), int(p["f1"])),
            "width": _entropy_width(prof, pk),                # SAMPLES (dimensionless) -- proxy
            "tau_decay": _tail_decay(prof, pk),               # SAMPLES -- tail decay rate
            "peak": pk,
            "coherence": float(sc.coherence),
            "contrast": float(sc.sigma_top / sc.noise_floor),
            "K": int(sc.K_signal),
            "noise_floor": float(sc.noise_floor),
            "phi_T": (float(lead.phi_T) if lead else float("nan")),
            "phi_F": (float(lead.phi_F) if lead else float("nan")),
        })
    return bands
