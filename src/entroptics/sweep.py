"""sweep.py -- the entroptics aperture swept over an unbounded ordered x feature field.

Entroptics reads a finite aperture, not an infinite field.  Forcing the full width through one
decomposition is both wrong (a faint spread signal is noise-dominated across the whole width) and
unnecessary.  Instead: set the aperture to a fixed capacity (a patch that keeps each read cheap),
then sweep it where there is coherence -- the coherence read gates out noise-only patches, and the
full read runs only on signal-bearing ones.

Coherence is the gate (not contrast): on a spread, band-limited signal it separates signal patches
(coherence high, flat across amplitude) from persistent contaminants (~ 0.8) and noise (~ 0) cleanly
and amplitude-independently, whereas contrast fades with the signal's strength.  (Contrast is then a
per-patch concentration read; phi_F > phi_T flags any persistent mode that slips through.)

Time is fluid: every width the aperture reads is in abstract window samples (dimensionless).  The
aperture has no physical clock and no physical frequency -- ``sweep`` returns each coherent band's
column ``span`` (index range) and its reads; the caller maps spans to physical units if it wants
them.

    bands = sweep(W)        # per-coherent-band reads: span (columns), width & tau_decay (samples)
"""
from __future__ import annotations

import numpy as np

from .projection import Projection
from .entropy import shannon_bits, MAD_SCALE
from .null_providers import reference_null, _norm_isf


def _on_pulse_threshold(prof: np.ndarray, far: float) -> float:
    """Where the profile stops being noise, read off the profile's OWN robust scale.

    The on-pulse reads below need to know where the pulse ends.  A fixed half-width or a fixed
    fraction of the peak is a guess about a signal that has not been measured yet -- and an
    amplitude-relative cut ignores the noise entirely, so the same tail is cut at a different
    place depending only on how bright its peak is.  The profile is already median-subtracted,
    so its MAD is the noise scale, and the reader's own ``far`` sets how far above it counts."""
    sigma = float(np.nanmedian(np.abs(prof - np.nanmedian(prof)))) * MAD_SCALE
    return _norm_isf(float(far)) * sigma


def _on_pulse(prof: np.ndarray, pk: int, thr: float) -> tuple[int, int]:
    """The contiguous run around ``pk`` that stays above ``thr`` -- the on-pulse extent as the profile actually draws it."""
    n = int(prof.size)
    lo = int(pk)
    while lo > 0 and prof[lo - 1] > thr:
        lo -= 1
    hi = int(pk)
    while hi + 1 < n and prof[hi + 1] > thr:
        hi += 1
    return lo, hi + 1


def _entropy_width(prof: np.ndarray, pk: int, thr: float) -> float:
    """Entropy width (participation length) of the on-pulse profile, in samples (dimensionless):
    2^H of the on-pulse power -- the aperture's own window units."""
    lo, hi = _on_pulse(prof, pk, thr)
    seg = np.clip(prof[lo:hi], 0.0, None)
    s = seg.sum()
    if s <= 0:
        return float("nan")
    return float(2 ** shannon_bits(seg / s))


def _tail_decay(prof: np.ndarray, pk: int, thr: float) -> float:
    """The exponential decay timescale of the post-peak tail, in samples.  A pure tail is
    exp(-t/tau), so a log-linear fit of the post-peak profile gives slope = -1/tau.  This reads
    the tail for what it is -- a decay rate (the entroptics ``attenuation`` -log|mu| of an
    exponential) -- not a width proxy, so it recovers a power-law tail far better than 2^H
    (which has a participation-length floor).

    The fit runs from the peak to where the tail meets the noise (``thr``), so both ends are
    measured: no fixed span to run past the pulse, and no fixed fraction of a peak whose height
    is the one thing a decay rate must not depend on."""
    _, hi = _on_pulse(prof, pk, thr)
    tail = prof[pk:hi]
    if tail.size < 4 or tail[0] <= 0:
        return float("nan")
    tt = np.arange(tail.size, dtype=float)
    # every tail value stands above `thr`, so the log needs no floor: a constant here would
    # substitute a value inside the fit and move the slope it returns.
    slope = float(np.polyfit(tt, np.log(tail), 1)[0])
    return (-1.0 / slope) if slope < 0 else float("nan")


def sweep(W: np.ndarray, mask: np.ndarray | None = None, *, patch: int = 1024, step: int | None = None,
          coherence: float | None = None, null="local", far: float = 0.05, local_window: int = 3):
    """Sweep a fixed ``patch``-column aperture across the feature axis of ``W`` (T, F); the entroptics
    coherence of each patch is the gate (signal >> contaminant/noise, amplitude-independent).  Each
    read is a bounded thin ``Projection`` of (T, patch) -- no full-width decomposition.  Returns per-band
    dicts with the column ``span`` and the on-pulse ``width`` / ``tau_decay`` in samples.

    ``coherence`` is the gate's z, and it defaults to the level ``far`` corrected for the sweep's
    own multiplicity: a sweep asks the question once per patch, so an uncorrected level fires
    about ``far * n_patches`` times on a field with nothing in it.  ``None`` (the default) derives
    ``z = norm_isf(far / n_patches)`` from the patches actually swept, so the FIELD-wide false-alarm
    rate stays bounded near ``far`` as the field widens instead of growing with it.  Measured on
    pure noise at ``far=0.05``, 200 trials, patch=256 -- field-wide rate over 2 / 8 / 32 patches:

        derived            0.070 / 0.080 / 0.120
        fixed z = 3.0      0.005 / 0.020 / 0.110     (10x too strict on a narrow field)
        uncorrected        0.140 / 0.400 / 0.840     (grows with the field)

    The residual drift is the coherence z's own tail: the permutation distribution is mildly
    right-skewed at small N, so the one-sided normal tail is approached (PAPER
    Remark 5.4).  Pass a float to state your own z.

    Null policy -- ``null`` chooses how each coherent patch's floor is calibrated:
      * a provider (callable / dict) or ``None`` -> global: that provider (or the ``mp`` default)
        thresholds every patch.  The regime for homogeneous noise -- a globally-injected level or a
        pinned caller reference (the sensitive, single-global-null case).
    A band is located and read off its projection, so ``span``, ``coherence``, ``contrast``, ``K``,
    ``noise_floor`` and ``phi_T``/``phi_F`` read the same on a field as on an intensity -- take the
    ``span`` back to the source and the original vectors are there, complex intact.  ``peak``,
    ``width`` and ``tau_decay`` are read off a per-sample brightness profile, which a complex record
    does not carry, and they come back ``-1``/``NaN`` for one; pass ``abs(W)`` or ``abs(W)**2`` when
    you want them.

    ``mask`` marks cells that were NOT observed (``True``); non-finite cells say the same thing.
    A patch is handed to :class:`Projection` as it was recorded; it reads absence itself, so the
    read is the one the surviving channels give.  The profile keeps the record's ordered axis, so
    ``peak`` is an index into it even where whole samples went unobserved.

      * ``"local"`` (default) -> region-dynamic: each coherent patch is thresholded by a
        ``reference_null`` calibrated on the top-mode values of the nearest signal-free (low-
        coherence) patches within +/- ``local_window`` steps -- machine-precision, no i.i.d.
        assumption, and it tracks the noise drifting across the band.  Falls back to the default
        where too few nearby noise patches exist."""
    F = W.shape[1]
    # A complex record is a FIELD: it carries no per-sample brightness, so the three profile
    # reads below stand down.  A dtype the caller chose, never a guess at what the values mean.
    carries_brightness = not np.iscomplexobj(np.asarray(W))
    step = step or patch
    n_patches = max(1, len(range(0, F, step)))
    if coherence is None:
        coherence = _norm_isf(float(far) / n_patches)      # the sweep's own multiplicity
    scan = []
    for f0 in range(0, F, step):
        f1 = min(F, f0 + patch)
        # Handed over as it was recorded.  `nan_to_num` here made a channel nothing was observed
        # in into a measurement of zero power, which diluted the patch -- 75% flagged read a
        # contrast of 1.01 where the surviving channels alone read 3.21.  `Projection` reads
        # absence itself and matches the deleted-channel truth exactly, so nothing is cleaned on
        # the way in; cleaning it here would also drop unobserved ROWS, and the profile below is
        # indexed on the record's own ordered axis.
        sub = np.asarray(W[:, f0:f1])
        if not np.issubdtype(sub.dtype, np.inexact):
            sub = sub.astype(float)                   # an integer dtype carries no NaN
        msk = None if mask is None else np.asarray(mask[:, f0:f1], dtype=bool)
        if msk is not None:
            sub = np.where(msk, np.nan, sub)
        if sub.size == 0 or not np.isfinite(sub).any():
            continue                                      # nothing was observed in this patch
        sc = Projection(sub, far=far)                             # cheap thin read: coherence + top SV
        scan.append({"f0": f0, "f1": f1, "sub": sub, "coh": float(sc.coherence),
                     "top": float(sc.sigma_top), "sc": sc})
    bands = []
    for i, p in enumerate(scan):
        if p["coh"] < coherence:                              # coherence gate -> noise/contaminant, skip
            continue
        if null == "local":
            near = [q["top"] for q in scan if q["coh"] < coherence
                    and abs(q["f0"] - p["f0"]) <= local_window * step]
            prov = reference_null(np.asarray(near), far=far) if len(near) >= 2 else None
            # RE-FLOOR, do not rebuild. The provider reaches one line of `Projection.__init__` --
            # `noise_floor(..., s=self.S)` -- which is handed the spectrum, so it moves the floor
            # inside a decomposition the scan pass has already paid for. Rebuilding recomputed an
            # identical `svdvals` once per coherent patch. ⚑ Measured 2026-08-27, 64 patches at
            # patch=256: 80 projections for 64 patches, SVD 66% of the run. `refloor` is an
            # identity, held by `tests/test_projection_refloor.py`.
            sc = p["sc"].refloor(prov) if prov is not None else p["sc"]
        else:                                                 # global caller-set provider (None -> mp)
            sc = p["sc"].refloor(null) if null is not None else p["sc"]
        # WHERE a band is, and what it resolves, is read off the projection and needs no brightness
        # -- those reads are the same on a field as on an intensity.  The three that DO need one
        # (`peak`, `width`, `tau_decay`) come from a per-sample profile, and a complex record does
        # not carry one: summing amplitudes lets a phase ramp across channels cancel, and a burst
        # planted at sample 100 read 70.  Neither repair is free -- |x| rectifies noise into the
        # baseline and read a planted tail timescale of 2.0 as 1.2, |x|**2 halves it outright --
        # and which brightness a field has is a fact about the instrument.  So those three are NaN
        # for a field and the rest of the band still reads; take `span` back to the source and hand
        # `sweep` the brightness you mean (`abs(W)` or `abs(W)**2`) when you want them.
        #
        # The profile keeps the record's ordered axis: a sample nothing was observed in carries
        # no profile value (NaN).  A sum of nothing would read as a deep trough, and dropping the
        # sample would renumber every index after it.
        if carries_brightness:
            seen = np.isfinite(p["sub"]).any(axis=1)
            prof = np.where(seen, np.nansum(p["sub"], 1), np.nan)
            prof = prof - np.nanmedian(prof)
            pk = int(np.nanargmax(prof))
            thr = _on_pulse_threshold(prof, far)              # where the profile leaves its own noise
            width, tau = _entropy_width(prof, pk, thr), _tail_decay(prof, pk, thr)
        else:
            pk, width, tau = -1, float("nan"), float("nan")
        fp = sc.footprints
        lead = fp[0] if fp else None
        bands.append({
            "span": (int(p["f0"]), int(p["f1"])),
            "width": width,                                   # samples; NaN for a field
            "tau_decay": tau,                                 # samples; NaN for a field
            "peak": pk,                                       # sample index; -1 for a field
            "coherence": float(sc.coherence),
            "contrast": float(sc.sigma_top / sc.noise_floor),
            "K": int(sc.K_signal),
            "noise_floor": float(sc.noise_floor),
            "phi_T": (float(lead.phi_T) if lead else float("nan")),
            "phi_F": (float(lead.phi_F) if lead else float("nan")),
        })
    return bands
