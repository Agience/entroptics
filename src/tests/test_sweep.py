"""The aperture sweep: a fixed-capacity aperture swept across the feature axis, gated by coherence.

The guarantees are structural, not physical: the sweep must (a) return bands only where there is
coherent structure -- a pure-noise field yields nothing; (b) locate the coherent band by its column
``span``; and (c) read a finite on-pulse width and tail decay there.  Deterministic seeds."""
import numpy as np
import pytest

from entroptics import Aperture
from entroptics.sweep import sweep


def _field(T=140, F=2048, lo=800, hi=1200, tau=3.0, amp=8.0, seed=0):
    """iid-noise field with a scattered, transient burst confined to columns [lo, hi)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((T, F))
    t = np.arange(T)
    prof = np.exp(-0.5 * ((t - 45) / 1.8) ** 2)
    k = np.exp(-np.arange(T) / tau); k /= k.sum()
    prof = np.convolve(prof, k)[:T]
    W[:, lo:hi] += amp * prof[:, None]
    return W


def test_sweep_gates_to_the_coherent_band():
    bands = Aperture(_field(), window=None).sweep( patch=256, coherence=3.0)
    assert bands, "must find the coherent band"
    for b in bands:
        s0, s1 = b["span"]
        assert s1 > 800 and s0 < 1200, "every returned band must overlap the signal columns"
        assert b["coherence"] >= 3.0, "the coherence gate must hold"
        assert np.isfinite(b["width"]), "on-pulse width must be read"


def test_sweep_skips_pure_noise():
    W = np.random.default_rng(1).standard_normal((140, 2048))
    assert Aperture(W, window=None).sweep( patch=256, coherence=3.0) == [], "no coherent structure -> no bands"


def test_sweep_tail_decay_grows_with_scattering():
    """A longer scattering tail reads a larger tau_decay (the sweep tracks the decay rate)."""
    def med_tau(tau):
        b = Aperture(_field(tau=tau, amp=10.0), window=None).sweep( patch=256, coherence=3.0)
        taus = [x["tau_decay"] for x in b if np.isfinite(x["tau_decay"])]
        return float(np.median(taus)) if taus else np.nan
    assert med_tau(6.0) > med_tau(2.0), "more scattering -> longer tail decay"


# ── the gate and the on-pulse reads carry no hand-set constants ───────────────

def test_the_coherence_gate_is_derived_from_far_and_the_field_width():
    """A sweep asks the coherence question once per patch, so the gate must carry the sweep's own
    multiplicity: an uncorrected level fires ~far*n_patches times on an empty field, and a FIXED z
    is only right at one field width -- 10x too strict on a narrow one.

    ``norm_isf(0.05/32)`` rounds to 3.0, so a fixed gate is a 32-patch sweep's level held fixed
    across every other width."""
    from entroptics.null_providers import _norm_isf
    from entroptics import Aperture as sweep_fn
    import inspect
    assert inspect.signature(Aperture.sweep).parameters["coherence"].default is None

    def rate(F, patch, coherence, trials=120):
        hits = 0
        for i in range(trials):
            W = np.random.default_rng(9000 + i).standard_normal((140, F))
            hits += bool(Aperture(W, window=None).sweep(patch=patch, coherence=coherence))
        return hits / trials

    narrow, wide = (512, 256), (8192, 256)
    n_wide = len(range(0, wide[0], wide[1]))
    # derived: bounded near far at both widths
    assert rate(*narrow, None) < 0.20 and rate(*wide, None) < 0.20
    # uncorrected: blows up as the field widens
    assert rate(*wide, _norm_isf(0.05)) > 0.5
    # the derived z at the wide field is the old hand-set constant, recovered
    assert _norm_isf(0.05 / n_wide) == pytest.approx(3.0, abs=0.05)


def test_the_on_pulse_reads_follow_the_profiles_own_noise():
    """``width`` and ``tau_decay`` are read over the run where the profile is above its own MAD
    noise at the reader's level -- not over a fixed half-window, and not above a fixed fraction of
    the peak.  So scaling the burst's AMPLITUDE must not move the decay timescale."""
    from entroptics.sweep import _on_pulse_threshold, _on_pulse, _tail_decay

    def taus(amp):
        b = Aperture(_field(tau=4.0, amp=amp), window=None).sweep( patch=256, coherence=3.0)
        return [x["tau_decay"] for x in b if np.isfinite(x["tau_decay"])]

    lo, hi = taus(8.0), taus(24.0)
    assert lo and hi
    # a 3x brighter burst decays at the same rate; a peak-fraction cut would move it
    assert float(np.median(hi)) == pytest.approx(float(np.median(lo)), rel=0.35)

    # the threshold is the profile's own scale: pure noise has no on-pulse run to speak of
    noise = np.random.default_rng(3).standard_normal(400)
    noise -= np.median(noise)
    thr = _on_pulse_threshold(noise, 0.05)
    assert thr > 0.0
    lo_i, hi_i = _on_pulse(noise, int(np.argmax(noise)), thr)
    assert hi_i - lo_i < 20                      # a noise spike is not a pulse


def test_a_flagged_channel_is_absent_from_a_patch_not_zero_in_it():
    """A swept patch used to be cleaned with `nan_to_num`, so a channel nothing was observed in
    became a measurement of exactly zero power sitting in the patch, diluting it.

    Checked against the only ground truth available: the same field with those channels deleted."""
    rng = np.random.default_rng(0)
    T, F = 200, 128
    t = np.arange(T)[:, None]
    W = 0.5 * rng.standard_normal((T, F))
    W[:, 30:90] += 4 * np.exp(-0.5 * ((t - 100) / 8.0) ** 2)

    for frac in (0.25, 0.5, 0.75):
        dead = np.sort(rng.choice(F, int(frac * F), replace=False))
        keep = np.setdiff1d(np.arange(F), dead)

        flagged = W.copy(); flagged[:, dead] = np.nan
        by_nan = sweep(flagged, patch=F)
        by_mask = sweep(W.copy(), np.isin(np.arange(F), dead)[None, :].repeat(T, 0), patch=F)
        truth = sweep(W[:, keep], patch=len(keep))

        assert len(by_nan) == len(truth) == len(by_mask)
        for a, b, c in zip(by_nan, truth, by_mask):
            assert a["contrast"] == pytest.approx(b["contrast"], rel=1e-9)
            assert c["contrast"] == pytest.approx(b["contrast"], rel=1e-9)
            assert a["K"] == b["K"] == c["K"]


def test_the_sweep_carries_the_apertures_mask():
    """`Aperture.sweep` reads the aperture's own frame, so it must read its mask too."""
    rng = np.random.default_rng(1)
    T, F = 200, 128
    t = np.arange(T)[:, None]
    W = 0.5 * rng.standard_normal((T, F))
    W[:, 30:90] += 4 * np.exp(-0.5 * ((t - 100) / 8.0) ** 2)
    dead = np.sort(rng.choice(F, 64, replace=False))
    keep = np.setdiff1d(np.arange(F), dead)

    garbage = W.copy(); garbage[:, dead] = 7.5          # finite, and not what was there
    m = np.zeros(W.shape, bool); m[:, dead] = True

    got = Aperture(garbage, mask=m, window=None).sweep(patch=F)
    truth = Aperture(W[:, keep], window=None).sweep(patch=len(keep))
    assert len(got) == len(truth)
    for a, b in zip(got, truth):
        assert a["contrast"] == pytest.approx(b["contrast"], rel=1e-9)


def test_the_peak_indexes_the_records_own_ordered_axis():
    """`peak` is a position in the record, so it has to survive samples that were not observed.

    Cleaning a patch before reading it dropped whole unobserved ROWS, which renumbered everything
    after them: with 40 samples blanked, a burst planted at 300 reported 259, and the caller had
    no way to map that back.  The profile now carries no value where nothing was observed rather
    than being collapsed."""
    rng = np.random.default_rng(0)
    T, F = 400, 256
    t = np.arange(T)[:, None]
    W = 0.5 * rng.standard_normal((T, F))
    W[:, 100:160] += 5 * np.exp(-0.5 * ((t - 300) / 6.0) ** 2)      # a burst at sample 300

    clean = sweep(W, patch=128)
    assert clean and all(abs(d["peak"] - 300) <= 2 for d in clean)

    blanked = W.copy()
    blanked[50:90, :] = np.nan                                      # 40 samples never observed
    got = sweep(blanked, patch=128)
    assert len(got) == len(clean)
    for d in got:
        assert abs(d["peak"] - 300) <= 2, f"peak {d['peak']} is not on the record's own axis"


def test_a_sample_nothing_was_observed_in_is_not_a_trough():
    """A profile value is a sum over the channels seen at that sample.  Where none were seen there
    is no value -- summing nothing to 0 would read as a deep trough and pull the peak and the
    on-pulse extent toward it."""
    rng = np.random.default_rng(1)
    T, F = 300, 128
    t = np.arange(T)[:, None]
    W = 0.5 * rng.standard_normal((T, F))
    W[:, 20:80] += 5 * np.exp(-0.5 * ((t - 150) / 5.0) ** 2)
    gapped = W.copy()
    gapped[140:145, :] = np.nan                 # unobserved samples right ON the burst

    ref, got = sweep(W, patch=F), sweep(gapped, patch=F)
    assert ref and got
    for a, b in zip(ref, got):
        assert abs(b["peak"] - a["peak"]) <= 5
        assert np.isfinite(b["width"]) and b["width"] > 0


def _planted_field(T=200, F=256):
    rng = np.random.default_rng(0)
    t = np.arange(T)[:, None]
    Z = 0.5 * rng.standard_normal((T, F)) + 1j * 0.5 * rng.standard_normal((T, F))
    Z[:, 100:160] += 4 * np.exp(-0.5 * ((t - 100) / 8.0) ** 2) * np.exp(1j * 2 * np.pi * np.arange(60) / 17.0)
    return Z


def test_a_field_is_swept_and_the_source_is_still_there_at_the_span():
    """Finding a band needs no brightness: `span` and everything read off the projection are the
    same on a field as on an intensity.  The point of the span is that the caller goes back to the
    record with it -- so the sweep must not be the thing that loses the imaginary part."""
    Z = _planted_field()
    bands = sweep(Z, patch=128)
    assert bands
    for b in bands:
        assert b["coherence"] > 3.0 and b["K"] >= 1 and np.isfinite(b["contrast"])
        assert np.isfinite(b["noise_floor"]) and np.isfinite(b["phi_F"])
        f0, f1 = b["span"]
        recovered = Z[:, f0:f1]                       # the original vectors, from the span alone
        assert np.iscomplexobj(recovered) and np.abs(recovered.imag).max() > 0


def test_the_brightness_reads_stand_down_on_a_field_and_return_on_one():
    """`peak`, `width` and `tau_decay` come off a per-sample brightness, which a field does not
    carry: summing amplitudes lets a phase ramp cancel and a burst planted at 100 read 70.  They
    say so instead of answering, and both named brightnesses bring them back."""
    Z = _planted_field()
    for b in sweep(Z, patch=128):
        assert b["peak"] == -1
        assert np.isnan(b["width"]) and np.isnan(b["tau_decay"])

    for brightness in (np.abs(Z), np.abs(Z) ** 2):
        bands = sweep(brightness, patch=128)
        assert bands and all(abs(b["peak"] - 100) <= 4 for b in bands)
        assert all(np.isfinite(b["width"]) for b in bands)


def test_an_integer_record_is_promoted_only_because_it_cannot_carry_nan():
    """The patch keeps the record's own dtype. An integer one is the single exception: a mask marks
    absence with NaN, which an integer array has no room for."""
    rng = np.random.default_rng(1)
    T, F = 120, 64
    W = (10 * np.abs(rng.standard_normal((T, F)))).astype(np.int32)
    mask = np.zeros(W.shape, bool); mask[:, :8] = True
    assert sweep(W, mask, patch=F) == sweep(W.astype(float), mask, patch=F)
