"""A read is a property of the signal, not of the units it was recorded in.

Entroptics whitens every channel by its own robust scale, so a read must be invariant when the
whole record is multiplied by a constant: the same instrument, in volts or in microvolts, reports
the same thing.  That invariance is only real if nothing in the path compares a DIMENSIONAL
quantity against a fixed number.  A MAD carries the record's units; a fixed cut on one is a
statement about units, and it silently reclassifies every channel of a record kept in small ones.

The failure mode these tests watch for, stated first so they can fail:
  - a signal read correctly at scale 1 reads as nothing at scale 1e-21 (strain-sized data);
  - a channel whose spread is exactly zero is given a manufactured scale and, divided by it,
    amplifies to something the screen then resolves as signal;
  - a round trip that returns the surface to floating-point accuracy is reported as lossy,
    because the residual -- pure round-off -- is whitened up to unit amplitude and read.

The floor separating "no spread" from "spread" is `resolution_floor`: the frame's own pooled MAD
times the working dtype's epsilon.  Both sides are measured, so it follows the data and the backend.
"""
import numpy as np
import pytest

from entroptics import Aperture, Projection, set_precision
from entroptics.entropy import macheps, resolution_floor, normalize, shannon_bits

SCALES = [1e12, 1e6, 1.0, 1e-6, 1e-15, 1e-21, 1e-30]
READS = ("phi", "phi_T", "phi_F", "etendue", "strehl", "a_delta", "focus")


@pytest.fixture
def signal():
    rng = np.random.default_rng(5)
    return rng.standard_normal((300, 3)) @ rng.standard_normal((3, 64)) \
        + 0.5 * rng.standard_normal((300, 64))


def test_every_read_is_invariant_to_the_recording_scale(signal):
    """The same signal in any units. A dimensional cut anywhere in the path shows up here."""
    ref = {k: float(getattr(Aperture(signal, window=None), k)) for k in READS}
    for s in SCALES:
        ap = Aperture(signal * s, window=None)
        for k in READS:
            assert float(getattr(ap, k)) == pytest.approx(ref[k], rel=1e-12), f"{k} moved at {s:g}"


def test_the_resolved_read_is_invariant_to_the_recording_scale(signal):
    """K_signal and the coherence go through the whitening, which is where a fixed cut would bite."""
    ref = Projection(signal)
    for s in SCALES:
        p = Projection(signal * s)
        assert p.K_signal == ref.K_signal
        assert p.coherence == pytest.approx(ref.coherence, rel=1e-9)


def test_a_channel_with_no_spread_gets_no_scale_rather_than_a_small_one(signal):
    """A flat channel has no scale to whiten by. Manufacturing a tiny one and dividing by it
    turns round-off into a mode -- the failure a fixed floor produced on coarsely quantized data,
    where more than half a channel's samples land on one level and its MAD is exactly zero."""
    W = signal.copy()
    W[:, 7] = 3.25                                     # measured, and never moved
    Z = np.asarray(normalize(W))
    assert np.all(Z[:, 7] == 0.0)                      # nothing to whiten -> exactly nothing
    assert np.abs(Z[:, [c for c in range(W.shape[1]) if c != 7]]).max() < 50.0


def test_coarse_quantization_does_not_blow_up_the_floor():
    """Quantization noise is real noise and the derived floor should measure it -- what it must not
    do is diverge. A channel whose samples collapse onto one level used to be divided by a
    manufactured scale of ~1e-11, lifting the screen to 1e10 and beyond."""
    rng = np.random.default_rng(2)
    clean = rng.standard_normal((300, 3)) @ rng.standard_normal((3, 64))
    lo, hi = clean.min(), clean.max()
    exact = Projection(clean)
    for bits in (8, 6, 4, 3, 2):
        lv = 2 ** bits - 1
        q = np.round((clean - lo) / (hi - lo) * lv) / lv * (hi - lo) + lo
        p = Projection(q)
        assert np.isfinite(p.noise_floor) and np.isfinite(p.sigma_top)
        assert p.noise_floor < 10.0 * exact.noise_floor, f"{bits}-bit floor diverged"
        assert p.sigma_top < 10.0 * exact.sigma_top, f"{bits}-bit screen diverged"


def test_the_resolution_floor_follows_the_dtype_and_the_data():
    """Both sides are measured: the pooled MAD from the frame, the epsilon from the array."""
    assert macheps(np, np.zeros(1, np.float64)) == np.finfo(np.float64).eps
    assert macheps(np, np.zeros(1, np.float32)) == np.finfo(np.float32).eps
    mad = np.ones(4)
    f64 = resolution_floor(np, 2.0, mad.astype(np.float64))
    f32 = resolution_floor(np, 2.0, mad.astype(np.float32))
    assert f32 > f64                                   # coarser arithmetic, higher floor
    assert resolution_floor(np, 2e-9, mad) == pytest.approx(1e-9 * f64, rel=1e-12)  # tracks the data


def test_shannon_bits_is_scale_free():
    """H(w) normalises internally, so a weight vector in any units carries the same entropy."""
    w = np.array([4.0, 3.0, 2.0, 1.0])
    ref = shannon_bits(w)
    for s in (1e30, 1e-30, 1e-300):
        assert shannon_bits(w * s) == pytest.approx(ref, rel=1e-12)


def test_a_float32_read_tracks_the_float64_one(signal):
    """The compute precision is a throughput choice, not a different instrument: the reads must
    agree to the precision actually being used, not merely be finite."""
    a64 = {k: float(getattr(Aperture(signal, window=None), k)) for k in READS}
    set_precision(32)
    try:
        a32 = {k: float(getattr(Aperture(signal.astype(np.float32), window=None), k)) for k in READS}
    finally:
        set_precision(64)
    band = 100.0 * np.finfo(np.float32).eps            # fp32 eps, with room for an O(T^3) eig
    for k in READS:
        assert abs(a32[k] - a64[k]) <= band * max(abs(a64[k]), 1e-12), f"{k} outside the fp32 band"
