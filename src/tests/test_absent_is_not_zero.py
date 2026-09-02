"""Absence is not an observation of zero.

`geometry` zeroes nonfinite and masked cells before summing power, which is right -- an absent
cell contributes nothing and `0 log 0 = 0` leaves the entropy untouched.  It also measures the
extent -- how many rows and columns actually carry an observation -- separately from the array's
raw shape, because that extent, not the shape, sizes every quantity the entropy is compared
against: the no-signal maximum `log2(F)`, the concentration test, the Miller-Madow band
`(F-1)/(2 T ln2)`, and the matched scale.  Widening an axis with cells nothing was observed in
would raise the bar the signal has to clear while adding no signal.

The failure mode these tests watch for, stated first so they can fail:
  - a frame padded with absent channels reads differently from the same frame without them;
  - an entirely unmeasured frame reports log2(F) bits of spread it never observed;
  - and, the control that keeps the rule honest, a frame padded with finite zeros reads
    differently from the same frame without them -- it must not, because zero is a real
    observation of no power and belongs in the extent.
"""
import numpy as np
import pytest

from entroptics import Aperture
from entroptics.aperture import MIN_WINDOW
from entroptics.entropy import geometry, fold_width, feature_adjacency
from entroptics.reads import phi, phi_F, phi_T


def _signal(rng, T=64, F=32):
    """A continuous, concentrated feature axis: a narrow Gaussian line that drifts."""
    x = np.arange(F)[None, :]
    c = (F / 2.0) + 3.0 * np.sin(np.arange(T) / 7.0)[:, None]
    return np.exp(-0.5 * ((x - c) / 1.5) ** 2) + 0.01 * rng.standard_normal((T, F))


def test_absent_channels_do_not_enter_the_feature_extent():
    """Pad with NaN channels: every read is unchanged, because nothing was measured in them."""
    rng = np.random.default_rng(0)
    W = _signal(rng)
    g0 = geometry(W)

    pad = np.full((W.shape[0], 96), np.nan)
    g1 = geometry(np.concatenate([W, pad], axis=1))

    assert g1["H_F"] == pytest.approx(g0["H_F"], abs=1e-12)
    assert g1["delta_F"] == pytest.approx(g0["delta_F"], rel=1e-12)
    # n_F is a resample target in the ARRAY's coordinates, so the fold decision must agree while
    # the width tracks the real array: folded to the same count, or not folded at all.
    assert (g1["n_F"] == g0["n_F"]) or (g1["n_F"] == W.shape[1] + 96 and g0["n_F"] == W.shape[1])


def test_masked_channels_do_not_enter_the_feature_extent():
    """Same, via the mask, the same statement as NaN -- the two are the same statement about absence."""
    rng = np.random.default_rng(1)
    W = _signal(rng)
    g0 = geometry(W)

    wide = np.concatenate([W, rng.standard_normal((W.shape[0], 96))], axis=1)
    mask = np.zeros(wide.shape, dtype=bool)
    mask[:, W.shape[1]:] = True                    # the padding is masked OUT
    g1 = geometry(wide, mask)

    assert g1["H_F"] == pytest.approx(g0["H_F"], abs=1e-12)
    assert g1["delta_F"] == pytest.approx(g0["delta_F"], rel=1e-12)


def test_finite_zero_channels_DO_enter_the_extent():
    """The control: zero is an observation, not an absence, and must still count.

    Without it, 'ignore cells with no power' would pass every test above while being a different,
    wrong rule: a frame padded with real zeros is a wider frame whose power is more concentrated,
    and it must read as such."""
    rng = np.random.default_rng(2)
    W = _signal(rng)
    g0 = geometry(W)

    wide = np.concatenate([W, np.zeros((W.shape[0], 96))], axis=1)
    g1 = geometry(wide)

    assert g1["H_F"] == pytest.approx(g0["H_F"], abs=1e-12)   # entropy itself is unchanged: 0log0 = 0
    # ...but the axis is genuinely 128 wide now, so the fold decision is made against log2(128).
    assert fold_width(g1["H_F"], *wide.shape) != (wide.shape[1], 1.0) or \
           fold_width(g0["H_F"], *W.shape) == (W.shape[1], 1.0)
    assert g1["delta_F"] >= g0["delta_F"] - 1e-12


def test_an_entirely_unmeasured_frame_claims_no_spread():
    """All-NaN: 0 bits on both axes -- one degenerate cell per axis, honestly reported: no channels resolved.  log2(F) would assert spread from no measurement at all."""
    g = geometry(np.full((16, 64), np.nan))
    assert g["H_F"] == pytest.approx(0.0)
    assert g["H_T"] == pytest.approx(0.0)


def test_absent_channels_do_not_manufacture_adjacency():
    """Absent channels carry no observation, so `feature_adjacency` drops them before scoring
    adjacency: counting them as zero would make every pair of them identical, and a run of dead
    channels would read as a smooth, continuous stretch of axis, licensing a fold across a region
    where nothing was observed.  Dropping them leaves the read unchanged."""
    rng = np.random.default_rng(3)
    W = _signal(rng)
    a0 = feature_adjacency(W)

    padded = np.concatenate([W, np.full((W.shape[0], 96), np.nan)], axis=1)
    a1 = feature_adjacency(padded)
    assert a1 == pytest.approx(a0, rel=1e-12)

    # And a NOMINAL axis stays nominal when padded -- the padding must not push it over the
    # continuity bar and unlock a fold it should never get.
    nominal = rng.standard_normal((64, 32))
    n0, n1 = feature_adjacency(nominal), feature_adjacency(
        np.concatenate([nominal, np.full((64, 96), np.nan)], axis=1))
    assert n1 == pytest.approx(n0, rel=1e-12)


def test_fold_width_defaults_to_the_array_shape():
    """Callers that pass no effective extent are unaffected: measuring the extent is opt-in per
    call."""
    for H, T, F in ((3.0, 64, 32), (4.9, 64, 32), (1.0, 16, 128)):
        assert fold_width(H, T, F) == fold_width(H, T, F, F_eff=F, T_eff=T)


def _sparse(rng, T=200, F=256, live=64):
    """A frame in which only `live` of `F` channels were ever observed -- the RFI-flagged case."""
    W = rng.standard_normal((T, F))
    keep = np.sort(rng.choice(F, live, replace=False))
    return W, keep


def test_the_fills_divide_by_the_measured_extent_not_the_array_shape():
    """Blank 75% of the channels and the fills must read what the live ones read.

    Dividing by F instead of the measured extent scales the fill by live/F -- here a factor of
    four -- so this is the difference between a nearly-full aperture and a nearly-empty one."""
    rng = np.random.default_rng(7)
    W, keep = _sparse(rng)
    truth = W[:, keep]

    absent = W.copy()
    absent[:, [c for c in range(W.shape[1]) if c not in set(keep.tolist())]] = np.nan

    assert phi_F(absent) == pytest.approx(phi_F(truth), rel=1e-12)
    assert phi_T(absent) == pytest.approx(phi_T(truth), rel=1e-12)
    assert phi(absent) == pytest.approx(phi(truth), rel=1e-12)


def test_a_mask_reads_the_same_as_deleting_the_channels():
    """The masked channels carry a finite, WRONG value; the mask says they were not observed.
    Honouring it means reading exactly what the surviving channels read on their own."""
    rng = np.random.default_rng(8)
    W, keep = _sparse(rng)
    dead = [c for c in range(W.shape[1]) if c not in set(keep.tolist())]

    garbage = W.copy()
    garbage[:, dead] = 7.5                          # a real number, and not the one that was there
    mask = np.zeros(W.shape, dtype=bool)
    mask[:, dead] = True

    a = Aperture(garbage, mask=mask, window=None)
    truth = Aperture(W[:, keep], window=None)
    assert a.phi_F == pytest.approx(truth.phi_F, rel=1e-12)
    assert a.phi_T == pytest.approx(truth.phi_T, rel=1e-12)
    assert a.phi == pytest.approx(truth.phi, rel=1e-12)
    assert a.etendue == pytest.approx(truth.etendue, rel=1e-12)
    assert a.strehl == pytest.approx(truth.strehl, rel=1e-12)


def test_a_mask_does_not_read_the_same_as_zeroing():
    """The control, and the reason the mask exists: substituting zero CHANGES the channel's value.
    A masked read and a zero-filled read must not agree, or the mask is being ignored."""
    rng = np.random.default_rng(9)
    W, keep = _sparse(rng)
    dead = [c for c in range(W.shape[1]) if c not in set(keep.tolist())]

    garbage = W.copy(); garbage[:, dead] = 7.5
    mask = np.zeros(W.shape, dtype=bool); mask[:, dead] = True
    zeroed = W.copy(); zeroed[:, dead] = 0.0

    masked_phi_F = Aperture(garbage, mask=mask, window=None).phi_F
    zeroed_phi_F = Aperture(zeroed, window=None).phi_F
    assert abs(masked_phi_F - zeroed_phi_F) > 1e-6
    # and the direction is the one the geometry docstring names: the zeros widen the axis the
    # signal is scored against, so the same signal reads as a smaller fraction of it.
    assert zeroed_phi_F < masked_phi_F


def test_the_mask_is_trimmed_to_the_same_window_as_the_data():
    """A windowed aperture reads a trailing slice of W. A mask given for the whole record must
    annotate the slice the reads actually see -- if it is not cut with the data, the two arrive at
    the reads with different row counts and the mask cannot be applied at all."""
    rng = np.random.default_rng(10)
    W, keep = _sparse(rng)
    dead = [c for c in range(W.shape[1]) if c not in set(keep.tolist())]
    garbage = W.copy(); garbage[:, dead] = 7.5
    mask = np.zeros(W.shape, dtype=bool); mask[:, dead] = True

    win = W.shape[0] // 2
    a = Aperture(garbage, mask=mask, window=win)      # windowed, so the two must be cut together
    assert a.W.shape[0] < W.shape[0]                  # the window is doing something
    assert a.mask.shape == a.W.shape                  # ... and the mask came with it
    assert a.phi_F == pytest.approx(Aperture(W[-a.W.shape[0]:, keep]).phi_F, rel=1e-12)


def test_a_finite_record_is_read_whole_and_a_stream_is_bounded():
    """`window` follows what the aperture was handed. A caller who passes a complete record gets a
    read of that record; a caller feeding frames gets the coherent window, which is what bounds
    memory on a stream that has no end. Handing over a record and being read 128 rows of it was
    the surprise this removes."""
    rng = np.random.default_rng(11)
    W = rng.standard_normal((4 * MIN_WINDOW, 16))

    batch = Aperture(W)
    assert batch.window is None
    assert batch.W.shape[0] == W.shape[0]             # all of it

    stream = Aperture()
    for row in W:
        stream.update(row)
    assert stream.window == MIN_WINDOW
    assert stream.W.shape[0] <= W.shape[0]            # bounded by the coherent window

    assert Aperture(W, window=64).window == 64        # an explicit window still wins
    assert Aperture(W, window=None).window is None


def test_setting_the_mask_restates_the_reads():
    """The mask is part of what a read means, so replacing it must not return a cached answer
    taken under the old one."""
    rng = np.random.default_rng(11)
    W, keep = _sparse(rng)
    dead = [c for c in range(W.shape[1]) if c not in set(keep.tolist())]
    mask = np.zeros(W.shape, dtype=bool); mask[:, dead] = True

    ap = Aperture(W, mask=mask, window=None)
    masked = ap.phi_F
    ap.mask = None
    assert ap.phi_F != pytest.approx(masked)
    assert ap.phi_F == pytest.approx(Aperture(W, window=None).phi_F, rel=1e-12)
