"""The FRAME ``extract`` returns its clean view on: the caller's own.

``Aperture.extract`` is the one read in the library whose output is DATA rather than a
measurement, so it is the only one where the frame it comes back on is a question at all.  The
modes are read on the screen, and the screen is ``project(normalize(W))`` -- per-channel median
removed, per-channel robust scale divided out -- so the projection lands on the whitened grid.
The whitening is undone before returning, because a denoised frame a caller cannot plot against
the frame they passed in is not a denoised frame.

There is ONE path: ``clean`` is always in the input's units.  What these tests pin:

  A. ``normalize(..., return_stats=True)`` reports the map it applied, and it inverts to machine
     precision -- on the clean path and on the masked one, which computes its own stats.
  B. ``clean`` lands on the input's scale, and ``info`` reports the map that put it there.
  C. It is a real denoising in those units: closer to the truth than the raw frame, where the
     un-inverted screen array would be far from it and no single rescale would repair it.
  D. ``W - clean`` is what the filter discarded -- the identity the input-units return exists to
     make true, and the reason the per-channel baseline is carried through rather than dropped.
  E. The map spans the SCREEN's feature axis, including when channels are folded or dropped.
"""
import numpy as np
import pytest

from entroptics import Aperture
from entroptics.entropy import normalize
from entroptics.projection import _fold_axis


def _frame(T=64, F=24, seed=0, gain=40.0, noise=1.0):
    """A broadband transient on channels with the per-channel offsets and gains real instrument
    data has.  Two properties are deliberate: the offsets and scales VARY, since a constant one
    would be invertible by a single rescale and would hide the defect these tests are about; and
    the SIGNAL is flat across the band, which keeps the feature axis at full entropy so the fold
    is the identity and ``clean`` comes back the same shape as ``W``.  That shape match is what
    made the units easy to miss, so it is the case worth testing on.

    The per-channel scale therefore comes from the NOISE rather than from a gain ramp on the
    burst.  ``geometry`` reads the raw frame, so a gain ramp would concentrate the frame's energy
    in the loud channels, drop the feature entropy and coarsen the fold -- which is a different
    test (see ``test_map_spans_a_folded_feature_axis``).  Whitening divides by each channel's own
    MAD either way, so this still yields a centre AND a scale that vary per channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    burst = np.exp(-((t - 0.47 * T) ** 2) / 8.0) * np.ones((1, F))
    offset = np.linspace(2.0, 9.0, F)[None, :]
    spread = np.linspace(0.5, 3.0, F)[None, :]
    truth = offset + gain * burst
    return truth + spread * rng.standard_normal((T, F)) * noise, truth


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ── A. the map normalize computes is the map it reports ───────────────────────────────────────
def test_normalize_stats_invert_exactly():
    """A. ``whitened * scale + centre`` returns the frame, to machine precision.

    This is the whole claim: the pair is not an estimate of the whitening, it IS the whitening,
    so the round trip is exact arithmetic and not a fit."""
    W, _ = _frame()
    wh, centre, scale = normalize(W, return_stats=True)
    assert centre.shape == (W.shape[1],) and scale.shape == (W.shape[1],)
    assert np.abs(wh * scale[None, :] + centre[None, :] - W).max() < 1e-9

    # opt-in only: the bare call is untouched, same array, same type
    plain = normalize(W)
    assert isinstance(plain, np.ndarray) and np.array_equal(plain, wh)


def test_normalize_stats_on_the_masked_path():
    """A. the masked branch reports ITS OWN stats.

    ``_normalize_masked_np`` takes a masked median and shrinks with a per-channel valid-cell
    count, so its centre and scale are not the clean path's.  Re-running the clean path on the
    gapped frame would give different numbers, which is why the branch has to return them."""
    W, _ = _frame()
    Wm = W.copy()
    Wm[3, 4] = np.nan            # a scattered gap
    Wm[7:12, 9] = np.nan         # a partly-gapped channel -> shrunk harder
    wh, centre, scale = normalize(Wm, return_stats=True)
    assert centre.shape == (Wm.shape[1],) and scale.shape == (Wm.shape[1],)

    ok = np.isfinite(wh) & np.isfinite(Wm)
    assert np.abs((wh * scale[None, :] + centre[None, :])[ok] - Wm[ok]).max() < 1e-9
    assert np.isnan(wh[3, 4]), "a masked cell stays missing; the stats do not fill it"


def test_zero_spread_channel_inverts():
    """A. a channel with no resolvable spread comes back with ``scale == 0``, and the map is
    still exact: every value in that channel IS its centre, which is what was measured.  A zero
    scale is a reading, not a gap."""
    W, _ = _frame()
    W[:, 5] = 3.25                                   # constant channel
    wh, centre, scale = normalize(W, return_stats=True)
    assert scale[5] == 0.0 and np.allclose(wh[:, 5], 0.0)
    assert np.abs(wh * scale[None, :] + centre[None, :] - W).max() < 1e-9


# ── B. extract lands on the input's scale and says how ────────────────────────────────────────
def test_clean_is_on_the_input_scale():
    """B. there is one path and it returns the caller's units.

    The screen array is whitened -- unit noise, zero centre -- and ``clean`` is not that: it
    carries the input's own offsets and spread.  Dividing the reported map back out is what
    recovers the screen-side array, which is the direction that is now the extra step."""
    W, _ = _frame()
    clean, info = Aperture(W, window=None).extract()
    assert info["centre"].shape == (clean.shape[1],)
    assert info["scale"].shape == (clean.shape[1],)

    # `clean` sits on the input's scale; the screen-side array sits on the WHITENED frame's.
    # Scored as scale ratios, so the two claims are the same measurement made twice.
    screen_side = (clean - info["centre"][None, :]) / info["scale"][None, :]
    whitened = np.asarray(normalize(W))
    assert 0.8 < float(clean.std()) / float(W.std()) < 1.2, "clean is on the input's scale"
    assert 0.8 < float(screen_side.std()) / float(whitened.std()) < 1.2, (
        "dividing the reported map out returns the array to the whitened screen's scale")

    assert "units" not in info, "there is no units switch -- one path only"
    assert "units_exact" not in info


def test_extract_takes_no_units_argument():
    """B. the switch is gone.  A caller who passes one gets a TypeError rather than a silently
    ignored keyword, which is the failure mode a leftover parameter name would create."""
    W, _ = _frame()
    with pytest.raises(TypeError):
        Aperture(W, window=None).extract(units="input")


# ── C. it is a real denoising, in those units ─────────────────────────────────────────────────
def test_extract_denoises_in_the_callers_units():
    """C. against a known truth, ``clean`` beats the raw frame -- and the whitened screen array
    it is built from does not, by a wide margin.  That gap is the finding this fixed: the
    filtering was always fine, the return value was on the wrong frame."""
    W, truth = _frame()
    clean, info = Aperture(W, window=None).extract()
    assert clean.shape == W.shape, "this frame's fold is the identity"

    raw_err = _rel(W, truth)
    assert _rel(clean, truth) < raw_err, "the filter must beat the raw frame in the caller's units"

    screen_side = (clean - info["centre"][None, :]) / info["scale"][None, :]
    assert _rel(screen_side, truth) > 3 * raw_err, (
        "the screen-side array is far from the truth -- inverting the map is what makes the "
        "filtering usable, not an optional presentation step")


def test_no_single_rescale_would_have_repaired_it():
    """C. the map is PER CHANNEL, so the best global affine rescale of the screen-side array
    still does not reach the caller's units.  This is why the map has to be applied inside the
    read, and why a scale matched by least squares at a call site was never enough."""
    W, truth = _frame()
    clean, info = Aperture(W, window=None).extract()
    screen_side = (clean - info["centre"][None, :]) / info["scale"][None, :]

    # least-squares best global (a, b) for a*screen_side + b against the truth
    A = np.stack([screen_side.ravel(), np.ones(screen_side.size)], axis=1)
    a, b = np.linalg.lstsq(A, truth.ravel(), rcond=None)[0]
    best_global = _rel(a * screen_side + b, truth)
    assert _rel(clean, truth) < best_global, (
        "the per-channel map must beat the best possible single rescale")


# ── D. W - clean is what the filter discarded ─────────────────────────────────────────────────
def test_removed_is_what_the_filter_discarded():
    """D. the identity that the input-units return exists to make true.

    ``W - clean`` holds the noise and nothing of the signal.  It is also why the per-channel
    baseline is carried through rather than dropped: subtracting a ``clean`` that had no baseline
    would put the whole baseline into the residual, where it was never the filter's to remove."""
    W, truth = _frame()
    clean, _ = Aperture(W, window=None).extract()
    removed = W - clean
    noise = W - truth

    def corr(a, b):
        a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    assert corr(removed, noise) > 0.8, "what was removed must be the noise"
    assert _rel(removed, noise) < 0.6, "and it must be the noise in the right units too"
    assert abs(float(removed.mean())) < 0.3 * float(W.std()), (
        "the residual must not carry the per-channel baseline")


# ── E. the map spans the screen's feature axis ────────────────────────────────────────────────
def test_map_spans_a_folded_feature_axis():
    """E. when the feature axis coarsens, the map is the fold group's and still spans the screen.

    The group map is the right one rather than a fallback: it is exact for whatever is common
    across a fold group, which is what a resolved mode is, and the fold merges only adjacent
    channels -- it coarsens precisely when the feature marginal is smooth, which is the same
    statement as neighbours being alike.  Scored against the truth folded the same way."""
    rng = np.random.default_rng(3)
    T, F = 200, 256
    t = np.arange(T)[:, None]; f = np.arange(F)[None, :]
    smooth = np.exp(-((t - 100) ** 2) / 50.0) * np.exp(-((f - 128) ** 2) / 2000.0)
    gain = np.linspace(1.0, 6.0, F)[None, :]
    truth = np.linspace(-3.0, 9.0, F)[None, :] + gain * 20.0 * smooth
    W = truth + gain * rng.standard_normal((T, F))

    clean, info = Aperture(W, window=None).extract()
    N, F_eff = info["screen_shape"]
    assert F_eff < F, "this frame is meant to coarsen"
    assert info["centre"].shape == (F_eff,) and info["scale"].shape == (F_eff,)

    # the truth and the raw frame on the screen's own grid -- the same fold, applied to both
    fold = lambda A: _fold_axis(_fold_axis(A, N, axis=0), F_eff, axis=1)
    assert _rel(clean, fold(truth)) < _rel(fold(W), fold(truth)), (
        "on the folded grid the filter must still beat the raw frame")
    assert _rel(clean, fold(truth)) < 0.15, "and the group map must land it close to the truth"


def test_dead_channels_do_not_misalign_the_map():
    """E. a fully-dead channel is dropped from the screen, and the map is aligned with what
    SURVIVED -- an off-by-one here would silently apply the wrong channel's gain."""
    W, _ = _frame(F=24)
    W[:, 6] = np.nan                                  # never measured
    clean, info = Aperture(W, window=None).extract()
    assert clean.shape[1] == info["centre"].shape[0] == info["scale"].shape[0]
    assert clean.shape[1] < W.shape[1], "the dead channel must be dropped"
