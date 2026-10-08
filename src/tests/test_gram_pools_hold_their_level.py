"""The pools -- ``ResolvedScreenBatch``, ``SpectralAccumulator`` and ``Dynamics`` -- hold their
false-alarm level on heavy right tails, on a window whose rows expire as the aperture forgets.

A Gram cannot hold the level: its closed-form edge reads a correlation at the extreme value of its
top eigenvalue, and on a heavy right tail that value sits in single rows -- a channel's largest cell
carries most of its energy, and two channels whose largest cells fall in one row read as a mode.
The Gram has summed the rows away, so each pool holds a window of rows and reads it at the library
floor: the exact permutation test where a single row could carry a noise eigenvalue over the closed
form, the closed form where none can.  Every level test fails on 0.2.7, where these pools read a
Gram at the closed form; the bound is ``far`` plus three binomial standard errors."""
import math

import numpy as np
import pytest

from entroptics import Aperture
from entroptics.batch import ResolvedScreenBatch, resolved_batch
from entroptics.dynamics import Dynamics, dynamics
from entroptics.null_providers import mp
from entroptics.reads import SpectralAccumulator, spectral_optics

FAR = 0.05


def _bound(n, far=FAR):
    return far + 3.0 * math.sqrt(far * (1.0 - far) / n)


def _lognormal(i, T, F):
    return np.random.default_rng(10_000 * T + 7 * F + i).lognormal(0.0, 1.5, (T, F))


def _pooled(x, plane=16):
    acc = SpectralAccumulator(x.shape[1])
    for k in range(0, x.shape[0], plane):
        acc.add(x[k:k + plane])
    return acc


def test_the_closed_form_alone_over_reads_a_heavy_tail():
    """The defect the pools repair: the closed-form edge on all of a lognormal record's rows."""
    n = 200
    hits = [spectral_optics(_lognormal(i, 256, 32), null=mp).resolved_modes > 0 for i in range(n)]
    assert np.mean(hits) > _bound(n)


def test_the_batched_screen_holds_its_level_on_a_heavy_tail():
    n, T, F = 200, 256, 32
    X = np.stack([_lognormal(i, T, F) for i in range(n)])
    rsb = ResolvedScreenBatch(n, F)
    rsb.update(X)
    assert np.mean(rsb.K_signal > 0) <= _bound(n)


def test_the_pooled_accumulator_holds_its_level_on_a_heavy_tail():
    n, T, F = 200, 256, 32
    hits = [_pooled(_lognormal(i, T, F)).spectral(seed=i).resolved_modes > 0 for i in range(n)]
    assert np.mean(hits) <= _bound(n)


@pytest.mark.parametrize("T,F", [(64, 128), (1024, 32)])
def test_the_operator_holds_its_level_on_a_heavy_tail(T, F):
    """Under-sampled (the DMD truncation's floor) and far past ``2F`` pairs, where it reads its
    window."""
    n = 200
    ds = [dynamics(_lognormal(i, T, F)) for i in range(n)]
    assert np.mean([d.resolved(seed=i) > 0 for i, d in enumerate(ds)]) <= _bound(n)
    if T - 1 < 2 * F:
        assert np.mean([d._signal_rank() > 0 for d in ds]) <= _bound(n)


def test_a_pool_of_planes_reads_the_closed_form_at_its_own_degrees_of_freedom():
    """Each plane is centred on its own mean, so a pool of M planes has T - M degrees of freedom,
    not T - 1; read at T - 1 the closed-form edge sits too low (complex Gaussian, 4 planes of 16 x
    128: 17.5% false alarms at 5%)."""
    n, T, F = 400, 64, 128
    rate = []
    for i in range(n):
        r = np.random.default_rng(i)
        acc = _pooled(r.standard_normal((T, F)) + 1j * r.standard_normal((T, F)))
        assert acc.dof == T - T // 16                     # F + 1 > T: every plane is held
        rate.append(acc.spectral(null=mp).resolved_modes > 0)
    assert np.mean(rate) <= _bound(n)


def test_one_plane_pooled_is_the_single_plane_read():
    """One plane pooled is the single-plane read under either floor; by default the pool -- the
    operator's cut point -- takes the closed form where no row can cross it, which a Gaussian plane
    of 200 rows is."""
    from entroptics.null_providers import permutation
    x = np.random.default_rng(3).standard_normal((200, 16))
    acc = SpectralAccumulator(16).add(x)
    assert acc.dof == 199
    assert acc.spectral(null=mp).noise_floor == spectral_optics(x, null=mp).noise_floor
    exact = permutation()
    assert acc.spectral(null=exact).noise_floor == pytest.approx(
        spectral_optics(x, null=exact).noise_floor, rel=1e-12)
    assert acc.spectral().noise_floor == acc.spectral(null=mp).noise_floor


def _burst(snr, i):
    t = np.arange(64)[:, None]; f = np.arange(256)[None, :]
    B = (np.exp(-0.5 * ((t - 0.46 * 64) / 3.0) ** 2) * np.exp(-0.5 * ((f - 128) / (0.22 * 256)) ** 2)
         + 0.7 * np.exp(-0.5 * ((t - 0.54 * 64) / 4.0) ** 2) * np.exp(-0.5 * ((f - 0.66 * 256) / (0.16 * 256)) ** 2))
    return B / B.max() + np.random.default_rng(i).standard_normal(B.shape) / snr


@pytest.mark.parametrize("snr", [2, 5, 10])
def test_the_windows_keep_the_burst(snr):
    """The calibration burst -- broadband, a few rows long, rank 2 -- is found by every pool."""
    X = np.stack([_burst(snr, i) for i in range(4)])
    rsb = ResolvedScreenBatch(4, 256)
    rsb.update(X)
    assert np.all(rsb.K_signal >= 1)
    for x in X[:2]:
        assert dynamics(x).resolved() >= 1


def test_a_burst_that_has_left_the_window_is_no_longer_read():
    """The window is what the aperture currently sees: a burst is read while it is in it, and once
    the aperture has forgotten it -- no mode coherent, the window back to ``F + 1`` frames -- it is
    not."""
    F = 16
    r = np.random.default_rng(7)
    burst = np.outer(r.standard_normal(12), r.standard_normal(F)) * 8 + r.standard_normal((12, F))
    d = Dynamics(F)
    for row in r.standard_normal((100, F)):
        d.update(row)
    for row in burst:
        d.update(row)
    assert d.resolved() >= 1
    for row in r.standard_normal((20 * (F + 1), F)):
        d.update(row)
    assert d.resolved() == 0
    assert len(d.window) == F + 1


def test_the_operator_window_is_the_aperture_window():
    """One derivation: the frames an aperture stream keeps are the frames its operator reads."""
    F = 8
    ap = Aperture()
    for row in np.random.default_rng(1).standard_normal((300, F)):
        ap.update(row)
    W = np.asarray(ap.W)
    assert np.array_equal(ap._core().window, W)
    assert W.shape[0] == F + 1                            # noise: nothing coherent, the minimum


def test_streamed_windows_are_bounded_on_noise():
    """Rows that stream in expire: a screen fed in blocks holds its latest block whole and, of what
    came before, only what the aperture still sees."""
    F = 12
    rsb = ResolvedScreenBatch(2, F)
    x = np.random.default_rng(2).standard_normal((2, 2000, F))
    for k in range(0, 2000, 50):
        rsb.update(x[:, k:k + 50])
    rsb.K_signal
    assert all(w.shape[0] == 50 for w in rsb.windows)


def test_a_batch_record_is_read_whole():
    """A record handed in at once is read whole -- a burst in the middle of it is found by every pool,
    as the whole-record reads find it -- and an ensemble pool keeps every plane."""
    T, F = 256, 32
    hits_b, hits_d = 0, 0
    for i in range(20):
        r = np.random.default_rng(300 + i)
        x = r.standard_normal((T, F))
        w = T // 64
        x[T // 2 - w // 2:T // 2 - w // 2 + w] += 2.5 * np.hanning(w)[:, None]
        b = ResolvedScreenBatch(1, F)
        b.update(x[None])
        assert b.windows[0].shape[0] == T
        hits_b += int(b.K_signal[0]) > 0
        hits_d += dynamics(x).resolved() > 0
    assert hits_b >= 17 and hits_d >= 17
    acc = _pooled(np.random.default_rng(1).standard_normal((2000, F)), plane=40)
    assert acc.T == 2000 and acc.n_planes == 50


def test_the_switch_reads_the_tail():
    """Below the record's resolution the default takes the closed form where no row can carry a
    noise eigenvalue over it: nearly always on Gaussian noise, never on lognormal noise (sigma =
    1.5)."""
    from entroptics.null_providers import (FloorContext, closed_form_holds, closed_form_far,
                                           row_influence)

    def ctx(x, far):
        Xc = x - x.mean(0)
        return FloorContext(spectrum=None, data=Xc, shape=x.shape, far=far, kind="spectral", rng=None)
    g = [closed_form_holds(ctx(np.random.default_rng(i).standard_normal((1024, 64)), 1e-4)) for i in range(40)]
    ln = [closed_form_holds(ctx(_lognormal(i, 1024, 64), far)) for i in range(40) for far in (0.05, 1e-4)]
    assert np.mean(g) >= 0.9 and not any(ln)
    # at a level the record resolves (far * T >= 1) the exact test is the read, light tail or not
    assert not closed_form_holds(ctx(np.random.default_rng(0).standard_normal((1024, 64)), 0.01))
    c = ctx(_lognormal(0, 1024, 64), 0.05)
    assert closed_form_far(c) < 0.05 and row_influence(c.data, "spectral") > 0


def test_a_strict_level_on_light_tails_takes_no_draws():
    """At far = 1e-6 the exact test would take 999 999 draws; on Gaussian noise no row can carry a
    noise eigenvalue over the closed form at that level, so the read is the closed form."""
    x = np.random.default_rng(4).standard_normal((1024, 64))
    so = spectral_optics(x, far=1e-6)
    assert so.noise_floor == spectral_optics(x, far=1e-6, null=mp).noise_floor


def test_the_operator_significance_is_the_floor_it_counts_against():
    """``resolved == #(p <= far)`` where the exact branch is taken."""
    for i in range(5):
        d = dynamics(_lognormal(i, 128, 16))
        sig = d.significance()
        assert d.resolved() == int(np.sum(sig.pvalue <= FAR))


def test_a_resumed_operator_keeps_its_window():
    x = np.random.default_rng(5).standard_normal((40, 8))
    d = dynamics(x)
    s = Dynamics.from_state(d.state())
    assert np.array_equal(s.window, d.window) and s.resolved() == d.resolved()
    m = dynamics(x[:20]).merge(dynamics(x[20:]), adjacent=True)
    assert len(m.window) <= len(x)
