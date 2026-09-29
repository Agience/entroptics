"""The read takes its input as given: any level the dtype holds, any memory layout, integer counts,
missing cells, and channels that never moved -- and every path agrees on it."""
import numpy as np
import pytest

from entroptics import Aperture
from entroptics import null_providers as NP
from entroptics.entropy import whiten_stats
from entroptics.projection import Projection, probe_signal


def _frame(seed=3, T=300, F=64):
    r = np.random.default_rng(seed)
    x = r.standard_normal((T, F))
    x[:, 10:30] += np.outer(r.standard_normal(T), np.hanning(20)) * 3
    return x


@pytest.mark.parametrize("level", [1e155, 1e300, 1e-300])
def test_the_read_does_not_move_with_the_recording_level(level):
    """The geometry squares the frame; carried by a power of two it reads the same at a level
    whose square leaves the float range (the fold collapsed to none there before)."""
    x = _frame()
    p, q = Projection(x), Projection(x * level)
    assert q.delta_F == pytest.approx(p.delta_F, rel=1e-12)
    assert q.coherence == pytest.approx(p.coherence, rel=1e-9)
    assert q.K_signal == p.K_signal


def test_the_read_does_not_depend_on_memory_layout():
    x = _frame()
    p, q = Projection(x), Projection(np.asfortranarray(x))
    assert (p.delta_F, p.coherence, p.sigma_top, p.noise_floor) == \
        (q.delta_F, q.coherence, q.sigma_top, q.noise_floor)


def test_whiten_stats_takes_counts_and_leaves_out_what_was_not_measured():
    x = np.random.default_rng(1).poisson(3, (50, 4))
    c, s = whiten_stats(np, x)
    assert np.allclose(c, x.mean(0)) and np.allclose(s, x.std(0))
    y = x.astype(float)
    y[3, 1], y[4, 2] = np.nan, np.inf
    c, s = whiten_stats(np, y)
    for j, i in ((1, 3), (2, 4)):
        keep = np.delete(y[:, j], i)
        assert c[j] == pytest.approx(keep.mean()) and s[j] == pytest.approx(keep.std())


def test_probe_signal_leaves_flat_channels_out_as_the_read_does():
    for i in range(60):
        r = np.random.default_rng(i)
        W = np.hstack([r.standard_normal((100, 10)), np.full((100, 3), 2.0)])
        if i % 2:
            W[:, :4] += np.outer(r.standard_normal(100), np.ones(4)) * 0.9
        assert probe_signal(W) is (Projection(W).K_signal > 0)


def test_a_row_measured_only_on_a_flat_channel_is_kept_whole():
    """The read drops the row (nothing but a constant was measured on it); the filter still returns
    it, at the channel's value, so clean + residual == W on every measured cell."""
    W = np.random.default_rng(0).standard_normal((64, 6))
    W[:, 2] = 5.0
    W[10, [0, 1, 3, 4, 5]] = np.nan
    clean, info = Aperture(W, window=None).extract()
    assert clean[10, 2] == 5.0 and info["residual"][10, 2] == 0.0
    m = np.isfinite(W)
    assert np.max(np.abs((clean + info["residual"] - W)[m])) < 1e-12


def test_a_small_weighted_ensemble_holds_its_level():
    """Each sample's deviation from the weighted mean it belongs to is short by 1 - 2/M + 1/eff;
    without that factor a small ensemble at unequal channel levels read structure in pure noise."""
    far, hits, n = 0.05, 0, 60
    for i in range(n):
        r = np.random.default_rng(500 + i)
        M, T, F = 8, 128, 64
        stack = r.standard_normal((M, T, F)) * np.exp(np.linspace(-1, 1, F))
        w = r.uniform(0.5, 1.5, M)
        agg = np.tensordot(w, stack, axes=(0, 0)) / w.sum()
        hits += Projection(agg, null=NP.weighted_effective(stack, w)).K_signal > 0
    assert hits / n <= 0.1
