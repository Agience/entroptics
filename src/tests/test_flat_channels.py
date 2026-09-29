"""A channel the whitening cannot give a noise scale (it never moved) leaves the screen the way a
dead channel does, and a filter keeps it at its centre.  A sparse channel moves, so it has a scale
and is read as the noise it is.

Before, a constant channel's level steered the entropy fold and diluted the columns it was folded
into: two constant channels beside eight of noise made the default read claim structure in 52% of
noise-only records (96% with eight), and eight sparse count channels, then left without a scale by
the median absolute deviation, in 38%."""
import numpy as np
import pytest

import entroptics as E
from entroptics import null_providers as NP
from entroptics.projection import Projection, read_batch

T = 256


def _rate(make, n=100, null=None):
    return np.mean([Projection(make(np.random.default_rng(i)), null=null).K_signal > 0 for i in range(n)])


@pytest.mark.parametrize("k", [2, 8])
def test_constant_channels_do_not_manufacture_structure(k):
    rate = _rate(lambda r: np.concatenate([r.standard_normal((T, 8)), np.full((T, k), 3.0)], 1))
    assert rate <= 0.08                        # 0.52 / 0.96 before; 0.03 with no constant channel


def test_sparse_channels_do_not_manufacture_structure():
    rate = _rate(lambda r: np.concatenate([r.standard_normal((T, 8)),
                                           r.poisson(0.2, (T, 8)).astype(float)], 1))
    assert rate <= 0.08                        # 0.385 before


def test_a_flat_channel_reads_as_if_it_were_absent():
    r = np.random.default_rng(1)
    N = r.standard_normal((T, 8))
    W = np.concatenate([N, np.full((T, 3), 3.0)], 1)
    p, q = Projection(W), Projection(N)
    assert np.array_equal(np.asarray(p.S), np.asarray(q.S)) and p.noise_floor == q.noise_floor
    assert p.K_signal == q.K_signal and list(np.flatnonzero(p._flat_cols)) == [8, 9, 10]


def test_every_read_path_agrees_on_a_frame_with_a_flat_channel():
    r = np.random.default_rng(2)
    W = np.concatenate([r.standard_normal((T, 6)), r.poisson(0.2, (T, 2)).astype(float),
                        np.full((T, 2), 3.0)], 1)
    V = r.standard_normal((T, 10))
    p = Projection(W)
    rb = read_batch([W, V])
    rs = E.resolved_batch(np.stack([W, V]))
    assert rb[0].K_signal == int(rs.K_signal[0]) == p.K_signal
    assert rb[0].noise_floor == float(rs.noise_floor[0]) == p.noise_floor
    assert rb[1].noise_floor == Projection(V).noise_floor


def test_a_filter_keeps_a_flat_channel_at_its_centre():
    r = np.random.default_rng(3)
    W = np.concatenate([r.standard_normal((T, 8)), np.full((T, 2), 3.0),
                        r.poisson(0.2, (T, 2)).astype(float)], 1)
    clean, info = E.Aperture(W).extract()
    assert np.all(clean[:, 8:10] == 3.0) and np.all(info["residual"][:, 8:10] == 0)
    assert np.nanmax(np.abs(clean + info["residual"] - W)) < 1e-12
    b = E.Aperture(W).basis()
    assert np.all(b.scale[8:10] == 0) and np.allclose(b.centre[8:10], [3, 3])
    assert np.all(b.scale[10:] > 0)                              # a sparse channel moves: it has a scale
    resolved, residual = b.split(W)
    assert np.max(np.abs(resolved + residual - W)) < 1e-12


def _median_row_floor(ctx):
    """The floor the default replaced: the MEDIAN row energy, de-biased by the Gaussian chi-square
    median.  Kept here only as the negative control that shows the level test can fail."""
    N, F = int(ctx.shape[0]), int(ctx.shape[1])
    re = np.sum(np.abs(np.asarray(ctx.data)) ** 2, axis=1)
    return float(np.sqrt(NP.screen_floor_sq(np.median(re) / NP.debias_denominator(N, F), N, F, ctx.far)))


def test_the_default_floor_holds_its_level_on_count_noise():
    """The default floor reads the mean cell energy, which estimates the per-cell variance for
    noise of any marginal; count noise holds the level, and a planted mode is still found."""
    pois = lambda r: r.poisson(1.0, (T, 8)).astype(float)        # noqa: E731
    assert _rate(pois) <= 0.08
    assert _rate(pois, null=_median_row_floor) >= 0.25            # negative control: the median floor

    def planted(r):
        x = r.poisson(3.0, (T, 8)).astype(float)
        return x + np.outer(r.standard_normal(T) * 1.2, np.ones(8) / np.sqrt(8)) * np.sqrt(3.0) * 3
    assert _rate(planted, n=40) == 1.0


def test_flat_channels_agree_on_every_path_and_a_small_channel_is_not_flat():
    """Only a channel that never moved is flat, at any level, and every path agrees on the set.  A
    channel at 1e-16 of its neighbours is a channel in its own units: it is whitened, not dropped."""
    r = np.random.default_rng(0)
    X = np.hstack([r.standard_normal((200, 3)), np.full((200, 1), 1e-300),
                   np.full((200, 2), -7.5e12)])
    p = Projection(X)
    assert list(np.flatnonzero(p._flat_cols)) == [3, 4, 5]
    Y = np.hstack([X[:, :3], 1.5e-16 * r.standard_normal((200, 1))])
    assert Projection(Y)._flat_cols is None
    q = Projection(X[:, :3])
    assert p.noise_floor == q.noise_floor == read_batch([X])[0].noise_floor
    assert float(E.resolved_batch(X[None]).noise_floor[0]) == p.noise_floor


def test_a_row_measured_only_on_flat_channels_is_dead():
    r = np.random.default_rng(1)
    W = np.hstack([r.standard_normal((200, 2)), np.full((200, 2), 3.0)])
    m = np.zeros(W.shape, bool)
    m[7, :2] = True
    p = Projection(W, m)
    q = Projection(np.delete(W, 7, 0)[:, :2])
    assert p.screen.shape == q.screen.shape and p.noise_floor == q.noise_floor
