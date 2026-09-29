"""cross_covariance: the lagged covariance of two records, with decay's conventions."""
import numpy as np
import pytest

from entroptics import cross_covariance
from entroptics.reads import decay


def _rec(T=64, F=5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, F))
    for t in range(1, T):
        x[t] = 0.7 * x[t - 1] + rng.standard_normal(F)
    return x + 1.0


@pytest.mark.parametrize("kw", [dict(), dict(periodic=True), dict(disconnected=0.5), dict(disconnected=None)])
def test_with_itself_it_is_decay(kw):
    X = _rec()
    assert np.allclose(cross_covariance(X, X, range(X.shape[0]), **kw), np.asarray(decay(X, **kw)), atol=1e-12)


def test_a_shifted_copy_peaks_at_its_shift():
    X = _rec(T=200, F=3, seed=1)
    Y = np.roll(X, 3, axis=0)
    lags = range(-10, 11)
    c = cross_covariance(X, Y, lags, periodic=True)
    assert list(lags)[int(np.argmax(c))] == 3


def test_negative_lags_are_the_swapped_pair():
    X, Y = _rec(seed=2), _rec(seed=3)
    assert np.allclose(cross_covariance(X, Y, [-4, -1]), cross_covariance(Y, X, [4, 1]), atol=1e-12)


def test_independent_noise_shows_no_lag_above_its_scatter():
    """Negative control: two independent white records.  At each lag the covariance of i.i.d.
    unit noise over F channels has standard deviation sqrt(F (T - |tau|)) / T; nothing may stand
    at five of those."""
    rng = np.random.default_rng(4)
    T, F = 400, 8
    X, Y = rng.standard_normal((T, F)), rng.standard_normal((T, F))
    lags = np.arange(-20, 21)
    c = cross_covariance(X, Y, lags)
    sd = np.sqrt(F * (T - np.abs(lags))) / T
    assert np.all(np.abs(c) < 5 * sd)


@pytest.mark.parametrize("level", [(0.1, 0.9), [0.1, 0.9], np.array([0.1, 0.9])])
def test_a_per_channel_level_is_decays(level):
    """A tuple is a level, one per channel, exactly as decay reads it -- never a pair of records."""
    X = _rec(T=24, F=2, seed=7)
    assert np.allclose(cross_covariance(X, X, range(24), disconnected=level),
                       np.asarray(decay(X, disconnected=level)), atol=1e-12)


def test_complex_records_match_decay():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((30, 3)) + 1j * rng.standard_normal((30, 3))
    assert np.allclose(cross_covariance(X, X, range(30)), np.asarray(decay(X)), atol=1e-12)


def test_refusals():
    X, Y = _rec(seed=5), _rec(seed=6)
    T = X.shape[0]
    with pytest.raises(ValueError):
        cross_covariance(X, Y[:10], [0])
    with pytest.raises(ValueError):
        cross_covariance(X, Y, [T])
    with pytest.raises(ValueError):
        cross_covariance(X, Y, [1.7])
    Z = Y.copy(); Z[3, 1] = np.nan
    with pytest.raises(ValueError):
        cross_covariance(X, Z, [0])
