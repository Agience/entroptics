"""integrated_autocorrelation and spread_over_chains: the error of a chain mean."""
import numpy as np
import pytest

from entroptics import IntegratedAutocorrelation, integrated_autocorrelation, spread_over_chains
from entroptics.reads import decay


def _ar1(phi, T, seed):
    rng = np.random.default_rng(seed)
    n = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = n[0] / np.sqrt(1.0 - phi ** 2)          # start in the stationary law
    for t in range(1, T):
        x[t] = phi * x[t - 1] + n[t]
    return x


def test_it_sums_decays_positive_lobe():
    rng = np.random.default_rng(9)
    x = 0.05 * rng.standard_normal(300).cumsum() + rng.standard_normal(300)
    cn = np.asarray(decay(x[:, None])) / np.asarray(decay(x[:, None]))[0]
    W = int(np.where(cn[1:] <= 0)[0][0])
    r = integrated_autocorrelation(x)
    assert isinstance(r, IntegratedAutocorrelation)
    assert r.window == W
    assert r.tau_int == pytest.approx(0.5 + cn[1:W + 1].sum(), rel=1e-12)
    assert integrated_autocorrelation(x, window=2).tau_int == pytest.approx(0.5 + cn[1:3].sum(), rel=1e-12)


@pytest.mark.parametrize("phi", [0.5, 0.8])
def test_an_ar1_chain_returns_its_known_time(phi):
    r = integrated_autocorrelation(_ar1(phi, 20000, seed=1))
    assert abs(r.tau_int - (1 + phi) / (2 * (1 - phi))) < 3 * r.tau_se
    assert r.tau_se < r.tau_int                     # resolved: the record holds many correlation times


def test_iid_sem_is_std_over_root_T():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(2000)
    r = integrated_autocorrelation(y)
    assert abs(r.sem - y.std() / np.sqrt(y.size)) < 3 * r.sem_se
    r0 = integrated_autocorrelation(y, window=0)
    assert r0.tau_int == 0.5 and r0.sem == pytest.approx(y.std() / np.sqrt(y.size), rel=1e-12)


def test_a_drift_is_flagged_by_its_own_error():
    """Negative control: a slow drift closes the window only where the mean removal forces it, and
    the correlation time's own error is then as large as the time itself -- not a small SEM."""
    rng = np.random.default_rng(3)
    T = 2000
    x = np.linspace(0.0, 3.0, T) + rng.standard_normal(T)
    r = integrated_autocorrelation(x)
    assert r.tau_se > r.tau_int
    assert r.sem_se > 0.5 * r.sem


def test_spread_over_chains():
    v = np.array([1.0, 2.0, 4.0, 5.0])
    assert spread_over_chains(v) == pytest.approx(v.std(ddof=1) / 2.0)
    with pytest.raises(ValueError):
        spread_over_chains([1.0])
    with pytest.raises(ValueError):
        spread_over_chains([1.0, np.nan])


def test_refusals():
    with pytest.raises(ValueError):
        integrated_autocorrelation(np.ones((10, 2)))
    with pytest.raises(ValueError):
        integrated_autocorrelation(np.ones(10))
    with pytest.raises(ValueError):
        integrated_autocorrelation(np.array([1.0, np.nan, 2.0]))
    with pytest.raises(ValueError):
        integrated_autocorrelation(np.arange(10.0), window=10)
    with pytest.raises(ValueError):
        integrated_autocorrelation(np.arange(10.0), window="auto")
    for bad in (True, np.True_, None, 2.5, -1):
        with pytest.raises(ValueError):
            integrated_autocorrelation(np.arange(10.0), window=bad)
    assert integrated_autocorrelation(np.arange(10.0), window=np.int64(3)).window == 3
