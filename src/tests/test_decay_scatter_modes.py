"""decay_scatter in every mode decay reads: the channel terms sum to decay's profile, and the
per-lag standard error is the channels' own spread."""
import numpy as np
import pytest

from entroptics.reads import decay, decay_scatter


def _record(T=48, F=12, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((T, F))
    for t in range(1, T):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal(F)
    return x + 2.0


MODES = [dict(), dict(periodic=True), dict(disconnected=1.5), dict(periodic=True, disconnected=1.5),
         dict(disconnected=None), dict(periodic=True, disconnected=np.linspace(1.0, 3.0, 12))]


@pytest.mark.parametrize("kw", MODES)
def test_the_channel_terms_sum_to_decay(kw):
    """The pinned identity, in every mode: the per-lag SE is the spread of terms that sum to
    decay's own profile, so the two cannot describe different reads."""
    W = _record()
    ds = decay_scatter(W, **kw)
    C = np.asarray(decay(W, **kw))
    assert ds.se.shape == C.shape and ds.channels == W.shape[1]
    total = float((C ** 2).sum())
    assert ds.tail_share == pytest.approx(float((C[1:] ** 2).sum()) / total, rel=1e-12)
    assert ds.noise_share == pytest.approx(float((ds.se ** 2).sum()) / total, rel=1e-12)


def test_the_standard_error_is_the_channels_spread():
    W = _record(T=20, F=8, seed=3)
    Xc = W - W.mean(axis=0)
    T, F = W.shape
    Cf = np.array([[np.sum(Xc[:T - k, f] * Xc[k:, f]) / T for f in range(F)] for k in range(T)])
    assert np.allclose(Cf.sum(axis=1), np.asarray(decay(W)), atol=1e-12)
    assert np.allclose(decay_scatter(W).se, np.sqrt(F * Cf.var(axis=1, ddof=1)), atol=1e-12)


def test_a_periodic_se_is_symmetric_about_the_ring():
    se = decay_scatter(_record(T=31), periodic=True).se
    assert np.array_equal(se[1:], se[1:][::-1])


def test_identical_channels_have_no_scatter():
    """Negative control: replicates that agree exactly carry no sampling error -- zero up to the
    round-off of averaging identical values (a mean of equal floats need not equal them bitwise)."""
    col = _record(T=40, F=1, seed=5)
    W = np.repeat(col, 6, axis=1)
    eps = np.finfo(float).eps
    for kw in MODES[:4]:
        ds, C = decay_scatter(W, **kw), np.asarray(decay(W, **kw))
        assert np.all(ds.se <= 8 * eps * np.abs(C).max())


def test_it_refuses_what_decay_refuses():
    W = _record()
    W[5] = np.nan                                     # a dead row
    with pytest.raises(ValueError):
        decay(W, periodic=True)
    with pytest.raises(ValueError):
        decay_scatter(W, periodic=True)
