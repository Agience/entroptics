"""Extract on noiseless rank-1 bursts: recovered to round-off at every width.

Every channel is scaled by its RMS about its mean, so a burst a sample wide is as resolvable as a
wide one, and a whitened column has sum |z|^2 = T: no cell exceeds sqrt(T), and nothing the screen
squares can leave the float range.  The floor's noise variance is the median singular value's, so a
noise-free record reads a noise variance of round-off and keeps its mode whole."""
import numpy as np
import pytest

from entroptics import Aperture, resolved_batch
from entroptics.entropy import normalize
from entroptics.projection import Projection, read_batch


def _burst(width, T=256, F=64, seed=5):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    return np.outer(np.exp(-0.5 * ((t - T / 2) / width) ** 2), np.abs(rng.standard_normal(F)) + 0.5)


@pytest.mark.parametrize("width", [0.5, 2.0, 8.0, 32.0, 128.0])
def test_a_noiseless_burst_is_recovered_exactly_at_every_width(width):
    W = _burst(width)
    clean, info = Aperture(W).extract()
    assert info["K_signal"] == 1
    assert np.linalg.norm(clean - W) / np.linalg.norm(W) < 1e-14


@pytest.mark.parametrize("width", [0.5, 2.0, 128.0])
def test_the_whitened_cells_are_bounded_by_the_record_length(width):
    W = _burst(width)
    Z = np.asarray(normalize(W))
    assert np.max(np.abs(Z)) <= np.sqrt(W.shape[0]) * (1 + 1e-12)
    assert np.allclose(np.sum(Z ** 2, axis=0), W.shape[0], rtol=1e-12)


def test_a_burst_reads_the_same_on_every_path():
    """Projection, read_batch and resolved_batch agree on a burst, and refloor reproduces the
    construction; an ordinary frame beside it is untouched."""
    W = _burst(2.0)
    N = np.random.default_rng(0).standard_normal(W.shape)
    p = Projection(W)
    assert p.K_signal == 1
    rb = read_batch(np.stack([W, N]))
    rs = resolved_batch(np.stack([W, N]))
    assert rb[0].K_signal == int(rs.K_signal[0]) == p.K_signal
    assert rb[0].sigma_top == pytest.approx(p.sigma_top, rel=1e-12)
    assert rb[0].noise_floor == pytest.approx(p.noise_floor, rel=1e-9, abs=1e-12 * p.sigma_top)
    q = p.refloor(None)
    assert q.K_signal == p.K_signal and q.noise_floor == p.noise_floor
    pn = Projection(N)
    assert rb[1].noise_floor == pn.noise_floor and rb[1].K_signal == pn.K_signal


def test_a_burst_reads_at_the_working_precision():
    import entroptics
    entroptics.set_precision(32)
    try:
        p = Projection(_burst(5.0))
        assert p.K_signal == 1 and np.isfinite(p.noise_floor) and np.isfinite(p.sigma_top)
    finally:
        entroptics.set_precision(64)


def test_a_bursts_energies_are_finite():
    rs = resolved_batch(_burst(2.0)[None], energy=True)
    en = np.asarray(rs.energy)[0]
    assert np.all(np.isfinite(en)) and np.any(en > 0)
