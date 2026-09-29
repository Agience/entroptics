"""ModePowers.spectrum: the Fourier view rendered from the operator's modes."""
import numpy as np
import pytest

from entroptics import Aperture, ModePowers


def _two_oscillators(T=4000, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([0.98 * np.exp(2j * np.pi * 0.05), 0.9 * np.exp(2j * np.pi * 0.2)])
    z = np.zeros(2, complex)
    x = np.zeros((T, 4))
    for i in range(1, T):
        z = mu * z + rng.standard_normal(2) + 1j * rng.standard_normal(2)
        x[i] = [z[0].real, z[0].imag, z[1].real, z[1].imag]
    return x


def test_it_is_the_transform_of_the_reconstructed_decay():
    dy = Aperture(_two_oscillators()).dynamics()
    m = dy.modes()
    f = np.linspace(-0.5, 0.5, 20001)[:-1]
    S = m.spectrum(f)
    assert S.mean() == pytest.approx(float(np.sum(m.power)), rel=1e-9)     # integrates to C(0)
    C = np.array([np.mean(S * np.cos(2 * np.pi * f * k)) for k in range(40)])
    assert np.allclose(C / C[0], np.asarray(dy.reconstruct_decay(40)), atol=1e-9)
    assert np.array_equal(m.spectrum(f), m.spectrum(-f))


def test_it_peaks_at_the_modes_frequencies_between_any_bins():
    m = Aperture(_two_oscillators()).dynamics().modes()
    f = np.linspace(0.0, 0.5, 50001)
    S = m.spectrum(f)
    top = f[np.argmax(S)]
    assert top == pytest.approx(abs(np.angle(m.mu[0])) / (2 * np.pi), abs=2e-5)


def test_continuous_across_the_unit_circle():
    """A mode at |mu| = 1/r and at r draw the same line: the width is |alpha| either way."""
    def mp(mu):
        mu = np.atleast_1d(np.asarray(mu, complex))
        return ModePowers(mu=mu, alpha=-np.log(np.abs(mu)), beta=np.angle(mu),
                          power=np.ones(mu.size), share=np.ones(mu.size) / mu.size)
    f = np.linspace(-0.5, 0.5, 1001)
    r = 0.97
    a = mp(r * np.exp(0.3j)).spectrum(f)
    b = mp(np.exp(0.3j) / r).spectrum(f)
    assert np.allclose(a, b, rtol=1e-12) and np.all(a > 0)


def test_no_modes_is_a_zero_view():
    z = np.zeros(0)
    assert np.array_equal(ModePowers(z, z, z, z, z).spectrum([0.1, 0.2]), np.zeros(2))


def test_a_pole_at_zero_is_white_and_quiet():
    """mu = 0 is a mode with no memory: a flat line of its power, with no divide warning."""
    mp = ModePowers(mu=np.array([0j]), alpha=np.array([np.inf]), beta=np.array([0.0]),
                    power=np.array([2.0]), share=np.array([1.0]))
    assert np.allclose(mp.spectrum(np.linspace(-0.5, 0.5, 11)), 2.0)
