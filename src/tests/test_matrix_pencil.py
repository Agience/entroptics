"""matrix_pencil: the generalised eigenproblem of a measured pair of correlator matrices -- the same
construction as hankel_spectrum, which uses it."""
import numpy as np
import pytest

from entroptics import MatrixPencil, hankel_spectrum, matrix_pencil


def _planted(E, t, seed=0):
    """C(t) = sum_k v_k v_k^T e^{-E_k t} with independent operator overlaps v_k."""
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((len(E) + 2, len(E)))
    return sum(np.outer(V[:, k], V[:, k]) * np.exp(-E[k] * t) for k in range(len(E)))


def test_a_planted_two_state_matrix_returns_its_levels():
    E = np.array([0.3, 1.1])
    mp = matrix_pencil(_planted(E, 0), _planted(E, 1))
    assert isinstance(mp, MatrixPencil)
    assert mp.evals[:2] == pytest.approx(np.exp(-E), abs=1e-10)


def test_the_vectors_solve_the_generalised_problem():
    E = np.array([0.2, 0.7, 1.5])
    C0, C1 = _planted(E, 0, seed=2), _planted(E, 1, seed=2)
    mp = matrix_pencil(C0, C1)
    for k in range(3):
        v = mp.vectors[:, k]
        assert np.allclose(C1 @ v, mp.evals[k] * (C0 @ v), atol=1e-9)


def test_it_agrees_with_hankel_spectrum_on_a_hankel_input():
    c = 0.6 * 0.9 ** np.arange(12) + 0.4 * 0.5 ** np.arange(12)
    n = 2
    idx = np.add.outer(np.arange(n + 1), np.arange(n + 1))
    hs = hankel_spectrum(c, n)
    mp = matrix_pencil(c[idx] / c[0], c[idx + 1] / c[0])
    assert np.array_equal(mp.evals, hs.evals) and mp.psd == hs.psd


def test_an_indefinite_C0_is_reported_and_its_direction_dropped():
    """Negative control: lost positivity shows as psd < 0, and the negative direction is not used."""
    C0 = np.diag([2.0, 1.0, -0.5])
    C1 = np.diag([1.0, 0.4, 0.3])
    mp = matrix_pencil(C0, C1)
    assert mp.psd < 0
    assert mp.evals.size == 2 and mp.evals == pytest.approx([0.5, 0.4])


def test_refusals():
    with pytest.raises(ValueError):
        matrix_pencil(np.eye(3), np.eye(2))
    with pytest.raises(ValueError):
        matrix_pencil(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        matrix_pencil(np.eye(2) + 0j, np.eye(2))


def test_the_cut_is_the_noise_c0_itself_shows():
    """C0's most negative eigenvalue is the noise the record shows: a direction at or below its size
    is dropped, one above it kept -- nothing chosen.  An explicit rcond is still the caller's."""
    C0 = np.diag([2.0, 0.3, 0.1, -0.2])
    C1 = np.diag([1.0, 0.2, 0.05, 0.1])
    assert matrix_pencil(C0, C1).evals.size == 2                   # 0.1 <= |-0.2| is noise
    assert matrix_pencil(C0, C1, rcond=1e-6).evals.size == 3       # the caller's own cut
    exact = np.diag([2.0, 0.3, 1e-9])
    assert matrix_pencil(exact, exact * 0.5).evals.size == 3       # PSD: only round-off is cut
