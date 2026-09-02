"""Degenerate inputs must not crash and must return sane, finite reads."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.screen import Screen, coherence
from entroptics.reads import decay, concentration, spectral_optics


@pytest.mark.parametrize("shape", [(1, 8), (16, 1), (2, 2), (3, 3), (2, 16), (16, 2)])
def test_degenerate_shapes_optics_runs(shape):
    o = A.Aperture(np.random.default_rng(0).standard_normal(shape)).optics()
    assert 0.0 < o["phi"] <= 1.0 + 1e-12
    assert o["n_T"] >= 1 and o["n_F"] >= 1


@pytest.mark.parametrize("shape", [(1, 8), (16, 1), (2, 2)])
def test_degenerate_shapes_screen_runs(shape):
    sc = Screen(np.random.default_rng(0).standard_normal(shape))
    assert sc.K_signal >= 0
    assert np.isfinite(sc.coherence)


def test_all_zeros_is_handled():
    o = A.Aperture(np.zeros((20, 10))).optics()
    assert np.isfinite(o["phi"])
    assert o["a_delta"] == 0.0            # no fluctuation -> no decay


def test_constant_signal_is_handled():
    o = A.Aperture(np.full((20, 10), 3.7)).optics()
    assert o["a_delta"] == 0.0
    assert np.isfinite(o["strehl"])


def test_complex_tiny():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    o = A.Aperture(W).optics()
    assert np.isfinite(o["phi"])


def test_masked_screen_runs():
    W = np.random.default_rng(0).standard_normal((40, 20))
    mask = np.zeros(W.shape, bool)
    mask[5:9, 3:7] = True
    sc = Screen(W, mask=mask)
    assert sc.K_signal >= 0
    assert np.isfinite(sc.coherence)


def test_coherence_below_threshold_is_zero():
    """N < 2*lag+2 has too few rows for a null -> coherence defined as 0."""
    assert coherence(np.random.default_rng(0).standard_normal((3, 6))) == 0.0


@pytest.mark.parametrize("lag", [1, 2])
def test_coherence_null_variance_is_exact(lag):
    """coherence() standardises by the EXACT permutation variance (Def 5.3): for a small
    screen the null mean and variance it uses match a FULL enumeration of all N! row
    permutations -- Theorem 5.2 (mean) and the Cliff-Ord/Mantel second moment (variance)."""
    from itertools import permutations
    N, F = 6, 3
    S = np.random.default_rng(1).standard_normal((N, F))
    R = np.real(S @ S.conj().T) ** 2
    M = N - lag
    As = np.array([R[np.asarray(p)[:M], np.asarray(p)[lag:lag + M]].mean()
                   for p in permutations(range(N))])           # exact null over all N! perms
    mu_true, var_true = float(As.mean()), float(As.var())
    d = np.diag(R)
    mu = (R.sum() - d.sum()) / (N * (N - 1))                    # mu independent of coherence()
    A = np.diag(R, lag).mean()
    z = coherence(S, lag=lag)
    assert mu == pytest.approx(mu_true, rel=1e-9)               # Theorem 5.2
    assert z != 0.0
    assert ((A - mu) / z) ** 2 == pytest.approx(var_true, rel=1e-9)   # Def 5.3 (exact variance)


def test_single_frame_dynamics_has_no_modes():
    from entroptics.dynamics import Dynamics
    d = Dynamics(4).update(np.random.default_rng(0).standard_normal(4))
    assert d.rates().n_modes == 0


def test_concentration_empty_and_single():
    assert concentration(np.zeros((0, 5))).n == 0
    c = concentration(np.random.default_rng(0).standard_normal((1, 5)))
    assert c.focus == pytest.approx(1.0, abs=1e-9)


def test_diffraction_limit_of_flat_decay():
    from entroptics.reads import diffraction_limit
    dl = diffraction_limit(np.zeros(8))
    assert dl.a_delta == 0.0 and dl.H == 0.0
