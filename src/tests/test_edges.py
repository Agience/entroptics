"""Degenerate inputs must not crash and must return sane, finite reads."""
import numpy as np
import pytest

from entroptics.entropy import shannon_bits

from entroptics import aperture as A
from entroptics.projection import Projection, coherence
from entroptics.reads import decay, concentration, spectral_optics


@pytest.mark.parametrize("shape", [(1, 8), (16, 1), (2, 2), (3, 3), (2, 16), (16, 2)])
def test_degenerate_shapes_optics_runs(shape):
    """Every shape reads without raising.  The fill is a number exactly when there is variation
    to read: the reads are taken on the centred block, so a frame of a SINGLE observation is
    identically zero once centred and has no structure to be a fraction of."""
    o = A.Aperture(np.random.default_rng(0).standard_normal(shape)).optics()
    if shape[0] > 1:
        assert 0.0 < o["phi"] <= 1.0 + 1e-12
    else:
        assert np.isnan(o["phi"])          # one row, centred, is nothing
    assert o["n_T"] >= 1 and o["n_F"] >= 1


def test_a_baseline_is_not_structure():
    """Adding a constant says nothing about the signal, so it must not move the fill.  The read
    used to be taken on the raw block, where an offset lands in a leading singular value: a
    global constant moved `phi` by more than a factor of three."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((200, 3)) @ rng.standard_normal((3, 32))         + 0.3 * rng.standard_normal((200, 32))
    ref = A.Aperture(W, window=None).phi
    for offset in (7.0, 50.0, rng.standard_normal(32) * 4):        # global, and per-channel
        assert A.Aperture(W + offset, window=None).phi == pytest.approx(ref, rel=1e-12)


@pytest.mark.parametrize("shape", [(1, 8), (16, 1), (2, 2)])
def test_degenerate_shapes_screen_runs(shape):
    sc = Projection(np.random.default_rng(0).standard_normal(shape))
    assert sc.K_signal >= 0
    assert np.isfinite(sc.coherence)


def test_all_zeros_is_handled():
    """A screen with no power has no active modes to take a fraction of, so the fill is not a
    number.  Reporting 1/n would be the rank-1 reading (PAPER Lemma 3.2) -- maximally coherent,
    maximum magnification -- claimed from a frame that carries nothing; reporting 0 says the same
    thing at the other end of the range.  The reads that are not ratios still answer."""
    o = A.Aperture(np.zeros((20, 10))).optics()
    assert np.isnan(o["phi"])
    assert o["a_delta"] == 0.0            # no fluctuation -> no decay
    assert o["strehl"] == 0.0             # no peak mode to carry power
    assert A.Projection(np.zeros((20, 10))).K_signal == 0


def test_a_powerless_screen_does_not_read_as_a_perfect_mode():
    """A genuine single mode and a frame carrying nothing must not produce the same aperture.

    Checked as the relation the paper states, not against a written-down number: Lemma 3.2 says
    phi = 1/n exactly when the rank is 1, so the rank-1 frame is compared with 1/n read off its
    own live width.  The powerless frames are then required only to DIFFER -- that is the whole
    claim, and it holds whatever value they take, so nothing here fixes one for them."""
    rng = np.random.default_rng(1)
    real = np.outer(rng.standard_normal(40), rng.standard_normal(8))     # rank 1 by construction
    r = A.Aperture(real, window=None)
    n = min(int(v) for v in r.W.shape)                # the singular spectrum's own length
    assert np.linalg.matrix_rank(real) == 1
    assert r.phi == pytest.approx(1.0 / n)            # Lemma 3.2, read off the frame
    assert r.magnification == pytest.approx(1.0 / r.phi)

    for dead in (np.full((40, 8), np.nan), np.zeros((40, 8))):
        d = A.Aperture(dead, window=None)
        for read in ("phi", "phi_F", "phi_T", "etendue", "magnification"):
            assert not float(getattr(d, read)) == pytest.approx(float(getattr(r, read))), read


def test_the_fill_fraction_is_formed_in_one_place():
    """The whole-screen fill and the per-axis fills are the same construction, 2^H/length, so they
    must agree about a spectrum with no power in it.  They disagreed while each formed the quotient
    for itself: one returned 0, the other 1/n."""
    from entroptics.reads import _fill_of
    ev = np.array([4.0, 1.0, 1.0])
    assert _fill_of(ev, 3) == pytest.approx(2.0 ** shannon_bits(ev) / 3)
    for empty in (np.zeros(3), np.zeros(0)):
        assert np.isnan(_fill_of(empty, 3))
    assert np.isnan(_fill_of(ev, 0))


def test_an_empty_axis_is_refused_and_says_why():
    """An axis of length 0 carries no cell at all, which is a different statement from an axis
    whose cells are all missing -- the second is a mask, the first is not a screen.  It used to
    surface as a bare ZeroDivisionError from inside the read."""
    for shape in ((0, 8), (8, 0), (0, 0)):
        with pytest.raises(ValueError, match="non-empty"):
            A.Aperture(np.zeros(shape))
        with pytest.raises(ValueError, match="non-empty"):
            A.Projection(np.zeros(shape))


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
    sc = Projection(W, mask=mask)
    assert sc.K_signal >= 0
    assert np.isfinite(sc.coherence)


def test_coherence_below_threshold_is_zero():
    """N < 2*lag+2 has too few rows for a null -> coherence defined as 0."""
    assert coherence(np.random.default_rng(0).standard_normal((3, 6))) == 0.0


@pytest.mark.parametrize("lag", [1, 2])
def test_coherence_null_variance_is_exact(lag):
    """coherence() standardises by the exact permutation variance (Def 5.3): for a small
    screen the null mean and variance it uses match a full enumeration of all N! row
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
