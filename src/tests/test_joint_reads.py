"""The joint reads — the joint power table and the entropies read off it.

Shannon 1948 sections 7 and 12, read on power, the intensity of each cell. Three entropies come
off one joint table so the identities that define these quantities hold exactly, to the last bit.

The failure modes these tests watch for:
  - the identities hold only approximately, because three code paths compute the pieces separately;
  - a coupled pair reads as independent, or an independent pair reads as coupled;
  - two frames that are not co-registered are silently truncated to a common length, publishing a
    number about a frame nobody handed in;
  - an absent cell contributes power to the joint;
  - equivocation is treated as symmetric.

`joint_entropies` is the one entry point onto the table and returns all six numbers, because they
are differences of the same three entropies computed once; `geometry()` sets the same precedent —
one call, one dict, the names in the keys.
"""
import numpy as np
import pytest

from entroptics.entropy import joint_entropies, joint_power


def _permutation_coupled(T=4000, F=8, seed=0):
    """The ceiling control: channel `i` of X fires exactly when channel `sigma(i)` of Y does, so one
    frame determines the other completely and nothing about X survives knowing Y."""
    rng = np.random.default_rng(seed)
    c = rng.integers(0, F, T)
    sigma = rng.permutation(F)
    X = np.zeros((T, F)); Y = np.zeros((T, F))
    X[np.arange(T), c] = 1.0
    Y[np.arange(T), sigma[c]] = 1.0
    return X, Y


def test_the_identities_are_exact_not_approximate():
    """`I = H_X + H_Y - H_XY` and `I = H_X - H_Y(X)` — both to the last bit.

    This is the whole reason the three entropies come off one table. Computed separately they would
    agree to float noise while three normalisations, three clips and three absent-cell rules stayed
    in step, and stop agreeing the moment one of them drifted."""
    rng = np.random.default_rng(1)
    g = joint_entropies(rng.random((64, 8)), rng.random((64, 5)))
    assert g["I_XY"] == g["H_X"] + g["H_Y"] - g["H_XY"]              # exactly
    assert g["I_XY"] == pytest.approx(g["H_X"] - g["H_X_given_Y"], abs=1e-15)
    assert g["I_XY"] == pytest.approx(g["H_Y"] - g["H_Y_given_X"], abs=1e-15)


def test_a_coupled_pair_reaches_the_ceiling():
    """Permutation coupling: `I = H(X) = H(Y)` and the equivocation is zero."""
    X, Y = _permutation_coupled()
    g = joint_entropies(X, Y)
    assert g["H_X"] == pytest.approx(g["H_Y"], abs=1e-12)
    assert g["I_XY"] == pytest.approx(g["H_X"], abs=1e-12)
    assert g["H_X_given_Y"] == pytest.approx(0.0, abs=1e-12)
    assert g["I_XY"] > 2.9                                            # 8 channels, near log2(8) = 3


def test_an_independent_pair_reads_near_zero():
    """The control that keeps the one above honest. Without it, a function returning `H(X)`
    unconditionally would pass every ceiling assertion."""
    rng = np.random.default_rng(2)
    assert joint_entropies(rng.random((4000, 8)), rng.random((4000, 5)))["I_XY"] < 0.01


def test_identical_frames_do_NOT_give_I_equals_H():
    """The textbook result `I(X;Y) = H(X)` when `X = Y` (Lesne, MSCS 2014, §2.3 eq. 7) holds for one
    random variable observed twice, where the joint sits on the diagonal; it does not hold here.

    The joint here is `p(i,j) = Σ_t p(t) p(i|t) p(j|t)` — conditionally independent given the ordered
    index, because the shared ordering is the only correspondence between an `F1`-alphabet and an
    `F2`-alphabet that co-registration gives us. Handing in the same frame twice therefore draws two
    channel indices independently from each row's profile, and a frame whose channels fire together
    correctly reads as telling little about itself.

    `test_a_coupled_pair_reaches_the_ceiling` is the control that does saturate — a per-row
    deterministic pairing. Both hold together, or the read means nothing."""
    rng = np.random.default_rng(3)
    W = rng.random((256, 8))
    g = joint_entropies(W, W)
    assert g["I_XY"] < 0.5 * g["H_X"], "self-coupling should NOT saturate on a dense frame"


def test_a_non_co_registered_pair_RAISES():
    """Two channels each ordered by its own private axis superpose into noise. Truncating to a
    common length here would publish a measurement of the misalignment."""
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="co-registered"):
        joint_entropies(rng.random((64, 8)), rng.random((63, 5)))
    with pytest.raises(ValueError, match="2-D"):
        joint_entropies(rng.random(64), rng.random((64, 5)))


def test_absent_cells_carry_no_power_into_the_joint():
    """Nonfinite and masked cells are absent from the joint, by either route, to the last bit."""
    rng = np.random.default_rng(5)
    X, Y = rng.random((64, 8)), rng.random((64, 5))
    bare = joint_entropies(X, Y)["I_XY"]

    padded = np.concatenate([X, np.full((64, 6), np.nan)], axis=1)
    assert joint_entropies(padded, Y)["I_XY"] == pytest.approx(bare, abs=1e-15)

    wide = np.concatenate([X, rng.random((64, 6))], axis=1)
    mask = np.zeros(wide.shape, dtype=bool); mask[:, 8:] = True
    assert joint_entropies(wide, Y, mask_x=mask)["I_XY"] == pytest.approx(bare, abs=1e-15)


def test_equivocation_is_directed():
    """`H_Y(X)` and `H_X(Y)` answer different questions and are different numbers. A caller reading
    one for the other gets a plausible bit-count about the wrong frame."""
    rng = np.random.default_rng(6)
    X, Y = rng.random((64, 8)), rng.random((64, 3))
    g = joint_entropies(X, Y)
    assert g["H_X_given_Y"] != pytest.approx(g["H_Y_given_X"], abs=1e-6)
    assert joint_entropies(Y, X)["H_X_given_Y"] == pytest.approx(g["H_Y_given_X"], abs=1e-15)


def test_joint_power_is_the_shape_the_alphabets_say():
    """`F1` and `F2` are separate alphabets; nothing requires them to be the same one."""
    rng = np.random.default_rng(7)
    J = joint_power(rng.random((32, 8)), rng.random((32, 5)))
    assert tuple(np.asarray(J).shape) == (8, 5)


def test_the_backends_agree():
    """numpy and torch, same read."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(8)
    X, Y = rng.random((64, 8)), rng.random((64, 5))
    gn = joint_entropies(X, Y)
    gt = joint_entropies(torch.from_numpy(X), torch.from_numpy(Y))
    for k in ("H_X", "H_Y", "H_XY", "I_XY", "H_X_given_Y"):
        assert float(gt[k]) == pytest.approx(gn[k], abs=1e-12), k
