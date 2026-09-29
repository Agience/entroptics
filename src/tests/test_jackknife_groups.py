"""jackknife over explicit groups (streams, chains, blocks) and for array-valued reads."""
import numpy as np
import pytest

from entroptics import jackknife


def _hand(x, read, groups):
    """The delete-one-group jackknife written out, for comparison."""
    N, G = len(x), len(groups)
    theta = np.array([read(x[np.setdiff1d(np.arange(N), g)]) for g in groups])
    return read(x), np.sqrt((G - 1) / G * np.sum((theta - theta.mean(axis=0)) ** 2, axis=0))


def test_contiguous_groups_reproduce_n_bins_bit_for_bit():
    x = np.random.default_rng(0).standard_normal(103)
    for G in (2, 5, 10):
        a = jackknife(x, np.mean, n_bins=G)
        b = jackknife(x, np.mean, groups=np.array_split(np.arange(103), G))
        assert a == b


def test_a_vector_read_equals_the_per_element_scalar_jackknife():
    X = np.random.default_rng(1).standard_normal((60, 4))
    full, se = jackknife(X, lambda s: s.mean(axis=0), n_bins=6)
    assert full.shape == (4,) and se.shape == (4,)
    for j in range(4):
        fj, sj = jackknife(X, lambda s, j=j: float(s[:, j].mean()), n_bins=6)
        assert full[j] == pytest.approx(fj, abs=1e-15) and se[j] == pytest.approx(sj, abs=1e-15)


def test_unequal_non_contiguous_groups():
    x = np.random.default_rng(2).standard_normal(20)
    groups = [np.array([0, 5, 9]), np.array([1, 2]), np.array([3, 11, 12, 19]), np.array([7, 15])]
    full, se = jackknife(x, np.mean, groups=groups)
    hf, hs = _hand(x, np.mean, groups)
    assert full == pytest.approx(hf) and se == pytest.approx(hs)


def test_a_sample_in_no_group_is_never_deleted():
    x = np.arange(10.0)
    seen = []
    jackknife(x, lambda s: seen.append(s.copy()) or float(s.sum()), groups=[[0, 1], [2, 3]])
    assert all(9.0 in s for s in seen)                               # row 9 is in every subset


def test_list_input_keeps_rows_in_order():
    planes = [np.full(2, float(i)) for i in range(6)]
    out = []
    jackknife(planes, lambda ps: out.append([float(p[0]) for p in ps]) or 0.0, groups=[[4, 1], [0]])
    assert out[1] == [0, 2, 3, 5] and out[2] == [1, 2, 3, 4, 5]


def test_a_read_constant_across_subsets_has_zero_error():
    """Negative control: the jackknife must not manufacture spread."""
    x = np.random.default_rng(3).standard_normal(30)
    assert jackknife(x, lambda s: 1.0, groups=[range(0, 10), range(10, 20), range(20, 30)])[1] == 0.0
    assert np.all(jackknife(x, lambda s: np.ones(3), n_bins=5)[1] == 0.0)


def test_a_non_finite_replicate_gives_a_non_finite_error():
    x = np.arange(12.0)
    read = lambda s: np.array([s.mean(), np.nan if 0.0 not in s else 1.0])
    full, se = jackknife(x, read, n_bins=4)
    assert np.isfinite(se[0]) and not np.isfinite(se[1])


def test_refusals():
    x = np.zeros(6)
    for bad in ([[0, 1, 2, 3, 4, 5]], [[0, 1], [6]], [[0, 1], [1, 2]], [[0, 0], [1]], [[0], []]):
        with pytest.raises(ValueError):
            jackknife(x, np.mean, groups=bad)
    with pytest.raises(ValueError):
        jackknife(x, np.mean, groups=[[0], [1]], n_bins=2)
