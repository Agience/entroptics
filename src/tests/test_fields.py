"""N-D field reduction: the geometry-preserving slab reader and the two reductions.
The load-bearing property is that keeping the plane intact is not the same as a naive
flatten: collapsing a plane's axes into one feature axis can change the sign of the
result, so callers that need the geometry preserved must use the slab reader."""
import numpy as np
import pytest

from entroptics import fields
from entroptics.reads import phi


def test_slabs_are_the_actual_sub_planes():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((5, 8, 8))          # 5 planes of 8x8
    sl = list(fields.slabs(field, plane_axes=(1, 2)))
    assert len(sl) == 5
    for k, plane in enumerate(sl):
        assert np.asarray(plane).shape == (8, 8)
        assert np.allclose(np.asarray(plane), field[k])   # each slab is the intact sub-plane


def test_slab_count_over_multiple_axes():
    field = np.random.default_rng(1).standard_normal((10, 6, 6, 4))
    assert sum(1 for _ in fields.slabs(field, plane_axes=(1, 2))) == 10 * 4


def test_over_planes_matches_manual_mean_and_differs_from_flatten():
    """Keeping each plane intact (over_planes) is the geometry-preserving reduction and
    is not equal to flattening all non-ordered axes into one feature axis."""
    rng = np.random.default_rng(2)
    planes = [np.outer(rng.standard_normal(8), rng.standard_normal(8)) for _ in range(6)]
    field = np.stack(planes, axis=0)                # (6, 8, 8), each plane rank-1
    op = fields.over_planes(field, plane_axes=(1, 2), read=phi, reduce="mean")
    assert op == pytest.approx(float(np.mean([phi(p) for p in planes])), rel=1e-9)
    flat = phi(field.reshape(6, -1))                # the naive (structure-destroying) reduction
    assert abs(op - flat) > 1e-3                    # geometry-preserving != flatten


def test_over_planes_reduce_options():
    field = np.random.default_rng(3).standard_normal((4, 8, 8))
    for r in ("mean", "median", "max", "min", "sum"):
        assert np.isfinite(fields.over_planes(field, (1, 2), reduce=r))
    with pytest.raises(ValueError):
        fields.over_planes(field, (1, 2), reduce="bogus")


def test_slabs_rejects_repeated_axis():
    with pytest.raises(ValueError):
        list(fields.slabs(np.zeros((3, 3, 3)), plane_axes=(1, 1)))


def test_pool_flattens_nonordered_as_samples():
    field = np.random.default_rng(4).standard_normal((10, 6, 6, 4))
    p = np.asarray(fields.pool(field, ordered_axis=0))
    assert p.shape == (10, 6 * 6 * 4)
    assert np.allclose(p[3], field[3].reshape(-1))   # rows preserve the ordered index


def test_pool_respects_ordered_axis():
    field = np.random.default_rng(5).standard_normal((6, 12, 3))
    p = np.asarray(fields.pool(field, ordered_axis=1))   # axis-1 is ordered
    assert p.shape == (12, 6 * 3)
