"""`reads.renyi_count` -- the Renyi effective mode count of a weight vector.  Pins it on the cases
whose answer is known exactly (equal modes, one mode), against the counts the library already
reads (``2^shannon_bits`` at order 1, ``proximity.effective_width`` at order 2), on the order of
the family, on the Lean witness that a bounded Shannon count does not bound the order-1/2 count,
and on its edge cases; and checks the top-level re-export."""
import math

import numpy as np
import pytest

import entroptics
from entroptics.entropy import shannon_bits
from entroptics.proximity import centre, effective_width
from entroptics.reads import renyi_count

ORDERS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 7.0, math.inf]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("n", [1, 2, 3, 7, 64])
def test_equal_modes_read_their_number_at_every_order(n, order):
    assert renyi_count(np.full(n, 0.37), order) == pytest.approx(n, rel=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_one_mode_reads_one(order):
    assert renyi_count([0.0, 5.0, 0.0, 0.0], order) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_zero_weights_do_not_change_the_count(order):
    rng = np.random.default_rng(3)
    w = rng.uniform(0.1, 2.0, 9)
    padded = np.concatenate([w, np.zeros(5)])
    assert renyi_count(padded, order) == pytest.approx(renyi_count(w, order), rel=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_scale_free(order):
    rng = np.random.default_rng(4)
    w = rng.exponential(size=12)
    assert renyi_count(1e6 * w, order) == pytest.approx(renyi_count(w, order), rel=1e-12)


def test_order_half_is_the_square_of_the_root_sum():
    w = np.array([4.0, 1.0, 1.0, 0.0])
    p = w / w.sum()
    assert renyi_count(w) == pytest.approx(float(np.sum(np.sqrt(p))) ** 2, rel=1e-14)
    assert renyi_count(w) == pytest.approx(16.0 / 6.0, rel=1e-14)       # (2 + 1 + 1)^2 / 6


def test_order_one_is_the_shannon_count():
    rng = np.random.default_rng(5)
    for _ in range(20):
        w = rng.exponential(size=int(rng.integers(2, 40)))
        assert renyi_count(w, 1.0) == pytest.approx(2.0 ** shannon_bits(w), rel=1e-12)


def test_the_general_order_meets_the_special_orders_continuously():
    """The closed forms at 0, 1/2, 1, 2 and inf are limits or values of the general formula."""
    w = np.random.default_rng(6).exponential(size=17)
    for a, eps in [(0.5, 1e-7), (1.0, 1e-7), (2.0, 1e-7)]:
        assert renyi_count(w, a + eps) == pytest.approx(renyi_count(w, a), rel=1e-5)
        assert renyi_count(w, a - eps) == pytest.approx(renyi_count(w, a), rel=1e-5)
    assert renyi_count(w, 1e-9) == pytest.approx(renyi_count(w, 0.0), rel=1e-6)
    assert renyi_count(w, 400.0) == pytest.approx(renyi_count(w, math.inf), rel=1e-2)


@pytest.mark.parametrize("seed", range(10))
def test_the_count_falls_with_the_order(seed):
    """R_0 >= R_1/2 >= exp(H) >= R_2 >= R_inf on every vector."""
    rng = np.random.default_rng(seed)
    w = rng.exponential(size=int(rng.integers(2, 60))) ** rng.uniform(0.5, 4.0)
    w[rng.random(w.size) < 0.2] = 0.0
    if w.sum() == 0.0:
        w[0] = 1.0
    counts = [renyi_count(w, a) for a in ORDERS]
    for hi, lo in zip(counts, counts[1:]):
        assert hi >= lo * (1.0 - 1e-12)
    assert renyi_count(w, 0.5) >= 2.0 ** shannon_bits(w) * (1.0 - 1e-12)
    assert 2.0 ** shannon_bits(w) >= renyi_count(w, 2.0) * (1.0 - 1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_order_two_is_effective_width(seed):
    """``effective_width`` is the order-2 count of the per-channel second moments."""
    rng = np.random.default_rng(seed)
    A = centre(rng.standard_normal((80, 24)) * rng.uniform(0.1, 3.0, 24))
    v = np.mean(A ** 2, axis=0)
    assert renyi_count(v, 2.0) == pytest.approx(effective_width(A), rel=1e-14)


def test_order_two_is_effective_width_on_its_two_valued_case():
    """The case ``effective_width`` was written for: ``p`` live channels at one level read ``p``."""
    v = np.zeros(32)
    v[:12] = 2.5
    A = np.sqrt(v)[None, :] * np.ones((10, 1))
    assert renyi_count(v, 2.0) == pytest.approx(12.0, rel=1e-12)
    assert effective_width(A) == pytest.approx(renyi_count(v, 2.0), rel=1e-12)


def test_a_bounded_shannon_count_does_not_bound_order_half():
    """The witness of ``spRenyi_unbounded_at_bounded_spDim``: one mode at ``1 - eps``, ``eps = 1/ln m``
    spread evenly over ``m`` others, has ``exp(H) <= e^2`` and ``R_1/2 >= m / ln m``."""
    previous = 0.0
    for m in [10, 100, 1_000, 10_000, 100_000]:
        eps = 1.0 / math.log(m)
        w = np.concatenate([[1.0 - eps], np.full(m, eps / m)])
        assert renyi_count(w, 1.0) <= math.exp(2.0)
        r = renyi_count(w, 0.5)
        assert r >= m / math.log(m)
        assert r > previous
        previous = r


@pytest.mark.parametrize("w", [[], [0.0, 0.0, 0.0], np.zeros((2, 3))])
def test_no_weight_has_no_count(w):
    for a in ORDERS:
        assert math.isnan(renyi_count(w, a))


def test_a_non_finite_weight_reads_nan():
    assert math.isnan(renyi_count([1.0, np.nan, 2.0]))
    assert math.isnan(renyi_count([1.0, np.inf, 2.0], 2.0))


def test_a_negative_weight_raises():
    with pytest.raises(ValueError):
        renyi_count([1.0, -0.5, 2.0])
    with pytest.raises(ValueError):
        renyi_count([1.0, -1e-18], 2.0)


@pytest.mark.parametrize("order", [-0.5, -math.inf, math.nan])
def test_a_negative_order_raises(order):
    with pytest.raises(ValueError):
        renyi_count([1.0, 2.0], order)


def test_the_input_is_flattened_and_returns_a_float():
    w = np.arange(1.0, 7.0).reshape(2, 3)
    r = renyi_count(w)
    assert isinstance(r, float)
    assert r == pytest.approx(renyi_count(w.ravel()), rel=0)


def test_reexported_at_top_level():
    assert entroptics.renyi_count is renyi_count
    assert "renyi_count" in entroptics.__all__
