"""The level edge — the best two-population split of an ordered profile, and its evidence."""
import numpy as np
import pytest

from entroptics import level_edge, occupied_modes


def test_a_planted_two_level_profile_is_recovered_exactly():
    for k, n in ((3, 12), (10, 64), (48, 64), (1, 8)):
        w = np.concatenate([np.full(k, 9.0), np.full(n - k, 1.0)])
        r = level_edge(w)
        assert r.k == k, (k, n, r)
        assert r.separability == pytest.approx(1.0)


def test_it_is_order_independent():
    w = np.concatenate([np.full(5, 9.0), np.full(11, 1.0)])
    rng = np.random.default_rng(0)
    for _ in range(5):
        assert level_edge(rng.permutation(w)).k == 5


def test_it_is_scale_free():
    w = np.concatenate([np.full(4, 9.0), np.full(10, 1.0)])
    a, b = level_edge(w), level_edge(w * 1e6)
    assert a.k == b.k and a.separability == pytest.approx(b.separability)


def test_a_single_population_reports_no_separation_rather_than_a_cut():
    for w in (np.ones(9), np.zeros(7), np.full(20, 3.5)):
        r = level_edge(w)
        assert r.k == len(w) and r.separability == 0.0


def test_degenerate_sizes():
    empty = level_edge([])
    assert empty.k == 0 and empty.separability == 0.0
    one = level_edge([5.0])
    assert one.k == 1 and one.separability == 0.0


def test_non_finite_is_reported_as_no_separation_not_as_a_split():
    r = level_edge([1.0, np.nan, 3.0])
    assert r.separability == 0.0


def test_it_measures_cleanness_not_distance():
    """An exact two-level profile reads 1.000 whatever the gap — two values explain all of
    the variance between them. This is the property that stops eta^2 being a measure of how
    BIG a break is, and it is why the ratio across the break is a separate number."""
    for hi in (9.0, 3.0, 1.5, 1.001):
        w = np.concatenate([np.full(12, hi), np.full(28, 1.0)])
        assert level_edge(w).separability == pytest.approx(1.0), hi


def test_with_spread_on_both_levels_it_does_fall_with_the_gap():
    """Which is the case real profiles are in."""
    rng = np.random.default_rng(0)
    seps = []
    for hi in (9.0, 3.0, 1.5, 1.05):
        draws = []
        for _ in range(8):
            w = np.concatenate([np.full(12, hi), np.full(28, 1.0)]) + 0.3 * rng.standard_normal(40)
            draws.append(level_edge(np.abs(w)).separability)
        seps.append(float(np.median(draws)))
    assert all(a >= b for a, b in zip(seps, seps[1:])), seps
    assert seps[0] > 0.95 and seps[-1] < 0.8, seps


def test_it_answers_where_occupied_modes_is_mute():
    """A smooth decay has no step, so the rank edge reports none; the partition still splits.

    The two reads are not substitutes — this is the case that shows it."""
    w = np.array([1.0 / i for i in range(1, 25)])
    assert occupied_modes(w).margin < 2.0          # no sharp step to find
    assert level_edge(w).separability > 0.5        # but a partition still explains the spread


def test_noise_reads_lower_than_planted_structure():
    rng = np.random.default_rng(3)
    noise = [level_edge(np.abs(rng.standard_normal(60))).separability for _ in range(20)]
    planted = []
    for _ in range(20):
        w = np.concatenate([np.full(12, 9.0), np.abs(rng.standard_normal(48))])
        planted.append(level_edge(w).separability)
    assert max(noise) < min(planted), (max(noise), min(planted))


def test_deterministic():
    rng = np.random.default_rng(4)
    w = np.abs(rng.standard_normal(50))
    first = level_edge(w)
    for _ in range(3):
        assert level_edge(w) == first
