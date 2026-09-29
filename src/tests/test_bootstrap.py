"""bootstrap: resampling with replacement. The draw pattern is a contract, the library supplies no
level, and it must never manufacture spread."""
import numpy as np
import pytest

from entroptics import bootstrap


def test_replicates_equal_a_hand_loop_of_the_same_draws():
    x = np.random.default_rng(3).standard_normal(50)
    got = bootstrap(x, np.mean, draws=40, rng=7)
    g = np.random.default_rng(7)
    want = np.array([np.mean(x[g.integers(0, 50, 50)]) for _ in range(40)])
    assert np.array_equal(got, want)                                  # bit for bit


def test_choice_with_replacement_draws_the_same_indices():
    x = np.arange(30.0)
    got = bootstrap(x, lambda s: s, draws=5, rng=11)
    g = np.random.default_rng(11)
    want = np.stack([x[g.choice(30, size=30, replace=True)] for _ in range(5)])
    assert np.array_equal(got, want)


def test_same_seed_repeats_and_a_generator_advances():
    x = np.random.default_rng(0).standard_normal(20)
    assert np.array_equal(bootstrap(x, np.mean, draws=10, rng=5), bootstrap(x, np.mean, draws=10, rng=5))
    g = np.random.default_rng(5)
    first, second = bootstrap(x, np.mean, draws=10, rng=g), bootstrap(x, np.mean, draws=10, rng=g)
    assert not np.array_equal(first, second)                          # the caller's stream continued
    assert np.array_equal(first, bootstrap(x, np.mean, draws=10, rng=5))


def test_spread_of_the_mean_matches_its_standard_error():
    """On N(0,1) samples the replicate std of the mean estimates s/sqrt(N). With B draws the std of a
    std estimate is about s_hat / sqrt(2(B-1)), so the check is at four of those (derived from draws)."""
    N, B = 400, 2000
    x = np.random.default_rng(1).standard_normal(N)
    reps = bootstrap(x, np.mean, draws=B, rng=2)
    target = x.std() / np.sqrt(N)
    assert abs(reps.std() - target) < 4 * target / np.sqrt(2 * (B - 1))


def test_array_reads_and_lists_of_planes():
    planes = [np.full((2, 2), float(i)) for i in range(6)]
    reps = bootstrap(planes, lambda ps: np.mean(np.stack(ps), axis=0), draws=4, rng=0)
    assert reps.shape == (4, 2, 2)
    g = np.random.default_rng(0)
    idx = g.integers(0, 6, 6)
    assert np.array_equal(reps[0], np.mean(np.stack([planes[i] for i in idx]), axis=0))


def test_prebuilt_indices_are_shared_across_reads():
    x = np.random.default_rng(4).standard_normal(25)
    g = np.random.default_rng(9)
    idx = [g.integers(0, 25, 25) for _ in range(8)]
    m, s = bootstrap(x, np.mean, indices=idx), bootstrap(x, np.std, indices=idx)
    assert np.array_equal(m, [np.mean(x[i]) for i in idx]) and np.array_equal(s, [np.std(x[i]) for i in idx])


def test_it_never_manufactures_spread():
    """Negative control: equal samples, or a read that ignores its input, give zero spread exactly."""
    assert bootstrap(np.full(30, 2.5), np.mean, draws=50, rng=0).std() == 0.0
    assert bootstrap(np.random.default_rng(0).standard_normal(30), lambda s: 1.0, draws=50, rng=0).std() == 0.0


def test_refusals():
    with pytest.raises(ValueError):
        bootstrap(np.zeros(0), np.mean, draws=5)
    with pytest.raises(ValueError):
        bootstrap(np.zeros(5), np.mean, draws=0)
    with pytest.raises(ValueError):
        bootstrap(np.zeros(5), np.mean)                               # the library has no default draws
    with pytest.raises(ValueError):
        bootstrap(np.zeros(5), np.mean, draws=3, indices=[np.arange(5)])
