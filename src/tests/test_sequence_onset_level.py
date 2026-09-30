"""``surrogate_test``'s onset holds the caller's level.

The onset was declared at a fixed ``p < 0.05`` at each of the orders n = 2 .. n_max in turn, so a
sequence with no order at all reported an onset far more often than 5%.  It is now read at the
caller's ``far``, Bonferroni over the orders, and the defaults (``n_max``, ``draws``) are derived
from the sequence and the level."""
import numpy as np
import pytest

from entroptics.sequence import sampled_order, surrogate_test


def test_an_iid_sequence_reports_an_onset_at_most_far_of_the_time():
    far = 0.05
    onsets = [surrogate_test(np.random.default_rng(s).integers(0, 3, 1500), far=far, n_max=5,
                             seed=s)["onset"] is not None for s in range(120)]
    assert np.mean(onsets) <= far, np.mean(onsets)


def test_the_defaults_are_derived():
    r = surrogate_test(np.random.default_rng(0).integers(0, 2, 1024), far=0.05)
    assert r["n_max"] == 10                          # 2^10 = 1024 windows' worth of words
    assert r["level"] == 0.05 / 9
    assert 1.0 / (r["draws"] + 1) <= r["level"] < 1.0 / r["draws"]   # the fewest that reach the level


def test_sampled_order_is_exact_at_powers():
    assert sampled_order(8, 2) == 3 and sampled_order(7, 2) == 2 and sampled_order(81, 3) == 4
    assert sampled_order(100, 1) == 1


def test_far_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        surrogate_test([0, 1, 0, 1], far=1.0)
