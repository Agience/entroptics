"""Profile reads: effective_rates (the lag-local decay rate) and crossing_lag (where a decay first
falls below a level)."""
import math

import numpy as np
import pytest

from entroptics import crossing_lag, effective_rates, hankel_spectrum


# ── effective_rates ──────────────────────────────────────────────────────────────────────────
def test_a_single_exponential_gives_its_rate_at_every_lag():
    m = 0.37
    c = 2.5 * np.exp(-m * np.arange(30))
    r = effective_rates(c)
    assert r.shape == (29,) and np.allclose(r, m, atol=1e-12)


def test_two_exponentials_fall_toward_the_slower_rate():
    t = np.arange(60)
    c = np.exp(-0.8 * t) + 0.3 * np.exp(-0.1 * t)
    r = effective_rates(c)
    assert np.all(np.diff(r) <= 1e-12) and r[-1] == pytest.approx(0.1, abs=1e-6) and r[0] > 0.1


def test_it_is_the_order_zero_hankel_read():
    c = np.exp(-0.25 * np.arange(12))
    r = effective_rates(c)
    for t in range(0, 8):
        lam = float(hankel_spectrum(c[t:t + 4], 1).leading)
        assert r[t] == pytest.approx(-math.log(lam), abs=1e-9)


def test_no_rate_across_a_sign_change_or_between_negatives():
    """Negative control: a sign change has no rate, and neither do two negative lags -- NaN, never
    a finite number.  A rise between two positive lags is a negative rate, returned as measured."""
    r = effective_rates(np.array([1.0, 0.5, -0.2, -0.4, 0.3, 0.6]))
    assert r[0] == pytest.approx(math.log(2.0))
    assert np.isnan(r[1]) and np.isnan(r[2]) and np.isnan(r[3])
    assert r[4] == pytest.approx(math.log(0.5))
    assert effective_rates(np.array([1.0])).size == 0


# ── crossing_lag ─────────────────────────────────────────────────────────────────────────────
def test_an_exponential_crosses_exactly():
    xi = 7.3
    c = np.exp(-np.arange(80) / xi)
    for level in (0.5, 0.1, 1 / math.e):
        assert crossing_lag(c, level) == pytest.approx(xi * math.log(1 / level), abs=1e-9)


def test_the_linear_branch_when_the_bracketing_value_is_not_positive():
    c = np.array([1.0, 0.6, -0.2, -0.5])
    assert crossing_lag(c, 0.2) == pytest.approx(1 + (0.6 - 0.2) / (0.6 + 0.2))


def test_a_profile_starting_at_or_below_the_level():
    assert crossing_lag(np.array([0.1, 0.05, 0.01]), 0.2) == 0.0
    assert crossing_lag(np.array([0.1, 0.5, 0.6]), 0.3) == 0.0      # rises after: still 0, not NaN
    assert crossing_lag(np.array([0.5, 0.6, 0.2]), 0.5) == 0.0      # starts ON the level


def test_never_crossing_is_censored_not_the_record_length():
    """Negative control: a profile that never reaches the level returns NaN, not its length."""
    assert math.isnan(crossing_lag(np.exp(-np.arange(10) / 100.0), 0.5))
    assert math.isnan(crossing_lag(np.ones(20), 0.5))
