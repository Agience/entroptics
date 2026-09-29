"""empirical_bernstein: Maurer-Pontil (2009) Thm 4 on a mean, with the a-priori width the theorem
needs and no plug-in."""
import math

import numpy as np
import pytest

from entroptics import empirical_bernstein


def _eb(x, delta, side, span):
    """Maurer & Pontil's bound written out by hand, as a caller would, for bit-identity."""
    n = len(x); m = x.mean(); v = x.var(ddof=1); Lc = math.log(2.0 / delta)
    R = span
    return m + side * (math.sqrt(2 * v * Lc / n) + 7 * R * Lc / (3 * (n - 1)))


def test_closed_form_on_a_fixed_vector():
    x = np.array([0.1, 0.4, 0.35, 0.8, 0.55])
    r = empirical_bernstein(x, 0.05, span=1.0)
    L = math.log(2 / 0.05)
    want = math.sqrt(2 * x.var(ddof=1) * L / 5) + 7 * 1.0 * L / (3 * 4)
    assert r.n == 5 and r.mean == pytest.approx(x.mean()) and r.radius == pytest.approx(want)
    assert r.lo == r.mean - r.radius and r.hi == r.mean + r.radius


def test_bit_identical_to_the_hand_written_bound():
    x = np.random.default_rng(0).uniform(0.2, 0.7, 300)
    for delta in (0.1, 0.01, 1e-6):
        r = empirical_bernstein(x, delta, span=1.0)
        assert r.hi == _eb(x, delta, +1, 1.0) and r.lo == _eb(x, delta, -1, 1.0)


def test_one_sided_coverage_at_its_level():
    """Uniform(0,1) with the true width: the upper endpoint falls below the true mean in at most
    a fraction delta of trials.  The slack is three binomial standard deviations at that count."""
    rng = np.random.default_rng(1)
    delta, trials, n = 0.1, 2000, 40
    miss = sum(empirical_bernstein(rng.uniform(0, 1, n), delta, span=1.0).hi < 0.5 for _ in range(trials))
    assert miss / trials <= delta + 3 * math.sqrt(delta * (1 - delta) / trials)


def test_constant_samples_in_a_zero_width_support_have_no_radius():
    assert empirical_bernstein(np.full(10, 3.0), 0.05, span=0.0).radius == 0.0


def test_no_plug_in_width():
    """Negative control: the theorem needs a width fixed in advance.  There is no default (the
    sample range would be a plug-in), and a width narrower than the samples' own range is refused."""
    x = np.array([0.0, 1.0, 0.5])
    with pytest.raises(TypeError):
        empirical_bernstein(x, 0.05)
    with pytest.raises(ValueError):
        empirical_bernstein(x, 0.05, span=0.5)


def test_refusals():
    with pytest.raises(ValueError):
        empirical_bernstein(np.array([1.0]), 0.05, span=1.0)
    for d in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            empirical_bernstein(np.array([0.1, 0.2]), d, span=1.0)
    with pytest.raises(ValueError, match="finite"):
        empirical_bernstein(np.array([0.1, np.nan, 0.2]), 0.05, span=1.0)
