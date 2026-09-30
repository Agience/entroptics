"""The Tracy-Widom laws every count is scored against, evaluated from their Fredholm
determinants with an Airy function built from its differential equation.  The quantiles land on the
published percentiles, the survival is the same law as the quantile at every level, and the far
tails -- where a coarse grid can agree with itself on a wrong answer -- match independent values."""
import math

import numpy as np

from entroptics import tracy_widom as tw
from entroptics.null_providers import tw1_quantile, tw1_sf, tw2_quantile, tw2_sf

# Ai, Ai' to 18 significant digits (mpmath at 25 digits)
AIRY = {0.0: (0.355028053887817239, -0.258819403792806798),
        1.0: (0.135292416312881416, -0.159147441296793213),
        -2.0: (0.227407428201685576, 0.618259020741691041),
        5.0: (1.08344428136074417e-4, -2.47413890868462476e-4),
        -8.0: (-0.0527050503563862026, 0.935560938198306551)}


def test_airy_from_its_differential_equation():
    for x, (ai, aip) in AIRY.items():
        a, ap = tw.airy(np.array([x]))
        scale = 1.0 if x < 0 else abs(ai)          # absolute where Ai oscillates, relative where it decays
        assert abs(a[0] - ai) / scale < 5e-14, (x, a[0], ai)
        assert abs(ap[0] - aip) / (1.0 if x < 0 else abs(aip)) < 5e-14, (x, ap[0], aip)


def test_the_975_quantile_is_the_true_one():
    q = tw1_quantile(0.025)
    assert abs(q - 1.45377) < 1e-4, q
    assert abs(tw1_sf(q) / 0.025 - 1.0) < 1e-12


def test_quantile_and_survival_are_one_law_at_every_level():
    for far in (0.5, 0.1, 0.05, 0.01, 1e-4, 1e-8, 1e-12):
        for q, sf in ((tw1_quantile(far), tw1_sf), (tw2_quantile(far), tw2_sf)):
            assert abs(sf(q) / far - 1.0) < 1e-11, (far, q, sf(q))


def test_survival_is_monotone_with_relative_precision_in_the_tail():
    g = np.linspace(-9.0, 12.0, 85)
    for sf in (tw1_sf, tw2_sf):
        p = np.asarray(sf(g))
        assert np.all(np.diff(p) < 0.0) or np.all(np.diff(p[p < 1.0]) < 0.0)
        assert 0.0 < p[-1] < 1e-12                 # the far tail is resolved, not flushed to zero
    # and resolved to relative precision thirty decades down
    assert abs(tw1_sf(tw1_quantile(1e-30)) / 1e-30 - 1.0) < 1e-11
    assert tw1_sf(-20.0) == 1.0 and tw2_sf(-20.0) == 1.0


def test_tw1_median_and_tail_against_published_values():
    # TW1 median -1.2686 (Bornemann 2010, Table 2 of Tracy-Widom F1 statistics), and the TW2
    # right tail is thinner than TW1's at every level
    assert abs(tw1_quantile(0.5) - (-1.2686)) < 1e-4
    for far in (0.1, 0.01, 1e-6):
        assert tw2_quantile(far) < tw1_quantile(far)


def test_far_tails_match_the_first_order_trace():
    # at these s the survival equals tr(K) to all digits: TW1 (1/2) int_s^inf Ai, TW2
    # int_s^inf (Ai'^2 - x Ai^2) -- independent references, by mpmath quadrature
    assert abs(tw1_sf(25.0) / 8.06825939784923e-39 - 1.0) < 1e-12
    assert abs(tw2_sf(17.0) / 7.18325514943437e-45 - 1.0) < 1e-12


def test_the_lower_tail_is_resolved_not_flushed():
    # a coarse grid over-resolves an eigenvalue here and read the survival as exactly 1
    assert abs((1.0 - tw2_sf(-6.2)) / 1.6457429863e-9 - 1.0) < 1e-6
    far = 1.0 - 1e-12
    assert abs((1.0 - tw2_sf(tw2_quantile(far))) / 1e-12 - 1.0) < 1e-3


def test_edge_inputs():
    import pytest
    assert math.isnan(tw1_sf(float("nan")))
    assert tw1_sf(float("inf")) == 0.0 and tw2_sf(1e300) == 0.0
    with pytest.raises(ValueError):
        tw.survival(0.0, 3)
    with pytest.raises(ValueError):
        tw.airy(np.array([-50.0]))
    assert tw1_quantile(5e-324) > tw1_quantile(1e-200) > tw1_quantile(1e-12)
    from entroptics.null_providers import _reg_gamma_upper
    assert _reg_gamma_upper(2.0, float("inf")) == 0.0 and math.isnan(_reg_gamma_upper(2.0, math.nan))
