"""The complex branch of the derived edge.

A complex Gaussian matrix's largest eigenvalue converges to Tracy-Widom_2 (GUE), not TW1
(GOE), and its row energy is a sum of exponentials rather than a chi^2.  Scoring a complex
screen at the TW1 quantile with the real Wishart centring delivered a false-alarm rate of
0.000 against a nominal 0.05 -- at every aspect ratio tested.  `far` is the reader's
Neyman-Pearson contract, so a delivered level that is not the requested one is the defect,
whether or not it happens to cost a detection.

References: Chiani (2014), arXiv:1209.3394 / J. Multivariate Anal. 129:69-81.
  Table 3  TW2 percentiles (a cross-check on the exact law; not read into the code)
  eq (33)-(34)  the Wishart centring adjustment: a1 = a2 = -1/2 real, a1 = a2 = 0 complex
"""
import math

import numpy as np
import pytest

from entroptics import Projection
from entroptics import null_providers as npv
from entroptics.entropy import MAD_SCALE, MAD_SCALE_C


def _far(N, F, complex_, trials=300):
    """P(K_signal >= 1) on pure iid Gaussian noise of the given field."""
    hits = 0
    for i in range(trials):
        g = np.random.default_rng(2000 + i)
        W = g.standard_normal((N, F))
        if complex_:
            W = (W + 1j * g.standard_normal((N, F))) / np.sqrt(2.0)   # same total power
        hits += Projection(W, null=npv.mp).K_signal >= 1        # the TW edge law under test
    return hits / trials


def _binom_two_sided_p(k, n, p):
    """Exact two-sided binomial p-value: the total probability of every outcome no more
    likely than the observed one.  No normal approximation, so it stays honest in the tail
    where a 4-sigma band on a rate reaches below zero and stops discriminating."""
    from math import comb
    pk = comb(n, k) * p ** k * (1 - p) ** (n - k)
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(n + 1)
               if comb(n, j) * p ** j * (1 - p) ** (n - j) <= pk * (1 + 1e-12))


def _binom_upper_p(k, n, p):
    """Exact one-sided binomial p-value: the probability of k or more hits at rate p."""
    from math import comb
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def test_complex_screens_hold_their_stated_false_alarm_level():
    """The defect this branch existed for: complex FAR was 0.000 at every aspect ratio.

    The screen's Gram is a sample correlation matrix, whose top eigenvalue sits below the
    Tracy-Widom law of the covariance at finite size, real and complex alike, so the read holds
    its level with room (research/benchmarks/screen_null.py).  Scored by
    the EXACT binomial tail at a stated false-failure budget: no shape exceeds the level, and
    the pooled rate is not the 0.000 of the defect."""
    trials, far, budget = 300, 0.05, 1e-4
    total = 0
    for N, F in ((200, 200), (400, 64), (64, 400), (128, 128)):
        k = int(round(_far(N, F, True, trials) * trials))
        assert _binom_upper_p(k, trials, far) > budget, (N, F, k, trials)
        total += k
    assert total > 0


def test_the_far_test_can_actually_fail():
    """A test that cannot fail is not a test.  A read that fires at over twice its level is rejected
    overwhelmingly, the nominal outcome passes, and zero hits out of 300 at a nominal 0.05 -- what
    the complex path delivered before the fix -- is rejected two-sided."""
    assert _binom_upper_p(33, 300, 0.05) < 1e-4            # 0.11: twice the level and more
    assert _binom_upper_p(15, 300, 0.05) > 0.05            # the nominal outcome passes
    assert _binom_two_sided_p(0, 300, 0.05) < 1e-6


def test_tw2_quantiles_match_chiani_table_3():
    """The exact TW2 quantile against the published TW2 percentiles (Chiani 2014 Table 3;
    Bornemann 2010) to the four decimals they are quoted to; and the quantile and the survival are
    one law, with no step at any level."""
    for far, published in ((0.10, -0.5969), (0.05, -0.2325), (0.01, 0.4776)):
        assert abs(npv.tw2_quantile(far) - published) < 1e-4, (far, npv.tw2_quantile(far))
    for far in (0.05, 0.0501, 0.0499, 1e-9):
        assert abs(npv.tw2_sf(npv.tw2_quantile(far)) / far - 1.0) < 1e-12
    # TW2's tail is thinner than TW1's
    assert npv.tw2_quantile(0.05) < npv.tw1_quantile(0.05)
    assert npv.tw_quantile(0.05, complex_=True) == npv.tw2_quantile(0.05)
    assert npv.tw_quantile(0.05, complex_=False) == npv.tw1_quantile(0.05)


def test_tw2_survival_is_monotone_and_bracketed():
    xs = [-4.0, -2.0, -1.0, 0.0, 1.0, 3.0]
    sf = [npv.tw2_sf(x) for x in xs]
    assert all(sf[i] > sf[i + 1] for i in range(len(sf) - 1))
    assert 0.0 <= sf[-1] < 0.01 and 0.9 < sf[0] <= 1.0


def test_complex_wishart_centring_drops_johnstones_minus_one():
    """Chiani eq. (33)-(34): a1 = a2 = 0 for the complex Wishart."""
    N, F = 200, 120
    mu_c, _ = npv.johnstone(N, F, complex_=True)
    assert mu_c == pytest.approx((math.sqrt(N) + math.sqrt(F)) ** 2)
    mu_r, _ = npv.johnstone(N, F)
    assert mu_r == pytest.approx((math.sqrt(N - 1) + math.sqrt(F)) ** 2)
    assert mu_c > mu_r


def test_complex_row_energy_debias_is_the_gamma_median():
    """A real row energy is sigma^2 chi^2_F; a complex one is sigma^2 chi^2_{2F}/2, so the
    Wilson-Hilferty correction is taken at twice the degrees of freedom."""
    N, F = 64, 32
    r = npv.debias_denominator(N, F)
    c = npv.debias_denominator(N, F, complex_=True)
    assert c > r                                        # less deflation at more dof
    dof = (N - 1) / N
    assert c == pytest.approx(F * (1 - 1 / (9 * F)) ** 3 * dof)
    assert r == pytest.approx(F * (1 - 2 / (9 * F)) ** 3 * dof)
    # and the complex de-bias recovers the true per-cell variance on complex noise
    g = np.random.default_rng(4)
    Z = (g.standard_normal((4000, F)) + 1j * g.standard_normal((4000, F))) / np.sqrt(2.0)
    s2 = npv.noise_sigma2(np, Z, 4000, F, complex_=True)
    assert 0.9 < s2 < 1.1, s2


def test_mad_scale_complex_is_one_over_sqrt_ln2():
    """Derived, not calibrated: |z|^2 ~ Exp(1) so median|z| = sqrt(ln 2)."""
    assert MAD_SCALE_C == pytest.approx(1.0 / math.sqrt(math.log(2.0)), rel=1e-12)
    n = 400000
    g = np.random.default_rng(1)
    z = (g.standard_normal(n) + 1j * g.standard_normal(n)) / np.sqrt(2.0)
    # the median of n draws has SE ~ 1/(2 f(m) sqrt(n)); for |z| (Rayleigh, scale 1/sqrt(2))
    # the density at the median is sqrt(2 ln 2) exp(-ln 2), so the tolerance is the estimator's
    # own error at this n, not a number chosen to make the line pass.
    se = 1.0 / (2 * math.sqrt(2 * math.log(2)) * math.exp(-math.log(2)) * math.sqrt(n))
    assert np.median(np.abs(z)) == pytest.approx(math.sqrt(math.log(2)), abs=5 * se)
    ratio = MAD_SCALE / MAD_SCALE_C                       # the size of the defect, derived
    assert ratio == pytest.approx(math.sqrt(math.log(2)) / (1 / 1.482602218505602), rel=1e-9)


def test_the_real_path_did_not_move():
    """Everything above is additive: real input takes the branch it always did."""
    g = np.random.default_rng(11)
    for shape in ((64, 24), (128, 128), (40, 90)):
        W = g.standard_normal(shape)
        sc = Projection(W, null=npv.mp)
        N, F = sc.screen.shape
        s2 = npv.noise_sigma2(np, sc.screen, N, F)
        expect = math.sqrt(npv.screen_floor_sq(s2, N, F, 0.05))
        assert float(sc.noise_floor) == pytest.approx(expect, rel=1e-12)
        assert npv.johnstone(N, F) == npv.johnstone(N, F, complex_=False)
        assert npv.debias_denominator(N, F) == npv.debias_denominator(N, F, complex_=False)


def test_complex_predicate_is_shared_not_reinlined():
    from entroptics import environment as env
    assert env.is_complex_obj(np.zeros((3, 3))) is False
    assert env.is_complex_obj(np.zeros((3, 3), complex)) is True
    assert env.is_complex_obj(None) is False
