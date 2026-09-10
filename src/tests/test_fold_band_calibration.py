"""`fold_band`'s stated guarantee, measured: size at most `far`, and power that survives it.

`fold_band`'s derivation makes a checkable promise about the significance term -- it is
`E[D] + k sd(D)` with `k` from Cantelli, so "a pure-noise record clears it with probability at
most `far`".  Nothing tested that, and a calibration claim that is never measured is an
assertion.  This file measures it.

Ground truth is exact rather than modelled: iid Gaussian noise has no feature structure, so
EVERY fold of a noise record is false, and the fold rate on noise IS the false-fold rate.

BOTH HALVES ARE REQUIRED.  A band set enormously wide has a false-fold rate of zero and is
useless, so size is only meaningful beside power -- the fold rate on a record whose feature
power genuinely concentrates into `K < F` channels, where a fold is correct.

WHAT THIS DOES AND DOES NOT SHOW DOWNSTREAM.  The band is one of TWO gates: `fold_width` also
requires `feature_axis_is_continuous`, and that second gate refuses the fold outright on narrow
feature axes.  Measured end to end -- 8 shapes x 3 data kinds x 40 draws -- swapping the derived
band for the capped Miller-Madow one changed the returned `F_eff` on ZERO of 960 records, and the
spurious-detection rate on pure noise was identical to four decimals (0.0245 either way).

So the rates below are properties of the BAND, measured in isolation, and no downstream benefit
over the predecessor has been demonstrated: the region where the capped band is miscalibrated
(`F = 4`) is exactly the region where the continuity gate refuses every fold anyway.  The band is
the more principled of the two and states a rate the other cannot; that is its case, and it is
not a performance claim.

WHAT THIS FILE CANNOT SEE, stated so it is not trusted for more than it holds.  Halving the
Cantelli multiplier in the significance term leaves every test here green.  That is not a gap to
be patched but a measured consequence: the realised false-fold rate sits at 0.0000 for every
`far <= 0.05` on these shapes, with a worst realised/nominal ratio of 0.263 on real input and
0.005 on complex, so there is 4x to 200x of headroom and a 2x change in the multiplier does not
reach the noise tail.  The significance bar's exact constant is therefore unobservable in the
regimes tested; what binds almost everywhere is the SUFFICIENCY bar, and that one IS pinned
below.

THE PREDECESSOR, for the record.  This band replaced a capped Miller-Madow one,
`min((F-1)/(2 T ln2), log2(F)/2)` -- `T` times a MEAN deficit, capped to stay inside the entropy
range, and stating no rate at all.  Measured at `F = 4` it folds pure noise 10.8% to 13.3% of the
time for every `T` from 32 to 512 (2.1x to 2.6x the stated level), because a mean deficit shrinks
as `1/T` while the tail of the deficit distribution does not.  Both bands reach power 1.0 on the
concentrated records below, so the derived band's calibration costs nothing here.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics.entropy import fold_band, shannon_bits

FAR = 0.05
DRAWS = 300
# Both terms must be exercised, or a gate cannot tell which one is doing the work.
# Measured crossover (where sufficiency overtakes significance): T = 8 at F = 8, 11 at 16,
# 14 at 32, 19 at 64, 26 at 128, 36 at 256, 51 at 512 -- roughly T ~ 2-3*sqrt(F).  So the
# few-row shapes below sit in the significance regime and the rest in the sufficiency one.
SHAPES = [(32, 4), (128, 4), (512, 4), (32, 8), (64, 8), (64, 16), (32, 128), (16, 64),
          (6, 64), (8, 128), (5, 16), (10, 256)]


def _deficit(W):
    """`log2 F - H_F` on the feature power marginal -- the quantity the band is a bar on."""
    P = np.abs(W) ** 2
    return float(np.log2(W.shape[1])) - float(shannon_bits(P.sum(axis=0)))


def _concentrated(T, F, K, rng):
    W = 0.05 * rng.standard_normal((T, F))
    W[:, :K] += rng.standard_normal((T, K))
    return W


@pytest.mark.parametrize("shape", SHAPES)
def test_false_fold_rate_is_at_most_far(shape):
    """The Cantelli promise: pure noise clears the band at most `far` of the time."""
    T, F = shape
    rng = np.random.default_rng(abs(hash(shape)) % (2 ** 32))
    band = fold_band(T, F, far=FAR)
    d = np.array([_deficit(rng.standard_normal((T, F))) for _ in range(DRAWS)])
    rate = float((d > band).mean())
    assert rate <= FAR, (
        f"T={T} F={F}: {rate:.4f} of pure-noise records cleared the band {band:.5f}, above the "
        f"stated far={FAR}. Every one is a false fold, and it sets the screen width.")


@pytest.mark.parametrize("shape", SHAPES)
def test_the_band_still_folds_real_concentration(shape):
    """The control without which the test above is vacuous -- a huge band never folds anything."""
    T, F = shape
    K = max(1, F // 4)
    rng = np.random.default_rng(abs(hash(shape)) % (2 ** 32) + 1)
    band = fold_band(T, F, far=FAR)
    d = np.array([_deficit(_concentrated(T, F, K, rng)) for _ in range(DRAWS)])
    rate = float((d > band).mean())
    assert rate >= 0.5, (
        f"T={T} F={F} K={K}: the band {band:.5f} folded only {rate:.4f} of records whose power "
        f"genuinely sits in {K} of {F} channels. Calibration bought by never folding is not "
        f"calibration.")


def test_the_band_is_not_vacuously_wide():
    """`fold_band` must stay inside the entropy range without needing a cap, as it claims."""
    for T, F in SHAPES:
        assert 0.0 < fold_band(T, F, far=FAR) < np.log2(F), (
            f"T={T} F={F}: the band must be a real bar strictly inside [0, log2 F)")


def test_a_looser_level_admits_more_folds():
    """`far` must actually steer the bar, or it is decoration rather than a level."""
    for T, F in [(64, 8), (128, 16), (32, 4)]:
        assert fold_band(T, F, far=0.20) < fold_band(T, F, far=0.01), (
            f"T={T} F={F}: a looser `far` must lower the bar")


# ── the sufficiency term has its own job, and calibration cannot test it ──────

@pytest.mark.parametrize("shape,boost", [((512, 4), 1.30), ((512, 4), 1.60),
                                         ((256, 8), 1.30), ((128, 4), 1.60)])
def test_a_fold_that_is_real_but_not_worth_making_is_refused(shape, boost):
    """The sufficiency half: a concentration can be statistically real and still not earn a fold.

    `fold_band` is the MAX of two bars. The significance bar asks "is the concentration real?";
    the sufficiency bar asks "would folding move the Marchenko-Pastur edge by more than the
    Tracy-Widom margin?" -- because a fold that changes the width by a fraction of a percent
    gains nothing and perturbs the very floor that depends on the shape.

    Calibration cannot test this half. Dropping the sufficiency term makes the band SMALLER, and
    a smaller band cannot raise the false-fold rate on noise, so `test_false_fold_rate_is_at_most_far`
    passes either way. The case that separates them is a mild but genuine concentration at a tall
    shape: its deficit clears significance and must still be refused.

    Measured deficits here sit between the two bars -- e.g. T=512 F=4 at boost 1.30 gives 0.044
    against significance 0.0096 and the full band 0.736.
    """
    T, F = shape
    rng = np.random.default_rng(3)
    band = fold_band(T, F, far=FAR)
    folded = 0
    for _ in range(120):
        W = rng.standard_normal((T, F))
        W[:, 0] *= boost
        folded += int(_deficit(W) > band)
    assert folded == 0, (
        f"T={T} F={F} boost={boost}: {folded}/120 records were folded on a concentration too "
        f"small to move the screen's edge past its own noise margin. The sufficiency bar exists "
        f"to refuse exactly these.")


def test_the_refused_folds_are_genuinely_concentrated():
    """Control: the records above must really carry concentration, or the refusal is trivial."""
    from entroptics.entropy import dirichlet_entropy_moments, _tail_multiplier
    T, F, boost = 512, 4, 1.30
    mean_H, sd_H = dirichlet_entropy_moments(F, T / 2.0)
    significance = float(np.log2(F)) - mean_H + _tail_multiplier(FAR) * sd_H
    rng = np.random.default_rng(3)
    d = []
    for _ in range(120):
        W = rng.standard_normal((T, F))
        W[:, 0] *= boost
        d.append(_deficit(W))
    assert float(np.mean(d)) > significance, (
        "these records must clear the SIGNIFICANCE bar, else they are just noise and the "
        "sufficiency term is not what refused them")


# ── the guarantee across LEVELS, not just at one, and for both input kinds ────

CURVE_FARS = [0.001, 0.01, 0.05, 0.20]
CURVE_SHAPES = [(32, 4), (64, 8), (128, 16), (16, 64)]


def _noise_deficits(T, F, draws, complex_, seed):
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for i in range(draws):
        if complex_:
            W = (rng.standard_normal((T, F)) + 1j * rng.standard_normal((T, F))) / np.sqrt(2.0)
        else:
            W = rng.standard_normal((T, F))
        out[i] = _deficit(W)
    return out


@pytest.mark.parametrize("complex_", [False, True], ids=["real", "complex"])
@pytest.mark.parametrize("shape", CURVE_SHAPES)
def test_realised_rate_never_exceeds_nominal_at_any_level(shape, complex_):
    """Cantelli is one-sided and distribution-free, so the realised rate must sit AT OR BELOW
    the nominal one at every level -- not only at the default `far`.

    Measured worst realised/nominal ratio over a wider grid: 0.263 on real input and 0.005 on
    complex.  The bound HOLDS everywhere and is LOOSE, which is what a distribution-free bound
    buys; the complex branch is the loosest because the null uses `a = T/2`, deliberately
    conservative for either input kind (see `fold_band`).
    """
    T, F = shape
    d = _noise_deficits(T, F, 800, complex_, abs(hash((T, F, complex_))) % (2 ** 32))
    for far in CURVE_FARS:
        rate = float((d > fold_band(T, F, far=far)).mean())
        assert rate <= far, (
            f"T={T} F={F} complex={complex_} far={far}: realised false-fold rate {rate:.4f} "
            f"exceeds the nominal level. The Cantelli bound is one-sided; it must not be crossed.")


def test_the_conservatism_has_a_dead_zone_and_it_is_documented():
    """What the loose bound costs, stated so a caller does not discover it in their own data.

    On narrow feature axes at tight levels the band never fires -- not even on records whose
    power genuinely sits in `K = F//4` channels.  Measured: at `F = 4` the fold is refused for
    every `far <= 0.01`, and at `F = 8` for `far <= 0.001`.  That is the price of a
    distribution-free bar on a short marginal, and it is a usable-range statement rather than a
    defect: `far = 0.05` folds these same records every time.

    This test pins the boundary in BOTH directions, so a change that widens the dead zone and a
    change that removes it both show up here rather than silently.
    """
    dead = _deficits_concentrated(32, 4, 1, 300, seed=11)
    assert float((dead > fold_band(32, 4, far=0.01)).mean()) == 0.0, (
        "F=4 at far=0.01 is documented as a dead zone; if it now folds, update the docstring")
    assert float((dead > fold_band(32, 4, far=0.05)).mean()) >= 0.99, (
        "the same records must fold at the default level, or the dead zone is not a level "
        "effect but a failure to detect concentration at all")


def _deficits_concentrated(T, F, K, draws, seed):
    rng = np.random.default_rng(seed)
    return np.array([_deficit(_concentrated(T, F, K, rng)) for _ in range(draws)])


# ── the band must be a real bar at EVERY level, including loose ones ──────────

LEVELS = [0.001, 0.01, 0.05, 0.10, 0.16, 0.17, 0.20, 0.30, 0.50]


@pytest.mark.parametrize("shape", [(32, 4), (64, 8), (128, 16), (16, 64), (256, 8)])
def test_the_band_is_positive_at_every_level(shape):
    """`q_TW1` is the (1-far) quantile of a law centred near -1.21, so it turns NEGATIVE above
    `far ~ 0.168`.  Left unclamped that made the sufficiency term negative -- not inactive but
    meaningless -- and `max(significance, sufficiency)` masked it silently, so nothing failed and
    the bar was quietly nonsense at every loose level.

    A band is a bar on a deficit; a deficit is non-negative; so the band must be too.
    """
    T, F = shape
    for far in LEVELS:
        b = fold_band(T, F, far=far)
        assert b > 0.0, (
            f"T={T} F={F} far={far}: band {b} is not a positive bar. Above far~0.168 the "
            f"Tracy-Widom quantile is negative; `dF` must be clamped at zero, not passed through.")


@pytest.mark.parametrize("shape", [(32, 4), (64, 8), (128, 16), (16, 64), (256, 8)])
def test_a_looser_level_never_raises_the_bar(shape):
    """Monotone in `far`, across the sign change too -- otherwise the level is not a level.

    This is the property the negative-`dF` bug broke in a way no single-level test could see:
    each level was individually plausible while the sequence was not.
    """
    T, F = shape
    bands = [fold_band(T, F, far=f) for f in LEVELS]
    for a, b, fa, fb in zip(bands, bands[1:], LEVELS, LEVELS[1:]):
        assert b <= a + 1e-12, (
            f"T={T} F={F}: band rose from {a:.6f} at far={fa} to {b:.6f} at far={fb}; "
            f"a looser level must not demand more evidence")


def test_the_refusal_regime_is_total_and_documented():
    """Where the required width change exceeds `F - 1`, the bar sits at exactly `log2 F`.

    That is the intended reading -- no achievable fold moves the Marchenko-Pastur edge past its
    own Tracy-Widom margin -- and it is why a tight level on a narrow feature axis folds nothing.
    Pinned so the boundary cannot move without notice.
    """
    assert fold_band(32, 4, far=0.01) == pytest.approx(np.log2(4), abs=1e-12)
    assert fold_band(64, 8, far=0.001) == pytest.approx(np.log2(8), abs=1e-12)
    assert fold_band(32, 4, far=0.05) < np.log2(4)      # and it must NOT be total at the default

