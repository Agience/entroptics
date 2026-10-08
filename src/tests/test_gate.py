"""The release gate's rule (``entroptics.gate.compare``), on both branches."""
import pytest

from entroptics.gate import compare


def _m(v, se, better="higher", **kw):
    return dict(value=v, se=se, better=better, **kw)


BASE = {"a": _m(0.8, 0.05), "b": _m(100.0, 1.0, "lower"), "c": _m(0.05, 0.01, "bound", bound=0.05)}


def test_an_unchanged_measurement_passes():
    assert compare(BASE, BASE).passed


def test_a_move_within_the_noise_passes_and_beyond_it_fails():
    within = dict(BASE, a=_m(0.8 - 0.05, 0.05))                 # one combined sigma down
    assert compare(BASE, within).passed
    beyond = dict(BASE, a=_m(0.2, 0.05))
    r = compare(BASE, beyond)
    assert not r.passed and [w[0] for w in r.worse] == ["a"]
    cost = dict(BASE, b=_m(150.0, 1.0, "lower"))                 # a lower-is-better metric rising
    assert not compare(BASE, cost).passed


def test_an_improvement_is_reported_and_passes():
    r = compare(BASE, dict(BASE, a=_m(0.99, 0.01), b=_m(50.0, 1.0, "lower")))
    assert r.passed and sorted(x[0] for x in r.better) == ["a", "b"]


def test_a_bound_must_hold_and_a_missing_metric_fails():
    assert not compare(BASE, dict(BASE, c=_m(0.2, 0.01, "bound", bound=0.05))).passed
    r = compare(BASE, {k: v for k, v in BASE.items() if k != "b"})
    assert not r.passed and r.missing == ["b"]


def test_the_noise_band_widens_with_the_number_of_metrics():
    # z at far / (2 M): more comparisons, a wider band, so the family errs at most far of the time
    assert compare({"a": BASE["a"]}, {"a": BASE["a"]}).z < compare(BASE, BASE).z
    with pytest.raises(ValueError):
        compare(BASE, BASE, far=1.0)


def _ladder(n, se=0.05):
    return {f"s{i}": _m(0.5, se) for i in range(n)}


def test_a_loss_spread_thinly_over_many_metrics_fails():
    """21 of 27 cells a little lower and 6 a little higher, none beyond its noise: each passes, the
    direction does not (P(>= 21 of 27 | no worse) = 0.003).  The floor that halved a weak line's
    power moved the release baseline this way and passed the per-metric test."""
    base = _ladder(27)
    cand = {k: _m(0.5 - 0.02 if i < 21 else 0.5 + 0.02, 0.05) for i, k in enumerate(base)}
    r = compare(base, cand)
    assert not r.worse and not r.passed and r.drift
    assert r.moved[:2] == (21, 6) and r.moved[2] < 0.025


def test_moves_either_way_as_often_pass():
    base = _ladder(27)
    cand = {k: _m(0.5 - 0.02 if i % 2 else 0.5 + 0.02, 0.05) for i, k in enumerate(base)}
    r = compare(base, cand)
    assert r.passed and not r.drift and r.moved[:2] == (13, 14)


def test_a_bound_does_not_vote():
    # a false-alarm level falling further under its bound is held to the bound, not counted as a gain
    base = dict(_ladder(10), **{f"l{i}": _m(0.05, 0.01, "bound", bound=0.05) for i in range(20)})
    cand = dict({k: _m(0.48, 0.05) for k in _ladder(10)},
                **{f"l{i}": _m(0.01, 0.01, "bound", bound=0.05) for i in range(20)})
    r = compare(base, cand)
    assert r.moved[:2] == (10, 0) and r.drift and not r.passed


def test_a_rate_read_over_a_broken_level_is_no_reference():
    """A baseline that claimed structure in 36% of noise records 'found' weak signals by its false
    alarms: its detection rates are set aside, not held against a candidate that holds the level."""
    lv = _m(0.36, 0.0154, "bound", bound=0.05)
    base = {"lv": lv, "s": _m(0.95, 0.02, level="lv")}
    cand = {"lv": _m(0.05, 0.0154, "bound", bound=0.05), "s": _m(0.23, 0.04, level="lv")}
    r = compare(base, cand)
    assert r.passed and r.unreferenced == [("s", "lv")] and not r.worse
    held = dict(base, lv=_m(0.05, 0.0154, "bound", bound=0.05))        # the same drop at a held level
    assert not compare(held, cand).passed
