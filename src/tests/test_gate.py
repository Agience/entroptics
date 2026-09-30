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
    # z at far / M: more comparisons, a wider band, so the family errs at most far of the time
    assert compare({"a": BASE["a"]}, {"a": BASE["a"]}).z < compare(BASE, BASE).z
    with pytest.raises(ValueError):
        compare(BASE, BASE, far=1.0)
