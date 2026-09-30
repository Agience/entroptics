"""No read may depend on the absolute size of the data.

Several guards floored a variance or a scale at an absolute ``1e-30`` (and a tolerance at ``1e-10`` or
``1e-9``).  Variances scale as the square of the data, so for a record scaled near 1e-15 or below the
floor replaced the true value and the read changed.  Each guard is now relative to the data (its
largest value times the format's eps) or the arithmetic's own round-off.  Each test here fails on
the absolute floors."""
import math

import numpy as np
import pytest

from entroptics.dynamics import dynamics
from entroptics.entropy import _tail_multiplier, fold_band
from entroptics.null_providers import ReferenceNull, reference_null, top_spectrum_value

TINY_SCALE = 1e-20      # variances ~1e-40, below the old 1e-30 floor


def _record(seed=0, T=400, F=12):
    rng = np.random.default_rng(seed)
    t = np.arange(T)[:, None]
    return (np.cos(0.21 * t + rng.uniform(0, 6, F)) + 0.5 * np.cos(0.05 * t + rng.uniform(0, 6, F))
            + 0.1 * rng.standard_normal((T, F)))


def test_the_streaming_count_does_not_see_the_data_scale():
    W = _record()
    assert dynamics(W * TINY_SCALE).resolved() == dynamics(W).resolved() > 0


def test_the_streaming_spectrum_does_not_see_the_data_scale():
    W = _record(1)
    a = np.asarray(dynamics(W)._feature_evals())
    b = np.asarray(dynamics(W * TINY_SCALE)._feature_evals())
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12)


def test_the_reference_statistic_does_not_see_the_data_scale():
    X = _record(2)
    for kind in ("spectral", "bulk"):
        assert top_spectrum_value(X * TINY_SCALE, kind) == pytest.approx(top_spectrum_value(X, kind),
                                                                           rel=1e-9)


def test_a_constant_reference_gives_its_value_as_the_floor():
    assert reference_null([2.5] * 50).scale == 0.0
    assert ReferenceNull([2.5] * 50).scale == 0.0


def test_far_is_honoured_not_clipped():
    # Cantelli's k = sqrt(1/far - 1) for any far in (0, 1); far = 0.9 was read as 0.5 (k = 1)
    assert _tail_multiplier(0.9) == pytest.approx(math.sqrt(1 / 0.9 - 1))
    assert fold_band(64, 16, far=0.9) != fold_band(64, 16, far=0.5)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            _tail_multiplier(bad)
        with pytest.raises(ValueError):
            fold_band(64, 16, far=bad)
