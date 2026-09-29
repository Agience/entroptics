"""entroptics.experimental.operator_read: the record as operator + exact residual."""
import numpy as np
import pytest

import entroptics as E
from entroptics.experimental import OperatorRead, operator_read


def _tones(T, tones, sigma, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = sum(A * np.exp(-a * t) * np.cos(2 * np.pi * f * t + i) for i, (f, a, A) in enumerate(tones))
    return x + sigma * rng.standard_normal(T)


def _fft_err(x, ft):
    T = x.size
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    k0 = int(round(ft * T))
    k = k0 - 3 + int(np.argmax(P[k0 - 3:k0 + 4]))
    y0, y1, y2 = (np.log(P[k + i]) for i in (-1, 0, 1))
    return abs((k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) / T - ft)


def _nearest(o, ft):
    f = o.frequency[o.frequency >= 0]
    return float(np.min(np.abs(f - ft)))


def test_two_tones_between_bins_beat_the_fft_and_read_the_decay():
    tones = [(0.0402, 0.002, 1.0), (0.0926, 0.0, 0.6)]
    x = _tones(2048, tones, 0.05)
    o = operator_read(x)
    assert isinstance(o, OperatorRead) and o.K == 4
    for ft, _, _ in tones:
        assert _nearest(o, ft) < _fft_err(x, ft)
    j = int(np.argmin(np.abs(o.frequency - 0.0402)))
    assert o.modes.alpha[j] == pytest.approx(0.002, abs=0.001)          # the FFT reads no decay


def test_a_weak_tone_beside_a_strong_one():
    """-50 dB, 10 bins away: the operator finds it; the FFT's leakage buries it."""
    fw = 0.1 + 10 / 2048
    x = _tones(2048, [(0.1, 0.0, 1.0), (fw, 0.0, 10 ** (-50 / 20))], 1e-4)
    o = operator_read(x)
    assert _nearest(o, fw) < 1e-4


def test_a_noiseless_tone_converges_far_ahead_of_the_fft():
    """The lags are sample means over T - k pairs, so a finite record biases them at order 1/T and
    a noiseless tone is not read to round-off.  Measured: 2.4e-7 at T = 1024 and 1.7e-8 at 4096,
    against the interpolated FFT's 1.2e-4 and 4.1e-5 -- falling faster than the FFT's, and with a
    decay read at zero."""
    errs = []
    for T in (1024, 4096):
        x = np.cos(2 * np.pi * 0.0402 * np.arange(T))
        o = operator_read(x)
        assert o.K == 2
        e = _nearest(o, 0.0402)
        assert e < _fft_err(x, 0.0402) / 100
        assert abs(o.modes.alpha[0]) < 1e-5
        errs.append(e)
    assert errs[1] < errs[0] / 4                            # faster than 1/T


def test_split_is_lossless_and_leaves_the_noise():
    x = _tones(2048, [(0.0402, 0.0, 1.0), (0.0926, 0.0, 0.7), (0.15, 0.0, 0.5)], 0.05)
    o = operator_read(x)
    resolved, residual = o.split(x)
    assert np.max(np.abs(resolved + residual - x)) < 1e-12
    assert np.sum(residual ** 2) / (0.05 ** 2 * x.size) == pytest.approx(1.0, abs=0.1)


def test_the_view_is_the_modes_view():
    o = operator_read(_tones(2048, [(0.0402, 0.002, 1.0)], 0.05))
    f = np.linspace(-0.5, 0.5, 101)
    assert np.array_equal(o.spectrum(f), o.modes.spectrum(f))


def test_gaps_are_left_out_not_filled():
    x = _tones(2048, [(0.0402, 0.0, 1.0), (0.0926, 0.0, 0.6)], 0.05)
    x[700:760] = np.nan
    o = operator_read(x)
    assert _nearest(o, 0.0402) < 1e-4 and _nearest(o, 0.0926) < 1e-4
    resolved, residual = o.split(x)
    assert np.isnan(resolved[700:760]).all() and np.isnan(residual[700:760]).all()
    ok = np.isfinite(x)
    assert np.max(np.abs((resolved + residual - x)[ok])) < 1e-12


def test_an_ar_process_reads_its_decay():
    rng = np.random.default_rng(3)
    T, F, rho = 2048, 16, 0.5
    x = np.zeros((T, F))
    e = rng.standard_normal((T, F))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    o = operator_read(x)
    assert o.K >= 1
    assert o.modes.alpha[0] == pytest.approx(-np.log(rho), rel=0.1)


def test_white_noise_claims_at_most_its_level():
    """Negative control: over 60 white records the claim rate stays within the binomial scatter of
    far = 0.05 (P(X >= 9 | n = 60, p = 0.05) < 0.5%)."""
    claims = sum(operator_read(np.random.default_rng(100 + i).standard_normal((512, 4))).K > 0
                 for i in range(60))
    assert claims <= 8


def test_nothing_counted_is_an_empty_read():
    o = operator_read(np.random.default_rng(7).standard_normal((256, 2)), far=1e-6)
    assert o.K == 0 and o.depth == 0 and np.asarray(o.modes.mu).size == 0
    x = np.random.default_rng(8).standard_normal(256)
    resolved, residual = o.split(x)
    assert np.allclose(resolved, x.mean()) and np.max(np.abs(resolved + residual - x)) < 1e-12


def test_refusals_and_its_place_in_the_surface():
    with pytest.raises(ValueError):
        operator_read(np.ones(64) + 1j)
    with pytest.raises(ValueError):
        operator_read(np.ones((4, 4, 4)))
    with pytest.raises(ValueError):
        operator_read(np.ones(64), far=0.0)
    assert not hasattr(E, "operator_read") and "operator_read" not in E.__all__
