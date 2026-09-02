"""The environmental compute precision (``set_precision`` / ``ENTROPTICS_PRECISION``).

Default 64 is bit-perfect (the reference results the rest of the suite pins); ``set_precision(32)``
computes the fold + screen in float32 (the GPU throughput path) with ``K_signal`` preserved and
the singular values within the fp32 band.  The global setting must never leak between tests."""
import numpy as np
import pytest

from entroptics import Projection, set_precision, precision
from entroptics import environment as _env


@pytest.fixture(autouse=True)
def _reset_precision():
    """No test may leave the process in fp32 -- reset to the bit-perfect default afterward."""
    yield
    set_precision(64)


def test_default_is_64():
    assert precision() == 64
    assert _env.rdtype(np) == np.float64 and _env.cdtype(np) == np.complex128


def test_64_is_float64_and_unchanged(make_W):
    W = make_W(3, 200, 48)
    ref = Projection(W)
    set_precision(64)
    s = Projection(W)
    assert s.screen.dtype == np.float64
    assert s.sigma_top == ref.sigma_top and s.K_signal == ref.K_signal   # bit-identical to default


def test_32_folds_in_float32():
    W = np.random.default_rng(3).standard_normal((200, 48)).astype(np.float32)
    set_precision(32)
    s = Projection(W)
    assert s.screen.dtype == np.float32               # the fold keeps its float32 dtype
    assert _env.rdtype(np) == np.float32 and _env.cdtype(np) == np.complex64


def test_32_preserves_k_signal_within_fp32_band():
    W = 5.0 * np.outer(np.random.default_rng(0).standard_normal(400),
                       np.random.default_rng(1).standard_normal(64)) \
        + np.random.default_rng(2).standard_normal((400, 64))
    set_precision(64); s64 = Projection(W)
    set_precision(32); s32 = Projection(W.astype(np.float32))
    assert s32.K_signal == s64.K_signal
    assert abs(s32.sigma_top - s64.sigma_top) / s64.sigma_top < 1e-3   # within the fp32 precision band


def test_as_compute_noop_at_64_casts_at_32():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    set_precision(64)
    assert _env.as_compute(np, x).dtype == np.float64
    set_precision(32)
    assert _env.as_compute(np, x).dtype == np.float32
    xc = x.astype(np.complex128)
    assert _env.as_compute(np, xc).dtype == np.complex64


def test_set_precision_only_accepts_32_or_64():
    set_precision(32); assert precision() == 32
    set_precision(7);  assert precision() == 64        # anything but 32 -> the bit-perfect default
