"""Periodic (wrapped-axis) decay: the circular autocorrelation and its image-folded
(cosh) decay rate.  The core guarantee is WINDOW-INVARIANCE -- the rate must not drift
with the window length L, unlike a pure-exponential fit or the DMD on a torus.  The
analytic-cosh identity lets us test that EXACTLY; a seeded synthetic ensemble exercises
the full periodic_decay -> effective_decay_rate pipeline."""
import math

import numpy as np
import pytest

from entroptics import periodic_decay, effective_decay_rate, Aperture


# ── the exact, deterministic guarantee: a single cosh reads its rate at ANY window ──
@pytest.mark.parametrize("D", [0.3, 0.5, 0.8, 1.2])
@pytest.mark.parametrize("L", [16, 32, 64, 128, 256])
def test_effective_rate_is_window_invariant_on_a_cosh(D, L):
    """C(t) = cosh(D (t - L/2)) is the exact periodic (forward + backward image) shape of a
    single mode of rate D.  effective_decay_rate must return D EXACTLY, for every L -- the
    identity cosh(a-D)+cosh(a+D) = 2 cosh(a) cosh(D) makes the arccosh read cosh(D) at
    every lag.  This is the window-invariance the pure-exp DMD lacks."""
    t = np.arange(L, dtype=float)
    C = np.cosh(D * (t - L / 2.0))
    assert effective_decay_rate(C) == pytest.approx(D, abs=1e-9)


def test_pure_exponential_reads_its_rate():
    """Even an APERIODIC one-sided exp e^{-Dt} reads D exactly: (C(t-1)+C(t+1))/(2C(t)) =
    cosh(D), so the estimator is correct for both the periodic cosh and a clean exp."""
    t = np.arange(64.0)
    assert effective_decay_rate(np.exp(-0.4 * t)) == pytest.approx(0.4, abs=1e-9)


def test_flat_correlator_reads_zero():
    """A flat correlator has no resolvable decay region -> rate 0."""
    assert effective_decay_rate(np.ones(64)) == 0.0


def test_degenerate_inputs_are_safe():
    assert effective_decay_rate(np.array([1.0])) == 0.0        # too short
    assert effective_decay_rate(np.array([0.0, 0.0, 0.0])) == 0.0  # zero peak
    C = periodic_decay(np.zeros((1, 4)))                       # a single dead row
    assert C.shape == (1,)


# ── the correlator itself: connected, circular (symmetric), normalized ──
def _periodic_field(L, D, F, rng):
    """F independent 1-D periodic scalar fields of extent L with a KNOWN decay rate D:
    draw in Fourier space with the 1-D lattice propagator power S(k) = 1/(4 sin^2(pi k/L)
    + M^2), M^2 = 2(cosh D - 1), whose circular autocorrelation is exactly cosh(D(t-L/2))."""
    M2 = 2.0 * (math.cosh(D) - 1.0)
    k = np.arange(L)
    S = 1.0 / (4.0 * np.sin(np.pi * k / L) ** 2 + M2)          # power spectrum, S(k)=S(L-k)
    out = np.empty((L, F))
    for f in range(F):
        eta = rng.standard_normal(L) + 1j * rng.standard_normal(L)
        phi_hat = np.sqrt(S) * eta
        out[:, f] = np.fft.ifft(phi_hat).real                  # real periodic field
    return out


def test_periodic_decay_is_normalized_and_symmetric():
    rng = np.random.default_rng(0)
    stack = [_periodic_field(64, 0.5, 8, rng) for _ in range(16)]
    C = periodic_decay(stack)
    assert C[0] == pytest.approx(1.0)                          # normalized to C(0)=1
    # circular symmetry C(t) = C(L-t) (the backward image) -- the hallmark of a wrapped axis
    L = C.size
    for t in (1, 2, 5, 10):
        assert C[t] == pytest.approx(C[L - t], abs=0.06)


def test_periodic_decay_accepts_single_screen_and_stack():
    rng = np.random.default_rng(1)
    W = _periodic_field(48, 0.6, 6, rng)
    Csingle = periodic_decay(W)                                # a single (T, F) screen
    Cstack = periodic_decay([W])                               # a one-element stack -> identical
    assert np.allclose(Csingle, Cstack)
    assert Csingle.shape == (48,)


def test_full_pipeline_recovers_the_rate_and_is_window_invariant():
    """periodic_decay (ensemble) -> effective_decay_rate recovers the known synthetic rate
    to within its calibration systematic, and the estimate does not drift with L."""
    rng = np.random.default_rng(7)
    D = 0.5
    got = {}
    for L in (32, 64, 128):
        stack = [_periodic_field(L, D, 16, rng) for _ in range(48)]
        got[L] = effective_decay_rate(periodic_decay(stack))
    vals = np.array(list(got.values()))
    # statistical bounds for this modest synthetic ensemble (48 cfgs x 16 cols); the
    # EXACT window-invariance guarantee is test_effective_rate_is_window_invariant_on_a_cosh.
    assert np.all(vals > 0.75 * D) and np.all(vals < 1.25 * D)   # recovers D (finite-stat scatter)
    assert vals.max() / vals.min() < 1.30                        # no systematic L drift


def test_aperture_exposes_periodic_reads():
    """The front door surfaces the periodic decay and its rate; effective_decay_rate is the
    image-folded rate of the window's periodic_decay."""
    rng = np.random.default_rng(3)
    W = _periodic_field(64, 0.5, 12, rng)
    ap = Aperture(W)
    C = ap.periodic_decay
    assert C[0] == pytest.approx(1.0)
    assert ap.effective_decay_rate == pytest.approx(effective_decay_rate(C), abs=1e-9)
    assert ap.effective_decay_rate >= 0.0
