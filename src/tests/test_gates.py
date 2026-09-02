"""The cheap signal gates for high-rate capture.

``Projection.has_signal``, and ``Aperture.has_signal()`` which delegates to it, must mirror
``K_signal > 0`` exactly, and the SVD-free ``probe_signal`` is conservative -- it may
over-fire but must never return False when a real mode is resolved (so gating on it cannot drop a
detection).  Deterministic seeds."""
import numpy as np
import pytest

from entroptics import Projection, Aperture
from entroptics.projection import probe_signal


def _mix(n=40, N=96, F=12, seed=0):
    """Alternating pure-noise (K=0 expected) and planted-mode (K>=1 expected) frames."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        amp = 6.0 if i % 2 else 0.0
        yield amp * np.outer(rng.standard_normal(N), rng.standard_normal(F)) + rng.standard_normal((N, F))


def test_screen_has_signal_matches_k_signal():
    hits = 0
    for W in _mix(seed=1):
        s = Projection(W)
        assert s.has_signal == (s.K_signal > 0)
        hits += s.has_signal
    assert 0 < hits < 40                               # both branches (signal / no signal) are exercised


def test_aperture_has_signal_matches_screen():
    # planes short enough (< the 128 window) that the Aperture never truncates -> equals Projection.
    for W in _mix(N=96, seed=2):
        ap = Aperture(W)
        assert ap.has_signal() == Projection(W).has_signal
        assert ap.has_signal() == (Projection(W).K_signal > 0)


def test_aperture_has_signal_is_bool_and_idempotent():
    W = 6.0 * np.outer(np.random.default_rng(0).standard_normal(96),
                       np.random.default_rng(1).standard_normal(12)) \
        + np.random.default_rng(2).standard_normal((96, 12))
    ap = Aperture(W)
    assert ap.has_signal() is True                     # planted mode -> True (a real bool)
    assert ap.has_signal() == ap.has_signal()          # repeatable
    noise = Aperture(np.random.default_rng(9).standard_normal((96, 12)))
    assert noise.has_signal() is False


def test_probe_signal_conservative_never_false_on_signal():
    for W in _mix(seed=3):
        if Projection(W).K_signal > 0:
            assert probe_signal(W) is True             # never drop a real detection


def test_probe_signal_defers_on_gaps():
    W = np.random.default_rng(1).standard_normal((40, 8)); W[0, 0] = np.nan
    assert probe_signal(W) is True                     # masked / gapped -> defer to the full Projection
