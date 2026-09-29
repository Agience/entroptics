"""A reference-calibrated floor counts structure, not the shape of the noise's marginal.

Under the median / MAD whitening before 0.2.5, a screen's energy tracked each column's (RMS / MAD)^2,
a one-point property of the marginal. A floor pinned on one marginal then read a different
marginal, with no correlation anywhere, as resolved modes: i.i.d. von Mises plaquettes against a
Haar-calibrated reference gave a mean K of 0.873. Under the mean / RMS whitening every column has
the same energy, and the read is of correlation. Negative control: the skewed marginal alone must
sit at the level. Positive control: a correlated mode must still be found. Both fail on the old
whitening (0.873 and 0.147), so the test can fail.

It also pins two guards: a reference calibrated at one screen shape refuses a screen of another,
and the moment pencil reports how many directions its cut kept."""
import numpy as np
import pytest

import entroptics as E
from entroptics import hankel_spectrum
from entroptics.null_providers import ReferenceNull, reference_null

N_FRAMES = 4000


def _plaquette(rng, kappa):
    theta = rng.vonmises(0.0, kappa, (8, 8)) if kappa else rng.uniform(-np.pi, np.pi, (8, 8))
    return 1.0 - np.cos(theta)


def _mean_K(frames, null):
    return float(np.mean([r.K_signal for r in E.read_batch(frames, null=null)]))


def test_a_skewed_marginal_alone_is_not_structure():
    rng = np.random.default_rng(0)
    ref = reference_null([r.sigma_top for r in E.read_batch([_plaquette(rng, 0) for _ in range(N_FRAMES)])])
    assert _mean_K([_plaquette(rng, 0) for _ in range(N_FRAMES)], ref) <= 0.08      # its own marginal
    assert _mean_K([_plaquette(rng, 4.0) for _ in range(N_FRAMES)], ref) <= 0.08    # 0.873 before 0.2.5


def test_a_correlated_mode_is_still_found():
    rng = np.random.default_rng(1)
    g = reference_null([r.sigma_top for r in E.read_batch([rng.standard_normal((8, 8)) for _ in range(N_FRAMES)])])
    sig = [rng.standard_normal((8, 8)) + np.outer(rng.standard_normal(8), np.ones(8)) for _ in range(N_FRAMES)]
    assert _mean_K(sig, g) >= 0.6                                                    # 0.147 before 0.2.5


def test_a_reference_refuses_a_screen_of_another_shape():
    rng = np.random.default_rng(2)
    tops = [r.sigma_top for r in E.read_batch([rng.standard_normal((8, 8)) for _ in range(200)])]
    for null in (reference_null(tops, shape=(8, 8)), ReferenceNull(tops, shape=(8, 8))):
        E.read_batch([rng.standard_normal((8, 8))], null=null)                       # same shape: read
        with pytest.raises(ValueError, match="calibrated on"):
            E.read_batch([rng.standard_normal((6, 6))], null=null)
    E.read_batch([rng.standard_normal((6, 6))], null=reference_null(tops))          # no shape: as before


def test_the_pencil_reports_the_order_it_kept():
    """The default cut drops H0 directions within the record's own noise, so an order-3 pencil can
    be an order-1 read; ``kept`` says so, and a fixed ``rcond`` keeps more."""
    c = 0.6 ** np.arange(12) + 1e-3 * np.random.default_rng(1).standard_normal(12)
    a, b = hankel_spectrum(c, 3), hankel_spectrum(c, 3, rcond=1e-6)
    assert a.kept == len(a.evals) == 1
    assert b.kept == len(b.evals) == 3
    assert hankel_spectrum(0.6 ** np.arange(12), 2).kept >= 1
