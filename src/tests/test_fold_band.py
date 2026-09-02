"""The fold band (Def 2.2) and the two derived criteria behind it.

The band replaced a capped Miller-Madow form -- ``T`` times the *mean* null deficit, capped at
``(1/2) log2 F`` because the multiplier ran past the entropy range on wide-short frames.  Both
numbers were chosen.  What replaces them is the larger of two bounds, each derived from a null
the instrument already carries:

    significance   the deficit exceeds what the null itself produces
    sufficiency    the fold moves the Marchenko-Pastur edge by more than its Tracy-Widom margin

These tests pin the properties that made the replacement worth making, and the ones that would
make it wrong if they stopped holding.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics import Projection
from entroptics.entropy import (dirichlet_entropy_moments, fold_band, fold_width,
                                _tail_multiplier, _digamma, _trigamma)

# tall, square, wide, and the two extremes the old cap existed for
SHAPES = [(64, 64), (200, 200), (300, 120), (600, 40), (40, 600),
          (64, 256), (64, 4096), (19, 16384)]


def _null_deficit(T, F, n, seed, complex_cells=False):
    """log2(F) - H_F on iid Gaussian frames, the null the band is set against."""
    g = np.random.default_rng(seed)
    a = T if complex_cells else T / 2.0
    x = g.gamma(shape=a, scale=1.0, size=(n, F))
    p = x / x.sum(1, keepdims=True)
    return np.log2(F) - (-(p * np.log2(np.clip(p, 1e-300, None))).sum(1))


# ── the closed forms ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("x", [0.5, 1.0, 2.5, 7.0, 100.0, 8192.0])
def test_special_functions_match_scipy(x):
    """The numpy-only psi and psi' carry the core; scipy is optional and must not be needed."""
    sp = pytest.importorskip("scipy.special")
    assert abs(_digamma(x) - sp.digamma(x)) < 1e-9
    assert abs(_trigamma(x) - sp.polygamma(1, x)) < 1e-8


@pytest.mark.parametrize("T,F", SHAPES)
def test_entropy_moments_match_simulation(T, F):
    """Wolpert-Wolf is exact, not an expansion: mean and sd must match a direct draw."""
    m, sd = dirichlet_entropy_moments(F, T / 2.0)
    D = _null_deficit(T, F, 4000, seed=11)
    assert abs((np.log2(F) - D.mean()) - m) < 6.0 * D.std() / np.sqrt(4000)
    assert 0.9 < D.std() / sd < 1.1        # a variance from 4000 draws carries ~1.1% error


def test_cantelli_multiplier_is_exact():
    """k = sqrt(1/far - 1) is the bound itself, not an approximation to a normal quantile."""
    for far in (0.5, 0.1, 0.05, 0.01, 1e-4):
        k = _tail_multiplier(far)
        assert abs(1.0 / (1.0 + k * k) - far) < 1e-12


# ── the properties the old cap was there to rescue ────────────────────────────────────────────
@pytest.mark.parametrize("T,F", SHAPES)
def test_band_cannot_reach_the_entropy_range(T, F):
    """The defect the (1/2)log2 F cap patched: a band above log2 F disables the fold vacuously.

    A tail bound on a deficit that lives in [0, log2 F] cannot do that, so no cap is needed.
    Measured well below 1% of the range at every shape here."""
    assert 0.0 < fold_band(T, F) < 0.05 * np.log2(F)


@pytest.mark.parametrize("T,F", SHAPES)
def test_noise_does_not_fold(T, F):
    """The band's first job. A pure-noise frame must not be folded -- folding noise blends
    adjacent cells and manufactures coherence, so this is the expensive direction to get wrong."""
    D = _null_deficit(T, F, 400, seed=23)
    assert float(np.mean(D > fold_band(T, F))) <= 0.01


def test_band_tightens_as_far_tightens():
    """far is the level, so a stricter level can only widen the no-fold band."""
    bands = [fold_band(200, 200, far=f) for f in (0.2, 0.05, 0.01, 1e-3)]
    assert bands == sorted(bands), "a smaller far must not shrink the band"


# ── the fold it now licenses is one worth making ──────────────────────────────────────────────
def test_a_concentrated_continuous_axis_folds_and_the_fold_is_faithful():
    """The band's second job: a record genuinely oversampled on a continuous axis SHOULD fold,
    and the fold must lose almost nothing -- otherwise the read is bought with signal."""
    T, F = 64, 256
    t = np.arange(T)[:, None]
    f = np.arange(F)[None, :]
    B = np.exp(-0.5 * ((t - 30) / 3.0) ** 2) * np.exp(-0.5 * ((f - 128) / 56.0) ** 2)

    P = np.abs(B) ** 2
    pF = P.sum(0) / P.sum()
    H_F = float(-(pF * np.log2(np.clip(pF, 1e-300, None))).sum())

    n_F, delta_F = fold_width(H_F, T, F, B)
    assert n_F < F and delta_F > 1.0, "an oversampled continuous axis must fold"

    idx = (np.arange(F) * n_F) // F
    folded = np.stack([B[:, idx == j].mean(1) for j in range(n_F)], axis=1)
    residual = np.linalg.norm(B - folded[:, idx]) / np.linalg.norm(B)
    assert residual < 0.05, f"the licensed fold must be near-lossless (got {residual:.3f})"


def test_folding_does_not_change_what_is_resolved():
    """A fold reads the same signal at its own width. It must not change the mode count -- if it
    did, the fold would be buying resolution with structure."""
    g = np.random.default_rng(5)
    T, F, K = 200, 200, 3
    U = np.linalg.qr(g.standard_normal((T, K)))[0]
    V = np.linalg.qr(g.standard_normal((F, K)))[0]
    edge = np.sqrt(T) + np.sqrt(F)
    W = U @ np.diag([3.0 * edge] * K) @ V.T + g.standard_normal((T, F))

    native = Projection(W).K_signal
    n_F = 160
    idx = (np.arange(F) * n_F) // F
    folded = np.stack([W[:, idx == j].mean(1) for j in range(n_F)], axis=1)
    assert Projection(folded).K_signal == native == K
