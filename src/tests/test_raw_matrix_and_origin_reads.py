"""Two reads for records the whitened, connected screen does not describe.

``null_providers.mp_median`` -- the closed-form edge of a matrix read as it stands, its noise
variance from the median row energy -- and ``Projection(..., disconnected=None)``, the read that
keeps the origin.  Both are additive: the default floor and the connected read are unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binom

import entroptics as E
from entroptics import null_providers as NP
from entroptics.projection import live_columns, mode_significance, noise_floor

FAR = 0.05


def _planted(d=256, rank=6, spike=None, seed=0):
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    V = np.linalg.qr(rng.standard_normal((d, rank)))[0]
    A = (U * np.linspace(60.0, 40.0, rank)) @ V.T + rng.standard_normal((d, d))
    if spike is not None:
        A[100, 200] = spike
    return A


# ── mp_median ──────────────────────────────────────────────────────────────────────────────────


def test_mp_median_is_the_median_row_energy_edge():
    """The floor is sqrt(sigma^2 * (mu + q sigma_J)) with sigma^2 the de-biased median row energy."""
    A = _planted()
    N, F = A.shape
    re = np.sum(A * A, axis=1)
    s2 = (np.median(re) + re.sum() * np.finfo(float).eps) / NP.debias_denominator(N, F)
    want = np.sqrt(NP.screen_floor_sq(s2, N, F, FAR))
    assert float(noise_floor(A, far=FAR, null=NP.mp_median)) == pytest.approx(want, rel=1e-15)


def test_mp_median_reads_a_raw_matrix_the_mean_energy_edge_does_not():
    """A raw matrix's mean cell energy is mostly its own signal: the planted rank with one giant
    cell is recovered by the median edge and lost under the whitened-screen ``mp`` edge."""
    A = _planted(spike=400.0)
    s = np.linalg.svd(A, compute_uv=False)
    k_median = int((s > noise_floor(A, far=FAR, null=NP.mp_median, s=s)).sum())
    k_mean = int((s > noise_floor(A, far=FAR, null=NP.mp, s=s)).sum())
    assert k_median == 7          # the six planted modes and the giant cell's own
    assert k_mean < 6


def test_mp_median_on_a_correlation_kind_is_mp():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 12))
    ctx = NP.FloorContext(spectrum=None, data=X, shape=X.shape, far=FAR, kind="spectral",
                          rng=np.random.default_rng(0))
    assert NP.mp_median(ctx) == NP.mp(ctx)


@pytest.mark.parametrize("far", [1e-6, 1e-3, 0.05, 0.2])
def test_mode_significance_on_the_median_variance_counts_what_mp_median_does(far):
    A = _planted(spike=400.0, seed=2)
    s = np.linalg.svd(A, compute_uv=False)
    s2 = NP.median_noise_sigma2(np, A, A.shape[0], live_columns(A))
    p = mode_significance(A, s, sigma2=s2).pvalue
    assert int((p < far).sum()) == int((s > noise_floor(A, far=far, null=NP.mp_median, s=s)).sum())


def test_mode_significance_default_is_unchanged():
    A = _planted(seed=3)
    s = np.linalg.svd(A, compute_uv=False)
    explicit = NP.noise_sigma2(np, A, A.shape[0], live_columns(A), s=s)
    a, b = mode_significance(A, s), mode_significance(A, s, sigma2=explicit)
    assert np.array_equal(a.pvalue, b.pvalue) and np.array_equal(a.deviate, b.deviate)


# ── Projection(disconnected=None) ──────────────────────────────────────────────────────────────


def _kinds(R, per=16, D=96, T=48, noise=0.05, seed=0):
    """Rows that each lie along one of ``R`` orthonormal directions, at a positive amplitude:
    ``R`` linearly independent directions, ``R - 1`` of spread about their mean."""
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((D, R)))[0].T
    rows = [Q[k] * (1.0 + 0.1 * rng.standard_normal()) + noise * rng.standard_normal(D)
            for k in range(R) for _ in range(per)]
    return np.array(rows)


@pytest.mark.parametrize("R", [1, 2, 3, 5, 8])
def test_the_origin_read_counts_the_linear_span(R):
    X = _kinds(R, seed=R)
    assert E.Projection(X, far=FAR, disconnected=None).K_signal == R


def test_the_connected_read_counts_the_spread_about_the_mean():
    """The same record about its own mean: the shared level is removed, and with it one direction.
    (The connected read itself is pinned by ``test_golden``.)"""
    X = _kinds(5, seed=5)
    assert E.Projection(X, far=FAR).K_signal == 4
    assert E.Projection(X, far=FAR, disconnected=None).K_signal == 5


def test_the_origin_read_keeps_zero_as_the_centre():
    X = _kinds(3, seed=7) + 0.0
    p = E.Projection(X, far=FAR, disconnected=None)
    assert np.all(np.asarray(p.centre) == 0.0)


def test_the_origin_read_is_deterministic_per_seed():
    X = np.random.default_rng(4).standard_normal((60, 30))
    a = E.Projection(X, far=FAR, disconnected=None, seed=3)
    b = E.Projection(X, far=FAR, disconnected=None, seed=3)
    assert a.noise_floor == b.noise_floor and a.K_signal == b.K_signal


def test_the_origin_read_holds_its_level_on_symmetric_noise():
    """Under the null it is exact for, the false alarms stay within the binomial spread of ``far``:
    the count of records claiming a mode is at most the ``1 - far`` quantile of Binomial(n, far)."""
    n = 100
    claims = sum(E.Projection(np.random.default_rng(1000 + i).standard_t(3, (64, 40)),
                              far=FAR, disconnected=None).K_signal > 0 for i in range(n))
    assert claims <= binom.ppf(1 - FAR, n, FAR)


@pytest.mark.parametrize("level", [0.0, 1.5, np.zeros(8)])
def test_a_level_to_subtract_is_refused(level):
    X = np.random.default_rng(0).standard_normal((40, 8))
    with pytest.raises(ValueError, match="takes None"):
        E.Projection(X, far=FAR, disconnected=level)
