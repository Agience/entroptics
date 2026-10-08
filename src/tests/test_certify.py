"""The certified-interval reads: resolved_dimension_interval (certified count), the
SpectralAccumulator (ensemble pooling of the feature correlation), and the attenuation interval on a
pooled spectrum. These certify the interval reads; the Lean side (`Entroptics/Spectrum.lean`,
`resolved_count_certified`, `attenuation_weyl_certified`) proves the enclosures are sound, and these
tests exercise the same enclosures on real data."""
import numpy as np
import pytest

from entroptics.reads import (spectral_optics, resolved_dimension_interval, CertifiedCount,
                              SpectralAccumulator, concentration_band, attenuation_interval)
from conftest import build_W


# ── resolved_dimension_interval: the certified count [K_lo, K_hi] ──────────────

def test_resolved_count_encloses_point(W):
    sg = spectral_optics(W)
    band = concentration_band(W.shape[0], W.shape[1])
    k = resolved_dimension_interval(W, band=band)
    assert isinstance(k, CertifiedCount)
    assert k.resolved_lo <= sg.resolved_modes <= k.resolved_hi
    assert k.resolved_lo <= k.resolved_hi
    assert k.band == pytest.approx(band)


def test_resolved_count_zero_band_is_exact(W):
    sg = spectral_optics(W)
    k = resolved_dimension_interval(W, band=0.0, sg=sg)
    assert k.resolved_lo == k.resolved_hi == sg.resolved_modes


def test_resolved_count_band_widens_interval(W):
    sg = spectral_optics(W)
    small = resolved_dimension_interval(W, band=0.01, sg=sg)
    large = resolved_dimension_interval(W, band=5.0, sg=sg)
    assert large.resolved_lo <= small.resolved_lo
    assert small.resolved_hi <= large.resolved_hi
    assert 0 <= large.resolved_lo <= large.resolved_hi <= np.asarray(sg.eigenvalues).size


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_resolved_count_soundness_under_perturbation(seed):
    """Weyl: with band >= the actual eigenvalue perturbation, the certified interval on the read
    spectrum encloses the true resolved count (the real-data instance of `resolved_count_certified`)."""
    rng = np.random.default_rng(seed)
    X = build_W(seed)
    Xp = X + 0.05 * rng.standard_normal(X.shape)
    sg_true = spectral_optics(X)
    sg_read = spectral_optics(Xp)
    ev_t = np.sort(np.asarray(sg_true.eigenvalues, float))[::-1]
    ev_r = np.sort(np.asarray(sg_read.eigenvalues, float))[::-1]
    band = float(np.max(np.abs(ev_r - ev_t)))          # a valid Weyl band (same shape -> same edge)
    k = resolved_dimension_interval(Xp, band=band, sg=sg_read)
    assert k.resolved_lo <= sg_true.resolved_modes <= k.resolved_hi


# ── SpectralAccumulator: pooling the feature correlation ──────────────────────

@pytest.mark.parametrize("floor", ["default", "mp"])
def test_accumulator_matches_spectral_optics_single_plane(floor):
    # one plane pooled is the single-plane read, under the library default (the accumulator holds
    # the plane and draws the same shuffles) and under the closed form
    from entroptics.null_providers import mp
    X = build_W(7)
    sg = spectral_optics(X, null=None if floor == "default" else mp)
    acc = SpectralAccumulator(X.shape[1])
    acc.add(X)
    sga = acc.spectral(null=mp if floor == "mp" else None)
    assert sga.attenuation == pytest.approx(sg.attenuation, rel=1e-9, abs=1e-12)
    assert sga.noise_floor == pytest.approx(sg.noise_floor, rel=1e-9)
    assert sga.resolved_modes == sg.resolved_modes
    assert np.allclose(np.asarray(sga.eigenvalues), np.asarray(sg.eigenvalues))


def test_accumulator_merge_is_additive():
    A1 = build_W(1, T=40)
    A2 = build_W(2, T=40)                               # both F=48
    accA = SpectralAccumulator(48)
    accA.add(A1)
    accB = SpectralAccumulator(48)
    accB.add(A2)
    accA.merge(accB)
    direct = SpectralAccumulator(48)
    direct.add(A1)
    direct.add(A2)
    assert accA.T == direct.T == 80
    assert np.allclose(accA._cov, direct._cov)
    assert accA.spectral().attenuation == pytest.approx(direct.spectral().attenuation, rel=1e-9)


def test_accumulator_band_tightens_with_pooling():
    """An ensemble pool holds every plane: nothing ages between independent realisations."""
    acc = SpectralAccumulator(48)
    acc.add(build_W(0, T=40))
    b1 = acc.band()
    for s in range(1, 12):
        acc.add(build_W(s, T=40))
    b2 = acc.band()
    assert b2 < b1                                       # more pooled samples -> tighter band
    assert acc.T == 12 * 40


def test_accumulator_whiten_runs():
    acc = SpectralAccumulator(48, whiten=True)
    acc.add(build_W(7))
    sg = acc.spectral()
    assert np.isfinite(sg.attenuation) and sg.resolved_modes >= 0


def test_accumulator_feature_mismatch_raises():
    acc = SpectralAccumulator(48)
    with pytest.raises(ValueError):
        acc.add(np.zeros((10, 7)))


def test_accumulator_rejects_non_2d():
    acc = SpectralAccumulator(4)
    with pytest.raises(ValueError):
        acc.add(np.zeros((3, 4, 5)))


def test_accumulator_empty_spectral_is_zero():
    acc = SpectralAccumulator(5)
    sg = acc.spectral()
    assert sg.resolved_modes == 0
    assert acc.T == 0


# ── attenuation interval on the pooled spectrum ──────────────────────────────

def test_attenuation_interval_encloses_point_on_pooled():
    acc = SpectralAccumulator(48)
    for s in range(6):
        acc.add(build_W(s, T=40))
    sg = acc.spectral()
    band = concentration_band(acc.T, acc.F)
    ci = attenuation_interval(None, band=band, sg=sg)
    assert ci.attenuation_lo <= sg.attenuation <= ci.attenuation_hi
    assert ci.band == pytest.approx(band)


@pytest.mark.parametrize("T,N,cx", [(4096, 4, False), (256, 32, False), (32, 128, False), (256, 32, True)])
def test_concentration_band_covers_at_its_level(T, N, cx):
    """The band is a guarantee at ``far`` for every shape, real or complex: a whitened Gaussian
    record's covariance leaves it at most a ``far`` fraction of the time.  A chosen constant in
    front of ``sqrt(N/T) + N/T`` under-covers when ``N << T``, where the fluctuation of order
    ``1/sqrt(T)`` dominates the ``sqrt(N/T)`` term."""
    far = 0.05
    band = concentration_band(T, N, far=far)
    rng = np.random.default_rng(T + N + cx)
    miss = 0
    n = 200
    for _ in range(n):
        Z = rng.standard_normal((T, N))
        if cx:
            Z = (Z + 1j * rng.standard_normal((T, N))) / np.sqrt(2.0)
        e = np.linalg.eigvalsh(Z.conj().T @ Z / T)
        miss += max(abs(e[-1] - 1.0), abs(e[0] - 1.0)) > band
    assert miss / n <= far


def test_concentration_band_scales_with_the_norm_and_level():
    b = concentration_band(256, 32, far=0.05)
    assert concentration_band(256, 32, spec_norm=3.0, far=0.05) == pytest.approx(3.0 * b)
    assert concentration_band(256, 32, far=0.01) > b                # a stricter level widens it
    with pytest.raises(ValueError):
        concentration_band(256, 32, far=1.0)
