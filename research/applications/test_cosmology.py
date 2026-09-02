"""Tests for the FRB channel-inversion stack in cosmology.py: find_dm / find_rm / derotate and the
composed unwind().  Each transform is INJECTED with a known parameter and must be recovered by
maximizing an entroptics read -- no template, no fit.  Deterministic seeds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cosmology as CO

DTM = 0.983                                                    # ms per sample
DT = DTM * 1e-3                                                # seconds


def _source(T=140, F=2048, rm=40.0, tau=2.0, psi0=0.6, amp=7.0, seed=0):
    """A ~100%-linearly-polarized, scattered, Faraday-rotated burst BEFORE dispersion (the source
    frame is I0; Q0/U0 already carry the Faraday rotation and scattering)."""
    freqs = np.linspace(800, 400, F)
    t = np.arange(T)
    lam2 = (299792458.0 / (freqs * 1e6)) ** 2
    I0 = np.zeros((T, F)); Q0 = np.zeros((T, F)); U0 = np.zeros((T, F))
    for c in range(F):
        ts = tau * (freqs[c] / 600) ** -4 / DTM
        k = np.exp(-np.arange(T) / max(ts, 0.4)); k /= k.sum()
        Ic = amp * np.convolve(np.exp(-0.5 * ((t - 45) / 1.8) ** 2), k)[:T] * ((freqs[c] / 600) ** -1.2)
        psi = psi0 + rm * lam2[c]
        I0[:, c] = Ic; Q0[:, c] = Ic * np.cos(2 * psi); U0[:, c] = Ic * np.sin(2 * psi)
    return freqs, I0, Q0, U0


def _observe(I0, Q0, U0, freqs, dm=1.5, seed=1):
    """Apply the channel: disperse every Stokes by ``dm`` and add unit noise."""
    rng = np.random.default_rng(seed)
    d = lambda X: CO.dedisperse(X, freqs, DT, -dm)             # disperse == dedisperse by -dm
    return (d(I0) + rng.standard_normal(I0.shape),
            d(Q0) + rng.standard_normal(I0.shape),
            d(U0) + rng.standard_normal(I0.shape))


def test_find_dm_recovers_dispersion():
    freqs, I0, _, _ = _source()
    I = CO.dedisperse(I0, freqs, DT, -1.5) + np.random.default_rng(0).standard_normal(I0.shape)
    dm, _, _ = CO.find_dm(I, freqs, DT, dm_lo=-3, dm_hi=3)
    assert dm == pytest.approx(1.5, abs=0.4)                   # small scattering-DM bias allowed


def test_find_rm_recovers_faraday():
    freqs, I0, Q0, U0 = _source(rm=40.0)
    on = slice(43, 48)
    rm, _, _ = CO.find_rm(Q0[on].mean(0), U0[on].mean(0), freqs, rm_lo=-400, rm_hi=400)
    assert rm == pytest.approx(40.0, abs=2.0)


def test_derotate_is_the_exact_inverse():
    freqs = np.linspace(800, 400, 512)
    rng = np.random.default_rng(2)
    Q, U = rng.standard_normal((30, 512)), rng.standard_normal((30, 512))
    Qr, Ur = CO.derotate(*CO.derotate(Q, U, freqs, 137.0), freqs, -137.0)
    assert np.allclose(Qr, Q, atol=1e-9) and np.allclose(Ur, U, atol=1e-9)


def test_unwind_recovers_the_full_channel():
    freqs, I0, Q0, U0 = _source(rm=40.0, tau=2.0)
    I, Q, U = _observe(I0, Q0, U0, freqs, dm=1.5)
    r = CO.unwind(I, freqs, DT, Q=Q, U=U, dm_search=(-3, 3), rm_search=(-400, 400),
                  patch=384, scattering_dt=DTM)
    assert r.dm == pytest.approx(1.5, abs=0.4)
    assert r.rm == pytest.approx(40.0, abs=2.0)
    assert r.tau_scatter == pytest.approx(2.0, rel=0.4)
    assert r.pol_fraction == pytest.approx(1.0, abs=0.15)
    # the source frame matches the intrinsic burst
    a = r.source - r.source.mean(); b = I0 - I0.mean()
    corr = float((a.ravel() @ b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert corr > 0.85


def test_unwind_intensity_only_no_polarization():
    """Without Q, U the stack still runs (DM + scattering + RFI-clean); rm / pol are None."""
    freqs, I0, _, _ = _source()
    I = CO.dedisperse(I0, freqs, DT, -1.5) + np.random.default_rng(4).standard_normal(I0.shape)
    r = CO.unwind(I, freqs, DT, dm_search=(-3, 3), patch=384, scattering_dt=DTM)
    assert r.rm is None and r.pol_fraction is None
    assert r.dm == pytest.approx(1.5, abs=0.4)
    assert np.isfinite(r.contrast) and r.source.shape == I.shape
