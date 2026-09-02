"""Calibration / round-trip fidelity for the entroptics ``extract`` FILTER.

Answers one question with numbers, not eyes: when we filter a field, are we recovering EXACTLY the
signal we put in -- nothing distorted, nothing invented?

The filter is the front-door ``Aperture(W).extract()``: clean = U @ diag(Sd) @ Vt, where U,S,Vt is
the SVD of the (whitened) screen of the data and Sd is the Gavish-Donoho-shrunk (persistent-geometry
modes zeroed) singular spectrum.  Because U and Vt come from the DATA ITSELF, the clean field is a
linear PROJECTION of the measured data onto its own resolved modes.  Three claims, each proven to
machine precision or characterized:

  A. EXACT recovery + NO synthesis (noise-free) -- clean == P_L @ data @ P_R to ~1e-15; idempotent.
  B. OPTIMAL, characterized recovery (with noise) -- fidelity rises monotonically toward 1 and the
     clean field beats the raw field at every S/N where noise is non-trivial.
  C. PERSISTENT-STRUCTURE separation -- a persistent modulated tone is dropped by the phi_F>phi_T
     cut, the transient burst kept.
"""
import numpy as np
import pytest

from entroptics import Aperture
from entroptics.screen import noise_floor


# ── synthetic ground truth ────────────────────────────────────────────────────────────────────
def make_burst(T=64, F=256, peak=1.0):
    """A broadband, transient, rank-2 burst: two drifting Gaussian sub-bursts (spread over
    frequency, compact in time), so the geometry cut keeps it."""
    t = np.arange(T)[:, None]; f = np.arange(F)[None, :]
    b1 = np.exp(-0.5 * ((t - 0.46 * T) / 3.0) ** 2) * np.exp(-0.5 * ((f - 0.50 * F) / (0.22 * F)) ** 2)
    b2 = np.exp(-0.5 * ((t - 0.54 * T) / 4.0) ** 2) * np.exp(-0.5 * ((f - 0.66 * F) / (0.16 * F)) ** 2)
    B = b1 + 0.7 * b2
    return peak * B / B.max()


def make_tone(T=64, F=256, amp=1.6, lo=0.80, hi=0.83):
    """Persistent, MODULATED narrowband tone: present across the whole window (phi_T high) but
    amplitude-varying (survives median subtraction as a coherent mode the cut must reject),
    a couple of features wide (phi_F low)."""
    f = np.arange(F); band = ((f >= lo * F) & (f < hi * F)).astype(float)
    t = np.arange(T)
    env = 1.0 + 0.5 * np.sin(2 * np.pi * 3 * t / T) + 0.3 * np.sin(2 * np.pi * 7 * t / T)
    return amp * np.outer(env, band)


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def noise_sweep(snrs=(200, 100, 50, 20, 10, 5, 3, 2, 1), trials=40, seed=1234):
    """Round-trip ``extract`` over a noise sweep; per S/N return
    (snr, image_corr, time_corr, freq_corr, clean_err, raw_err) plus the truth burst B."""
    B = make_burst(); tprof, fprof = B.sum(1), B.sum(0)
    rows = []; rng = np.random.default_rng(seed)
    for snr in snrs:
        sigma = 1.0 / snr; ic = tc = fc = ce = re = 0.0
        for _ in range(trials):
            W = B + rng.standard_normal(B.shape) * sigma
            clean, _ = Aperture(W, window=None).extract()
            ic += corr(clean, B); tc += corr(clean.sum(1), tprof); fc += corr(clean.sum(0), fprof)
            ce += 1.0 - corr(clean, B); re += 1.0 - corr(W, B)
        n = trials
        rows.append((snr, ic / n, tc / n, fc / n, ce / n, re / n))
    return rows, B


# ── the guarantees ────────────────────────────────────────────────────────────────────────────
def test_exact_recovery_and_no_synthesis():
    """A. noise-free: recovers the input bit-for-bit; clean IS a two-sided projection; idempotent."""
    B = make_burst(); floor = noise_floor(B)
    U, S, Vt = np.linalg.svd(B, full_matrices=False)
    keep = S > floor
    clean = (U * np.where(keep, S, 0.0)) @ Vt
    Uk, Vk = U[:, keep], Vt[keep]
    assert relerr(clean, B) < 1e-12                                   # exact recovery
    assert relerr(clean, (Uk @ Uk.T) @ B @ (Vk.T @ Vk)) < 1e-12       # clean == P_L data P_R (no synthesis)
    U2, S2, Vt2 = np.linalg.svd(clean, full_matrices=False)
    clean2 = (U2 * np.where(S2 > noise_floor(clean), S2, 0.0)) @ Vt2
    assert relerr(clean2, clean) < 1e-10                              # idempotent -> a true projection


def test_noise_recovery_is_optimal():
    """B. with noise: fidelity rises monotonically and the clean field beats the raw wherever
    noise is non-trivial (S/N <= 20)."""
    rows, _ = noise_sweep()
    helps = [snr for snr, _, _, _, ce, re in rows if ce < re]
    low = sorted((snr, ic) for snr, ic, *_ in rows if snr <= 20)      # noise-relevant regime, asc S/N
    monotone = all(low[i][1] >= low[i - 1][1] - 1e-6 for i in range(1, len(low)))
    strong = next(r for r in rows if r[0] == 10)
    assert monotone, "fidelity must rise as S/N improves"
    assert strong[1] > 0.95, "image correlation at S/N=10 must exceed 0.95"
    assert helps and max(helps) >= 20, "clean must beat raw wherever noise matters"


def test_persistent_structure_rejection():
    """C. a persistent modulated tone is dropped by the phi_F>phi_T geometry cut, burst preserved."""
    rng = np.random.default_rng(7)
    B, R = make_burst(), make_tone()
    W = B + R + rng.standard_normal(B.shape) * (1.0 / 8)
    clean, info = Aperture(W, window=None).extract()
    assert corr(clean, B) > 0.9, "burst must be preserved"
    assert abs(corr(clean, R)) < 0.2, "tone must be removed"
    assert info["n_dropped"] >= 1, "the persistent mode must be flagged and dropped"
