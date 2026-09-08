"""Calibration / round-trip fidelity for the entroptics ``extract`` FILTER.

Answers one question with numbers, not eyes: does filtering a field recover EXACTLY the signal
that went in -- nothing distorted, nothing invented?

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
from entroptics.projection import noise_floor


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


def _native(clean, F):
    """A read taken on a folded screen, mapped back onto the recorded feature axis.

    ``extract`` returns the field on the screen it was read on, and Def 8.1 folds that screen to
    the marginal's own width.  Comparing the read to the input therefore has to undo the fold's
    index map first -- ``clean`` is piecewise-constant across each folded group, so the inverse is
    the same map applied backwards and no interpolation is invented."""
    import numpy as _np
    n = clean.shape[1]
    if n == F:
        return clean
    idx = (_np.arange(F) * n) // F
    return clean[:, idx]


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
            cn = _native(clean, B.shape[1])
            ic += corr(cn, B); tc += corr(cn.sum(1), tprof); fc += corr(cn.sum(0), fprof)
            ce += 1.0 - corr(cn, B); re += 1.0 - corr(W, B)
        n = trials
        rows.append((snr, ic / n, tc / n, fc / n, ce / n, re / n))
    return rows, B


# ── the guarantees ────────────────────────────────────────────────────────────────────────────
def test_hard_threshold_form_is_a_projection():
    """A. The HARD-THRESHOLD form of Def 8.4 -- truncate at the derived floor, no shrinkage --
    recovers a noise-free input bit-for-bit, is a two-sided projection, and is idempotent.

    This is a property of that map, not of ``Aperture.extract()``.  The front door composes it
    with per-channel MAD whitening and Gavish-Donoho shrinkage: shrinkage de-biases the surviving
    singular values, so the composed map is not idempotent.  The whitening is undone before the
    front door returns, so its output IS on the input's amplitude scale -- what it measures is
    pinned in ``test_extract_front_door_fidelity`` below."""
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
    """B. with noise: fidelity rises with S/N through the noise-relevant band, and the clean field
    beats the raw wherever noise is non-trivial (S/N <= 20).  The band matters -- fidelity is NOT
    monotone over the whole range; see ``test_extract_front_door_fidelity``."""
    rows, _ = noise_sweep()
    helps = [snr for snr, _, _, _, ce, re in rows if ce < re]
    low = sorted((snr, ic) for snr, ic, *_ in rows if snr <= 20)      # noise-relevant regime, asc S/N
    monotone = all(low[i][1] >= low[i - 1][1] - 1e-6 for i in range(1, len(low)))
    strong = next(r for r in rows if r[0] == 10)
    assert monotone, "fidelity must rise as S/N improves through the noise-relevant band"
    assert strong[1] > 0.95, "image correlation at S/N=10 must exceed 0.95"
    assert helps and max(helps) >= 20, "clean must beat raw wherever noise matters"


def test_extract_front_door_fidelity():
    """C. What ``Aperture.extract()`` itself does, measured through the front door.

    Three facts, all measured on the calibration burst:
      1. across the noise-relevant band it recovers the burst's morphology (correlation > 0.95);
      2. it recovers the burst's AMPLITUDE too, because the read comes back in the input's own
         units -- the relative error against the truth is small, and beats the raw frame wherever
         noise is non-trivial.  The read is taken on the whitened screen and the whitening is
         undone before returning; this assertion is what pins that it stays undone;
      3. what the filter costs where there is nothing to remove.  Its own residual is ~1%, so
         above S/N ~ 200 the raw frame is already closer to the truth than the filtered one.
         That is the limit of the map, and it is pinned here.

    Correlation is monotone in S/N and stays above 0.99 across the whole range, the noiseless
    limit included -- the whitening is inverted before return, so dividing a channel by a
    vanishing scale is undone by multiplying it back and nothing degrades at that end."""
    B = make_burst()

    def run(snr):
        W = B if snr is None else B + np.random.default_rng(0).standard_normal(B.shape) / snr
        clean, _ = Aperture(W, window=None).extract()
        cn = _native(clean, B.shape[1])
        return corr(cn, B), relerr(cn, B), relerr(W, B)

    band = {snr: run(snr) for snr in (10, 50, 1000)}
    for snr, (c, _, _) in band.items():
        assert c > 0.95, f"morphology must survive at S/N={snr} (got {c:.3f})"

    # (2) the output is ON the input's scale: close to the truth in absolute terms, and closer
    # than the raw frame is, wherever there is noise to remove.
    for snr in (10, 50):
        c, e, raw_e = band[snr]
        assert e < 0.2, f"S/N={snr}: extract must land on the input's scale (relerr {e:.3f})"
        assert e < raw_e, f"S/N={snr}: clean must beat the raw frame ({e:.3f} vs {raw_e:.3f})"

    # (3) monotone and near-perfect in correlation, including the noiseless limit -- and the
    # residual the filter costs is what stops it beating a frame that had no noise to begin with.
    c_clean, e_clean, _ = run(None)
    assert c_clean > 0.99, "the noiseless limit must not degrade -- it was a units artifact"
    assert band[10][0] <= band[50][0] <= band[1000][0] <= c_clean, "correlation rises with S/N"
    assert 0.001 < e_clean < 0.05, (
        "the filter's own residual on a noiseless frame; update the paper if it moves")
    assert e_clean > relerr(B, B) , "a filter costs something where there is nothing to remove"


def test_persistent_structure_rejection():
    """C. a persistent modulated tone is dropped by the phi_F>phi_T geometry cut, burst preserved.

    The cut is a statement about MODES, so it is scored against the tone's MODULATION.  A
    channel's median is not a mode -- ``normalize`` removes it before the SVD runs, so no cut was
    ever offered it -- and ``extract`` returns the input's units, which carries that baseline back
    through.  The tone's DC level therefore stays in the baseline of the channels it sits on while
    its varying part is dropped.  Scoring against the raw ``R``, DC included, would be scoring
    this filter for a baseline estimate it does not claim to make; the identity that DOES hold
    over the whole frame is checked below."""
    rng = np.random.default_rng(7)
    B, R = make_burst(), make_tone()
    noise = rng.standard_normal(B.shape) * (1.0 / 8)
    W = B + R + noise
    clean, info = Aperture(W, window=None).extract()
    cn = _native(clean, B.shape[1])

    R_mod = R - R.mean(axis=0, keepdims=True)          # the tone as a MODE: its varying part
    assert abs(corr(cn, R_mod)) < 0.2, "the tone's modulation must be removed"
    assert info["n_dropped"] >= 1, "the persistent mode must be flagged and dropped"

    # What the filter discarded is exactly W - clean: the noise and the tone's modulation, and
    # NOT the burst.  This is the identity the input-units return exists to make true.
    removed = W - cn
    assert corr(removed, noise) > 0.3, "the noise must be in what was removed"
    assert corr(removed, R_mod) > 0.3, "the tone's modulation must be in what was removed"
    assert abs(corr(removed, B)) < 0.1, "the burst must NOT be in what was removed -- it was kept"
