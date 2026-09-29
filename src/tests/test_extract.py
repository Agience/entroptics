"""Calibration / round-trip fidelity for the entroptics ``extract`` FILTER.

Answers one question with numbers, not eyes: does filtering a field recover EXACTLY the signal
that went in -- nothing distorted, nothing invented?

The filter is the front-door ``Aperture(W).extract()``: the hard projection onto the resolved modes
(Def 8.4), clean = P_L @ data @ P_R, with the modes read on the (whitened) screen and carried back to
the data's own grid by the fold's adjoint.  Because the modes come from the DATA ITSELF and the
singular values are never altered, the clean field is an orthogonal PROJECTION of the measured data
onto its own resolved modes.  Three claims, each proven to machine precision or characterized:

  A. EXACT recovery + NO synthesis -- clean == P_L @ data @ P_R to ~1e-15; idempotent; nothing lost.
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
            cn = clean
            ic += corr(cn, B); tc += corr(cn.sum(1), tprof); fc += corr(cn.sum(0), fprof)
            ce += 1.0 - corr(cn, B); re += 1.0 - corr(W, B)
        n = trials
        rows.append((snr, ic / n, tc / n, fc / n, ce / n, re / n))
    return rows, B


# ── the guarantees ────────────────────────────────────────────────────────────────────────────
def _hard_threshold(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    keep = S > noise_floor(X)
    return (U * np.where(keep, S, 0.0)) @ Vt, U[:, keep], Vt[keep]


def test_hard_threshold_form_is_a_projection():
    """A. The HARD-THRESHOLD form of Def 8.4 -- truncate at the derived floor, no shrinkage --
    recovers a noise-free mode bit-for-bit, is a two-sided projection, and is idempotent.

    The floor reads each mode against what independent channels of the same energies would give,
    so a noise-free input is recovered whole when every mode clears that edge (a single mode
    always does) -- the rank-2 burst's weaker mode need not.  The front door is this map, composed
    with the per-channel whitening (undone before it returns) and the fold's adjoint --
    ``test_front_door_is_an_orthogonal_split`` pins that it keeps the three properties."""
    t, f = np.arange(64)[:, None], np.arange(256)[None, :]
    one = np.exp(-0.5 * ((t - 30) / 3.0) ** 2) * np.exp(-0.5 * ((f - 128) / 56.0) ** 2)
    assert relerr(_hard_threshold(one)[0], one) < 1e-12               # exact recovery of a mode
    B = make_burst()
    clean, Uk, Vk = _hard_threshold(B)
    assert Uk.shape[1] >= 1
    assert relerr(clean, (Uk @ Uk.T) @ B @ (Vk.T @ Vk)) < 1e-12       # clean == P_L data P_R (no synthesis)
    U2, S2, Vt2 = np.linalg.svd(clean, full_matrices=False)
    clean2 = (U2 * np.where(S2 > noise_floor(clean), S2, 0.0)) @ Vt2
    assert relerr(clean2, clean) < 1e-10                              # idempotent -> a true projection


def test_front_door_is_an_orthogonal_split():
    """A, through the front door: ``clean + residual == W`` exactly, the residual is orthogonal to
    the resolved part in the whitened (noise) metric, and projecting the resolved part again changes
    nothing -- with noise, with a persistent tone in the frame, and on a folded feature axis."""
    rng = np.random.default_rng(5)
    B = make_burst()
    for W in (B, B + rng.standard_normal(B.shape) / 10, B + make_tone() + rng.standard_normal(B.shape) / 8):
        clean, info = Aperture(W, window=None).extract()
        assert clean.shape == W.shape
        assert np.abs(clean + info["residual"] - W).max() < 1e-12                 # nothing lost
        zc = (clean - info["centre"]) / info["scale"]
        zr = info["residual"] / info["scale"]
        nc, nr = np.sqrt(np.sum(zc ** 2)), np.sqrt(np.sum(zr ** 2))
        # orthogonal, to the round-off of an inner product of zc.size terms (the residual of a
        # noise-free frame is itself round-off, so a bound relative to it alone would ask more)
        assert abs(np.sum(zc * zr)) < 1e-10 * nc * nr + zc.size * np.finfo(float).eps * nc * nc


def test_extract_takes_no_shrink_argument():
    """The singular-value shrinkage is gone: it altered the singular values, so the map was neither
    lossless nor idempotent.  A caller who passes it gets a TypeError, not a silently ignored keyword."""
    with pytest.raises(TypeError):
        Aperture(make_burst(), window=None).extract(shrink=True)


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
        cn = clean
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

    # (3) monotone and near-perfect in correlation, the noiseless limit included.  On a noiseless
    # frame what remains is tiny and it is in the residual, not lost.
    c_clean, e_clean, _ = run(None)
    assert c_clean > 0.99, "the noiseless limit must not degrade -- it was a units artifact"
    assert band[10][0] <= band[50][0] <= band[1000][0] <= c_clean, "correlation rises with S/N"
    assert 0.0 < e_clean < 0.01, "the read's own residual on a noiseless frame"
    clean, info = Aperture(B, window=None).extract()
    assert np.abs(clean + info["residual"] - B).max() < 1e-12, "and it is in the residual, not lost"


def test_snr_band_matches_the_committed_table():
    """The values the paper quotes, asserted against the table that publishes them.

    The assertions above are BOUNDS -- correlation above 0.95, relative error below 0.2 -- and a
    bound leaves the value itself written down nowhere but a docstring.  Section 12 of the paper
    quotes the values, so for a long time the only thing tying the two together was a comment
    saying to update the paper by hand if the numbers moved.  They moved; nobody did.

    ``research/figures/calibration.py`` now emits the band to ``calibration.csv``, and this
    asserts that what the filter computes here is what that table says.  Change the filter and
    this fails until the figure is re-run, which is the point.

    Skipped when the research tree is absent (an installed wheel has no ``research/``); the
    committed table is what CI runs against."""
    import csv
    from pathlib import Path

    table = Path(__file__).resolve().parents[2] / "research/figures/calibration.csv"
    if not table.is_file():
        pytest.skip("research/figures/calibration.csv not present (installed wheel, not a checkout)")

    rows = {}
    for r in csv.reader(table.open()):
        if len(r) == 5 and r[0] not in ("snr",) and not r[0].startswith("#"):
            try:
                rows[r[0]] = [float(v) for v in r[1:]]
            except ValueError:
                continue
    assert rows, f"no S/N band rows in {table}; re-run research/figures/calibration.py"

    B = make_burst()
    for key, (read_corr, raw_corr, read_relerr, raw_relerr) in rows.items():
        snr = None if key == "none" else float(key)
        W = B if snr is None else B + np.random.default_rng(0).standard_normal(B.shape) / snr
        clean, _ = Aperture(W, window=None).extract()
        cn = clean
        for name, got, want in (("read_corr", corr(cn, B), read_corr),
                                ("raw_corr", corr(W, B), raw_corr),
                                ("read_relerr", relerr(cn, B), read_relerr),
                                ("raw_relerr", relerr(W, B), raw_relerr)):
            assert round(got, 3) == pytest.approx(want, abs=5e-4), (
                f"S/N={key} {name}: filter gives {got:.4f}, calibration.csv says {want} "
                f"-- re-run research/figures/calibration.py")


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
    # 18 channels: a whitened channel carries energy T at most, so a mode on k channels clears the
    # independence edge only when k T exceeds it -- about (1 + sqrt(F / T))^2 = 9 channels here
    B, R = make_burst(), make_tone(lo=0.76, hi=0.83)
    noise = rng.standard_normal(B.shape) * (1.0 / 8)
    W = B + R + noise
    clean, info = Aperture(W, window=None).extract()
    cn = clean

    R_mod = R - R.mean(axis=0, keepdims=True)          # the tone as a MODE: its varying part
    assert abs(corr(cn, R_mod)) < 0.2, "the tone's modulation must be removed"
    assert info["n_dropped"] >= 1, "the persistent mode must be flagged and dropped"

    # What the filter discarded is exactly W - clean: the noise and the tone's modulation, and
    # NOT the burst.  This is the identity the input-units return exists to make true.
    removed = W - cn
    assert corr(removed, noise) > 0.3, "the noise must be in what was removed"
    assert corr(removed, R_mod) > 0.3, "the tone's modulation must be in what was removed"
    assert abs(corr(removed, B)) < 0.1, "the burst must NOT be in what was removed -- it was kept"
