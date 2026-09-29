"""operator_vs_fft.py -- the library's operator reads beside the FFT, on the README's cases.

Every number the README quotes against the FFT comes from this script, run through the library only:
  read A    ``entroptics.experimental.operator_read`` -- the order, the depth and the modes all read
            from the record; the only input is far;
  lift      ``koopman_lift`` at one fixed delay depth ``d = 16`` (the caller's choice, the same for
            every case), its modes ``Dynamics.modes()`` ranked by power;
  FFT       standard practice, with two windows: a 16x zero-padded periodogram of the mean-removed
            record (rectangular, and Hann) with log-parabolic peak interpolation of the strongest
            local maxima, a gap filled with the record's mean.
  FFT pipeline  the same outputs the reads return -- a count at ``far``, each peak's frequency,
            decay rate and power -- by standard practice from a periodogram (``fft_pipeline``),
            Hann-windowed and unwindowed: the like-for-like comparison, in accuracy and in cost.
The Fourier views are ``ModePowers.spectrum``.

    python operator_vs_fft.py            # prints one JSON line per case, then the timings

Seeded and single-threaded, so it reproduces.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json      # noqa: E402
import time      # noqa: E402
from functools import lru_cache  # noqa: E402

import numpy as np  # noqa: E402
from scipy import fft as sfft  # noqa: E402
from scipy.stats import chi2  # noqa: E402

import entroptics as E  # noqa: E402
from entroptics.experimental import operator_read  # noqa: E402


def emit(**kw):
    print(json.dumps(kw), flush=True)


def record(rng, T, tones, sigma, gap=None):
    t = np.arange(T)
    x = sum(A * np.exp(-a * t) * np.cos(2 * np.pi * f * t + i) for i, (f, a, A) in enumerate(tones))
    x = x + sigma * rng.standard_normal(T)
    if gap is not None:
        x[gap[0]:gap[1]] = np.nan
    return x


def fft_peaks(x, K, window=None):
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    n = 16 * x.size
    w = np.ones(x.size) if window is None else window(x.size)
    P = np.abs(np.fft.rfft((x - x.mean()) * w, n)) ** 2
    lp = np.log(P + np.finfo(float).tiny)
    pk = np.flatnonzero((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:])) + 1
    pk = pk[np.argsort(-P[pk])][:K]
    out = []
    for i in pk:
        a, b, c = lp[i - 1], lp[i], lp[i + 1]
        out.append((i + 0.5 * (a - c) / (a - 2 * b + c)) / n)
    return np.array(out)


@lru_cache(maxsize=None)
def _edge_scale(dof, bins, far):
    """The noise law's quantile at ``far`` (Bonferroni over ``bins``) over its median."""
    return chi2.isf(far / bins, dof) / chi2.ppf(0.5, dof)


def fft_pipeline(W, far=0.05, pad=16, window="hann"):
    """What the reads return -- a count at ``far``, and each counted peak's frequency, decay rate
    and power -- by standard practice from the FFT, from one FFT per channel (``scipy.fft``, one
    worker) of the mean-removed record, ``pad``-times zero-padded.  The periodogram, averaged over
    the channels, is unwindowed (``window="rect"``) or Hann-windowed (``"hann"``, the periodic
    window, applied as its three-tap kernel in frequency).  The noise's mean ordinate comes from the
    median ordinate (chi-square with 2F degrees of freedom); a local maximum of the unpadded
    periodogram counts where it clears that law's quantile at ``far``, Bonferroni over its T/2
    bins; its frequency by log-parabolic interpolation at its top on the padded grid; its decay
    rate from the half-power width of its line in the unwindowed periodogram (a decay alpha gives a
    line alpha / pi wide; a taper would narrow it), the record's own 0.886-bin width taken out in
    quadrature.  A gap is filled with the channel mean.  Returns ``(f, alpha, power)``, strongest
    first."""
    X = np.asarray(W, float)
    X = X[:, None] if X.ndim == 1 else X
    X = np.where(np.isfinite(X), X, np.nanmean(X, axis=0))
    T, F = X.shape
    n = pad * T
    S = sfft.rfft(X - X.mean(axis=0), n, axis=0, workers=1)
    R = np.mean(S.real ** 2 + S.imag ** 2, axis=1)                  # unwindowed
    if window == "hann":
        Z = np.concatenate([np.conj(S[pad:0:-1]), S, np.conj(S[-2:-2 - pad:-1])])
        H = 0.5 * S - 0.25 * (Z[:-2 * pad] + Z[2 * pad:])           # x[t] (1 - cos 2 pi t / T) / 2
        P = np.mean(H.real ** 2 + H.imag ** 2, axis=1)
    else:
        P = R
    B = P[::pad]                                                    # the T/2 bins the law holds on
    edge = np.median(B) * _edge_scale(2 * F, T // 2, far)
    pk = np.flatnonzero((B[1:-1] > B[:-2]) & (B[1:-1] >= B[2:]) & (B[1:-1] > edge)) + 1
    pk = pk[np.argsort(-B[pk])] * pad
    for m, i in enumerate(pk):                                      # each counted peak's top
        while i + 1 < P.size - 1 and P[i + 1] > P[i]:               # on the padded grid
            i += 1
        while i > 1 and P[i - 1] > P[i]:
            i -= 1
        pk[m] = i
    lp = np.log(P + np.finfo(float).tiny)
    f, a = [], []
    for i in pk:
        l, c, r = lp[i - 1], lp[i], lp[i + 1]
        f.append((i + 0.5 * (l - r) / (l - 2 * c + r)) / n)
        j = i                                             # the same line's top, unwindowed
        while j + 1 < R.size and R[j + 1] > R[j]:
            j += 1
        while j > 0 and R[j - 1] > R[j]:
            j -= 1
        h = R[j] / 2
        lo = j - np.argmax(R[j::-1] <= h)                 # first bin at or below half power, left
        hi = j + np.argmax(R[j:] <= h)                    # and right
        x_lo = lo + (h - R[lo]) / (R[lo + 1] - R[lo]) if R[lo] <= h else lo
        x_hi = hi - (h - R[hi]) / (R[hi - 1] - R[hi]) if R[hi] <= h else hi
        a.append(np.pi * np.sqrt(max(((x_hi - x_lo) / n) ** 2 - (0.886 / T) ** 2, 0.0)))
    return np.array(f), np.array(a), P[pk]


def operator_modes(x, d, K):
    """The K strongest positive-frequency modes of the lifted operator: (f, alpha, power)."""
    m = E.koopman_lift(x[:, None], d).modes()
    mu, P = np.asarray(m.mu), np.asarray(m.power)
    keep = np.angle(mu) > 0
    f = np.angle(mu[keep]) / (2 * np.pi)
    a = -np.log(np.abs(mu[keep]))
    p = P[keep]
    o = np.argsort(-p)[:K]
    return f[o], a[o], p[o], m


CASES = [
    # name, T, [(f, decay, amplitude)], sigma, gap, d
    ("damped tones between bins", 1024, [(0.0402, 0.010, 1.0), (0.0926, 0.030, 0.8)], 0.02, None, 16),
    ("the same, noisier", 1024, [(0.0402, 0.010, 1.0), (0.0926, 0.030, 0.8)], 0.1, None, 16),
    ("a short record (T = 64)", 64, [(0.0402, 0.010, 1.0), (0.0926, 0.030, 0.8)], 0.02, None, 16),
    ("a 30-sample gap", 1024, [(0.0402, 0.010, 1.0), (0.0926, 0.030, 0.8)], 0.02, (500, 530), 16),
    ("close tones (0.006 apart)", 2048, [(0.040, 0.0, 1.0), (0.046, 0.0, 1.0)], 0.05, None, 16),
    ("a -50 dB tone 10 bins from a strong one", 2048,
     [(0.1, 0.0, 1.0), (0.1 + 10 / 2048, 0.0, 10 ** (-50 / 20))], 1e-4, None, 16),
    ("three tones", 2048, [(0.0402, 0.0, 1.0), (0.0926, 0.0, 0.7), (0.15, 0.0, 0.5)], 0.05, None, 16),
]


def run_cases():
    rng = np.random.default_rng(0)
    for name, T, tones, sigma, gap, d in CASES:
        x = record(rng, T, tones, sigma, gap)
        n = len(tones)
        fo, ao, po, m = operator_modes(x, d, n)
        fk = fft_peaks(x, n)
        fh = fft_peaks(x, n, np.hanning)
        fp, ap, _ = fft_pipeline(x)
        fr, ar, _ = fft_pipeline(x, window="rect")
        s = time.perf_counter()
        a = operator_read(x)
        a_ms = (time.perf_counter() - s) * 1e3
        pos = a.frequency >= 0
        fa, aa = a.frequency[pos], np.asarray(a.modes.alpha)[pos]
        _, res = a.split(x)
        rows = []
        for (ft, at, At) in tones:
            i = int(np.argmin(np.abs(fa - ft))) if fa.size else None
            j = int(np.argmin(np.abs(fo - ft))) if fo.size else None
            k = int(np.argmin(np.abs(fk - ft))) if fk.size else None
            h = int(np.argmin(np.abs(fh - ft))) if fh.size else None
            q = int(np.argmin(np.abs(fp - ft))) if fp.size else None
            u = int(np.argmin(np.abs(fr - ft))) if fr.size else None
            rows.append(dict(
                f_true=ft, decay_true=at,
                a_f_err=None if i is None else float(abs(fa[i] - ft)),
                a_decay=None if i is None else float(aa[i]),
                lift_f_err=None if j is None else float(abs(fo[j] - ft)),
                lift_decay=None if j is None else float(ao[j]),
                fft_f_err=None if k is None else float(abs(fk[k] - ft)),
                hann_f_err=None if h is None else float(abs(fh[h] - ft)),
                pipe_f_err=None if q is None else float(abs(fp[q] - ft)),
                pipe_decay=None if q is None else float(ap[q]),
                rect_f_err=None if u is None else float(abs(fr[u] - ft)),
                rect_decay=None if u is None else float(ar[u])))
        emit(case=name, T=T, sigma=sigma, lift_d=d, fft_bin=1.0 / T, a_K=a.K, a_depth=a.depth,
             pipe_K=int(fp.size), rect_K=int(fr.size),
             a_ms=a_ms, a_residual_over_noise=float(np.nanmean(res ** 2) / sigma ** 2), tones=rows)


def run_views():
    """The Fourier view from the operator, against the periodogram, on the -50 dB case: what each
    reads at the weak tone's frequency, relative to the strong tone's peak."""
    rng = np.random.default_rng(1)
    T = 2048
    fs, fw = 0.1, 0.1 + 10 / 2048
    x = record(rng, T, [(fs, 0.0, 1.0), (fw, 0.0, 10 ** (-50 / 20))], 1e-4)
    m = operator_read(x).modes
    S = m.spectrum(np.array([fs, fw, 0.5 * (fs + fw)]))
    out = dict(view="-50 dB tone: level at the weak tone and midway, relative to the strong tone",
               read_a_weak_dB=float(10 * np.log10(S[1] / S[0])),
               read_a_between_dB=float(10 * np.log10(S[2] / S[0])))
    for name, w in (("fft", np.ones(T)), ("hann", np.hanning(T))):
        P = np.abs(np.fft.rfft((x - x.mean()) * w, 16 * T)) ** 2
        Pat = lambda f: float(P[int(round(f * 16 * T))])                      # noqa: E731
        out[name + "_weak_dB"] = float(10 * np.log10(Pat(fw) / Pat(fs)))
        out[name + "_between_dB"] = float(10 * np.log10(Pat(0.5 * (fs + fw)) / Pat(fs)))
    emit(**out)


def run_round_trip():
    """The write path on a lifted record: the residual's energy per cell against the noise's."""
    rng = np.random.default_rng(2)
    T, sigma, d = 1024, 0.05, 16
    x = record(rng, T, [(0.0402, 0.002, 1.0), (0.0926, 0.0, 0.6)], sigma)
    Z = E.delay_embed(x[:, None], d)
    b = E.Aperture(Z).basis()
    resolved, residual = b.split(Z)
    c = b.certify(Z)
    emit(round_trip="two tones, lifted d=16", K=b.K,
         residual_over_noise=float(np.mean(residual ** 2) / sigma ** 2),
         lossless=c.lossless, inverse=c.inverse, idempotent=c.idempotent, defect=c.defect)


def run_null_and_ar():
    """Read A on noise: the claim rate on white records (far = 0.05) and the decay of AR(1)."""
    rng = np.random.default_rng(4)
    claims = [operator_read(rng.standard_normal((2048, 16))).K > 0 for _ in range(200)]
    emit(null="white noise, T = 2048, F = 16, 200 draws", claim_rate=float(np.mean(claims)), far=0.05)
    for rho in (0.9, 0.5):
        x = np.zeros((2048, 16))
        e = rng.standard_normal((2048, 16))
        for t in range(1, 2048):
            x[t] = rho * x[t - 1] + e[t]
        a = operator_read(x)
        emit(ar=rho, true_decay=float(-np.log(rho)), K=a.K, decay=[float(v) for v in np.asarray(a.modes.alpha)[:2]])


def run_timings():
    """Seconds per call, best of five, one thread: the batch operator read and the FFT."""
    rng = np.random.default_rng(3)
    for T in (1024, 4096, 16384):
        for F in (1, 16):
            W = rng.standard_normal((T, F))
            best_op = best_fft = np.inf
            for _ in range(5):
                s = time.perf_counter()
                E.Aperture(W).dynamics().modes()
                best_op = min(best_op, time.perf_counter() - s)
                s = time.perf_counter()
                np.fft.rfft(W, axis=0)
                best_fft = min(best_fft, time.perf_counter() - s)
            emit(timing="dynamics().modes() vs rfft", T=T, F=F, operator_ms=best_op * 1e3,
                 fft_ms=best_fft * 1e3)
            if F == 16:
                best_opt = np.inf
                for _ in range(3):
                    s = time.perf_counter()
                    E.Aperture(W).optics()
                    best_opt = min(best_opt, time.perf_counter() - s)
                emit(timing="Aperture(W).optics()", T=T, F=F, ms=best_opt * 1e3)
        x = rng.standard_normal(T)
        best = np.inf
        for _ in range(5):
            s = time.perf_counter()
            E.koopman_lift(x[:, None], 16).modes()
            best = min(best, time.perf_counter() - s)
        emit(timing="koopman_lift(d=16).modes()", T=T, operator_ms=best * 1e3)
        t = np.arange(T)
        x = np.cos(2 * np.pi * 0.0402 * t) + 0.6 * np.cos(2 * np.pi * 0.0926 * t + 1) + 0.05 * rng.standard_normal(T)
        best = np.inf
        for _ in range(3):
            s = time.perf_counter()
            a = operator_read(x)
            best = min(best, time.perf_counter() - s)
        emit(timing="operator_read (two persistent tones)", T=T, operator_ms=best * 1e3, depth=a.depth)


def _best(fn, reps):
    best, out = np.inf, None
    for _ in range(reps):
        s = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - s)
    return best * 1e3, out


def _counted(D):
    return D.resolved(), D.modes()


def run_like_for_like():
    """Milliseconds per call, best of five (three from T = 2^18), one thread, each method returning
    a count at far = 0.05 and each component's frequency, decay rate and power; the bare FFT, which
    returns a spectrum only, for scale.  The record: two persistent tones (0.0402 and 0.0926, the
    second at 0.6) in white noise 0.05, each channel at its own phase.  The whole sweep runs three
    times over the same records (``rep``), so the spread between runs is in the output."""
    for rep, F, T in ((r, F, T) for r in range(3) for F in (1, 16)
                      for T in (1024, 4096, 16384, 65536, 262144, 1048576)):
        if F == 1 and T == 1024:
            rng = np.random.default_rng(5)
        reps = 5 if T <= 65536 else 3
        t = np.arange(T)[:, None]
        ph = rng.uniform(0, 2 * np.pi, (2, F))
        X = (np.cos(2 * np.pi * 0.0402 * t + ph[0]) + 0.6 * np.cos(2 * np.pi * 0.0926 * t + ph[1])
             + 0.05 * rng.standard_normal((T, F)))
        row = dict(timing="like for like", rep=rep, T=T, F=F)
        row["fft_ms"], _ = _best(lambda: sfft.rfft(X, axis=0, workers=1), reps)
        row["pipeline_ms"], (fq, _, _) = _best(lambda: fft_pipeline(X, pad=1), reps)
        row["pipeline_K"] = int(fq.size)
        if T * F <= 2 ** 22:
            row["pipeline_pad16_ms"], (fq, _, _) = _best(lambda: fft_pipeline(X), reps)
            row["pipeline_pad16_K"] = int(fq.size)
        if F == 1:
            row["fixed_depth_ms"], (k, _) = _best(lambda: _counted(E.koopman_lift(X, 16)), reps)
            row["fixed_depth_K"] = int(k)
            if T <= 65536:
                row["operator_read_ms"], a = _best(lambda: operator_read(X[:, 0]), 1 if T > 16384 else 3)
                row["operator_read_K"], row["operator_read_depth"] = int(a.K), int(a.depth)
        else:
            row["streaming_ms"], (k, _) = _best(lambda: _counted(E.Aperture(X).dynamics()), reps)
            row["streaming_K"] = int(k)
        emit(**row)


if __name__ == "__main__":
    import hashlib
    from pathlib import Path
    h = hashlib.sha256()
    for p in sorted(Path(E.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))   # git's form: a checkout reproduces it
    emit(file=E.__file__, source_sha256=h.hexdigest())    # installed metadata can be stale
    run_cases()
    run_views()
    run_round_trip()
    run_null_and_ar()
    run_timings()
    run_like_for_like()
