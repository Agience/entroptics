"""operator_read.py -- the Entroptics operator read (research; accepted 2026-09-28).

Record = operator + residual. The operator is the record's modes (poles mu_k, powers P_k); the residual is
what they do not account for, so the round trip is lossless by construction. The Fourier view is a function
of the operator; any FFT belongs after the read, on the residual.

Inputs: the record W (T, F) and far. Nothing else.
  order K   the count of past-future canonical correlations above the Tracy-Widom edge (Johnstone 2008,
            Jacobi ensemble; exact under the i.i.d. null on non-overlapping windows), at every zoom level
            (block means at b = 1, 2, 4, ...) and every dyadic window depth, each look paid for; K is the
            largest count;
  depth d   the window span (depth x block) at the coarsest level the count still holds: the objects'
            own coherence, capped where the pencil is solvable;
  modes     the matrix pencil on the lags gamma(1..2d) (lags >= 1: white noise is at lag 0), truncated to
            K; powers from the Vandermonde fit to the lags;
  view      S(f) = sum_k P_k (1 - |mu_k|^2) / |1 - mu_k e^{-2 pi i f}|^2, at any resolution, no leakage.
The FFT appears below only as the reference it is compared against.

    python operator_read.py
"""
import sys
import time

import numpy as np

from entroptics.null_providers import tw1_quantile

FAR = 0.05


# -- the count: canonical correlations against the Jacobi-ensemble Tracy-Widom edge --------------------
def halve(X):
    h = X.shape[0] // 2
    return (X[0:2 * h:2] + X[1:2 * h:2]) / 2.0


def deepest(Tj, F):
    """The deepest window m the level supports: error dof N - F - m >= m + 1, N = F floor(Tj / 2m)."""
    best = None
    for m in range(1, Tj // 2 + 1):
        if F * (Tj // (2 * m)) - F - m >= m + 1:
            best = m
    return best


def edge(p, me, nh, q):
    """Johnstone (2008): logit of the largest root of (A + B)^-1 B, A ~ W_p(me), B ~ W_p(nh), ~ mu + sigma TW1."""
    s = me + nh - 1
    g = 2 * np.arcsin(np.sqrt((min(p, nh) - 0.5) / s))
    ph = 2 * np.arcsin(np.sqrt((max(p, nh) - 0.5) / s))
    mu = 2 * np.log(np.tan((ph + g) / 2))
    sig = (16.0 / s ** 2 / (np.sin(ph + g) ** 2 * np.sin(ph) * np.sin(g))) ** (1 / 3)
    return 1 / (1 + np.exp(-(mu + sig * q)))


def count(Y, m, q):
    """Canonical correlations of each window's past m values against its next m, above the edge."""
    Tj, F = Y.shape
    G = Tj // (2 * m)
    W = Y[:G * 2 * m].reshape(G, 2 * m, F).transpose(2, 0, 1)            # (F, G, 2m)
    ok = np.all(np.isfinite(W), axis=2)                                   # windows with no missing sample
    rows_p, rows_f = [], []
    for f in range(F):
        w = W[f][ok[f]]
        if w.shape[0]:
            w = w - w.mean(axis=0)
            rows_p.append(w[:, :m])
            rows_f.append(w[:, m:])
    if not rows_p:
        return 0
    Xp, Xf = np.concatenate(rows_p), np.concatenate(rows_f)
    me = Xp.shape[0] - len(rows_p) - m
    if me < m + 1:
        return 0
    Lx, Ly = np.linalg.cholesky(Xp.T @ Xp), np.linalg.cholesky(Xf.T @ Xf)
    M = np.linalg.solve(Ly, np.linalg.solve(Lx, Xp.T @ Xf).T).T
    rho2 = np.linalg.svd(M, compute_uv=False) ** 2
    return int(np.sum(rho2 > edge(m, me, m, q)))


def counts(X):
    """Per zoom level b: the largest count over the dyadic depths (every look paid for) and its depth."""
    Y, b, out = X, 1, []
    while True:
        m_max = deepest(Y.shape[0], Y.shape[1])
        if m_max is None or m_max < 2:
            break
        depths = [2 ** i for i in range(1, int(np.log2(m_max)) + 1)]
        q = tw1_quantile(1 - (1 - FAR) ** (1.0 / len(depths)))
        ks = [count(Y, m, q) for m in depths]
        K = max(ks)
        if K == 0:
            break
        out.append((b, depths[int(np.argmax(ks))], K))
        Y, b = halve(Y), b * 2
    return out


# -- the operator ---------------------------------------------------------------------------------------
def lags(X, n):
    """gamma(1..n), pooled over channels, over the observed pairs only."""
    out = []
    for k in range(1, n + 1):
        p = X[k:] * X[:-k]
        ok = np.isfinite(p)
        out.append(float(np.sum(np.where(ok, p, 0.0))) / max(int(ok.sum()), 1))
    return np.array(out)


def operator(W):
    """The modes (f, a, power) of the record, with the order K and the depth d used."""
    X = np.asarray(W, float)
    X = X - np.nanmean(X, axis=0)
    T = X.shape[0]
    lv = counts(X)
    if not lv:
        return [], 0, 0
    K = max(k for _, _, k in lv)
    b, m, _ = max(lv)
    d = max(K + 1, min(m * b, (T - 2) // 4))
    g = lags(X, 2 * d + 1)
    idx = np.arange(d)[:, None] + np.arange(d)[None, :]
    U, s, Vh = np.linalg.svd(g[idx])
    mu = np.linalg.eigvals((U[:, :K].T @ g[idx + 1] @ Vh[:K].T) / s[None, :K])
    V = mu[None, :] ** np.arange(1, 2 * d + 2)[:, None]
    c = np.linalg.lstsq(V, g.astype(complex), rcond=None)[0]
    modes = [(abs(np.angle(u)) / (2 * np.pi), -np.log(abs(u)) if abs(u) > 0 else np.inf, abs(ci))
             for u, ci in zip(mu, c) if np.angle(u) >= -1e-12]
    return modes, K, d


def view(modes, f):
    """The Fourier view as a function of the operator: each mode is a line of width a at frequency f_k."""
    S = np.zeros_like(f, dtype=float)
    for fk, a, p in modes:
        for u in (np.exp(-a + 2j * np.pi * fk), np.exp(-a - 2j * np.pi * fk)):
            S += p * (1 - abs(u) ** 2) / np.abs(1 - u * np.exp(-2j * np.pi * f)) ** 2
    return S


def residual(x, modes):
    """Record minus the rendered modes (coordinates by projection onto each mode's e^{-a t} cos / sin)."""
    t = np.arange(x.size)
    cols = [np.ones(x.size)]
    for fk, a, _ in modes:
        env = np.exp(-max(a, 0.0) * t)
        cols += [env * np.cos(2 * np.pi * fk * t), env * np.sin(2 * np.pi * fk * t)]
    B = np.stack(cols, axis=1)
    return x - B @ np.linalg.lstsq(B, x, rcond=None)[0]


# -- evidence against the FFT (the reference) ------------------------------------------------------------
def fft_peak(x, ft):
    T = x.size
    P = np.abs(np.fft.rfft(x - x.mean())) ** 2
    k0 = int(round(ft * T))
    lo, hi = max(1, k0 - 3), min(P.size - 2, k0 + 3)
    k = lo + int(np.argmax(P[lo:hi + 1]))
    y0, y1, y2 = (np.log(P[k + i] + 1e-300) for i in (-1, 0, 1))
    den = y0 - 2 * y1 + y2
    return (k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)) / T


def ar(rng, T, F, rho):
    x = np.zeros((T, F))
    e = rng.standard_normal((T, F))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    recs = [
        ("two tones (one decaying)", 2048, [(0.0402, 0.002, 1.0), (0.0926, 0.0, 0.6)], 0.05),
        ("close tones, df = 0.006", 2048, [(0.040, 0.0, 1.0), (0.046, 0.0, 1.0)], 0.05),
        ("strong + weak (-50 dB, 10 bins)", 2048, [(0.1, 0.0, 1.0), (0.1 + 10 / 2048, 0.0, 0.00316)], 1e-4),
        ("damped pair", 2048, [(0.0402, 0.010, 1.0), (0.0926, 0.030, 0.8)], 0.02),
        ("three tones", 2048, [(0.0402, 0.0, 1.0), (0.0926, 0.0, 0.7), (0.15, 0.0, 0.5)], 0.05),
        ("two tones, T = 8192", 8192, [(0.0402, 0.0005, 1.0), (0.0926, 0.0, 0.6)], 0.05),
    ]
    for name, T, tones, sig in recs:
        t = np.arange(T)
        x = sum(A * np.exp(-a * t) * np.cos(2 * np.pi * f * t + i) for i, (f, a, A) in enumerate(tones))
        x = x + sig * rng.standard_normal(T)
        s0 = time.perf_counter()
        modes, K, d = operator(x[:, None])
        ms = (time.perf_counter() - s0) * 1e3
        r = residual(x, modes)
        print(f"=== {name}: K={K} d={d}, {len(modes)} modes, {ms:.1f} ms, residual / noise energy "
              f"{np.sum(r ** 2) / (sig ** 2 * T):.2f}", flush=True)
        for ft, at, _ in tones:
            j = int(np.argmin([abs(md[0] - ft) for md in modes])) if modes else None
            op = "none" if j is None else f"f err {abs(modes[j][0] - ft):.1e}, a {modes[j][1]:.4f}"
            print(f"  true f {ft:.5f}, a {at:.4f} | operator {op} | FFT f err {abs(fft_peak(x, ft) - ft):.1e}, no decay",
                  flush=True)
    for rho in (0.9, 0.5):
        modes, K, d = operator(ar(rng, 2048, 16, rho))
        print(f"=== AR({rho}), F = 16: a {[round(a, 4) for _, a, _ in modes]} (true {-np.log(rho):.4f})", flush=True)
    claims = np.mean([operator(rng.standard_normal((2048, 16)))[1] > 0 for _ in range(200)])
    print(f"=== white noise, F = 16, 200 draws: object claimed {claims:.3f} (far {FAR})", flush=True)
    for T in (1024, 4096, 16384):
        t = np.arange(T)
        x = (np.cos(2 * np.pi * 0.0402 * t) + 0.6 * np.cos(2 * np.pi * 0.0926 * t + 1) + 0.05 * rng.standard_normal(T))[:, None]
        s0 = time.perf_counter(); _, _, d = operator(x); to = time.perf_counter() - s0
        s0 = time.perf_counter(); np.fft.rfft(x[:, 0]); tf = time.perf_counter() - s0
        print(f"T = {T}: operator {to * 1e3:.1f} ms (d = {d}), FFT {tf * 1e3:.2f} ms", flush=True)
    sys.exit(0)
