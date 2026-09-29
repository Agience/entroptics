"""experimental.py -- reads that are not yet part of the stable surface.

EXPERIMENTAL.  Names, signatures and returned numbers here may change in any release, and nothing
is re-exported at the top level: import it as ``from entroptics.experimental import operator_read``.

The operator read
-----------------
A record is an operator plus an exact residual.  The operator is the record's modes -- poles
``mu_k`` and powers ``P_k`` -- and the residual is what they do not account for, so the round trip
is lossless by construction.  The Fourier view is a function of the operator
(:meth:`dynamics.ModePowers.spectrum`): no bins, no leakage, and each line's width is its decay.
Any FFT belongs after the read, on the residual.

Inputs: the record and ``far``.  Nothing else.

  order K   At every zoom level (block means at ``b = 1, 2, 4, ...``) and every dyadic window depth
            ``m``, the record is cut into non-overlapping windows of ``2m`` samples, and the
            canonical correlations of each window's past ``m`` values against its next ``m`` are
            counted above the Tracy-Widom edge of the Jacobi ensemble (Johnstone 2008), exact under
            the i.i.d. null because the windows do not overlap.  The depths at one level are paid
            for together (``far_D = 1 - (1 - far)^(1/D)``); a level is read only if the one before
            it counted, so the zoom costs nothing more.  ``K`` is the largest count.
  depth d   The window span ``m b`` at the coarsest level the count still holds: the objects' own
            coherence.  It is capped so every lag used lies in the first half of the record, where
            each lag is averaged over at least as many pairs as it spans, and it is at least
            ``K + 1``, so the pencil has room for ``K`` modes.
  modes     The matrix pencil on the lags ``gamma(1 .. 2d)`` -- lags >= 1 only, because white noise
            lives at lag 0 -- truncated to ``K``; the powers are the magnitudes of the Vandermonde
            fit of the modes to the lags.

Measured against the FFT, the benchmark ``research/benchmarks/operator_vs_fft.py`` holds the cases.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import environment as _env
from .dynamics import ModePowers
from .null_providers import tw1_quantile

__all__ = ["operator_read", "OperatorRead"]


def _round_off_rank(s: np.ndarray, shape) -> int:
    # the round-off rank of an SVD: numpy.linalg.matrix_rank's default tolerance
    return int((s > s.max(initial=0.0) * max(shape) * np.finfo(float).eps).sum())


def _span(X: np.ndarray) -> np.ndarray:
    """An orthonormal basis of X's column space, at its round-off rank."""
    if not X.size:
        return X
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :_round_off_rank(s, X.shape)]


def _halve(X: np.ndarray) -> np.ndarray:
    """One zoom level coarser: the means of consecutive pairs."""
    h = X.shape[0] // 2
    return (X[0:2 * h:2] + X[1:2 * h:2]) / 2.0


def _deepest(T: int, F: int):
    """The deepest window ``m`` a level supports: ``N - F - m >= m + 1`` error degrees of freedom,
    with ``N = F floor(T / 2m)`` windows.  None if not even ``m = 1`` fits."""
    best = None
    for m in range(1, T // 2 + 1):
        if F * (T // (2 * m)) - F - m >= m + 1:
            best = m
    return best


def _edge(p: int, me: int, nh: int, q: float) -> float:
    """Johnstone (2008): the largest root of ``(A + B)^-1 B``, ``A ~ W_p(me)``, ``B ~ W_p(nh)``, has
    logit ``~ mu + sigma TW1``; this is its ``q`` quantile on the root's own scale."""
    s = me + nh - 1
    g = 2 * np.arcsin(np.sqrt((min(p, nh) - 0.5) / s))
    ph = 2 * np.arcsin(np.sqrt((max(p, nh) - 0.5) / s))
    mu = 2 * np.log(np.tan((ph + g) / 2))
    sig = (16.0 / s ** 2 / (np.sin(ph + g) ** 2 * np.sin(ph) * np.sin(g))) ** (1 / 3)
    return float(1 / (1 + np.exp(-(mu + sig * q))))


def _count(Y: np.ndarray, m: int, q: float) -> int:
    """How many canonical correlations of each window's past ``m`` against its next ``m`` stand
    above the edge.  A window with a missing sample is left out; each channel's windows are centred
    on their own mean (one degree of freedom per channel)."""
    T, F = Y.shape
    G = T // (2 * m)
    Wn = Y[:G * 2 * m].reshape(G, 2 * m, F).transpose(2, 0, 1)            # (F, G, 2m)
    ok = np.all(np.isfinite(Wn), axis=2)
    past, fut = [], []
    for f in range(F):
        w = Wn[f][ok[f]]
        if w.shape[0]:
            w = w - w.mean(axis=0)
            past.append(w[:, :m])
            fut.append(w[:, m:])
    if not past:
        return 0
    Xp, Xf = np.concatenate(past), np.concatenate(fut)
    me = Xp.shape[0] - len(past) - m
    if me < m + 1:
        return 0
    Up, Uf = _span(Xp), _span(Xf)
    if not (Up.shape[1] and Uf.shape[1]):
        return 0
    rho2 = np.linalg.svd(Up.T @ Uf, compute_uv=False) ** 2
    return int(np.sum(rho2 > _edge(m, me, m, q)))


def _levels(X: np.ndarray, far: float) -> tuple:
    """Per zoom level ``b``: ``(b, m, K)``, the largest count over the dyadic depths and its depth,
    for every level from the finest until one counts nothing."""
    Y, b, out = X, 1, []
    while True:
        m_max = _deepest(Y.shape[0], Y.shape[1])
        if m_max is None or m_max < 2:
            break
        depths = [2 ** i for i in range(1, int(np.log2(m_max)) + 1)]
        q = tw1_quantile(1 - (1 - far) ** (1.0 / len(depths)))
        ks = [_count(Y, m, q) for m in depths]
        K = max(ks)
        if K == 0:
            break
        out.append((b, depths[int(np.argmax(ks))], K))
        Y, b = _halve(Y), b * 2
    return tuple(out)


def _lags(X: np.ndarray, n: int) -> np.ndarray:
    """``gamma(1 .. n)``, pooled over channels, each the mean over the pairs both measured."""
    out = np.empty(n)
    for k in range(1, n + 1):
        p = X[k:] * X[:-k]
        ok = np.isfinite(p)
        out[k - 1] = float(np.sum(np.where(ok, p, 0.0))) / max(int(ok.sum()), 1)
    return out


def _signals(mu: np.ndarray, T: int) -> np.ndarray:
    """The record-length signals the modes span: a constant, then ``Re mu^t`` for each pole with
    ``Im mu >= 0`` and ``Im mu^t`` for each with ``Im mu > 0``.

    The poles are read from the lags of a stationary record, and a stationary autocovariance admits
    no growing mode: a pole the pencil places outside the unit circle is estimation error, so its
    signal is taken at the nearest admissible pole, on the circle (a persistent line)."""
    t = np.arange(T)
    cols = [np.ones(T)]
    for u in mu[np.imag(mu) >= 0]:
        r = abs(u)
        if r == 0.0:
            cols.append((t == 0).astype(float))
            continue
        env = np.exp(t * min(np.log(r), 0.0))
        ph = np.angle(u) * t
        cols.append(env * np.cos(ph))
        if np.imag(u) > 0:
            cols.append(env * np.sin(ph))
    return np.stack(cols, axis=1)


@dataclass(frozen=True)
class OperatorRead:
    """The operator read of a record (EXPERIMENTAL).  ``modes`` holds every pole -- both members
    of a conjugate pair -- largest power first, so ``modes.spectrum(f)`` is the Fourier view and
    ``C(tau) ~ Re sum_k P_k mu_k^tau`` the decay the lags carry."""
    modes:  ModePowers
    K:      int                      # the order counted above the edge
    depth:  int                      # d: the lag window the pencil read
    levels: tuple = field(default=())  # (block, depth, count) at each zoom level that counted

    @property
    def frequency(self) -> np.ndarray:
        """Each mode's frequency, cycles per step (``beta / 2 pi``), in ``modes`` order."""
        return np.asarray(self.modes.beta) / (2 * np.pi)

    def spectrum(self, f) -> np.ndarray:
        """The Fourier view at frequencies ``f``: :meth:`dynamics.ModePowers.spectrum`."""
        return self.modes.spectrum(f)

    def split(self, W, mask=None):
        """``(resolved, residual)`` of a record ``W`` ``(T,)`` or ``(T, F)``: each channel projected
        orthogonally, over its measured samples, onto a constant and the modes' own signals, and
        what that leaves.  ``resolved + residual == W`` wherever ``W`` was measured; both are NaN
        where it was not."""
        W = np.asarray(_env.to_numpy(W), float)
        one = W.ndim == 1
        X = W[:, None] if one else W
        if mask is not None:
            X = np.where(np.asarray(_env.to_numpy(mask), bool).reshape(X.shape), np.nan, X)
        S = _signals(np.asarray(self.modes.mu, complex), X.shape[0])
        resolved = np.full(X.shape, np.nan)
        for f in range(X.shape[1]):
            o = np.isfinite(X[:, f])
            if o.any():
                c = np.linalg.lstsq(S[o], X[o, f], rcond=None)[0]
                resolved[o, f] = S[o] @ c
        residual = X - resolved
        return (resolved[:, 0], residual[:, 0]) if one else (resolved, residual)


def operator_read(W, *, far: float = 0.05) -> OperatorRead:
    """The operator read (EXPERIMENTAL): the record's modes, their order ``K`` counted against the
    Tracy-Widom edge at level ``far``, and the lag depth ``d`` its own coherence sets.  See the
    module notes for the read and :class:`OperatorRead` for what it returns.

    ``W``: a real record ``(T,)`` or ``(T, F)``, channels as independent views of one process; a
    missing sample is NaN (it leaves its window out of the count and its pairs out of the lags).
    A record in which nothing counts returns ``K = 0`` and no modes."""
    far = float(far)
    if not 0.0 < far < 1.0:
        raise ValueError(f"far must be in (0, 1); got {far!r}")
    A = np.asarray(_env.to_numpy(W))
    if np.iscomplexobj(A):
        raise ValueError("operator_read reads real records; split a complex one into its parts")
    A = A.astype(float)
    if A.ndim == 1:
        A = A[:, None]
    if A.ndim != 2:
        raise ValueError(f"W must be (T,) or (T, F); got shape {A.shape}")
    live = np.isfinite(A).any(axis=0)
    X = A[:, live]
    X = X - np.nanmean(X, axis=0) if X.shape[1] else X
    T = int(X.shape[0])
    empty = ModePowers(np.zeros(0, complex), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
    lv = _levels(X, far) if X.shape[1] else ()
    if not lv:
        return OperatorRead(modes=empty, K=0, depth=0, levels=())
    K = max(k for _, _, k in lv)
    b, m, _ = max(lv)                                     # the coarsest level that still counted
    d = max(K + 1, min(m * b, (T - 2) // 4))
    g = _lags(X, 2 * d + 1)
    idx = np.arange(d)[:, None] + np.arange(d)[None, :]
    U, s, Vh = np.linalg.svd(g[idx])
    Kp = min(K, int(np.sum(s > 0)))
    mu = np.linalg.eigvals((U[:, :Kp].T @ g[idx + 1] @ Vh[:Kp].T) / s[None, :Kp])
    V = mu[None, :] ** np.arange(1, 2 * d + 2)[:, None]
    c = np.linalg.lstsq(V, g.astype(complex), rcond=None)[0]
    P = np.abs(c)
    o = np.argsort(-P, kind="stable")
    mu, P = mu[o], P[o]
    tot = float(P.sum())
    with np.errstate(divide="ignore"):
        alpha = -np.log(np.abs(mu))
    modes = ModePowers(mu=mu, alpha=alpha, beta=np.angle(mu), power=P,
                       share=P / tot if tot > 0 else P * 0.0)
    return OperatorRead(modes=modes, K=int(K), depth=int(d), levels=lv)
