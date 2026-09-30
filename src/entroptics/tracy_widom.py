"""The Tracy-Widom laws TW1 (real matrices, GOE) and TW2 (complex, GUE), to near float precision.

Every count in the library compares a largest eigenvalue with a quantile of one of these laws, so
the law is evaluated exactly rather than approximated or tabulated:

    F1(s) = det(I - B),  F2(s) = det(I - B^2)  on L2(0, inf),  B(x, y) = Ai(x + y + s)

(Ferrari-Spohn for F1; the Airy kernel is B^2), by Nystrom discretisation with Gauss-Legendre nodes
(Bornemann 2010): one symmetric eigensolve gives the eigenvalues lambda of B, and the survival is
``1 - F = -expm1(sum log1p(-lambda))`` (TW1) or ``-expm1(sum log1p(-lambda^2))`` (TW2), so the upper
tail keeps its relative precision.  The quadrature size is doubled until the survival
stops moving at round-off.  The result carries a few hundred eps (about 1e-13): the round-off of
forming the kernel and its eigenvalues.

The Airy function comes from nothing but its differential equation ``y'' = x y``:
- at large x, from its asymptotic series, summed while its terms fall;
- below that, by exact Taylor steps of the equation.  Their coefficients follow the three-term
  recurrence ``c(n) = (x0 c(n-2) + c(n-3)) / (n (n - 1))`` and are summed to round-off.  Stepping
  downward is the stable direction for Ai, which grows there while Bi decays.
- On the interval the laws need, a Chebyshev expansion of those exact values, at the first degree
  whose trailing coefficients are below round-off, evaluates it vectorised.

Every bound here is read off the float format (``eps``), not chosen:
- the left end is where F1 and F2 fall below eps / 2, so the survival rounds to 1;
- the right end is where the kernel falls below round-off of its value at the evaluation point.

numpy only.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_EPS = float(np.finfo(float).eps)
_LNE = -math.log(_EPS)

# F1(s) <= exp(-|s|^3 / 24) and F2(s) <= exp(-|s|^3 / 12) in the left tail: below this, F < eps/2
# and the survival is 1 to the last bit.
_S_LO = -(24.0 * math.log(2.0 / _EPS)) ** (1.0 / 3.0)
# Ai(x) ~ exp(-(2/3) x^1.5): at this x it is below round-off of its size at the origin -- where the
# exact Taylor sweep starts.
_X_HI = (1.5 * _LNE) ** (2.0 / 3.0)
# The asymptotic series' smallest term, relative to its sum, is ~ exp(-(4/3) x^1.5): from here up
# it is at round-off in relative terms, and it serves every larger x.
_X_ASY = (0.75 * _LNE) ** (2.0 / 3.0)


def _ai_asymptotic(x: float) -> tuple[float, float]:
    """(Ai(x), Ai'(x)) from the asymptotic series at large positive x, summed while terms fall."""
    z = 2.0 / 3.0 * x ** 1.5
    su, sv = _asymptotic_sums(z)
    pre = math.exp(-z) / (2.0 * math.sqrt(math.pi))
    return pre * x ** -0.25 * su, -pre * x ** 0.25 * sv


def _asymptotic_sums(z: float) -> tuple[float, float]:
    """The two sums of the asymptotic series at zeta = z: Ai(x) = exp(-z) x^(-1/4) su / (2 sqrt(pi))
    and Ai'(x) = -exp(-z) x^(1/4) sv / (2 sqrt(pi)), summed while their terms fall."""
    su = sv = 1.0
    u = 1.0
    k = 0
    last = math.inf
    while True:
        k += 1
        u = u * (6 * k - 5) * (6 * k - 3) * (6 * k - 1) / ((2 * k - 1) * 216 * k)
        v = -(6 * k + 1) / (6 * k - 1) * u
        tu = u * (-1) ** k / z ** k
        tv = v * (-1) ** k / z ** k
        size = abs(tu) + abs(tv)
        if size >= last:                       # the asymptotic series has turned: stop before it
            break
        su += tu
        sv += tv
        last = size
        if size <= _EPS * (abs(su) + abs(sv)):
            break
    return su, sv


def _taylor_step(x0: float, a: float, ap: float, h: float) -> tuple[float, float]:
    """Advance (Ai, Ai') from x0 by h, the Taylor series of y'' = x y summed to round-off."""
    size = abs(a) + abs(ap * h)
    c2, c1, c0 = a, ap, 0.5 * x0 * a            # c(n-3), c(n-2), c(n-1) as n advances
    y = a + ap * h + c0 * h * h
    yp = ap + 2.0 * c0 * h
    n, hp = 2, h * h
    quiet = 0
    while True:
        n += 1
        cn = (x0 * c1 + c2) / (n * (n - 1))
        c2, c1, c0 = c1, c0, cn
        hp *= h
        ty, typ = cn * hp, n * cn * hp / h
        y += ty
        yp += typ
        # the coefficients fall factorially, so three successive terms below round-off of the
        # local size put the whole tail below it
        quiet = quiet + 1 if abs(ty) + abs(typ * h) <= _EPS * size else 0
        if quiet == 3:
            return y, yp


def _airy_exact(xs: np.ndarray) -> np.ndarray:
    """(Ai, Ai') at points xs <= _X_HI: one downward sweep of Taylor steps, point to point."""
    out = np.empty((xs.size, 2))
    x = _X_HI
    a, ap = _ai_asymptotic(x)
    for i in np.argsort(-xs):
        h = float(xs[i]) - x
        if h:
            a, ap = _taylor_step(x, a, ap, h)
            x = float(xs[i])
        out[i] = a, ap
    return out


def _fit(lo: float, hi: float, scaled: bool) -> np.ndarray:
    """Chebyshev coefficients of (Ai, Ai') on [lo, hi] -- or, when ``scaled``, of
    (Ai, Ai') exp((2/3) x^1.5) as functions of u = sqrt(x) on [sqrt(lo), sqrt(hi)]: analytic in u
    (the exponent is (2/3) u^3) and of order one, so on x >= 0, where Ai decays, the expansion
    carries relative rather than absolute precision.  The degree doubles until the trailing
    coefficients stop shrinking inside the round-off of the sum: the round-off plateau."""
    deg = 4
    prev, prev_tail = None, math.inf
    while True:
        t = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1))
        if scaled:
            ulo, uhi = math.sqrt(lo), math.sqrt(hi)
            u = (uhi - ulo) / 2 * t + (uhi + ulo) / 2
            v = _airy_exact(u * u) * np.exp(2.0 / 3.0 * u ** 3)[:, None]
        else:
            v = _airy_exact((hi - lo) / 2 * t + (hi + lo) / 2)
        # the exact discrete orthogonality of Chebyshev polynomials at these nodes: c_j = (2/N)
        # sum_k f(t_k) T_j(t_k), c_0 halved -- no least-squares solve and its round-off
        n_pts = deg + 1
        T = np.cos(np.outer(np.arange(n_pts), np.pi * (np.arange(n_pts) + 0.5) / n_pts))
        c = (2.0 / n_pts) * (T @ v)
        c[0] *= 0.5
        # each component's tail against its size on the interval: absolute precision for (Ai, Ai'),
        # relative for the scaled pair, whose size stays within a small factor of its smallest
        unit = np.max(np.abs(v), axis=0)
        tail = float(np.max(np.max(np.abs(c[-3:]), axis=0) / unit))
        # the round-off plateau: once the tail is inside the sum's round-off (deg eps) and doubling
        # no longer shrinks it, the previous expansion is as exact as the values it was fitted to
        if prev is not None and prev_tail <= deg * _EPS and tail >= prev_tail:
            return prev
        prev, prev_tail = c, tail
        deg *= 2


def _cheb_plateau(values_at) -> np.ndarray:
    """Chebyshev coefficients of the columns ``values_at(t)`` on t in [-1, 1], at the degree where
    doubling stops shrinking the trailing coefficients inside the sum's round-off."""
    deg = 4
    prev, prev_tail = None, math.inf
    while True:
        n_pts = deg + 1
        t = np.cos(np.pi * (np.arange(n_pts) + 0.5) / n_pts)
        v = values_at(t)
        T = np.cos(np.outer(np.arange(n_pts), np.pi * (np.arange(n_pts) + 0.5) / n_pts))
        c = (2.0 / n_pts) * (T @ v)
        c[0] *= 0.5
        tail = float(np.max(np.max(np.abs(c[-3:]), axis=0) / np.max(np.abs(v), axis=0)))
        if prev is not None and prev_tail <= deg * _EPS and tail >= prev_tail:
            return prev
        prev, prev_tail = c, tail
        deg *= 2


@lru_cache(maxsize=None)
def _asymptotic_expansion() -> np.ndarray:
    """The asymptotic series' two sums as smooth functions of tau = 1/zeta on [0, 1/zeta(_X_ASY)],
    where the series is at round-off: one Chebyshev evaluation in place of a term-by-term sum."""
    tau_max = 1.0 / (2.0 / 3.0 * _X_ASY ** 1.5)

    def values(t):
        tau = tau_max * (t + 1.0) / 2.0
        return np.array([_asymptotic_sums(1.0 / x) if x > 0 else (1.0, 1.0) for x in tau])

    return _cheb_plateau(values)


@lru_cache(maxsize=None)
def _airy_expansions() -> tuple[np.ndarray, np.ndarray]:
    """(Ai, Ai') on [_S_LO, 0], where Ai oscillates at order one, and the scaled pair in
    u = sqrt(x) on [0, _X_ASY], where it decays."""
    return _fit(_S_LO, 0.0, False), _fit(0.0, _X_ASY, True)


def airy(x) -> tuple[np.ndarray, np.ndarray]:
    """``(Ai(x), Ai'(x))`` to float precision, elementwise, for real x >= the laws' left end:
    absolute precision where x < 0, relative precision where x >= 0."""
    return _airy(x, 2)


def _ai(x) -> np.ndarray:
    """``Ai(x)`` alone: half the work of :func:`airy` where the derivative is not needed."""
    return _airy(x, 1)[0]


def _airy(x, parts: int):
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    if flat.size and float(np.nanmin(flat)) < _S_LO:
        raise ValueError(f"airy is expanded on [{_S_LO:.4g}, inf), the laws' domain; got {np.nanmin(flat)}")
    out = np.empty((flat.size, 2))
    neg, pos = _airy_expansions()
    below = flat < 0.0
    above = flat > _X_ASY
    mid = ~below & ~above
    if below.any():
        t = (2.0 * flat[below] - _S_LO) / (0.0 - _S_LO)
        out[below, :parts] = np.polynomial.chebyshev.chebval(t, neg[:, :parts]).T
    if mid.any():
        u = np.sqrt(flat[mid])
        uhi = math.sqrt(_X_ASY)
        g = np.polynomial.chebyshev.chebval((2.0 * u - uhi) / uhi, pos[:, :parts]).T
        out[mid, :parts] = g * np.exp(-2.0 / 3.0 * u ** 3)[:, None]
    if above.any():
        xa = flat[above]
        z = 2.0 / 3.0 * xa ** 1.5
        tau_max = 1.0 / (2.0 / 3.0 * _X_ASY ** 1.5)
        su, sv = np.polynomial.chebyshev.chebval(2.0 * (1.0 / z) / tau_max - 1.0, _asymptotic_expansion())
        pre = np.exp(-z) / (2.0 * math.sqrt(math.pi))
        out[above] = np.stack([pre * xa ** -0.25 * su, -pre * xa ** 0.25 * sv], axis=1)
    out = out.reshape(x.shape + (2,))
    return out[..., 0], out[..., 1]


# The smallest positive float: a survival below it is not representable, so it is 0.
_LN_SUB = -math.log(float(np.nextafter(0.0, 1.0)))


def _check_beta(beta: int) -> int:
    if beta not in (1, 2):
        raise ValueError(f"beta must be 1 (real, TW1) or 2 (complex, TW2); got {beta!r}")
    return int(beta)


def _length(s: np.ndarray) -> np.ndarray:
    """The length L of [0, L] on which B(x, y) = Ai(x + y + s) is kept: Ai(x) ~ exp(-(2/3) x^1.5),
    so the kernel's row at y = 0 has fallen below round-off of its largest value, Ai(max(s, 0)),
    where (2/3) (s + L)^1.5 - (2/3) max(s, 0)^1.5 = ln(1/eps)."""
    return (np.maximum(s, 0.0) ** 1.5 + 1.5 * _LNE) ** (2.0 / 3.0) - s


def _s_hi(beta: int) -> float:
    """Above this the survival is below the smallest positive float: the right tail decays as
    exp(-(2 beta / 3) s^1.5) times a factor below one there."""
    return (1.5 / beta * _LN_SUB) ** (2.0 / 3.0)


def _survival(s: np.ndarray, beta: int, m: int) -> np.ndarray:
    """The m-node Nystrom survival at each point of s (1-D), all in one batched eigensolve.

    Both laws come from the one Hankel operator B(x, y) = Ai(x + y + s) on L2(0, inf):
    F1 = det(I - B) (Ferrari-Spohn) and F2 = det(I - B^2), since the Airy kernel is B^2.  With the
    eigenvalues lambda of the symmetrised B, F1 = prod(1 - lambda) and F2 = prod(1 - lambda^2) --
    no divided difference, so nothing cancels.  The value is left signed where a coarse grid gives
    a factor at or below zero, so the convergence test sees an under-resolved grid rather than a
    clipped agreement."""
    t, w = np.polynomial.legendre.leggauss(m)
    L = _length(s)
    u = L[:, None] * (t + 1.0) / 2.0                                  # (n, m) nodes on [0, L]
    sw = np.sqrt(w[None, :] * L[:, None] / 2.0)
    B = _ai(u[:, :, None] + u[:, None, :] + s[:, None, None])
    lam = np.linalg.eigvalsh(sw[:, :, None] * B * sw[:, None, :])
    f = lam if beta == 1 else lam * lam                                # the factors are 1 - f
    below = np.all(f < 1.0, axis=1)
    out = np.empty(s.shape)
    # the upper tail keeps relative precision through log1p / expm1
    out[below] = -np.expm1(np.sum(np.log1p(-f[below]), axis=1))
    out[~below] = 1.0 - np.prod(1.0 - f[~below], axis=1)
    return out


def survival(s, beta: int = 1):
    """``P(TW_beta > s)``, elementwise; a float for a scalar ``s``.  Accurate to a few hundred eps
    (about 1e-13, relative in the upper tail).  NaN gives NaN."""
    beta = _check_beta(beta)
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    out = np.where(np.isnan(s_arr), np.nan, np.where(s_arr >= _s_hi(beta), 0.0, 1.0))
    live = (s_arr > _S_LO) & (s_arr < _s_hi(beta))
    if live.any():
        out[live] = _converged(s_arr[live], beta)
    return float(out[0]) if np.ndim(s) == 0 else out.reshape(np.shape(s))


def _converged(s: np.ndarray, beta: int) -> np.ndarray:
    # Gauss-Legendre converges exponentially: double the nodes until a point's change is inside the
    # round-off of an m-term sum, or until it stops shrinking -- the round-off plateau, where the
    # previous (less noisy) value is the answer.
    m = 8
    prev = _survival(s, beta, m)
    out = prev.copy()
    step = np.full(s.shape, np.inf)
    todo = np.arange(s.size)
    while todo.size:
        m *= 2
        cur = _survival(s[todo], beta, m)
        change = np.abs(cur - prev[todo])
        converged = change <= m * _EPS * np.abs(cur)
        plateau = change >= step[todo]
        out[todo] = np.where(plateau, prev[todo], cur)
        prev[todo], step[todo] = cur, change
        todo = todo[~(converged | plateau)]
    return np.clip(out, 0.0, 1.0)                    # a probability


@lru_cache(maxsize=None)
def quantile(far: float, beta: int = 1) -> float:
    """The ``q`` with ``P(TW_beta > q) = far``: the Illinois method on log-survival (monotone and
    smooth in s), with a bisection step where the log-survival is not finite, until the bracket's
    midpoint is no longer strictly inside it.  The first upper end is where the tail's leading term
    equals ``far``; it is extended by the bracket's own width while the survival there is above."""
    beta = _check_beta(beta)
    far = float(far)
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    target = math.log(far)

    def f(x):
        sf = survival(x, beta)
        return math.log(sf) - target if sf > 0.0 else -math.inf

    lo = _S_LO
    hi = min(max((1.5 / beta * -target) ** (2.0 / 3.0), _X_HI), _s_hi(beta))
    f_hi = f(hi)
    while f_hi > 0.0:
        lo, hi = hi, min(hi + (hi - lo), _s_hi(beta))
        f_hi = f(hi)
    f_lo = -target                                   # log 1 - log far
    side = 0
    while True:
        mid = 0.5 * (lo + hi)
        if not (lo < mid < hi):
            return mid
        if math.isfinite(f_hi) and math.isfinite(f_lo) and f_hi != f_lo:
            x = hi - f_hi * (hi - lo) / (f_hi - f_lo)
            if not (lo < x < hi):
                x = mid
        else:
            x = mid
        fx = f(x)
        if fx == 0.0:
            return x
        if fx > 0.0:                                 # survival above far: the root is above x
            lo, f_lo = x, fx
            if side == -1 and math.isfinite(f_hi):
                f_hi *= 0.5
            side = -1
        else:
            hi, f_hi = x, fx
            if side == 1:
                f_lo *= 0.5
            side = 1
