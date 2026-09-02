"""
common.py -- shared, deterministic helpers for the entroptics validation harness.

Everything here is seeded (numpy Generator) so every experiment is bit-reproducible.
No entroptics internals are imported; only the public front door is used by the
experiment scripts.  Ground-truth signal generators live here so the reads can be
checked against KNOWN planted structure.
"""
from __future__ import annotations

import numpy as np


# ── determinism ───────────────────────────────────────────────────────────────

def rng(seed: int) -> np.random.Generator:
    """A fresh, independent PCG64 generator for a given integer seed."""
    return np.random.default_rng(seed)


# ── ground-truth signal generators ────────────────────────────────────────────

def ar1(T: int, F: int, phi: float, seed: int) -> np.ndarray:
    """F independent real AR(1) channels of length T with lag-1 coefficient ``phi``
    and unit stationary variance.  The population autocorrelation is exactly
    C(tau) = phi**tau = exp(-tau / rho) with correlation length rho = -1/ln(phi),
    so this plants a KNOWN exponential correlation length."""
    g = rng(seed)
    e = g.standard_normal((T, F))
    x = np.empty((T, F))
    x[0] = e[0]
    s = np.sqrt(1.0 - phi * phi)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + s * e[t]
    return x


def phi_for_rho(rho: float) -> float:
    """AR(1) coefficient giving exponential correlation length rho (phi = e^{-1/rho})."""
    return float(np.exp(-1.0 / rho))


def rot_block(r: float, theta: float) -> np.ndarray:
    """A 2x2 scaled-rotation block with eigenvalues r*e^{+-i theta}
    (|mu| = r  ->  alpha = -ln r ;  arg mu = +-theta  ->  beta = +-theta)."""
    c, s = r * np.cos(theta), r * np.sin(theta)
    return np.array([[c, -s], [s, c]])


def oscillator_operator(specs) -> np.ndarray:
    """Block-diagonal operator A from a list of (r, theta) oscillator specs.
    Eigenvalues are the r*e^{+-i theta}; dim(A) = 2 * len(specs)."""
    blocks = [rot_block(r, th) for (r, th) in specs]
    n = 2 * len(blocks)
    A = np.zeros((n, n))
    for k, B in enumerate(blocks):
        A[2 * k:2 * k + 2, 2 * k:2 * k + 2] = B
    return A


def linear_trajectory(A: np.ndarray, T: int, seed: int, x0=None) -> np.ndarray:
    """The exact noise-free trajectory x_{t+1}=A x_t as a (T, F) waterfall.
    x0 is a random unit-ish vector (excites every eigenvector) unless given."""
    F = A.shape[0]
    x = rng(seed).standard_normal(F) if x0 is None else np.asarray(x0, float)
    rows = [x.copy()]
    for _ in range(T - 1):
        x = A @ x
        rows.append(x.copy())
    return np.asarray(rows)


def planted_lowrank(T: int, F: int, K: int, snr: float, seed: int,
                    *, complex_bg: bool = False) -> np.ndarray:
    """K orthogonal rank-1 modes planted into an iid Gaussian background.

    Each planted mode is a random (orthonormal) temporal pattern u_k (length T) times
    a random (orthonormal) feature pattern v_k (length F).  Every mode is given the
    SAME singular value s = snr * edge, where edge = sqrt(median row power)*(1+sqrt(N/F))
    is the Marchenko-Pastur / Bai-Yin bulk singular-value edge of the iid background;
    so ``snr`` is the planted mode strength in units of the noise floor and K modes
    should be resolved once snr > 1.  Deterministic given the seed."""
    g = rng(seed)
    if complex_bg:
        bg = (g.standard_normal((T, F)) + 1j * g.standard_normal((T, F))) / np.sqrt(2.0)
    else:
        bg = g.standard_normal((T, F))
    W = bg.astype(complex) if complex_bg else bg.astype(float)
    if K <= 0:
        return W
    # orthonormal temporal and feature bases via QR of Gaussian matrices
    U, _ = np.linalg.qr(g.standard_normal((T, K)))
    V, _ = np.linalg.qr(g.standard_normal((F, K)))
    # iid-Gaussian bulk singular edge (real per-entry sigma = 1 here)
    edge = 1.0 * (np.sqrt(T) + np.sqrt(F))
    s = snr * edge
    for k in range(K):
        W = W + s * np.outer(U[:, k], V[:, k])
    return W


def regime_switch(T: int, F: int, phis, seed: int) -> np.ndarray:
    """A NONSTATIONARY AR(1) whose lag-1 coefficient switches through ``phis`` in
    equal-length segments along the ordered axis (a piecewise-stationary record of
    the same size as a stationary one).  The correlation length jumps at each seam."""
    g = rng(seed)
    x = np.empty((T, F))
    e = g.standard_normal((T, F))
    x[0] = e[0]
    seg = T // len(phis)
    for t in range(1, T):
        phi = phis[min(t // seg, len(phis) - 1)]
        x[t] = phi * x[t - 1] + np.sqrt(1.0 - phi * phi) * e[t]
    return x


def ordered_smooth(T: int, F: int, n_modes: int, seed: int) -> np.ndarray:
    """A smoothly ORDERED signal: a few low-frequency temporal modes (so adjacent
    rows are alike) with random feature loadings, plus light noise.  Row t is a
    continuous function of t, so lag-1 rows are correlated (high coherence)."""
    g = rng(seed)
    ts = np.linspace(0.0, 1.0, T)[:, None]
    freqs = (0.5 + np.arange(n_modes))[None, :]
    phases = g.uniform(0, 2 * np.pi, size=(1, n_modes))
    temporal = np.cos(2 * np.pi * freqs * ts + phases)      # (T, n_modes), smooth in t
    loadings = g.standard_normal((n_modes, F))
    sig = temporal @ loadings
    sig = sig / sig.std()
    return sig + 0.25 * g.standard_normal((T, F))


def band_limited(T: int, F: int, n_active: int, seed: int) -> np.ndarray:
    """A feature-BANDLIMITED signal: only a contiguous BAND of ``n_active`` of the F
    feature channels carries signal power (the rest is a tiny broadband noise floor).
    The active band is independent Gaussian per channel, so the feature aperture has a
    KNOWN width n_active -- both the power marginal (n_F ~ n_active) and the feature
    correlation fill (phi_F ~ n_active/F) read the bandwidth directly."""
    g = rng(seed)
    W = 0.02 * g.standard_normal((T, F))
    b = min(n_active, F)
    lo = (F - b) // 2
    W[:, lo:lo + b] = W[:, lo:lo + b] + g.standard_normal((T, b))
    return W


# ── stats helpers ─────────────────────────────────────────────────────────────

def spearman(x, y) -> float:
    """Spearman rank correlation (monotonicity), scipy-free."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def loglog_fit(x, y) -> tuple[float, float]:
    """Least-squares slope and R^2 of log(y) vs log(x)."""
    lx = np.log(np.asarray(x, float)); ly = np.log(np.asarray(y, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(((ly - pred) ** 2).sum())
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r2)


def pearson(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


# ── markdown table helper ─────────────────────────────────────────────────────

def md_table(headers, rows) -> str:
    """Render a markdown table from a header list and a list of row lists."""
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)
