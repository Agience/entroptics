"""
tensor.py -- delay-embedded Tucker (HOSVD) decomposition of a screen.

The standard screen construction (normalize -> project -> SVD) averages delta_T
consecutive ordered steps into one row, recovering feature shapes but losing the
within-window fine structure.  ``tensor_embed`` works at native resolution by
embedding consecutive feature vectors into a 3-way tensor and taking its
Higher-Order SVD -- exposing how the feature spectrum evolves within a window,
not just what it is on average.

Standalone, backend-agnostic (numpy on CPU, torch on its device -- one code path):
general linear algebra (randomised range-finding + Tucker/HOSVD), nothing
domain-specific.  Reachable from the front door as ``Aperture.tensor()`` /
``Projection.tensor()``.

  tensor_read(W[, mask], d)     -- whiten a raw (T, F) waterfall, then HOSVD it.
  tensor_embed(data, d)         -- HOSVD of the delay-embedded (already-normalized) tensor.
  tensor_reconstruct(te)        -- inverse (overlap-add).
  tensor_fidelity(data, te)     -- round-trip fidelity in [0, 1].
  pack_factors / unpack_factors -- (de)serialise the Tucker factors to a 2-D matrix.
"""
from __future__ import annotations

import math

import numpy as np

from . import environment as _env

# Default delay-embedded Tucker ranks.
TENSOR_EMBED_D:   int = 16   # default delay-window width in tensor_embed
TENSOR_EMBED_R_T: int = 8    # default Tucker rank along the ordered axis
TENSOR_EMBED_R_F: int = 16   # default Tucker rank along the feature axis


def _iscomplex(M) -> bool:
    xp = _env.ns(M)
    return xp.is_complex(M) if _env.is_torch(xp) else np.iscomplexobj(M)


def _truncated_left_svd(M, k: int, oversample: int = 10):
    """Top-k left singular vectors of M via randomised range-finding
    (Halko/Martinsson/Tropp 2011).  Build an orthonormal basis Q for the column
    range of M*Omega (Omega a small random matrix), then SVD the projected
    Q^H*M.  Cost O(m*n*(k+oversample)) instead of O(m*n*min(m,n)).  Falls back to
    a full SVD when k_eff >= max(min(M.shape)//2, 16) (no asymptotic advantage
    there, and the deterministic SVD is more stable).  Real- and complex-safe,
    backend-agnostic.  Omega is seeded (numpy) and injected into M's backend, so
    the result is deterministic on numpy and torch."""
    xp = _env.ns(M)
    m, n = int(M.shape[0]), int(M.shape[1])
    is_c = _iscomplex(M)
    k_eff = min(k, m, n)
    if k_eff <= 0:
        return _env.zeros(xp, (m, 0), complex=is_c, ref=M), _env.zeros(xp, (0,), ref=M)
    if k_eff >= max(min(m, n) // 2, 16):
        U, S, _ = xp.linalg.svd(M, full_matrices=False)
        return U[:, :k_eff], S[:k_eff]
    over = min(oversample, max(0, min(m, n) - k_eff))
    p = k_eff + over
    rng = np.random.default_rng(0)  # deterministic across runs and backends
    if is_c:
        Omega_np = (rng.standard_normal((n, p)) + 1j * rng.standard_normal((n, p)))
    else:
        Omega_np = rng.standard_normal((n, p))
    Omega = _env.asdtype_of(M, Omega_np)
    Y = M @ Omega
    Q, _ = xp.linalg.qr(Y)
    B = Q.conj().T @ M
    U_b, S, _ = xp.linalg.svd(B, full_matrices=False)
    U = (Q @ U_b)[:, :k_eff]
    return U, S[:k_eff]


def tensor_embed(data, d: int = TENSOR_EMBED_D, rank: tuple | None = None) -> dict:
    """Delay-embedded Tucker (HOSVD) decomposition of a normalized array.

    Embed consecutive feature vectors into a 3-way tensor at native resolution:

        T[t, delta, f] = data[t + delta, f]     shape (T', d, F_eff),  T' = T - d + 1

    then decompose by unfolding along each mode:

        U_time  (T' x r_t)     -- when each pattern fires
        U_lag   (d  x r_d)     -- how the feature spectrum evolves within a window
        U_freq  (F_eff x r_f)  -- feature shapes
        core    (r_t x r_d x r_f) -- coupling strengths

    U_lag is the quantity absent from the plain screen:
      * flat column     -> spectrum is constant within the window (one emitter)
      * monotonic ramp  -> the feature centroid drifts across the window
      * peaked column   -> a transient sub-pulse within the window

    Parameters
    ----------
    data : (T, F_eff) normalized array (e.g. from entropy.normalize).  numpy or torch.
    d    : delay window width in ordered steps.
    rank : (r_t, r_d, r_f) Tucker ranks.  Default (TENSOR_EMBED_R_T, d, min(F_eff, TENSOR_EMBED_R_F)).

    Returns
    -------
    dict with keys U_time, U_lag, U_freq, core, sv_time, sv_lag, sv_freq, T_prime, d.
    """
    xp = _env.ns(data)
    T, F_eff = int(data.shape[0]), int(data.shape[1])

    T_prime = T - d + 1
    if T_prime < 2:
        raise ValueError(f"d={d} too large for T={T}: need d <= T-2")

    if rank is None:
        r_t = min(T_prime, TENSOR_EMBED_R_T)
        r_d = d            # keep all lag modes by default
        r_f = min(F_eff, TENSOR_EMBED_R_F)
    else:
        r_t, r_d, r_f = int(rank[0]), int(rank[1]), int(rank[2])

    # Build the delay-embedded tensor: T[anchor, lag, freq] = data[anchor+lag, freq]
    T_arr = _env.stack(xp, [data[delta: delta + T_prime] for delta in range(d)], axis=1)

    # HOSVD: unfold each mode, take truncated left singular vectors.
    T0 = T_arr.reshape(T_prime, d * F_eff)                              # mode 0 (time)
    U_time, sv_time = _truncated_left_svd(T0, r_t)
    T1 = _env.permute(xp, T_arr, (1, 0, 2)).reshape(d, T_prime * F_eff)  # mode 1 (lag)
    U_lag, sv_lag = _truncated_left_svd(T1, r_d)
    T2 = _env.permute(xp, T_arr, (2, 0, 1)).reshape(F_eff, T_prime * d)  # mode 2 (freq)
    U_freq, sv_freq = _truncated_left_svd(T2, r_f)

    # Tucker core (Hermitian-adjoint projector; conj is a no-op for real factors
    # and preserves phase coherence for complex ones).
    G1 = xp.einsum('tdf,fc->tdc', T_arr, U_freq.conj())
    G2 = xp.einsum('tdc,db->tbc', G1, U_lag.conj())
    core = xp.einsum('tbc,ta->abc', G2, U_time.conj())

    return {
        'U_time': U_time, 'U_lag': U_lag, 'U_freq': U_freq, 'core': core,
        'sv_time': sv_time, 'sv_lag': sv_lag, 'sv_freq': sv_freq,
        'T_prime': T_prime, 'd': d,
    }


def tensor_reconstruct(te: dict, T_out: int | None = None):
    """Reconstruct normalized data from Tucker factors (inverse of tensor_embed).

    Reverses the delay-embedding by overlap-add: anchor-time / lag pair (t, delta)
    contributes to absolute time t + delta; cells covered by multiple pairs are
    averaged.  ``T_out`` defaults to T_prime + d - 1.  Preserves the factor dtype,
    so complex inputs round-trip without phase loss.  Backend-agnostic.
    """
    U_time = te['U_time']
    U_lag = te['U_lag']
    U_freq = te['U_freq']
    core = te['core']
    d = int(te['d'])
    T_prime = int(te['T_prime'])
    F_eff = int(U_freq.shape[0])
    xp = _env.ns(U_time)

    if T_out is None:
        T_out = T_prime + d - 1

    A = xp.einsum('abc,ta->tbc', core, U_time)
    B = xp.einsum('tbc,db->tdc', A, U_lag)
    T_hat = xp.einsum('tdc,fc->tdf', B, U_freq)

    is_c = _iscomplex(T_hat)
    data_hat = _env.zeros(xp, (T_out, F_eff), complex=is_c, ref=T_hat)
    count = _env.zeros(xp, (T_out,), ref=U_time)
    for delta in range(d):
        t_end = min(T_prime, T_out - delta)
        if t_end <= 0:
            break
        data_hat[delta:delta + t_end] += T_hat[:t_end, delta, :]
        count[delta:delta + t_end] += 1

    safe = count > 0
    den = count[safe]
    data_hat[safe] = data_hat[safe] / den[:, None]
    return data_hat


def tensor_fidelity(data, te: dict) -> float:
    """Round-trip fidelity for the tensor path: 1 - RMSE(data_norm, data_hat_norm),
    in [0, 1] (1.0 = perfect).  Complex-safe (compares magnitudes of the residual)."""
    xp = _env.ns(data)
    T = int(data.shape[0])
    data_hat = tensor_reconstruct(te, T_out=T)

    def _norm(x):
        m = float(xp.abs(x).max())
        return x / m if m > 0 else x

    diff = _norm(data) - _norm(data_hat)
    rmse = float(xp.sqrt(xp.mean(xp.abs(diff) ** 2)))
    return float(1.0 - rmse)


def tensor_read(W, mask=None, d: int | None = None, *, rank: tuple | None = None) -> dict:
    """Convenience front door: whiten a raw (T, F) waterfall to native resolution
    (entropy.normalize) and return its delay-embedded Tucker (HOSVD).  Masked /
    non-finite cells are filled with 0 after whitening.  Backend-agnostic; this is
    what ``Aperture.tensor()`` / ``Projection.tensor()`` call."""
    from .entropy import normalize
    data = normalize(W, mask)
    xp = _env.ns(data)
    finite = xp.isfinite(xp.abs(data))                 # complex-safe gap fill (no NaN into HOSVD)
    data = xp.where(finite, data, xp.zeros_like(data))
    return tensor_embed(data, TENSOR_EMBED_D if d is None else int(d), rank)


def pack_factors(te: dict) -> np.ndarray:
    """Serialise Tucker factors into a 2-D numpy matrix for transport / storage
    (materialised to numpy; the factors may live on any backend).

    Layout (each row is F_eff wide):
        Row 0        : header [r_t, r_d, r_f, d, F_eff, T_prime, 0 ...]
        Rows 1..r_f  : U_freq columns
        next         : U_lag, zero-padded
        next         : core flattened
        next         : U_time, zero-padded

    Output dtype tracks the factors: real -> float64, complex -> complex128
    (full precision -- lossless round-trip).
    """
    U_time = _env.to_numpy(te['U_time'])
    U_lag = _env.to_numpy(te['U_lag'])
    U_freq = _env.to_numpy(te['U_freq'])
    core = _env.to_numpy(te['core'])
    d = int(te['d'])
    T_prime = int(te['T_prime'])
    F_eff = U_freq.shape[0]
    r_t = U_time.shape[1]
    r_d = U_lag.shape[1]
    r_f = U_freq.shape[1]

    is_complex = any(np.iscomplexobj(f) for f in (U_time, U_lag, U_freq, core))
    out_dtype = np.complex128 if is_complex else np.float64   # full precision (lossless round-trip)

    def _rows(M: np.ndarray, cols: int) -> np.ndarray:
        flat = M.ravel()
        n_rows = math.ceil(len(flat) / cols)
        padded = np.zeros(n_rows * cols, dtype=out_dtype)
        padded[:len(flat)] = flat
        return padded.reshape(n_rows, cols)

    header = np.zeros(F_eff, dtype=out_dtype)
    meta = [r_t, r_d, r_f, d, F_eff, T_prime]
    header[:len(meta)] = meta

    u_freq_rows = U_freq.T.astype(out_dtype, copy=False)
    u_lag_rows = _rows(U_lag, F_eff)
    core_rows = _rows(core, F_eff)
    u_time_rows = _rows(U_time.T, F_eff)

    rows = np.vstack([header[None, :], u_freq_rows, u_lag_rows, core_rows, u_time_rows])
    return rows.astype(out_dtype)


def unpack_factors(rows: np.ndarray,
                   r_t: int, r_d: int, r_f: int,
                   d: int, F_eff: int, T_prime: int,
                   skip_header: bool = True) -> dict:
    """Inverse of pack_factors().  Pass the shapes from the pre-agreed parameters
    (or read them from ``rows[0]`` if the header was kept).  Returns a dict
    compatible with tensor_reconstruct (sv_* keys absent -- not needed)."""
    pos = 1 if skip_header else 0

    n_freq = r_f
    U_freq = rows[pos:pos + n_freq].T.copy()
    pos += n_freq

    n_lag_rows = math.ceil(r_d * d / F_eff)
    flat_lag = rows[pos:pos + n_lag_rows].ravel()[:r_d * d]
    U_lag = flat_lag.reshape(d, r_d)
    pos += n_lag_rows

    n_core_elems = r_t * r_d * r_f
    n_core_rows = math.ceil(n_core_elems / F_eff)
    flat_core = rows[pos:pos + n_core_rows].ravel()[:n_core_elems]
    core = flat_core.reshape(r_t, r_d, r_f)
    pos += n_core_rows

    n_time_rows = math.ceil(r_t * T_prime / F_eff)
    flat_time = rows[pos:pos + n_time_rows].ravel()[:r_t * T_prime]
    U_time = flat_time.reshape(r_t, T_prime).T.copy()

    return {'U_time': U_time, 'U_lag': U_lag, 'U_freq': U_freq, 'core': core,
            'd': d, 'T_prime': T_prime}
