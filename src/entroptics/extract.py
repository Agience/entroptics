"""extract.py -- the read-side filter: split a signal into its own resolved modes and the rest.

This module holds the filter -- :func:`filter_projection` -- and the parameter-free piece it uses.
The one public way to reach it is :meth:`entroptics.Aperture.extract`, which hands this its
projection.

The filter is the hard form of the projection onto the resolved modes (Def 8.4).  On the data's
own SVD, ``P_L data P_R = P_L data`` (``data V_k = U_k S_k``), so it is the projection of every
channel onto the resolved modes' ordered-axis profiles:

    resolved = P_L data,        residual = data - resolved,

an orthogonal projection, so nothing is lost (``resolved + residual`` is the data), nothing is
synthesised (every value is a projection of measured data) and ``P_L`` is idempotent: projecting
the resolved part again onto the same modes returns it.  Reading ``clean`` afresh through a new
``Aperture`` is a new read (its own centre, scale and modes) and need not return it.
The singular values are never altered; each channel's amplitude on each mode is read from the data.

The modes are read on the entropy-matched screen, which is a linear fold of the whitened data.
Their ordered-axis profiles are carried back to the data's own grid by the fold's adjoint, and the
channels are never folded, so the split is on the caller's grid at the channels' native resolution.
(Carrying the feature profiles back as well, ``P_L data P_R`` with ``P_R`` lifted, would limit every
mode to the feature fold's resolution and push native channel detail into the residual.)

The mode cut: the entropy footprint fill ``mode_fill`` separates a transient event (spread across
features, compact along the ordered axis: phi_F > phi_T) from persistent narrowband structure
(phi_F <= phi_T).  ``reject_persistent`` moves the latter into the residual.
"""
from __future__ import annotations

import numpy as np

from . import environment as _env
from .entropy import normalize, shannon_bits


def mode_fill(v: np.ndarray) -> float:
    """Entropy fill fraction (participation ratio) of a mode vector: 2^H(|v|^2) / len.
    ~1 => spread over the whole axis (broadband / persistent), ~0 => concentrated (compact)."""
    p = np.abs(v) ** 2
    return float(2 ** shannon_bits(p) / p.size)


def _adjoint_lift(n_in: int, n_out: int, V: np.ndarray) -> np.ndarray:
    """``R @ V``, where ``R`` is the ``(n_in, n_out)`` matrix of the screen's fold along one axis (a
    row ``z`` folds to ``z @ R``): the fold's adjoint, carrying screen vectors ``V`` (``n_out, k``)
    back to the data's grid.  ``R`` is built from the fold itself, a block of unit rows at a time,
    so memory stays bounded however wide the axis is."""
    from .projection import _fold_axis          # deferred import (projection imports this module's peers)
    out = np.empty((n_in, V.shape[1]), dtype=np.result_type(V, float))
    step = max(1, (1 << 22) // max(n_in, 1))    # rows per block: about 4M cells of the unit block
    for i0 in range(0, n_in, step):
        i1 = min(n_in, i0 + step)
        E = np.zeros((i1 - i0, n_in))
        E[np.arange(i1 - i0), np.arange(i0, i1)] = 1.0
        out[i0:i1] = np.asarray(_fold_axis(E, n_out, axis=1), float) @ V
    return out


def _orth(A: np.ndarray) -> np.ndarray:
    return np.linalg.qr(A)[0] if A.shape[1] else A


def _flat_centres(sc, W):
    """``{channel: centre}`` for the channels the projection left out because every measured value
    is the same (``Projection._flat_cols``): measured, so a filter keeps each at that value, over
    every row it was measured on -- including a row the read dropped because its only measured
    cells were flat."""
    flat = getattr(sc, "_flat_cols", None)
    if flat is None:
        return {}
    mask = None if sc.mask is None else np.asarray(sc.mask)
    out = {}
    for f in np.flatnonzero(flat):
        col = W[:, f]
        ok = np.isfinite(np.abs(col))
        if mask is not None:
            ok &= ~mask[:, f]
        if ok.any():
            out[int(f)] = col[ok][0]
    return out


def _whitened_live(sc):
    """The projection's record on its live rows and channels, whitened exactly as the screen
    whitens it: ``(W, rows, cols, Z, centre, scale, missing)``.  A missing cell stands at its
    channel's centre (0 in ``Z``), as it enters the screen, and ``missing`` marks it."""
    W = np.asarray(_env.to_numpy(sc.W))
    mask = None if sc.mask is None else np.asarray(sc.mask)
    rows = np.ones(W.shape[0], bool) if sc._live_rows is None else np.asarray(sc._live_rows)
    cols = np.ones(W.shape[1], bool) if sc._live_cols is None else np.asarray(sc._live_cols)
    W_live = W[np.ix_(rows, cols)]
    m_live = None if mask is None else mask[np.ix_(rows, cols)]
    if m_live is not None and not m_live.any():
        m_live = None
    Z, centre, scale = normalize(W_live, m_live, return_stats=True)
    Z = np.asarray(_env.to_numpy(Z))
    centre = np.asarray(_env.to_numpy(centre))
    scale = np.asarray(_env.to_numpy(scale), float)
    missing = ~np.isfinite(np.abs(W_live))
    if m_live is not None:
        missing |= m_live
    Z = np.where(missing | ~np.isfinite(np.abs(Z)), 0.0, Z)       # the channel's centre, as on the screen
    return W, rows, cols, Z, centre, scale, missing


def filter_projection(sc, *, reject_persistent: bool = True):
    """THE read-side filter: split a signal into its resolved modes and the residual, on the
    caller's grid and in the caller's units.

    Takes the :class:`projection.Projection` -- the filter is a statement about a projection, and
    every caller already holds one.  The resolved modes are the ``K_signal`` screen modes above the
    derived floor; ``reject_persistent`` drops those with ``phi_F <= phi_T`` (persistent and
    narrowband, i.e. RFI) into the residual.

    Returns ``(clean, info)``.  ``clean`` has ``W``'s shape and units: the per-channel baseline
    plus the resolved part.  ``info["residual"]`` is ``W - clean``, so ``clean + residual == W``
    wherever ``W`` was measured.  A missing cell is missing in both: it enters the projection at
    its channel's centre, as it enters the screen, and is never filled in.  A channel or row the
    read dropped (never measured) is NaN in both.  ``info`` also carries K_signal, contrast,
    coherence, the screen shape, the kept and dropped modes with their phi_T / phi_F, and the
    per-channel ``centre`` and ``scale`` of the whitening.

    Why the baseline is in ``clean``: a channel's mean is not a mode -- ``normalize`` removes it
    before the SVD runs, so no cut was ever offered it -- and carrying it through is what makes the
    residual mean "what the modes do not account for" rather than that plus a baseline the filter
    never looked at.  The residual is orthogonal to the resolved part in the whitened (noise)
    metric over the full grid, each missing cell standing at its channel centre; over the measured
    cells alone the two are exactly orthogonal only when every cell was measured."""
    U = np.asarray(_env.to_numpy(sc.U))
    Vt = np.asarray(_env.to_numpy(sc.Vt))                           # footprints only
    N, F_eff = (int(v) for v in sc.screen.shape)
    K = int(sc.K_signal)
    kept, dropped, phis = [], [], []
    for k in range(K):
        pT, pF = mode_fill(U[:, k]), mode_fill(Vt[k, :])
        phis.append((k, pT, pF))
        if reject_persistent and pF <= pT:            # persistent + narrowband -> residual
            dropped.append(k)
        else:
            kept.append(k)

    W, rows, cols, Z, centre, scale, missing = _whitened_live(sc)
    QL = _orth(_adjoint_lift(Z.shape[0], N, U[:, kept]))       # resolved profiles on the data's grid
    resolved = QL @ (QL.conj().T @ Z)                            # each channel onto them

    clean_live = resolved * scale[None, :] + centre[None, :]      # back to the caller's units
    clean_live = np.where(missing, np.nan, clean_live)
    clean = np.full(W.shape, np.nan, dtype=np.result_type(clean_live, W))
    clean[np.ix_(rows, cols)] = clean_live
    centre_f = np.full(W.shape[1], np.nan, dtype=centre.dtype)
    scale_f = np.full(W.shape[1], np.nan)
    centre_f[cols], scale_f[cols] = centre, scale
    flat = _flat_centres(sc, W)
    for f, c in flat.items():                    # measured, but no noise scale: its centre, whole
        meas = np.isfinite(np.abs(W[:, f]))
        if sc.mask is not None:
            meas &= ~np.asarray(sc.mask)[:, f]
        clean[meas, f] = c
        centre_f[f], scale_f[f] = c, 0.0
    residual = W - clean
    info = {
        "K_signal": K,
        "contrast": float(sc.sigma_top / sc.noise_floor) if sc.noise_floor > 0 else 0.0,
        "coherence": float(sc.coherence),
        "screen_shape": (N, F_eff),
        "n_kept": len(kept), "n_dropped": len(dropped),
        "kept": kept, "dropped": dropped, "phis": phis,
        "residual": residual,
        # The whitening map, per channel of W (NaN where a channel was never measured) --
        # reported so the read is auditable, not so there is a second way to call this.
        "centre": centre_f, "scale": scale_f,
    }
    return clean, info
