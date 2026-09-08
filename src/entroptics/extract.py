"""extract.py -- the read-side filter: project a signal onto its own resolved modes.

This module holds the filter -- :func:`filter_projection` -- and the two parameter-free pieces it
uses.  A clean view is a projection of a signal's own resolved modes onto the data, never a
synthesis.  The one public way to reach it is :meth:`entroptics.Aperture.extract`, which hands
this its projection:

  1. Optimal shrinkage (Gavish & Donoho 2017, "Optimal Shrinkage of Singular Values") of the
     singular values against the derived floor -- the minimum-MSE low-rank estimate.  Noise modes
     are attenuated to zero and the surviving signal modes are de-biased toward the floor.

  2. The entropy footprint fill ``mode_fill`` -- the discriminator for the geometry cut: a transient
     event is spread across features but compact along the ordered axis (phi_F > phi_T); persistent
     narrowband structure is the opposite (phi_F <= phi_T).

:func:`filter_projection` applies both to a :class:`projection.Projection`.  It takes the read it is given, so there is one filter and one way in: ``Aperture.extract`` builds the
projection and hands it over.  Nothing here constructs an ``Aperture``.
"""
from __future__ import annotations

import numpy as np

from .entropy import shannon_bits


def mode_fill(v: np.ndarray) -> float:
    """Entropy fill fraction (participation ratio) of a mode vector: 2^H(|v|^2) / len.
    ~1 => spread over the whole axis (broadband / persistent), ~0 => concentrated (compact)."""
    p = np.abs(v) ** 2
    return float(2 ** shannon_bits(p) / p.size)


def gavish_donoho(S: np.ndarray, floor: float, T: int, F: int) -> np.ndarray:
    """Frobenius-optimal singular-value shrinker (Gavish & Donoho 2017) against the derived floor.

    The per-entry noise sigma is backed out of the Marchenko-Pastur edge (floor ~ sigma*(sqrt T +
    sqrt F)); singular values are put in bulk-edge units, shrunk by the optimal nonlinearity, and
    returned to physical units.  Values at/below the bulk edge map to 0."""
    m, n = min(T, F), max(T, F)
    beta = m / n
    sigma = floor / (np.sqrt(T) + np.sqrt(F))
    y = S / (sigma * np.sqrt(n))
    edge = 1.0 + np.sqrt(beta)
    eta = np.where(y > edge, np.sqrt(np.maximum((y ** 2 - beta - 1.0) ** 2 - 4.0 * beta, 0.0)) / y, 0.0)
    return eta * sigma * np.sqrt(n)


def filter_projection(sc, *, reject_persistent: bool = True, shrink: bool = True):
    """THE read-side filter: project a signal onto its own resolved modes, IN THE CALLER'S UNITS.

    Takes the :class:`projection.Projection` -- the filter is a statement about a projection, and
    every caller already holds one, so this takes the read it is given.  ``shrink``
    applies Gavish & Donoho (2017) optimal singular-value shrinkage against the derived floor (else
    a hard floor cut); ``reject_persistent`` drops the ``phi_F <= phi_T`` modes -- persistent and
    narrowband, i.e. RFI.

    Nothing is synthesised: ``clean`` is a linear PROJECTION of the MEASURED data onto its own
    resolved modes (``clean = U diag(Sd) Vt``, U/Vt from the data's own projection).  Returns
    ``(clean, info)``; ``info`` carries K_signal, contrast, and the kept/dropped modes with their
    phi_T/phi_F.

    Units.  This is the one read in the library whose output is DATA rather than a measurement,
    so it is the only one where the frame it comes back on is a question at all.  The modes are
    the screen's, and the screen is ``project(normalize(W))`` -- the per-channel median removed
    and the per-channel robust scale divided out before any mode is taken -- so the projection
    lands on the whitened grid.  The whitening is then UNDONE before returning, because a
    denoised frame a caller cannot plot against the frame they passed in is not a denoised frame:
    it differs from ``W`` by a PER-CHANNEL affine map, which no single rescale repairs, and it
    comes back the same shape as ``W`` whenever the entropy fold is the identity, so the error is
    silent.  ``info`` reports the ``centre`` and ``scale`` that were applied.

    The whitening is undone; the entropy FOLD is not.  The fold is what makes the screen the
    screen, it is reported in ``screen_shape``, and inverting it would be the synthesis this
    filter does not do.  Where the feature axis folded, ``centre`` and ``scale`` are the fold
    group's -- see :meth:`projection.Projection._screen_units` for why that is the right map for
    what ``clean`` actually carries.

    What this means for the mode cuts, and it is not a caveat but the definition: ``clean`` is
    the per-channel BASELINE plus the resolved modes.  The filter is a statement about modes, and
    a channel's median is not one -- it is removed by ``normalize`` before the SVD ever runs, so
    no cut here was ever offered the chance to reject it, and it is carried through untouched.  A
    persistent narrowband tone therefore has its MODULATION dropped by the ``phi_F <= phi_T`` cut
    while its DC level stays in the channel's baseline.  This is what makes ``W - clean`` mean
    "what the filter discarded" rather than "what the filter discarded, plus a baseline it never
    looked at".  A caller who wants the baseline gone as well is asking for baseline estimation,
    which is a different read and not this one."""
    U, S, Vt = sc.U, sc.S, sc.Vt
    T, F = sc.screen.shape
    Sd = gavish_donoho(S, sc.noise_floor, T, F) if shrink else np.where(S > sc.noise_floor, S, 0.0)
    kept, dropped, phis = [], [], []
    for k in np.nonzero(Sd > 0)[0]:
        pT, pF = mode_fill(U[:, k]), mode_fill(Vt[k, :])
        phis.append((int(k), pT, pF))
        if reject_persistent and pF <= pT:            # persistent + narrowband -> drop it
            Sd[k] = 0.0
            dropped.append(int(k))
        else:
            kept.append(int(k))
    centre, scale = sc.centre, sc.scale
    clean = ((U * Sd) @ Vt) * scale[None, :] + centre[None, :]   # back to the caller's units
    info = {
        "K_signal": int(sc.K_signal),
        "contrast": float(sc.sigma_top / sc.noise_floor) if sc.noise_floor > 0 else 0.0,
        "coherence": float(sc.coherence),
        "screen_shape": (int(T), int(F)),
        "n_kept": len(kept), "n_dropped": len(dropped),
        "kept": kept, "dropped": dropped, "phis": phis,
        # The whitening map that was applied -- reported so the read is auditable and so a
        # caller who wants the whitened screen back can divide it out, not so there is a
        # second way to call this.
        "centre": centre, "scale": scale,
    }
    return clean, info
