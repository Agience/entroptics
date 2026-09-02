"""extract.py -- the parameter-free shrinker/geometry for the read-side FILTER.

The FILTER itself is a method of the front door -- :meth:`entroptics.Aperture.extract` -- because a
clean view is a PROJECTION of the aperture's own resolved screen modes onto the data, never a
synthesis.  This module holds the two parameter-free pieces that method uses:

  1. Optimal shrinkage (Gavish & Donoho 2017, "Optimal Shrinkage of Singular Values") of the
     singular values against the derived floor -- the minimum-MSE low-rank estimate.  Noise modes
     are attenuated to zero and the surviving signal modes are de-biased toward the floor.

  2. The entropy footprint fill ``mode_fill`` -- the discriminator for the geometry cut: a transient
     event is SPREAD across features but COMPACT along the ordered axis (phi_F > phi_T); persistent
     narrowband structure is the opposite (phi_F <= phi_T).

plus a thin functional shortcut ``extract(W)`` == ``Aperture(W, window=None).extract()`` for callers
who want the full frame in one call.  Everything still funnels through the aperture front door.
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


def extract(W: np.ndarray, *, far: float = 0.05, reject_persistent: bool = True, shrink: bool = True):
    """Functional shortcut for ``Aperture(W, window=None).extract(...)`` -- routes through the front
    door on the FULL frame (no coherence windowing).  See :meth:`entroptics.Aperture.extract`."""
    from .aperture import Aperture
    return Aperture(W, window=None).extract(far=far, reject_persistent=reject_persistent, shrink=shrink)
