"""
fields.py -- N-D field handling: the geometry-preserving reduction to the 2-D screen.

``Aperture`` and ``Projection`` are strictly 2-D -- the screen is a two-axis aperture.
A higher-dimensional field (a video T x H x W, a volume, a multichannel series, any
tensor field) must be reduced to 2-D before it can be read.  The naive reduction --
flatten every non-ordered axis into one feature axis -- destroys the within-plane
correlation and can silently invert a read.  This module is the bridge: it keeps each
screen plane intact -- use it in place of a bare ``reshape``.

Two reductions, and which one is correct depends on the read:

  * ``over_planes`` / ``slabs`` -- keep each plane intact, iterate the other axes and
    aggregate.  The correct reduction for feature / plane reads (``phi_F``,
    ``spectral_optics``, ``concentration``): each plane is its own screen.
  * ``pool`` -- flatten all non-ordered axes into the feature axis, so every off-axis
    site becomes a sample of the same ordered-axis process.  The correct reduction for
    ordered reads (``phi_T``, ``decay``, ``rates``): the sites are samples of one decay.

Getting these backwards changes the answer, so the choice lives here (documented).  Backend-agnostic (numpy or torch).
"""
from __future__ import annotations

import numpy as np

from . import environment as _env
from .reads import phi

_REDUCERS = {"mean": np.mean, "median": np.median, "max": np.max,
             "min": np.min, "sum": np.sum}


def slabs(field, plane_axes):
    """Yield each 2-D ``(plane_axes[0], plane_axes[1])`` slice of an N-D ``field``,
    iterating over all other axes -- geometry-preserving (each plane stays intact).
    ``plane_axes`` are the two axes that form the screen plane.  Backend-agnostic:
    each yielded plane is in ``field``'s backend (numpy or torch)."""
    xp = _env.ns(field)
    nd = len(field.shape)
    a, b = int(plane_axes[0]) % nd, int(plane_axes[1]) % nd
    if a == b:
        raise ValueError(f"plane_axes must be two distinct axes; got {plane_axes}")
    other = [ax for ax in range(nd) if ax not in (a, b)]
    P = _env.permute(xp, field, [a, b] + other)          # (La, Lb, *rest)
    La, Lb = int(P.shape[0]), int(P.shape[1])
    n_slab = 1
    for s in P.shape[2:]:
        n_slab *= int(s)
    R = P.reshape(La, Lb, n_slab)
    for k in range(n_slab):
        yield R[:, :, k]


def over_planes(field, plane_axes, read=None, reduce: str = "mean") -> float:
    """Apply a scalar 2-D ``read`` to each plane of ``field`` (default ``phi``) and
    aggregate over the slabs.  ``reduce`` in {mean, median, max, min, sum}.  The
    geometry-preserving reduction for feature / plane reads: each plane is read as its
    own screen, so within-plane correlation is preserved.  Returns a Python float
    (nan if the field has no planes)."""
    read = phi if read is None else read
    if reduce not in _REDUCERS:
        raise ValueError(f"reduce must be one of {sorted(_REDUCERS)}; got {reduce!r}")
    vals = [float(read(sl)) for sl in slabs(field, plane_axes)]
    if not vals:
        return float("nan")
    return float(_REDUCERS[reduce](vals))


def pool(field, ordered_axis: int):
    """Flatten all non-ordered axes of an N-D ``field`` into the feature axis: return a
    2-D ``(L_ordered, F')`` array whose rows are the ordered index and whose columns
    pool every off-axis site as a sample of the same ordered-axis process.  The correct
    reduction for ordered reads (``phi_T``, ``decay``, ``rates``) -- not for feature /
    plane reads (use ``over_planes`` there).  Backend-agnostic."""
    xp = _env.ns(field)
    o = int(ordered_axis) % len(field.shape)
    P = _env.movedim(xp, field, o, 0)                    # ordered axis first
    return P.reshape(int(P.shape[0]), -1)                # (L_ordered, F')
