"""
lift.py -- the Koopman observable lift: turn a nonlinear trajectory into a linear operator.

Linear Koopman / DMD (:class:`entroptics.dynamics.Dynamics`) fits ``x_{t+1} = A x_t``.  For a
nonlinear domain -- text / symbol trajectories, near-orthogonal token embeddings -- a single-frame
state resolves no operator (near-orthogonal vectors have no linear one-step map, and the screen of
such a trajectory resolves no modes until lifted): the dynamics live in a higher-dimensional
observable space.  The blessed lift is delay embedding (Takens / Hankel-DMD / HAVOK): stack ``d``
consecutive frames as the observable, so a linear operator in the delay coordinates approximates the
nonlinear flow -- the same construction as ``Projection.tensor`` (delay-embedded Tucker) but pointed at
the streaming :class:`Dynamics` operator instead of a within-window HOSVD.

    from entroptics import koopman_lift, delay_embed

    dyn = koopman_lift(W, d=16)          # W: (T, F) trajectory  ->  Dynamics fitted on delay coords
    dyn.rates()                          # exact per-mode decay rates / frequencies in the lift
    z0 = delay_embed(W, 16)[-1]          # the current delay-coordinate state
    dyn.predict(z0, steps=h)             # h-step forecast (spectral: exact mu^h, no A^h blowup)
    dyn.rollout(z0, h)                   # the whole predicted delay-coordinate trajectory

The recipe, one line:  trajectory (T, F) -> ``delay_embed(W, d)`` -> ``dynamics(...)`` ->
``rates()`` / ``predict(steps=h)`` / ``rollout(h)``.  Takens' condition is ``d`` at least twice
the intrinsic dimension, which a signal does not announce, so the depth is stated.

What the lift buys is the forecast.  The resolved-mode count is not the measure of it: the count
rises with the observable's dimension ``d*F`` and with whiteness, so a time-shuffled trajectory
can resolve more modes than the real one (measured in research/validation/exp10: 15 against 5).
Forecast against a shuffled control separates dynamics from disorder; a count does not.
Backend-agnostic (numpy on CPU, torch on its device).
"""
from __future__ import annotations

import numpy as np

from . import environment as _env
from .dynamics import Dynamics, dynamics   # noqa: F401 (Dynamics re-exported for convenience)


def delay_embed(W, d: int):
    """Delay-embed a trajectory ``W`` ``(T, F)`` into Hankel observables ``(T-d+1, d*F)``: row ``t``
    is the window ``[w_t, w_{t+1}, ..., w_{t+d-1}]`` flattened -- the Takens / Hankel coordinates in
    which nonlinear dynamics become approximately linear (Koopman).  ``d = 1`` returns ``W``
    unchanged.  Backend-agnostic (numpy or torch, on its device)."""
    xp = _env.ns(W)
    if len(getattr(W, "shape", ())) != 2:
        raise ValueError(f"W must be 2-D (T, F); got shape {getattr(W, 'shape', None)}")
    T, F = int(W.shape[0]), int(W.shape[1])
    d = int(d)
    if d < 1:
        raise ValueError("delay depth d must be >= 1")
    if T < d:
        raise ValueError(f"trajectory length {T} < delay depth {d}")
    if d == 1:
        return W
    m = T - d + 1
    cols = [W[i:i + m] for i in range(d)]                     # cols[i] row t = w_{t+i}, each (m, F)
    return xp.cat(cols, dim=1) if _env.is_torch(xp) else np.concatenate(cols, axis=1)   # (m, d*F)


def koopman_lift(W, d: int, *, forgetting: float = 1.0, rank: int | None = None,
                 far: float = 0.05, null=None) -> Dynamics:
    """Fit a :class:`Dynamics` operator on the delay-embedded trajectory -- the blessed lift from a
    nonlinear ``(T, F)`` trajectory to a linear Koopman operator.  Equivalent to
    ``dynamics(delay_embed(W, d), ...)``; the returned operator's ``rates`` / ``predict(steps=h)`` /
    ``rollout`` / ``resolved`` act in the ``d*F`` delay coordinates.  ``far`` / ``null`` set the
    DMD-truncation operating point (the resolved-mode count of the lifted operator).

    ``d`` is the depth of the observable: ``delay_embed`` builds a different frame and this reads
    it, so the depth belongs to the signal a caller submits."""
    Z = delay_embed(W, d)
    return dynamics(Z, forgetting=forgetting, rank=rank, far=far, null=null)
