"""gate.py -- a measurement may only improve on its baseline.

A metric is ``{"value", "se", "better"}``: ``better`` is ``"higher"``, ``"lower"``, or ``"bound"``
for a level that must hold rather than improve (then ``"bound"`` names the level).  ``se`` is the
metric's own sampling uncertainty.  :func:`compare` holds a candidate set of metrics against a
baseline set and fails it when any metric is worse than its baseline by more than the two
measurements' combined noise can explain:

    higher  candidate < baseline - z * sqrt(se_b^2 + se_c^2)
    lower   candidate > baseline + z * sqrt(se_b^2 + se_c^2)
    bound   candidate > bound    + z * se_c

and when a baseline metric is missing from the candidate (what is measured may not shrink).

A loss spread thinly over many metrics passes every one of those tests and is still a loss: a floor
that finds a weak signal half as often as before moves each cell of a sensitivity ladder by less
than its noise.  So the metrics that moved are also tested together.  Every ``higher`` / ``lower``
metric whose value changed is a draw from the change: a change that is no worse moves each such
metric up as often as down, so the number moved the worse way is at most Binomial(n, 1/2), and the
candidate fails when that many or more is a rarer event than the level allows (an exact tail, no
normal approximation).  A ``bound`` metric is held to its bound alone.

A power is only a reference at the level it was measured at.  A metric may name the ``bound`` metric
it was measured under (``"level"``: a detection rate names the false-alarm rate of the same read on
the same noise); where the baseline's own level broke its bound -- above it by more than ``z`` of
its noise -- the baseline's rate holds false alarms as well as detections, and is no reference.
That comparison is set aside and listed (``unreferenced``), not taken.

The two families -- each metric against its noise, and the direction of the moved ones -- share
the level: each is taken at ``far / 2``, and ``z`` is the normal quantile at ``far / (2 M)`` for ``M``
compared metrics, so the gate errs at most ``far`` of the time on a measurement that did not change.
The tolerance is the measurements' own noise; no margin is chosen.  Domain-free: the metrics are the
caller's.  A metric that a seeded measurement reproduces exactly does not move at all when the
library does not change, so on the release baseline every moved metric is the change's own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from statistics import NormalDist

__all__ = ["compare", "GateResult"]


@dataclass
class GateResult:
    """The verdict of :func:`compare`: ``passed``, the ``z`` it was taken at, and the metrics that
    moved beyond the noise (``worse`` / ``better``: ``(name, baseline, candidate, noise)``) or went
    missing, and the comparisons set aside because the baseline broke the level its metric names
    (``unreferenced``: ``(name, level)``).  ``moved`` is ``(worse, better, p)``: how many ``higher`` / ``lower`` metrics changed the
    worse and the better way, and the chance of at least that many the worse way from a change that
    is no worse; ``drift`` is whether that chance is within the level (a loss spread over many
    metrics)."""
    passed:  bool
    z:       float
    worse:   list = field(default_factory=list)
    better:  list = field(default_factory=list)
    missing: list = field(default_factory=list)
    unreferenced: list = field(default_factory=list)
    moved:   tuple = (0, 0, 1.0)
    drift:   bool = False


def _metrics(m):
    return m["metrics"] if isinstance(m, dict) and "metrics" in m else m


def compare(baseline: dict, candidate: dict, *, far: float = 0.05) -> GateResult:
    """Hold ``candidate`` against ``baseline`` (each a ``{name: metric}`` mapping, or a document with
    a ``"metrics"`` key) at family-wise level ``far``.  See the module docstring for the rule."""
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    base, cand = _metrics(baseline), _metrics(candidate)
    M = max(len(base), 1)
    z = NormalDist().inv_cdf(1.0 - far / (2 * M))
    res = GateResult(passed=True, z=z)
    n_worse = n_better = 0

    def broke(level):
        lv = base.get(level)
        return lv is not None and lv["value"] > lv["bound"] + z * lv["se"]
    for name, b in sorted(base.items()):
        c = cand.get(name)
        if c is None:
            res.missing.append(name)
            continue
        if b.get("level") is not None and broke(b["level"]):
            res.unreferenced.append((name, b["level"]))
            continue
        if b["better"] == "bound":
            lim = b["bound"] + z * c["se"]
            if c["value"] > lim:
                res.worse.append((name, b["bound"], c["value"], lim))
            continue
        tol = z * math.sqrt(b["se"] ** 2 + c["se"] ** 2)
        diff = c["value"] - b["value"]
        if b["better"] == "lower":
            diff = -diff
        n_worse += diff < 0
        n_better += diff > 0
        if diff < -tol:
            res.worse.append((name, b["value"], c["value"], tol))
        elif diff > tol:
            res.better.append((name, b["value"], c["value"], tol))
    n = n_worse + n_better
    tail = sum(math.comb(n, j) for j in range(n_worse, n + 1))          # P(W >= worse) * 2^n
    res.moved = (n_worse, n_better, tail / 2 ** n if n else 1.0)
    res.drift = n > 0 and 2 * tail <= Fraction(far) * 2 ** n           # P <= far / 2
    res.passed = not res.worse and not res.missing and not res.drift
    return res
