"""gate.py -- a measurement may only improve on its baseline.

A metric is ``{"value", "se", "better"}``: ``better`` is ``"higher"``, ``"lower"``, or ``"bound"``
for a level that must hold rather than improve (then ``"bound"`` names the level).  ``se`` is the
metric's own sampling uncertainty.  :func:`compare` holds a candidate set of metrics against a
baseline set and fails it when any metric is worse than its baseline by more than the two
measurements' combined noise can explain:

    higher  candidate < baseline - z * sqrt(se_b^2 + se_c^2)
    lower   candidate > baseline + z * sqrt(se_b^2 + se_c^2)
    bound   candidate > bound    + z * se_c

and when a baseline metric is missing from the candidate (what is measured may not shrink).  ``z``
is the normal quantile at ``far / M`` for ``M`` compared metrics: the family of comparisons errs at
most ``far`` of the time on a measurement that did not change.  The tolerance is the measurements'
own noise; no margin is chosen.  Domain-free: the metrics are the caller's.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

__all__ = ["compare", "GateResult"]


@dataclass
class GateResult:
    """The verdict of :func:`compare`: ``passed``, the ``z`` it was taken at, and the metrics that
    moved beyond the noise (``worse`` / ``better``: ``(name, baseline, candidate, noise)``) or went
    missing."""
    passed:  bool
    z:       float
    worse:   list = field(default_factory=list)
    better:  list = field(default_factory=list)
    missing: list = field(default_factory=list)


def _metrics(m):
    return m["metrics"] if isinstance(m, dict) and "metrics" in m else m


def compare(baseline: dict, candidate: dict, *, far: float = 0.05) -> GateResult:
    """Hold ``candidate`` against ``baseline`` (each a ``{name: metric}`` mapping, or a document with
    a ``"metrics"`` key) at family-wise level ``far``.  See the module docstring for the rule."""
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    base, cand = _metrics(baseline), _metrics(candidate)
    M = max(len(base), 1)
    z = NormalDist().inv_cdf(1.0 - far / M)
    res = GateResult(passed=True, z=z)
    for name, b in sorted(base.items()):
        c = cand.get(name)
        if c is None:
            res.missing.append(name)
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
        if diff < -tol:
            res.worse.append((name, b["value"], c["value"], tol))
        elif diff > tol:
            res.better.append((name, b["value"], c["value"], tol))
    res.passed = not res.worse and not res.missing
    return res
