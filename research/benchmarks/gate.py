"""gate.py -- a release may only improve on the baseline.

    python gate.py baseline_metrics.json candidate_metrics.json [more pairs ...]

Each file is the output of ``baseline.py`` or ``cost.py``.  Pass the committed baseline files and
the candidate's, in pairs (baseline, candidate).  The rule is the library's own,
:func:`entroptics.gate.compare`: a candidate fails when any metric is worse than its baseline by
more than the two measurements' combined noise can explain, or when a baseline metric is missing;
``z`` is the normal quantile at ``far / M`` for ``M`` metrics, ``far = 0.05``.  Exit status 0 on pass,
1 on fail; every regression and improvement is listed.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from entroptics.gate import compare  # noqa: E402


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["metrics"]


def main(argv):
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    base, cand = {}, {}
    for b, c in zip(argv[0::2], argv[1::2]):
        base.update(_load(b))
        cand.update(_load(c))
    r = compare(base, cand)
    print(f"{len(base)} metrics, z = {r.z:.3f} (family-wise far = 0.05)")
    for n, bv, cv, t in r.better:
        print(f"  better   {n}: {bv:.6g} -> {cv:.6g}  (noise {t:.3g})")
    for n, bv, cv, t in r.worse:
        print(f"  WORSE    {n}: {bv:.6g} -> {cv:.6g}  (noise {t:.3g})")
    for n in r.missing:
        print(f"  MISSING  {n}")
    print("PASS" if r.passed else "FAIL")
    return 0 if r.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
