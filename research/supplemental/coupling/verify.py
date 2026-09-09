"""Check every number quoted in PAPER_coupling.md against the tables reproduce.py wrote.

    python research/supplemental/coupling/reproduce.py   # writes the tables
    python research/supplemental/coupling/verify.py      # checks the prose against them

Exits non-zero if any claim in the paper has drifted from the table behind it.  The claims are
transcribed here by hand, which is the point: if a figure in the prose is edited without
re-running the script, this is what catches it.

Requires a FULL reproduce.py run.  After ``--quick`` the 100k and 200k columns are empty and
the checks that need them are reported as missing rather than passed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"

if not (TABLES / "brute_force.csv").is_file():
    raise SystemExit(f"No tables under {TABLES}.  Run reproduce.py first.")

brute = list(csv.DictReader(open(TABLES / "brute_force.csv")))
sign = list(csv.DictReader(open(TABLES / "sign.csv")))
null = list(csv.DictReader(open(TABLES / "null.csv")))[0]
embed = {r["quantity"]: float(r["value"]) for r in csv.DictReader(open(TABLES / "embedding.csv"))}

_ok = _fail = _skip = 0


def check(label, claimed, actual, tol=5e-5):
    global _ok, _fail
    good = abs(claimed - actual) <= tol
    _ok += good
    _fail += not good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:48s} paper {claimed:<10g} table {actual:.4f}")


def missing(label):
    global _skip
    _skip += 1
    print(f"  [SKIP] {label:48s} needs a full run (no --quick)")


def row(shape):
    return next(r for r in brute if r["shape"] == shape)


def ratios(n):
    vs = [r[f"ratio_{n}"] for r in brute]
    return [float(v) for v in vs] if all(vs) else None


# -- section 2.5: the counterexample the real embedding exists for -----------------------
check("true Var[Re S] over both re-pairings", 0.0,
      embed["true Var[Re S] over all re-pairings"])
check("Hermitian Gram form reports", 4.0, embed["Hermitian Gram form"])
check("real embedding reports", 0.0, embed["real embedding (Theorem 2.2)"])

# -- section 5.1, table 1: the brute force ------------------------------------------------
for shape, closed in [("(40, 5)", 173.47), ("(64, 8)", 486.79), ("(120, 3)", 440.69),
                      ("(64, 6)", 791.68), ("(96, 4)", 780.68)]:
    check(f"closed form, {shape}", closed, float(row(shape)["closed_form"]), 5e-3)

for shape, r20 in [("(40, 5)", 1.0054), ("(64, 8)", 1.0094), ("(120, 3)", 1.0044),
                   ("(64, 6)", 1.0056), ("(96, 4)", 1.0291)]:
    check(f"ratio at 20k, {shape}", r20, float(row(shape)["ratio_20000"]))

if ratios(100_000) is None:
    missing("ratios at 100k")
else:
    for shape, r in [("(40, 5)", 1.0001), ("(64, 8)", 0.9987), ("(120, 3)", 1.0012),
                     ("(64, 6)", 0.9994), ("(96, 4)", 1.0197)]:
        check(f"ratio at 100k, {shape}", r, float(row(shape)["ratio_100000"]))
    # "within 2.0% at every shape tested", at 100k
    check("worst deviation at 100k (%)", 2.0,
          100 * max(abs(v - 1.0) for v in ratios(100_000)), 0.05)

if ratios(200_000) is None:
    missing("ratios at 200k")
else:
    for shape, r in [("(40, 5)", 1.0046), ("(64, 8)", 0.9985), ("(120, 3)", 1.0012),
                     ("(64, 6)", 0.9995), ("(96, 4)", 1.0088)]:
        check(f"ratio at 200k, {shape}", r, float(row(shape)["ratio_200000"]))

# the worst case across shapes, which is the sequence the paper quotes: 1.029, 1.020, 1.009
if ratios(100_000) and ratios(200_000):
    worst = [max(ratios(n)) for n in (20_000, 100_000, 200_000)]
    for claimed, actual, n in zip((1.029, 1.020, 1.009), worst, (20, 100, 200)):
        check(f"worst case over shapes at {n}k", claimed, actual, 5e-4)
    assert worst[0] > worst[1] > worst[2], "the worst case must fall monotonically"
    print("  [OK ] worst case falls monotonically with the draw count")
    _ok += 1
else:
    missing("worst-case convergence sequence")

# -- section 5.2, table 2: planted sign recovery ------------------------------------------
for rho, agree, strength, resolved in [("+1.0", 1.000, 0.693, 1.000),
                                       ("+0.5", 1.000, 0.508, 1.000),
                                       ("+0.0", 0.967, 0.003, 0.033),
                                       ("-0.5", 1.000, -0.497, 1.000),
                                       ("-1.0", 1.000, -0.678, 1.000)]:
    r = next(x for x in sign if x["planted_rho"] == rho)
    check(f"sign agreement, rho {rho}", agree, float(r["sign_agreement"]), 5e-4)
    check(f"mean strength, rho {rho}", strength, float(r["mean_strength"]), 5e-4)
    check(f"resolved rate, rho {rho}", resolved, float(r["resolved_rate"]), 5e-4)

# -- sections 4 and 5.3: null calibration --------------------------------------------------
check("null pairs", 1600, float(null["n_pairs"]), 0.5)
check("null mean z", 0.011, float(null["mean_z"]), 5e-4)
check("null std z", 1.018, float(null["std_z"]), 5e-4)
check("null firing rate", 0.0488, float(null["fire_rate"]), 5e-5)
check("nominal two-sided level", 0.05, float(null["nominal_far"]), 1e-9)

print(f"\n{_ok} checks passed, {_fail} failed, {_skip} skipped")
sys.exit(1 if _fail else 0)
