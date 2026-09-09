"""Check every number quoted in PAPER_frb.md against the tables reproduce.py wrote.

    python research/supplemental/frb/reproduce.py   # writes the tables
    python research/supplemental/frb/verify.py      # checks the prose against them

Exits non-zero if any claim in the paper has drifted from the table behind it.  This is the
mechanical half of "no number appears in the paper unless a script regenerates it": reproduce.py
makes the numbers, and this asserts the paper still says what they say.

The claims are transcribed here by hand, which is the point -- if someone edits a figure in the
prose without re-running the script, this is what catches it.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"

if not (TABLES / "agreement.csv").is_file():
    raise SystemExit(f"No tables under {TABLES}.  Run reproduce.py first.")

_ag = list(csv.DictReader(open(TABLES / "agreement.csv")))
# the last row is the full-precision mean over the four events, written by reproduce.py.
# The paper quotes THAT, not the mean of the four printed 3-decimal values -- the two differ
# in the third decimal, which is where 0.215-vs-0.214 came from.
means = {k: float(v) for k, v in _ag[-1].items() if k != "event" and v}
agreement = [r for r in _ag if r["event"] != "mean"]
# frb_panel.csv carries a trailing full-precision mean row for the correlation columns;
# the per-event checks below are over the events only.
events = [r for r in csv.DictReader(open(TABLES / "events.csv")) if r["event"] != "mean"]
_sp = list(csv.reader(open(TABLES / "spotcheck.csv")))
spot = [dict(zip(_sp[1], r)) for r in _sp[2:]]          # row 0 is the provenance comment


def _col(rows, k):
    return [float(r[k]) for r in rows]


def _rng(rows, k):
    v = _col(rows, k)
    return min(v), max(v)


def _mean(rows, k):
    v = _col(rows, k)
    return sum(v) / len(v)


_ok = _fail = 0


def check(label, claimed, actual, tol=5e-4):
    """One claim in the paper against one number in a table."""
    global _ok, _fail
    good = abs(claimed - actual) <= tol
    _ok += good
    _fail += not good
    print(f"  [{'OK ' if good else 'FAIL'}] {label:50s} paper {claimed:<10g} table {actual:.4f}")


# -- section 3.3, table 2: correlation against the forward model -------------------------
lo, hi = _rng(agreement, "corr_raw_model")
check("raw waterfall, range", 0.115, lo)
check("raw waterfall, range", 0.373, hi)
check("raw waterfall, mean", 0.198, means["corr_raw_model"], 5e-4)

lo, hi = _rng(agreement, "corr_rebinned_model")
check("box-average control, range", 0.118, lo)
check("box-average control, range", 0.427, hi)
check("box-average control, mean", 0.214, means["corr_rebinned_model"], 5e-4)

lo, hi = _rng(agreement, "corr_read_model")
check("the read, range", 0.515, lo)
check("the read, range", 0.609, hi)
check("the read, mean", 0.560, means["corr_read_model"], 5e-4)

# -- section 3.3: the premise of the baseline removal -------------------------------------
check("model per-channel baseline, worst (1e-6)", 3.9,
      1e6 * max(_col(agreement, "model_baseline_max")), 0.05)
lo, hi = _rng(agreement, "model_max")
check("model maximum, range", 0.108, lo)
check("model maximum, range", 0.506, hi)

# -- section 4, table 3: the two reference points ----------------------------------------
lo, hi = _rng(agreement, "ref_noise_limited")
check("noise-limited reference, range", 0.117, lo)
check("noise-limited reference, range", 0.439, hi)
check("noise-limited reference, mean", 0.219, means["ref_noise_limited"], 5e-4)

lo, hi = _rng(agreement, "ref_fold_ceiling")
check("fold-resolution ceiling, range", 0.9853, lo)
check("fold-resolution ceiling, range", 0.9987, hi)
check("fold-resolution ceiling, mean", 0.994, means["ref_fold_ceiling"], 5e-4)

# -- section 4, quantities derived in the prose ------------------------------------------
check("on-burst raw, mean", 0.297, means["corr_raw_model_onburst"], 5e-4)
check("on-burst read, mean", 0.468, means["corr_read_model_onburst"], 5e-4)
check("read / raw, full frame", 2.8,
      means["corr_read_model"] / means["corr_raw_model"], 0.05)
check("read / raw, on-burst", 1.6,
      means["corr_read_model_onburst"] / means["corr_raw_model_onburst"], 0.05)
check("fold cost, worst event (%)", 1.5, 100 * (1 - min(_col(agreement, "ref_fold_ceiling"))), 0.05)
check("fold cost, best event (%)", 0.1, 100 * (1 - max(_col(agreement, "ref_fold_ceiling"))), 0.05)
_gain = means["corr_read_model"] - means["corr_raw_model"]
_fold = means["corr_rebinned_model"] - means["corr_raw_model"]
check("fold's share of the gain (%)", 4.6, 100 * _fold / _gain, 0.05)

# -- section 3.1, table 1: the per-burst reads -------------------------------------------
lo, hi = _rng(events, "n_live")
check("live channels, range", 9760, lo, 0.5)
check("live channels, range", 11696, hi, 0.5)
lo, hi = _rng(events, "n_F_read")
check("width read at, range", 5949, lo, 0.5)
check("width read at, range", 10972, hi, 0.5)
lo, hi = _rng(events, "contrast")
check("four-event contrast, range", 1.30, lo)
check("four-event contrast, range", 4.22, hi)
lo, hi = _rng(events, "T")
check("four-event T, range", 19, lo, 0.5)
check("four-event T, range", 38, hi, 0.5)
check("four events on the figure", 4, len(events), 0.5)

# -- section 3.2: the random spot-check --------------------------------------------------
resolved = [r for r in spot if int(r["K_signal"]) >= 1]
blank = [r for r in spot if int(r["K_signal"]) == 0]
check("waterfalls drawn", 12, len(spot), 0.5)
check("resolved K_signal >= 1", 8, len(resolved), 0.5)
lo, hi = _rng(blank, "contrast")
check("unresolved contrast, range", 0.98, lo)
check("unresolved contrast, range", 0.99, hi)
lo, hi = _rng(blank, "T")
check("unresolved T, range", 19, lo, 0.5)
check("unresolved T, range", 57, hi, 0.5)
lo, hi = _rng(resolved, "T")
check("resolved T, range", 19, lo, 0.5)
check("resolved T, range", 95, hi, 0.5)

print(f"\n{_ok} checks passed, {_fail} failed")
sys.exit(1 if _fail else 0)
