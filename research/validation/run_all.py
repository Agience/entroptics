"""
run_all.py -- run every entroptics validation experiment and (re)write RESULTS.md.

Deterministic: every experiment is seeded AND the BLAS pool is pinned below, so RESULTS.md is
byte-reproducible.  Seeding alone is not enough -- see the pin.
    python run_all.py            # run all, rewrite RESULTS.md

Each experiment module exposes run() -> dict(title, setup, table, headline,
conclusion, metrics, [provisional]).  This driver only formats; the numbers are
produced by the experiment modules and their common.py generators.
"""
from __future__ import annotations

import os

# ---- the BLAS thread pin, before anything can import numpy -------------------------------
#
# RESULTS.md is committed and diffed, so it has to be reproducible across machines, and
# seeding does not get there on its own: OpenBLAS splits a reduction across its pool, so the
# summation order -- and the last bit of every eigen/SVD read -- depends on how many threads
# the pool happens to have.  Measured here, one run at the default pool against one pinned to
# a single thread: exp2 |alpha err| 3.5e-16 vs 4.5e-16, exp9 conservation 0.0e+00 vs 2.0e-16.
# Each is reproducible on its own and they disagree with each other, which is the failure a
# seed cannot catch.
#
# `entroptics/__init__` sets this too, and that pin cannot help here: OpenBLAS sizes its pool
# when the library loads, and every module below reaches numpy before it reaches entroptics.
# This is the same reason `src/tests/conftest.py` repeats the pin for the test suite; this is
# the other entry point that writes an artifact meant to be compared byte for byte.
# `setdefault`, so an operator who chose a value keeps it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import datetime as _dt
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import exp1_correlation_length as e1
import exp2_dmd_recovery as e2
import exp3_planted_modes as e3
import exp4_coherence as e4
import exp5_mercer_stationarity as e5
import exp6_etendue_sbw as e6
import exp7_coupling as e7
import exp8_fold_continuity as e8
import exp9_screen_crossing as e9
import exp10_koopman_lift as e10
import exp11_sequence_order as e11
import exp12_scale_profile as e12
import exp13_measured_extent as e13
import exp14_scale_invariance as e14
import exp15_decay_scatter as e15
import exp16_scale_before_whitening as e16
import exp17_rank_baselines as e17
import exp18_coloured_null as e18
import miller_madow_check as mm

MODULES = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16,
           e17, e18, mm]

def _environment() -> str:
    """The environment this run happened in.

    RESULTS.md is committed and diffed, and its numbers are BLAS-pool dependent (see the pin in
    ``_bootstrap``).  A reader comparing a diff needs to know whether they are looking at drift
    or at a different machine, so the run stamps itself.
    """
    import numpy as _np
    import platform as _pl
    try:
        import entroptics as _e
        ver = getattr(_e, "__version__", "unknown")
    except Exception:                                   # pragma: no cover - reporting only
        ver = "unimportable"
    try:
        blas = _np.__config__.show(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:                                   # pragma: no cover - numpy-version dependent
        blas = "unknown"
    return (f"Produced {_dt.date.today().isoformat()} by `python research/validation/run_all.py` "
            f"with entroptics {ver} (numpy backend), numpy {_np.__version__}, BLAS {blas}, "
            f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}, "
            f"Python {_pl.python_version()} on {_pl.system()} {_pl.machine()}.")


INTRO = """# Entroptics -- empirical validation

Quantitative ground-truth evidence for the reads in the entroptics paper.  Each
experiment **plants a KNOWN ground truth** (a correlation length, a linear operator's
eigenvalues, a mode count, ordered vs permuted structure, stationary vs regime-switch,
a rank/bandwidth) and shows the corresponding read **recovers it**.  Everything is
seeded and deterministic; regenerate with `python research/validation/run_all.py`.

Scripts: `common.py` (seeded ground-truth generators), `exp1..exp18_*.py`,
`miller_madow_check.py`.

{ENVIRONMENT}
"""


def main():
    t0 = time.time()
    results = [m.run() for m in MODULES]

    lines = [INTRO.replace("{ENVIRONMENT}", _environment()), "\n## Headline numbers\n"]
    for r in results:
        tag = " *(provisional; see note)*" if r.get("provisional") else ""
        lines.append(f"- **{r['title']}**{tag} -- {r['headline']}")

    lines.append("\n---\n")
    for r in results:
        lines.append(f"\n## {r['title']}\n")
        lines.append(f"**Setup.** {r['setup']}\n")
        lines.append(r["table"])
        lines.append(f"\n**Conclusion.** {r['conclusion']}\n")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
