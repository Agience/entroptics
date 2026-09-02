"""Put the repo's local ``src/`` ahead of any installed ``entroptics`` on sys.path.

The validation is run against the tip of the working tree, not the published wheel,
so a bump under development is exercised before it ships.  Import this module before
importing ``entroptics`` anywhere in the validation suite (it mirrors what
``src/tests/conftest.py`` does for pytest).  Side-effect import: no public API.
"""
import os

# ---- the BLAS thread pin, before anything can import numpy -------------------------------
#
# This module is imported first by every experiment, which makes it the only place the pin
# reaches BOTH `run_all.py` and a single `python expN_*.py`.  It used to live only in
# run_all.py, so running one script by hand did not reproduce the committed RESULTS.md:
# OpenBLAS splits a reduction across its pool, so the summation order -- and the last bit of
# every eigen/SVD read -- depends on how many threads the pool happens to have.  Measured, one
# run at the default pool against one pinned to a single thread: exp2 |alpha err| 3.5e-16 vs
# 4.5e-16, exp9 conservation 0.0e+00 vs 2.0e-16.  Each is reproducible on its own and they
# disagree with each other, which is the failure a seed cannot catch.
#
# `entroptics/__init__` sets this too and cannot help: OpenBLAS sizes its pool when the library
# loads, and every experiment reaches numpy before it reaches entroptics.  `setdefault`, so an
# operator who chose a value keeps it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys

# _bootstrap.py lives in research/validation/ -> repo root is two levels up.
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
