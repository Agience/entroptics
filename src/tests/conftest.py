"""Shared test fixtures for the Entroptics suite.

Puts ``src/`` on the import path so ``import entroptics`` resolves without an
editable install, and provides reproducible signal factories.
"""
# ── the BLAS thread pin, again — because the package pin cannot reach a test suite ───────────────
#
# `entroptics/__init__.py` pins OpenBLAS to one thread, and that is worth nothing here. Of the 23
# test files in this suite that import numpy, all 23 import it before they import entroptics —
# including this conftest, which imports numpy below. OpenBLAS sizes its pool when the library
# loads, so by the time any `import entroptics` runs the pool is already 8 wide and the package
# pin is inert.
#
# A run that depends on whoever remembered to export the variable is unreliable in a way testing
# cannot catch: the defect hangs as often as it faults, so an unpinned run and a pinned run can both
# come back green.
#
# conftest.py is imported by pytest before the test modules in its directory, so this is the first
# opportunity in-process. `setdefault`, for the same reason as in the package: an operator who set a
# value keeps it.
import os as _os

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
del _os

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/ (tests live under src/tests)


def build_W(seed: int = 7, T: int = 80, F: int = 48, amp: float = 0.6) -> np.ndarray:
    """A reproducible structured (T, F) waterfall: iid noise + one slow drift along
    the ordered axis.  The single canonical test signal (also used by the golden
    generator, so goldens and tests stay in lock-step)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, F)) + amp * np.sin(np.linspace(0, 25, T))[:, None]


def build_Wc(seed: int = 11, T: int = 64, F: int = 32) -> np.ndarray:
    """A reproducible complex (T, F) field."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, F)) + 1j * rng.standard_normal((T, F))


@pytest.fixture
def make_W():
    return build_W


@pytest.fixture
def W():
    return build_W(7)


@pytest.fixture
def Wc():
    return build_Wc(11)
