"""Shared test fixtures for the Entroptics suite.

Puts ``src/`` on the import path so ``import entroptics`` resolves without an
editable install, and provides reproducible signal factories.
"""
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
