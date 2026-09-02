"""Golden / contract tests: pin the full optics read so nothing drifts silently.

Regenerate the golden with ``python src/tests/golden/_generate.py`` only when a change
to the reads is intended and reviewed.
"""
import json
from pathlib import Path

import pytest

from entroptics import aperture as A
from conftest import build_W

GOLDEN = json.loads((Path(__file__).parent / "golden" / "optics.json").read_text())


def _assert_match(got: dict, exp: dict):
    assert set(got) == set(exp), f"key set drift: {set(got) ^ set(exp)}"
    for k, v in exp.items():
        if isinstance(v, bool):
            assert bool(got[k]) == v, f"{k}: {got[k]} != {v}"
        elif isinstance(v, int):
            assert got[k] == v, f"{k}: {got[k]} != {v}"
        else:
            assert got[k] == pytest.approx(v, rel=1e-8, abs=1e-10), f"{k}: {got[k]} != {v}"


@pytest.mark.parametrize("seed", [7, 3])
def test_optics_golden(seed):
    got = A.Aperture(build_W(seed)).optics()
    _assert_match(got, GOLDEN[str(seed)]["optics"])


@pytest.mark.parametrize("seed", [7, 3])
def test_screen_golden(seed):
    sc = A.Aperture(build_W(seed)).projection()
    exp = GOLDEN[str(seed)]["screen"]
    assert sc.K_signal == exp["K_signal"]
    assert sc.coherence == pytest.approx(exp["coherence"], rel=1e-8, abs=1e-10)
    assert sc.noise_floor == pytest.approx(exp["noise_floor"], rel=1e-8, abs=1e-10)
    assert sc.sigma_top == pytest.approx(exp["sigma_top"], rel=1e-8, abs=1e-10)


def test_free_and_method_optics_agree():
    """reads.optics(W) and Aperture(W).optics() share one canonical schema."""
    from entroptics.reads import optics as reads_optics
    W = build_W(7)
    a = A.Aperture(W).optics()
    b = reads_optics(W)
    assert set(a) == set(b)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12) if isinstance(a[k], float) else a[k] == b[k]
