"""MAD_SCALE is a literal. This is the derivation that says the literal is right.

`entropy.py` carries `MAD_SCALE = 1/Phi^{-1}(0.75)` as a float64 literal rather than computing it
at import. Until 2026-09-02 it computed it, via `from scipy.stats import norm` at module level --
which cost **5 seconds on every import of entroptics** (scipy.stats is 5.0 s cold and accounts for
4.4 s of the 5.4 s that `-X importtime` attributes to the package) to produce a number the file
already contained, bit for bit.

That branch was also not a check. It USED scipy's value when scipy was present and the literal
when it was absent, so the two were never compared and a disagreement would have been adopted in
silence. This is the comparison it was meant to be, in the place a comparison belongs.

The constant propagates through the per-channel whitening into `noise_sigma2`, into the floor, and
out through `k`, so it is asserted EXACTLY rather than to a tolerance: a 5-dp abbreviation is
~1.5e-7 relative and moves resolved ranks.
"""
import pytest

from entroptics.entropy import MAD_SCALE


def test_the_literal_is_what_scipy_derives():
    scipy_stats = pytest.importorskip("scipy.stats")
    derived = float(1.0 / scipy_stats.norm.ppf(0.75))
    assert MAD_SCALE == derived, (
        f"MAD_SCALE literal {MAD_SCALE!r} disagrees with 1/Phi^-1(0.75) = {derived!r}. This is a "
        f"finding to act on, not a tolerance to widen: the constant reaches the noise floor and "
        f"the resolved rank."
    )


def test_importing_entroptics_does_not_pull_in_scipy_stats():
    """The point of the change. If something reintroduces a module-level scipy import, this fails
    and the 5 seconds come back unnoticed."""
    import subprocess
    import sys

    code = "import entroptics, sys; print('scipy.stats' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", (
        "importing entroptics pulled in scipy.stats, which costs ~5 s of process start for a "
        "constant that is a literal in entropy.py"
    )


def test_the_constant_is_the_full_float64_value():
    """Guards the precision itself: a shortened literal would still pass a loose comparison."""
    assert repr(MAD_SCALE) == "1.482602218505602"
