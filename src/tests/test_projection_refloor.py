"""`Projection.refloor(null)` must read EXACTLY as a projection built with that null.

`refloor` exists to remove a redundant `svdvals` from `sweep(null="local")`, which built a second
whole `Projection` of every coherent patch only to move its noise floor. That is only admissible if
it is an identity, so this file compares the two constructions field by field rather than checking
that the fast one merely looks reasonable.

The comparison is EXACT (`==`, not `approx`) on purpose: both paths are handed the same `self.S`,
so any difference would be a structural mistake -- a field left stale, a cache carried that should
not have been -- and not floating-point drift. A tolerance here would hide the only bug class this
test exists to catch.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics.projection import Projection
from entroptics.null_providers import reference_null, robust


def _screen(seed: int = 0, T: int = 96, F: int = 192) -> np.ndarray:
    """A screen with resolved structure, so `K_signal > 0` and the floor actually bites."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(T, F))
    t = np.exp(-((np.arange(T) - T / 3) ** 2) / (2 * (T / 12) ** 2))
    W[:, F // 4:F // 2] += 5.0 * np.outer(t, np.ones(F // 4))
    return np.abs(W)


def _providers():
    """Each admissible null shape: the analytic default, a robust provider, and a reference null
    calibrated on sampled top-mode values -- which is the one `sweep(null="local")` actually uses."""
    rng = np.random.default_rng(7)
    yield "mp-default", None
    yield "robust", robust
    yield "reference", reference_null(np.abs(rng.normal(size=32)) * 3.0, far=0.05)


@pytest.mark.parametrize("name,null", list(_providers()), ids=lambda v: v if isinstance(v, str) else "")
def test_refloor_reads_the_same_as_a_fresh_projection(name, null):
    W = _screen()
    base = Projection(W, far=0.05)          # built with NO null, as the sweep's scan pass does
    fast = base.refloor(null)
    slow = Projection(W, far=0.05, null=null)

    assert fast.noise_floor == slow.noise_floor, f"{name}: floor differs"
    assert fast.K_signal == slow.K_signal, f"{name}: K_signal differs"
    assert fast.H_screen == slow.H_screen, f"{name}: H_screen differs"
    assert fast.sigma_top == slow.sigma_top
    assert fast.has_signal == slow.has_signal
    assert fast.null is null
    np.testing.assert_array_equal(fast.S, slow.S)
    np.testing.assert_array_equal(fast.screen, slow.screen)
    assert fast.coherence == slow.coherence
    # the derived reads that stand on K_signal must follow, with nothing to invalidate
    assert len(fast.footprints) == len(slow.footprints)
    assert fast.read() == slow.read()


def test_refloor_actually_moves_the_floor():
    """The guard on the test above: if every provider produced the same floor, the equality
    assertions would pass on a `refloor` that ignored its argument entirely."""
    W = _screen()
    base = Projection(W, far=0.05)
    moved = base.refloor(reference_null(np.full(32, 0.001), far=0.05))
    assert moved.noise_floor != base.noise_floor, (
        "a provider pinned far below the data must move the floor; if it does not, this "
        "file cannot detect a refloor that drops its argument")


def test_refloor_does_not_disturb_the_projection_it_came_from():
    """`refloor` shares the screen and the spectrum by reference. It must not write through
    them -- the scan pass keeps using the original after re-flooring it."""
    W = _screen()
    base = Projection(W, far=0.05)
    floor_before, k_before = base.noise_floor, base.K_signal
    S_before = np.array(base.S, copy=True)

    base.refloor(robust)

    assert base.noise_floor == floor_before
    assert base.K_signal == k_before
    assert base.null is None
    np.testing.assert_array_equal(base.S, S_before)


def test_refloor_carries_the_basis_cache_without_recomputing_it():
    """The full SVD basis is a factorization of the screen; the floor only decides how many of
    its modes are resolved. A re-floored projection may therefore inherit it -- and must, or the
    saving is given back at the first `footprints` access."""
    W = _screen()
    base = Projection(W, far=0.05)
    _ = base.U                                        # force the heavy basis once
    fast = base.refloor(robust)
    assert fast._U is not None and fast._Vt is not None, "basis cache was dropped"
    assert fast._U is base._U, "basis was recomputed rather than shared"
