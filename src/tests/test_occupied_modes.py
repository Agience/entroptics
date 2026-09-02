"""occupied_modes: the rank edge of an ordered spectrum.

The two counts that already exist answer a filled aperture badly.  ``phi`` (equivalently 2^H) is
the entropic effective mode count and is EXACT for an occupied set at one level, but entropy
counts every mode carrying anything, so noise in the empty modes lifts it.
``resolved_dimension`` counts modes standing above a noise bulk, which needs a bulk to stand
above, so it fades as the aperture fills.  The edge is invariant to both: occupied modes sit at
one level and empty ones at another, and a step does not care how many modes lie on each side.
"""
from __future__ import annotations

import numpy as np
import pytest

from entroptics import occupied_modes
from entroptics.entropy import shannon_bits


@pytest.mark.parametrize("k", [1, 2, 10, 32, 60, 63])
def test_reads_the_occupancy_of_a_filled_then_empty_spectrum(k):
    """The shape the read exists for, at every occupancy including nearly-full."""
    w = np.r_[np.ones(k), np.zeros(64 - k)]
    o = occupied_modes(w)
    assert o.k == k
    assert o.margin > 1.0, "a hard edge must report itself as one"


def test_a_flat_spectrum_is_fully_occupied_and_says_there_is_no_edge():
    """Every mode as lit as every other: there is no boundary to find, and inventing one would
    be worse than reporting the truth.  ``margin == 1`` is the read declining to guess."""
    o = occupied_modes(np.ones(64))
    assert o.k == 64
    assert o.margin == 1.0 and o.step == 0.0


def test_a_smooth_decay_reports_no_edge():
    """A geometric spectrum has the SAME step everywhere, so no step is the boundary.  The count
    is meaningless here and the margin is what says so -- the reader is told, not misled."""
    o = occupied_modes(2.0 ** -np.arange(64))
    assert o.margin == pytest.approx(1.0)


def test_an_empty_spectrum_occupies_nothing():
    o = occupied_modes(np.zeros(32))
    assert o.k == 0 and np.isnan(o.margin)


def test_it_is_invariant_to_scale_and_to_ordering():
    """A spectrum is a multiset: the read must not depend on the order it arrives in, nor on the
    units it was measured in."""
    rng = np.random.default_rng(0)
    w = np.r_[rng.uniform(0.8, 1.2, 12), np.full(52, 1e-6)]
    base = occupied_modes(w)
    assert occupied_modes(rng.permutation(w)).k == base.k
    assert occupied_modes(w * 1e7).k == base.k


def test_it_survives_noise_in_the_empty_modes_where_the_entropic_count_does_not():
    """The claim that earns the read its place.  Fill 10 of 64 and put noise in the rest: 2^H
    climbs monotonically with the noise, because entropy counts every mode carrying anything,
    while the edge does not move at all, because a step is a ratio."""
    rng = np.random.default_rng(0)
    k, n = 10, 64
    effective = []
    for noise in (1e-8, 1e-6, 1e-4, 1e-2):
        w = np.r_[np.ones(k), rng.uniform(0, noise, n - k)]
        assert occupied_modes(w).k == k, f"the edge moved at noise={noise}"
        effective.append(2.0 ** shannon_bits(w))
    assert effective == sorted(effective), f"2^H should climb with noise, got {effective}"
    assert effective[0] == pytest.approx(k, abs=1e-3), "and start exact when the rest is empty"
    assert effective[-1] > k, f"and end above it, got {effective[-1]:.2f} against k={k}"


def test_one_mode_and_one_weight_are_not_errors():
    assert occupied_modes(np.r_[1.0, np.zeros(63)]).k == 1
    assert occupied_modes(np.array([5.0])).k == 1
    assert occupied_modes(np.array([])).k == 0
