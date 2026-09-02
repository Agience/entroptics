"""
beam.py -- the beam: what a signal carries through an aperture and lands on a screen.

A beam is the force carrier of its region, and a bundle of beams: it decomposes into ``modes``,
each itself a :class:`Beam`, down to a leaf that spans one direction.  The same record reads at
every depth, which is what lets a mode be split off, passed along, and placed on another screen.

Its fields sit on a cost ladder, so the expensive ones resolve on access and a beam is cheap to
obtain.  At ``T=800, D=64``:

    energy, flow       the frame's own power                           ~0.3 ms
    basis, profile     which directions, and the amplitude on them     carried with the beam
    phi_T, phi_F       the axis fills, hence ``etendue``               ~51 ms (a T x T
                       eigendecomposition, inherent to the read)
    modes              the constituent beams                           ~18 ms

So there is one read per side -- ``Screen.beam(lens)`` -- and touching a field pays for that
field alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Beam:
    """What one side carries -- the force carrier of this region, and a bundle of beams.

    A beam is a string in the string-theory sense: the carrier itself, and an extended object.
    It decomposes into ``modes``, and **each mode is a Beam**, so the same record describes a
    side, a beam, and a single resolved direction.  The recursion bottoms out where a beam
    resolves one direction and its ``modes`` is empty.

    ``etendue == phi_T * phi_F`` by construction, at every depth: a mode occupies its own patch
    of phase space exactly as the whole beam occupies its own.  The modes span the resolved
    sector and etendue is an area, so overlapping modes share area, and each level reports its
    own energy and its own area on its own terms.

    Beams superpose: several placed on one screen add, which is what ``Screen.resolution``
    computes.  Where a beam meets a boundary its energy splits by direction of travel, which is
    what ``Transfer`` reports.
    """
    lens:    str          # the side carrying it
    index:   int          # position within the parent bundle (-1 for a whole side)
    energy:  float        # total energy, by its own law, about its own zero
    flow:    np.ndarray   # (T,) that energy per ordered-axis step
    basis:   np.ndarray   # (D, k) the directions it spans, unit-norm; (D, 1) at a leaf
    profile: np.ndarray   # (T, k) the amplitude carried along each of those directions
    # Resolved on access: a value once read, a zero-argument callable until then.
    _fills:  object = field(default=None, repr=False, compare=False)
    _modes:  object = field(default=None, repr=False, compare=False)

    # ── the axis fills, and the phase-space area they imply ──────────────────
    def _fill_pair(self):
        if callable(self._fills):
            self._fills = tuple(float(v) for v in self._fills())
        return self._fills if self._fills is not None else (0.0, 0.0)

    @property
    def phi_T(self) -> float:
        """Ordered-axis fill of this beam."""
        return self._fill_pair()[0]

    @property
    def phi_F(self) -> float:
        """Feature-axis fill of this beam."""
        return self._fill_pair()[1]

    @property
    def etendue(self) -> float:
        """``phi_T * phi_F``: the phase space this beam occupies, and the conserved invariant a
        crossing is settled by.  Derived, so the identity holds at every depth by construction."""
        pt, pf = self._fill_pair()
        return pt * pf

    # ── the bundle ───────────────────────────────────────────────────────────
    @property
    def modes(self) -> list:
        """The constituent beams, strongest first; empty at a leaf."""
        if callable(self._modes):
            self._modes = list(self._modes())
        return self._modes if self._modes is not None else []

    @property
    def is_leaf(self) -> bool:
        """True where this beam resolves one direction and decomposes no further."""
        return len(self.modes) == 0

    # ── the signal itself ────────────────────────────────────────────────────
    @property
    def direction(self) -> np.ndarray:
        """The single direction a leaf beam spans."""
        return self.basis[:, 0]

    @property
    def frame(self) -> np.ndarray:
        """The ``(T, D)`` frame this beam is, on the screen's coordinates -- its amplitude laid
        back along its directions, ``profile @ basis^H``.

        This is what makes beams split and merge.  A beam's directions are unit-norm, so the
        basis is pure information -- which concepts it occupies -- and the profile is the
        amplitude riding them.  Splitting a mode off is reading ``modes[k].frame``; merging is
        summing frames; and the two are exact inverses, because ``frame`` equals the sum of its
        modes' frames by construction (an orthonormal decomposition).  A frame taken off one
        screen is placeable on any other through a lens.

        What the split conserves is the resolved sector: the modes span what stands above the
        floor, so their frames sum to this one and their energies sum to its resolved energy."""
        if int(self.basis.shape[1]) == 0:
            return np.zeros((int(self.flow.shape[0]), int(self.basis.shape[0])))
        return self.profile @ self.basis.conj().T


__all__ = ["Beam"]
