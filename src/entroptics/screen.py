"""screen.py -- the two-way screen: lenses meet on a shared basis and a signal entering from
either side can be read, coupled, converted or carried across to the other.

Vocabulary
----------

Four nouns, one job each, and they are the optical chain:

    [ aperture  >  beam ] --lens-->  screen  <--lens-- [ beam  <  aperture ]

  beam      What passes through: the carrier, with energy and etendue (:class:`Beam`, from
            ``Screen.beam(lens)``).  A string in the string-theory sense -- the carrier itself,
            and an extended object: it decomposes into ``modes``, each a resolved direction
            with its own energy and its own etendue.  Each mode is a :class:`Beam`, so the
            same three quantities -- how much, how much room, where -- describe a mode, a beam
            and a side.  A beam carries energy.  One per side.

  aperture  What bounds the beam, and the read of everything about it (:class:`aperture.
            Aperture`, from ``Screen.aperture(lens)``).  Its size is the beam's etendue, which
            is why a crossing is settled by comparing two of them.  Seen from the sending side
            it is the entrance pupil, from the receiving side the exit pupil; their ratio is
            ``Transfer.concentration``.

  lens      What converts (:class:`Lens`): ``entry`` (surface -> the screen's coordinates) and
            ``inverse`` (back out), plus that system's own laws (``energy`` / ``zero`` /
            ``null``).  The pair exists because an optical system is reciprocal -- run it
            backwards and you have the inverse.  Every line of domain code lives here and
            nowhere else in this file.  ``certify`` asks whether a lens is aberration-free;
            ``realise`` measures how close it comes to the etendue ceiling.

  screen    What receives: the surface beams land on and are read at (:class:`Screen`), whose
            coordinates are ``basis()``.  Where a beam carries, a screen is where carrying is
            read: it holds coordinates and no energy of its own.  One screen; one beam per
            side.  (:class:`projection.Projection` is the same noun for one signal -- a
            signal's own landing surface and the factorization on it.  This class is the
            surface shared between sides.)

Two more words, each a place in the chain:

  concept   A direction on the screen -- universal, the same point no matter who looks.  A
            column of ``basis()``.  ``dog`` entered from language and ``dog`` entered from
            vision are one concept.
  signal    A concept as one lens carries it -- the ``(T, D)`` frame a side places, and the
            surface it renders back out.  Directional: the same concept is a different signal
            from each side, which is the whole reason the conversion is per-lens.

A screen is viewed differently from each side
----------------------------------------------

Four nouns, and :class:`projection.Projection` is a fifth class and the same noun: it is
this screen as one side sees it, on that side's own entropy-matched grid.

    Screen()        the surface itself, in coordinates every side shares.  ``basis()`` is those
                    coordinates; ``directions(lens)`` is what one side resolves of them, which
                    is that side's view.
    Projection(W)   one signal on its own matched grid: whitened per channel, folded to its own
                    width, and read against a singular-value floor.  A distinct calibrated
                    pipeline -- the whitening is what makes that floor's Tracy-Widom edge hold
                    -- where a screen balances and reads against a correlation floor.

Both answer "what does this signal carry"; they differ in the grid and the floor, so their
K_signal agrees while their coherence does not.  A screen never folds, because it has to keep
the coordinates its sides agreed on; a projection folds precisely because it answers for one
side alone and can match that side's own scale.

Consuming: what crosses the boundary, and who owns the loop
--------------------------------------------------------------

The handoff is a frame.  A plain ``(T, D)`` array crosses in either direction and nothing
else has to: ``place(lens, surface)`` and ``update`` take one in, ``Beam.frame`` and
``uncondensed`` hand one back.  A beam split off one screen is a frame, so it places on any
other.  Everything inside the boundary is measurement; everything outside is the caller's.

The caller owns the loop.  These reads are pull: ask, and you get the reading as of now.  A
runner, an event bus, or a callback would put *when to read* and *what to do next* inside a
measurement instrument, and that is control-flow policy -- the same reason a ``mix`` hook stays
out and ``far`` stays with the reader.  The instrument answers questions; sequencing them is
the caller's program.

Noise is the system's, risk is the reader's.  A lens declares its own ``null`` -- a detector
and a market count signal differently, and the side that owns the physics owns the floor that
scores it.  The false-alarm level ``far`` is the reader's single decision input; by
Neyman-Pearson it belongs to whoever is asking.  So substrate-specific
noise is per lens, and appetite for risk is per read.

One read per side, priced by field.  ``beam(lens)`` is the read; its fields resolve on access,
so the answer costs what the question costs.  At ``T=800, D=64``::

    beam(lens)              ~3.8 ms    the beam itself
      .energy  .flow        included   how much, and how much per step
      .basis   .profile     included   which directions, and the amplitude on them
      .modes               ~3.6 ms     the constituent beams
      .etendue .phi_T/F    ~70 ms      a T x T eigendecomposition, inherent to the read

Reach for ``etendue`` when a crossing is in question, since that is what ``transfer`` settles
by; a monitor that wants energy alone never pays for it.

Streaming.  Ingest is incremental on the aperture: ``Aperture.update(frame)`` folds a frame
into the operator in ~0.03 ms and keeps a coherent window.  A screen is placed -- ``place`` replaces a side and re-reads it, ~2.3 ms per frame at ``T=200, D=8`` --
so a running stream drives the aperture and places on the screen at the cadence it wants a
crossing measured.

One beam, many screens
------------------------

A screen is one meeting.  A lens that meets several others in several places is registered
on several screens -- the same conversions, the same laws, a beam per meeting.  The topology
is however many screens you build, and each one measures its own crossing alone.

Why the reads are shaped this way
------------------------------------

  * **Un-folded.**  Every screen read runs on the native frame (the unit-diagonal
    correlation path).  The entropy fold sends a concentrated frame to ``F_eff = 1`` and
    ``K_signal = 0``; a screen must keep the measured basis it meets on.
  * **Measured coupling.**  What two sides do to each other is read (``reads.coupling``:
    signed, exact permutation null), never declared.
  * **A crossing is a nonimaging problem.**  ``certify`` is the imaging objective (reconstruct
    the surface); ``transfer`` is the nonimaging one (deliver the energy).  Imaging is
    the stricter requirement -- a lens can be an excellent conduit and a poor imager.  Etendue
    conservation is the invariant, ``tau = min(1, G_to/G_from)`` the transmissible fraction,
    and ``radiance_to <= radiance_from`` holds identically (the brightness theorem), with
    equality exactly at the concentration limit.
  * **Only matching signal is in an interaction.**  Energy off the receiver's resolved
    directions is ``bystanding`` -- present, real, and pertinent to some other pairing.
  * **Access is a threshold crossing.**  The structure follows the global
    workspace model (Dehaene & Naccache, *Cognition* 79 (2001) 1-37): lenses are the modular
    processors, the screen is the workspace, broadcast is one settled state rendered to any
    side, and the access (ignition) threshold is the derived noise floor.

Domain-agnostic, concretely: what each side owns, and what the screen keeps
--------------------------------------------------------------------------

Each side owns its conversion (``entry`` / ``inverse``), its energy law (``energy``), its zero
(``zero``) and its noise floor (``null``); the library's derived defaults apply only where a
system declares nothing.  Every energy and every fraction of an energy goes through that
side's own law, so a share is always a share of the quantity it is reported against.

Three things are the screen's and stay that way, each for a reason:

  * **Superposition.**  ``resolution`` sums the balanced sides as they are -- the linearity of a shared basis. A system needing nonlinear mixing belongs on a different basis, and a ``mix`` hook is where domain logic would come back in.
  * **Etendue.**  Measured off the frame, never declared.  It is the currency two sides are
    compared in; a side that could declare its own phase space could claim more brightness
    than it was sent, and the second-law bound would stop holding.
  * **``far``.**  The false-alarm level is the reader's one decision input (by Neyman-Pearson
    it belongs to the reader).  A side's ``null`` says what its noise is; ``far`` says how
    sure the reader wants to be.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import environment as _env
from .environment import ns as _ns
from .entropy import macheps
from .reads import Coupling, coupling, _centred
from .beam import Beam
from .projection import Projection, coherence as _coherence, _vector_fill
from .null_providers import _reg_gamma_upper


@dataclass
class Lens:
    """One side of a screen: its name, its own conversions, and its own laws.  A lens
    is exactly how it maps its surface into the shared basis, so the lens is its
    conversion -- and an emergent system carries its own physics with it, so it supplies the
    laws with it.

    ``entry``    surface -> (T, D) frame on the shared basis (forward; required).
    ``inverse``  (T, D) frame on the shared basis -> surface (required only to render out
                 through this side or to certify its losslessness).
    ``energy``   balanced frame -> ``(T,)`` energy per ordered-axis step: this system's
                 energy, by its own calculation.  ``None`` uses the library's derived read,
                 the frame's own power ``sum_d |x_td|^2`` -- correct for a system with no
                 separate law, and wrong for one that has (a price series carries its energy
                 somewhere other than its variance).  It receives the frame at the side's own ``zero``.
    ``zero``     frame -> the zero to remove: where this system balances, by its own law.
                 ``None`` uses the derived default, the column mean (the connected frame).
                 Must broadcast against ``(T, D)``.
    ``null``     the noise-floor provider that scores this side (a ``FloorContext -> float``
                 callback; see :mod:`null_providers`).  ``None`` inherits the screen's.
                 A detector and a market each count signal their own way, so the system
                 that owns the physics owns the floor that scores it.

    ``energy`` and ``zero`` are supplied exactly the way a null provider is (see
    :mod:`null_providers`): the system owns the law, the library owns the measurement."""
    name:    str
    entry:   Callable[[Any], Any]
    inverse: Callable[[Any], Any] | None = None
    energy:  Callable[[Any], Any] | None = None
    zero:    Callable[[Any], Any] | None = None
    null:    Any = None


@dataclass
class ScreenRead:
    """The aperture measurement of the screen's joint frame (see ``Screen.read``).
    Read on the un-folded frame: the spectral quantities are the unit-diagonal correlation
    reads, so a concentrated frame is never collapsed away."""
    n_lenses:          int     # sides placed on the screen
    T:                 int     # shared ordered-axis length
    D:                 int     # shared-basis dimension (per side)
    K_signal:          int     # resolved modes of the joint frame (= spectral resolved_modes)
    contrast:          float   # top correlation eigenvalue / noise floor (>1 => structure)
    top_share:         float   # dominant-mode power fraction of the joint frame
    noise_floor:       float   # the correlation-edge floor the count was taken against
    attenuation:       float   # attenuation constant alpha of the joint dominant mode
    coherence:         float   # ordered-axis coherence z-score of the balanced joint frame
    a_delta:           float   # diffraction limit from the joint frame's own decay
    correlation_length: float  # xi, the integral correlation length of that decay
    basis_dim:         int     # k: dimension of the shared basis that resolves (0 => nothing)


@dataclass
class Balance:
    """The screen's self-balancing zero (see ``Screen.balance``).  ``frame`` is the joint
    frame with each side at ITS OWN zero.

    ``offsets`` says how much DC a side gave up to reach its zero.  ``residual`` says what its
    zero left behind, and ``pvalue``/``closed`` decide whether that remainder is what a
    balanced side would show anyway.  Under the derived default zero (the column mean) the
    residual is 0 by construction and every side reads closed; a side that declares its OWN
    zero is genuinely tested, and a zero placed elsewhere than where the system balances shows
    up here."""
    total:    float   # max |column sum| of the balanced frame (0 => it closes)
    offsets:  dict    # lens -> per-row RMS of the zero removed, in data units
    residual: dict    # lens -> ||column mean remaining after that zero||, in data units
    pvalue:   dict    # lens -> tail probability of that residual under no drift
    closed:   dict    # lens -> pvalue >= far: the side closes at the zero it declared
    frame:    object  # the balanced joint frame, or None across orders


@dataclass
class Transfer:
    """What crosses the screen from one side to the other (see ``Screen.transfer``).

    Two independent things gate the crossing, and confusing them loses the physics:

    **1. Only matching signal is in the interaction at all.**  Plenty of energy exists on a
    side that has nothing to do with this pairing.  The part that participates is the part
    lying along directions the receiving side actually resolves -- ``participation``, the
    energy fraction of the sender inside the receiver's resolved subspace (measured against
    the receiver's own derived floor, so participation runs over its resolved directions).  The
    remainder is ``bystanding``: present, real, and pertinent to some other pairing -- it stands by, entering no part of this one.

    **The two root behaviours.**  Wave energy meeting a boundary does exactly one of two
    things, and everything below is a sub-case of these:

        absorption     the energy stops propagating and becomes structure on the other side.
                       Here that is condensation: it resolves into the receiver's concepts.
                       ``absorbed`` (== ``delivered``, itemised by ``condensation``).
        transmission   the energy continues propagating.  ``transmitted``, and it continues
                       one of two ways -- forward as ``bystanding`` (it never coupled: the
                       receiver resolves nothing matching, so it passes straight through
                       intact, retrievable as a frame from ``Screen.uncondensed``) or
                       backward as ``reflected`` (it coupled and exceeded the receiver's
                       phase space, so it is turned back).

    Reflection is transmission with a direction flip, so absorption alone removes energy from
    the propagating field.  ``flux`` carries that flip as a sign: forward-going ``bystanding``
    counts positive and backward-going ``reflected`` counts negative, so ``flux`` is the net
    energy still travelling in the direction the beam arrived from.  Absorption leaves the
    field entirely and carries no direction, which is why ``energy == absorbed +
    transmitted`` is the conservation that matters and the four-way split below merely says
    how.  (``reflected`` is a capacity shortfall: the etendue bound says how much fits and
    leaves the directions open, so ``bystanding`` carries a frame and ``reflected`` carries an
    energy alone.)

    **What a crossing is.**  A condensation; most systems have no response to give.  The energy
    that crosses condenses as a concept in the receiving lens: it arrives along directions that
    side resolves, and becomes structure there.  ``condensation`` reports which concepts it matched into and how much matched into each, so
    "the cat is meowing" arrives as energy condensing into a concept the observer's lens already
    resolves.  Those energies sum to ``pertinent`` for a quadratic law (an orthogonal
    decomposition), and each agrees with its own ``frame``.  The capacity limit ``tau`` scales
    the total that crosses: the etendue bound fixes how much fits and leaves open which concepts
    it favours, so ``delivered`` is reported for the crossing as a whole.

    **2. Of what does participate, only what fits crosses.**  The carrying capacity of a
    screen is its etendue ``G = phi_T * phi_F`` -- ordered fill times featured fill, the
    conserved invariant of a lossless system (the discrete Smith-Helmholtz / Lagrange
    invariant).  The transmissible fraction is the etendue that fits, ``tau = min(1, G_to /
    G_from)``; the pertinent energy beyond that fraction is ``reflected``.  This is the
    brightness theorem: a passive screen preserves the phase space a signal arrived with, so
    pushing a wide signal into a narrow side costs exactly the ratio.

    The accounting closes exactly, at both levels:
    ``energy == absorbed + transmitted`` and ``transmitted == bystanding + reflected``.

    **Concentration and dilution are the two directions of the same ratio.**  This is the
    nonimaging optics problem exactly -- optimal radiative transfer from a source to a
    target, where forming an image of the source is neither required nor helpful.
    ``concentration = G_from / G_to`` is that field's figure of merit: ``> 1`` the signal is
    being squeezed into less phase space (and ``tau`` caps it at the etendue limit -- what
    exceeds it is reflected); ``< 1`` it is being diluted, and all the energy crosses but
    spreads, so radiance falls by exactly that factor.

    ``radiance = pertinent energy / etendue`` is the brightness, and the brightness theorem
    holds identically here: ``radiance_to <= radiance_from`` for every crossing, with
    equality exactly when concentrating at the etendue limit (the ideal concentrator).  A
    passive screen leaves a side at most as bright as the side that fed it -- the same
    second-law ceiling nonimaging optics derives its concentration limit from.

    Full transfer both ways holds when ``G_from == G_to`` -- the two sides are
    etendue-matched, which is what it means for the screen to be balanced between them.
    ``match`` is that ratio in ``(0, 1]``; ``1`` is the matched screen."""
    absorbed:      float   # root 1: energy that stopped here and became structure (= delivered)
    transmitted:   float   # root 2: energy still propagating (= bystanding + reflected)
    flux:          float   # signed against propagation: bystanding (forward) - reflected (back)
    energy:        float   # the sending side's total energy, by its own law
    participation: float   # in [0,1]: the fraction of it the receiving side can resolve at all
    pertinent:     float   # energy * participation: the energy actually in this interaction
    tau:           float   # min(1, G_to/G_from) in (0,1]: the fraction of that which fits
    delivered:     float   # pertinent * tau: what crosses
    reflected:     float   # pertinent * (1 - tau): pertinent energy with nowhere to fit
    bystanding:    float   # energy - pertinent: present, but not part of this interaction
    etendue_from:  float   # G of the sending side (phi_T * phi_F of its balanced frame)
    etendue_to:    float   # G of the receiving side
    match:         float   # min(G)/max(G) in (0,1]: 1 == matched, full transfer both ways
    concentration: float   # G_from/G_to: >1 concentrating (etendue-limited), <1 diluting
    radiance_from: float   # pertinent / G_from -- the brightness offered
    radiance_to:   float   # delivered / G_to  -- the brightness received (<= radiance_from)
    modes_from:    int     # resolved dimension the sending side carries
    modes_to:      int     # resolved dimension the receiving side carries
    condensation:  list    # list[Beam]: which concepts it matched into, and how much matched


@dataclass
class Realisation:
    """What a crossing actually delivers through the receiving lens's own conversion, against
    what the etendue bound permits (see ``Screen.realise``).

    ``Transfer`` answers what CAN cross -- a bound set by the two beams' phase space, with no
    conversion in it.  A real lens reaches that bound only if its ``inverse`` puts the energy
    where its ``entry`` can pick it up again.  The shortfall belongs to the lens, and it is exactly what the nonimaging design methods (CPC, flow-line, SMS) exist to remove.

        realised = ideal * efficiency,   ideal = the etendue-bounded `Transfer.delivered`

    ``efficiency`` above 1 is a real finding about that lens: the registered conversion puts
    out more than it took in, so it is active, and the value is reported as measured."""
    ideal:      float   # Transfer.delivered -- what the etendue bound permits
    realised:   float   # what survives the receiver's own inverse . entry
    efficiency: float   # realised / ideal in [0, 1] for a passive conversion
    shortfall:  float   # ideal - realised: energy the conversion lost (phase space aside)
    passive:    bool    # efficiency <= 1: the conversion returns no more than it was given


@dataclass
class Linearity:
    """Whether a lens passes a beam's modes independently (see ``Screen.linear``).

    A lens acts on the whole frame, and nothing in it is per-mode.  One lens therefore serves a
    beam of any number of modes -- because it is linear, so each mode transforms on its own.
    That is what lets a mode be split off, converted, and recombined, and what
    ``Screen.resolution``'s superposition rests on.

    Two departures, measured on the side's own signal:
    ``additivity`` compares converting the modes together against converting them separately,
    and ``homogeneity`` compares converting a scaled frame against scaling the conversion.  Both
    are measured in balance, as every screen read is, so an affine conversion -- which departs
    by a constant its side's zero absorbs -- reads as the linear one the screen sees.
    ``linear`` holds where neither departure resolves structure on its own screen -- the same
    derived decision ``certify`` uses."""
    additivity:  float   # ||entry(sum modes) - sum entry(mode)|| / ||entry(sum modes)||
    homogeneity: float   # ||entry(cX) - c entry(X)|| / ||c entry(X)||
    modes:       int     # modes the additivity check had to work with (it needs 2)
    linear:      bool    # neither departure carries resolvable structure


@dataclass
class Losslessness:
    """The conversion certificate of one lens (see ``Screen.certify``): is
    ``inverse(entry(surface))`` the surface again, within the resolution the aperture
    reports?  The decision is the conjunction of two derived checks, neither of them a
    chosen tolerance:

      1. the residual resolves nothing -- ``K_signal == 0`` on the residual's own screen, so every structure the instrument could see survived the round trip;
      2. the round trip beats the null conversion -- ``residual < 1``, the residual of the trivial inverse that returns nothing -- a reference point the instrument supplies.

    Check 1 alone would pass a conversion that replaced the surface with unstructured noise
    of any amplitude; check 2 alone would pass one that lost a resolved mode but kept the
    energy.  Both fields are reported either way."""
    residual:    float   # ||inverse(entry(x)) - x||_F / ||x||_F  (0.0 = perfect, 1.0 = the null)
    sigma_top:   float   # top singular value of the residual's screen
    noise_floor: float   # that screen's own singular-value noise floor
    K_signal:    int     # modes the residual resolves above it (0 = nothing was taken)
    lossless:    bool    # K_signal == 0 and residual < 1: nothing resolvable leaked


class Screen:
    """The two-way screen: lenses meeting on a shared basis, read from either side.

    ``far`` is the false-alarm level every read on this screen applies (the reader's one
    decision input); ``null`` is the noise-floor provider (``None`` = the derived ``mp``
    default -- see :mod:`null_providers`), ``seed`` makes a resampling provider
    deterministic.  All three are plumbed to every read, so the screen has ONE floor.
    """

    def __init__(self, *, far: float = 0.05, null=None, seed: int = 0) -> None:
        self._far = float(far)
        self._null = null
        self._seed = int(seed)
        self._lenses: dict[str, Lens] = {}
        self._placed: dict[str, Any] = {}
        # Per-side memo.  A side's balanced frame, resolved directions, etendue and energy are
        # each an eigendecomposition or two, and `transfer` alone wants eight of them; they are
        # pure functions of the placement, so they are computed once and dropped the moment
        # anything is placed, cleared or re-registered.
        self._cache: dict = {}

    def _c(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ── wiring a side in ──────────────────────────────────────────────────────
    def register(self, lens: str, *, entry: Callable, inverse: Callable | None = None,
                 energy: Callable | None = None, zero: Callable | None = None,
                 null=None) -> Lens:
        """Register a lens's own conversions and own laws: ``entry`` (surface -> shared
        basis), ``inverse`` (shared basis -> surface), ``energy`` (the system's energy per
        ordered-axis step, by its own calculation) and ``zero`` (where it balances, by its
        own law).  Registered once per lens; the screen applies each on the correct side,
        so ``N`` lenses carry ``N`` conversions and no pairwise table ever exists.

        Only ``entry`` is required.  Without ``inverse`` the side is entry-only: placeable, readable and couplable, with rendering reserved for the sides that carry one.  Without ``energy`` / ``zero`` the side
        inherits the library's derived defaults (frame power; the column mean; the
        screen's floor), which are the right laws for a system that has no others and the
        wrong ones for a system that does.  A system reading or writing several forces registers a lens per force.
        Re-registering a lens clears anything it had placed."""
        for label, fn, required in (("entry", entry, True), ("inverse", inverse, False),
                                    ("energy", energy, False), ("zero", zero, False)):
            if (required or fn is not None) and not callable(fn):
                raise ValueError(f"lens {lens!r}: {label} must be callable")
        if null is not None and not callable(null):
            raise ValueError(f"lens {lens!r}: null must be a FloorContext -> float provider")
        self._lenses[lens] = Lens(name=str(lens), entry=entry, inverse=inverse,
                                    energy=energy, zero=zero, null=null)
        self._placed.pop(lens, None)
        self._cache.clear()
        return self._lenses[lens]

    def _lens_of(self, lens: str) -> Lens:
        if lens not in self._lenses:
            raise KeyError(f"no lens {lens!r} registered; have {sorted(self._lenses)}")
        return self._lenses[lens]

    @property
    def lenses(self) -> list[str]:
        """The registered lenses, in registration order."""
        return list(self._lenses)

    @property
    def placed(self) -> list[str]:
        """The lenses currently holding a placement, in placement order."""
        return list(self._placed)

    # ── entering from a side ──────────────────────────────────────────────────
    def place(self, lens: str, surface) -> Any:
        """Enter from a side: convert ``surface`` through that lens's ``entry`` and place
        the resulting ``(T, D)`` frame on the screen.  Every side must land on the same
        shared basis (same ``D``) -- that is what makes the screen one meeting place.

        The ordered axis is each side's own: bank transactions are ordered by real
        time, language by information flow, and they still meet.  A side carried on its own
        order is containment, and it is ordinary here -- every read that meets on the basis
        works across it (``transfer``, ``beam``, ``directions``, ``basis``, ``energy``,
        ``balance``, ``render(g, concept)``).  Only the row-paired reads (``couple``,
        ``joint``, ``read``, ``resolution``) need a common order, and they say so."""
        g = self._lens_of(lens)
        X = g.entry(surface)
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) != 2:
            raise ValueError(f"lens {lens!r}: entry must return a 2-D (T, D) frame on the "
                             f"shared basis; got shape {getattr(X, 'shape', None)}")
        T, D = int(X.shape[0]), int(X.shape[1])
        for name, other in self._placed.items():
            if int(other.shape[1]) != D:
                raise ValueError(
                    f"lens {lens!r} places D={D} columns but {name!r} placed "
                    f"D={int(other.shape[1])}: sides meet on ONE shared basis. A lens's "
                    f"entry must land on that basis -- differing widths are two bases, and "
                    f"a coupling across them reports magnitude alone (see reads.coupling).")
        self._placed[lens] = X
        # A stateful null provider tracks what it is scoring: the same contract the streaming
        # aperture honours, so a floor that sharpens on its own local sample sharpens here too.
        if g.null is not None and hasattr(g.null, "update"):
            g.null.update(X)
        self._cache.clear()
        return X

    def update(self, lens: str, surface) -> Any:
        """Extend a side by further ordered steps, keeping what is there.

        ``place`` sets a side and invalidates every read on the screen, because a new side can
        move the shared coordinates.  ``update`` appends to one side and invalidates only what
        that side can change: the other sides keep their beams, apertures and energies, and only
        the shared reads (the basis and the joint frame) are dropped.  For a stream of frames
        against several standing sides, that is the difference between re-reading the screen and
        re-reading one side of it.

        Returns the side's extended frame."""
        g = self._lens_of(lens)
        X = g.entry(surface)
        if not hasattr(X, "shape"):
            X = np.asarray(X)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if len(X.shape) != 2:
            raise ValueError(f"lens {lens!r}: entry must return a 2-D (T, D) frame on the shared "
                             f"basis; got shape {getattr(X, 'shape', None)}")
        prev = self._placed.get(lens)
        if prev is not None:
            if int(X.shape[1]) != int(prev.shape[1]):
                raise ValueError(f"lens {lens!r} extends with D={int(X.shape[1])} but carries "
                                 f"D={int(prev.shape[1])}: a side keeps one basis as it grows.")
            X = _env.cat0(_ns(prev), [prev, X])
        else:
            for name, other in self._placed.items():
                if int(other.shape[1]) != int(X.shape[1]):
                    raise ValueError(
                        f"lens {lens!r} places D={int(X.shape[1])} but {name!r} placed "
                        f"D={int(other.shape[1])}: sides meet on ONE shared basis.")
        self._placed[lens] = X
        if g.null is not None and hasattr(g.null, "update"):
            g.null.update(X)
        self._invalidate(lens)
        return X

    def _invalidate(self, lens: str) -> None:
        """Drop what this side can change, and the shared reads that span every side."""
        for k in [k for k in self._cache
                  if (isinstance(k, tuple) and len(k) > 1 and k[1] == lens)]:
            self._cache.pop(k, None)
        for k in ("basis", "joint_ap"):
            self._cache.pop(k, None)

    def clear(self, lens: str | None = None) -> None:
        """Drop one lens's placement, or every placement (``None``).  The registered
        conversions are untouched."""
        if lens is None:
            self._placed.clear()
        else:
            self._placed.pop(lens, None)
        self._cache.clear()

    # ── the two frames ────────────────────────────────────────────────────────
    @property
    def orders(self) -> dict:
        """Each placed side's ordered-axis length.  Beams may each carry their own -- see ``place``."""
        return {g: int(X.shape[0]) for g, X in self._placed.items()}

    @property
    def shares_order(self) -> bool:
        """True iff every placed side is carried on the same ordered axis, so a row pairing
        between sides exists.  When False the sides still meet -- on the shared basis."""
        return len(set(self.orders.values())) <= 1

    def _require_common_order(self, read: str):
        """Guard the row-paired reads.  Never fit: the row pairing IS the quantity, so where
        the orders differ there is no reading to take."""
        if not self.shares_order:
            raise ValueError(
                f"{read} pairs the sides ROW BY ROW, so it needs one ordered axis; the placed "
                f"beams carry {self.orders}. Beams on different orders still MEET, on the "
                f"shared basis, which is order-free -- use transfer / beam / directions / "
                f"basis / energy / balance / render(lens, concept). Only couple, joint, read "
                f"and resolution need a common order.")

    def _require_placed(self):
        if not self._placed:
            raise ValueError("nothing placed on the screen; call place(lens, surface) first")
        return list(self._placed.values())

    def joint(self):
        """The joint frame ``(T, n*D)``: the placed sides side by side on the shared ordered
        axis.  Cross-lens structure is visible only here -- this is what ``read()`` and the
        couplings measure on."""
        self._require_common_order("joint()")
        parts = self._require_placed()
        return parts[0] if len(parts) == 1 else _env.cat1(_ns(parts[0]), parts)

    def balanced_joint(self):
        """The joint frame with each side brought to its own zero -- what every screen read
        actually measures on.  Side by side, each by its own law."""
        self._require_common_order("balanced_joint()")
        self._require_placed()
        parts = [self.balanced(g) for g in self._placed]
        return parts[0] if len(parts) == 1 else _env.cat1(_ns(parts[0]), parts)

    def pooled(self):
        """The pooled frame ``(n*T, D)``: the placed sides stacked in the shared basis, which
        is what the shared basis itself is read off (``basis()``).  A concept entered from
        two sides is the same point here -- the frame holds the concept itself, whichever lens
        placed it."""
        parts = self._require_placed()
        return parts[0] if len(parts) == 1 else _env.cat0(_ns(parts[0]), parts)

    # ── each side's OWN laws (the library supplies only the default) ──────────
    def _zero_of(self, lens: str):
        """The zero this side balances at, by its own ``zero`` law -- broadcast to ``(T, D)``.
        The derived default is the column mean (the connected frame)."""
        X = self._placed[lens]
        xp = _ns(X)
        Xf = _env.asnum(X)
        law = self._lenses[lens].zero
        if law is None:
            return _env.nanmean0(xp, Xf) + xp.zeros_like(Xf)
        z = law(Xf)
        if not hasattr(z, "shape"):
            z = np.asarray(z)
        try:
            return z + xp.zeros_like(Xf)                 # broadcast to the frame
        except Exception as exc:
            raise ValueError(f"lens {lens!r}: zero returned shape "
                             f"{getattr(z, 'shape', None)}, which fails to broadcast against "
                             f"the placed frame {tuple(int(s) for s in Xf.shape)}") from exc

    def balanced(self, lens: str):
        """One side's frame brought to its own zero -- the side as it sits on the screen.
        Every screen read runs on balanced frames, so a system that balances somewhere
        other than its arithmetic mean is read where IT balances."""
        return self._c(("balanced", lens), lambda: self._balance_frame(lens))

    def _balance_frame(self, lens: str):
        if lens not in self._placed:
            raise KeyError(f"lens {lens!r} has nothing placed; call place({lens!r}, surface) first")
        X = self._placed[lens]
        xp = _ns(X)
        if self._lenses[lens].zero is None:
            return _centred(xp, X)          # the library's ONE centring, shared with the reads
        diff = _env.asnum(X) - self._zero_of(lens)
        return xp.where(xp.isnan(diff), xp.zeros_like(diff), diff)

    def energy(self, lens: str | None = None):
        """The energy flow of a side: ``(T,)``, one value per ordered-axis step, by that
        system's own ``energy`` law.  ``None`` returns ``{lens: flow}`` for every placed
        side.  The derived default is the frame's own power ``sum_d |x_td|^2``.

        Measured on the balanced frame -- about the side's own zero, the same frame every
        other screen read uses.  It has to be: a fraction of the energy (``transfer``'s
        ``participation``) is measured there, so an energy measured anywhere else would not be
        the quantity that fraction is a fraction OF.  A system that means to measure energy
        about the raw origin instead declares ``zero`` returning 0 -- it owns that choice too.

        Energy is what makes two sides commensurable, and the coupling stays free of it:
        ``couple`` is scale-invariant by construction (a signed cosine), so a side's choice of
        units leaves the sign and the strength where they are.  That is deliberate -- the
        magnitude lives here, where each system reports it in its own terms, and the sign lives
        there, where it is unit-free."""
        if lens is None:
            return {g: self.energy(g) for g in self._placed}
        if lens not in self._placed:
            raise KeyError(f"lens {lens!r} has nothing placed; call place({lens!r}, surface) first")
        return self._c(("energy", lens),
                       lambda: self._energy_law(lens, self.balanced(lens)))

    def _energy_law(self, lens: str, frame):
        """This side's energy law applied to any frame in its own coordinates -- so the total
        and every fraction of it (``transfer``'s participation) are measured the same way.
        Routing both through here is what keeps an L1 law from being divided by an L2 share."""
        law = self._lenses[lens].energy
        if law is not None:
            e = law(frame)
            return e if hasattr(e, "shape") else np.asarray(e)
        xp = _ns(frame)
        return _env.sum_ax(xp, xp.abs(frame) ** 2, 1)    # derived: the frame's own power

    def _total_energy(self, lens: str, frame) -> float:
        return float(np.sum(np.asarray(_env.to_numpy(self._energy_law(lens, frame)))))

    def _null_of(self, lens: str):
        """The floor provider for ONE side: its own ``null`` if it declared one, else the
        screen's.  A system's noise model is its own -- a detector and a market each
        count signal their own way -- so the side that owns the physics owns the
        floor that scores it.  The screen's ``null`` still scores anything joint (the joint
        frame spans every side, so no single side's noise model governs it)."""
        g = self._lenses.get(lens)
        return self._null if (g is None or g.null is None) else g.null

    # ── the measurements ──────────────────────────────────────────────────────
    def _aperture_on(self, frame, null=None):
        """An :class:`aperture.Aperture` on a frame.  ``window=None`` because a placement is a whole
        whole frame -- all of it is the read; ``null`` is the floor that governs it and
        ``far`` the screen's level.  A side IS an aperture, so every optics read comes from
        one."""
        from .aperture import Aperture           # deferred: aperture imports screen
        return Aperture(frame, window=None, null=(self._null if null is None else null),
                        far=self._far, seed=self._seed)

    def _scatter(self, V, frame) -> np.ndarray:
        """Put a resolved basis back on the full shared basis (``principal_directions`` drops
        fully-dead columns) so it composes with a ``(T, D)`` frame in the caller's
        coordinates."""
        D = int(frame.shape[1])
        if int(V.shape[1]) == 0:
            return np.zeros((D, 0))
        if int(V.shape[0]) == D:
            return V
        bad = ~np.isfinite(np.abs(np.asarray(_env.to_numpy(frame))))
        live = np.nonzero(~bad.all(axis=0))[0]
        out = np.zeros((D, int(V.shape[1])), dtype=V.dtype)
        out[live] = V
        return out

    def aperture(self, lens: str):
        """The full :class:`aperture.Aperture` of ONE side, on its balanced frame and its own
        floor.  A side is an aperture plus its own laws, so the entire optics surface is here
        -- ``etendue``, ``phi_T`` / ``phi_F``, ``spectral``, ``decay``, ``rates()`` -- and the
        screen reads its own etendue and directions off exactly this object."""
        if lens not in self._placed:
            raise KeyError(f"lens {lens!r} has nothing placed; call place({lens!r}, surface) first")
        return self._c(("ap", lens),
                       lambda: self._aperture_on(self.balanced(lens), null=self._null_of(lens)))

    def basis(self) -> np.ndarray:
        """The screen -- the shared surface both sides land on, as the ``(D, k)`` coordinates it
        actually resolves.  A beam carries (it has energy and etendue); a screen is where
        carrying is read (it has coordinates and no energy of its own).  There is one screen
        and one beam per side.

        Concretely: the ``(D, k)`` directions of the
        pooled frame's resolved correlation modes (``reads.principal_directions``), ordered
        by descending eigenvalue.  Derived: ``k`` is what stands above the
        floor, and ``(D, 0)`` when nothing does.  Read on the un-folded correlation, so a
        concentrated frame keeps its carrier."""
        return self._c("basis", lambda: self._scatter(
            self._aperture_on(self.pooled()).principal_directions, self.pooled()))

    def directions(self, lens: str) -> np.ndarray:
        """The ``(D, k)`` directions one side resolves, on its own balanced frame, against its
        own floor (``_null_of``).  What this side can see -- and therefore what another side's
        energy has to lie along to be pertinent to an interaction with it (see ``transfer``)."""
        return self._c(("directions", lens),
                       lambda: self._scatter(self.aperture(lens).principal_directions,
                                             self.balanced(lens)))

    def _etendue_of(self, lens: str) -> float:
        """The phase space this side occupies -- off its own Aperture, which caches it."""
        return float(self.aperture(lens).etendue)

    def beam(self, lens: str | None = None):
        """The beam this side carries -- energy (how much), etendue (how much room it needs),
        basis (where it sits).  ``None`` returns ``{lens: beam}`` for every placed side.
        Everything else a screen reports about one side is derived from these three; a
        crossing (``transfer``) needs exactly them and nothing more."""
        if lens is None:
            return {g: self.beam(g) for g in self._placed}
        if lens not in self._placed:
            raise KeyError(f"lens {lens!r} has nothing placed; call place({lens!r}, surface) first")
        X = self.balanced(lens)
        flow = self._energy_law(lens, X)
        V = self.directions(lens)
        Vx = _env.asdtype_of(X, np.asarray(V)) if _env.is_torch(_ns(X)) else np.asarray(V)
        return Beam(lens=str(lens), index=-1,
                    energy=float(np.sum(np.asarray(_env.to_numpy(flow)))),
                    flow=flow, basis=V,
                    profile=np.asarray(_env.to_numpy(X @ Vx)),
                    # the fills need a T x T eigendecomposition, so they resolve on access
                    _fills=lambda: (self.aperture(lens).T.phi, self.aperture(lens).F.phi),
                    _modes=lambda: self._bundle(lens, X, V))

    def _bundle(self, lens: str, X, V) -> list:
        """Decompose a side into its constituent :class:`Beam` modes -- one per resolved
        direction, each with its own energy (through the side's own law) and its own etendue
        ``phi_T * phi_F``.  The per-mode fills are the ordinary entropic fills of the mode's
        ordered profile and of its direction (``screen._vector_fill``), read on the un-folded
        frame -- the per-mode footprint of ``Aperture.footprints``, without the fold."""
        xp = _ns(X)
        D = int(X.shape[1])
        out = []
        for k in range(int(V.shape[1])):
            v = np.asarray(V[:, k])
            vx = _env.asdtype_of(X, v) if _env.is_torch(xp) else v
            profile = X @ vx                             # (T,) this mode along the ordered axis
            only = xp.outer(profile, xp.conj(vx))        # the frame restricted to this mode
            pT = _vector_fill(xp, profile)
            pF = _vector_fill(xp, vx)
            e = float(np.sum(np.asarray(_env.to_numpy(self._energy_law(lens, only)))))
            out.append(Beam(lens=str(lens), index=k, energy=e,
                            flow=self._energy_law(lens, only), basis=v.reshape(-1, 1),
                            profile=np.asarray(_env.to_numpy(profile)).reshape(-1, 1),
                            _fills=(pT, pF), _modes=[]))
        return out

    def uncondensed(self, lens_from: str, lens_to: str):
        """What carries on -- the sending beam restricted to the directions outside what the
        receiver resolves, as a ``(T, D)`` frame ready to be placed on another screen.

        A crossing condenses where the receiver resolves a match.  Energy that meets a side which
        resolves nothing matching survives intact: it passes through, still carrying its information, and can go on to another screen where it may condense.
        This is the complementary projection to ``transfer``'s ``condensation``, so its energy
        is ``bystanding`` (exactly, for a quadratic law); placing it elsewhere is how a beam
        that found no home here continues to look for one."""
        for g in (lens_from, lens_to):
            if g not in self._placed:
                raise KeyError(f"lens {g!r} has nothing placed; call place({g!r}, surface) first")
        Xf = self.balanced(lens_from)
        Vt = self.directions(lens_to)
        if int(Vt.shape[1]) == 0:
            return Xf                                   # nothing resolved here: all of it goes on
        xp = _ns(Xf)
        Vx = _env.asdtype_of(Xf, np.asarray(Vt)) if _env.is_torch(xp) else np.asarray(Vt)
        return Xf - (Xf @ Vx) @ Vx.conj().T

    def transfer(self, lens_from: str, lens_to: str) -> Transfer:
        """The energy accounting of a crossing: how much of one side's energy reaches the
        other, and where the rest went.  See :class:`Transfer` -- the short form is that only
        the energy lying where the receiver can resolve is in the interaction
        (``participation``; the rest is ``bystanding``), and of that only the part
        that fits the receiver's etendue crosses (``tau``; the rest is ``reflected``).

        Both gates are measured: participation against the receiving side's own derived
        floor, tau from the two sides' etendue, which is the conserved invariant of a
        lossless system.  Nothing here is a coefficient anyone chose."""
        for g in (lens_from, lens_to):
            if g not in self._placed:
                raise KeyError(f"lens {g!r} has nothing placed; call place({g!r}, surface) first")
        Xf, Xt = self.balanced(lens_from), self.balanced(lens_to)
        xp = _ns(Xf)
        # Every energy here goes through the sender's own law, including the fraction: a share
        # measured one way and a total measured another do not describe the same quantity.
        E = self._total_energy(lens_from, Xf)
        Vt = self.directions(lens_to)                  # what the receiver resolves, by its floor
        Vf = self.directions(lens_from)
        if int(Vt.shape[1]) == 0 or E <= 0.0:
            part = 0.0                                  # the receiver resolves nothing: nothing matches
        else:
            Vx = _env.asdtype_of(Xf, np.asarray(Vt)) if _env.is_torch(xp) else np.asarray(Vt)
            matched = (Xf @ Vx) @ Vx.conj().T           # the sender, restricted to what the receiver sees
            part = min(1.0, max(0.0, self._total_energy(lens_from, matched) / E))
        Gf, Gt = self._etendue_of(lens_from), self._etendue_of(lens_to)
        tau = min(1.0, Gt / Gf) if Gf > 0 else 0.0
        pertinent = E * part
        delivered = pertinent * tau
        hi = max(Gf, Gt)
        # Which concepts it condenses into: the sender's energy resolved onto the receiver's
        # directions (the same bundle machinery, the sender's law), scaled by what crosses.
        cond = self._bundle(lens_from, Xf, Vt) if int(Vt.shape[1]) else []
        return Transfer(absorbed=delivered, transmitted=(E - pertinent) + pertinent * (1.0 - tau),
                        flux=(E - pertinent) - pertinent * (1.0 - tau),
                        energy=E, participation=part, pertinent=pertinent, tau=tau,
                        condensation=cond,
                        delivered=delivered, reflected=pertinent * (1.0 - tau),
                        bystanding=E - pertinent, etendue_from=Gf, etendue_to=Gt,
                        match=(min(Gf, Gt) / hi if hi > 0 else 0.0),
                        concentration=(Gf / Gt if Gt > 0 else float("inf")),
                        radiance_from=(pertinent / Gf if Gf > 0 else 0.0),
                        radiance_to=(delivered / Gt if Gt > 0 else 0.0),
                        modes_from=int(Vf.shape[1]), modes_to=int(Vt.shape[1]))

    def resolution(self):
        """The screen's settled state: the sides superposed and projected onto the resolved
        shared basis.  ``None`` when nothing stands above the floor -- the null is no outgoing
        signal, never a written one.

        The mixing law: each side is brought to its own zero
        (``balanced``) and the balanced frames are summed as they are.  Summing raw destroys
        nothing -- each side enters at its own energy, and ``energy()`` reports what that is,
        so ``energy()`` reports the weighting each side enters at.  A system that means to enter
        at a different weight scales in its OWN ``entry``, which keeps the law with the system.

        Nothing is synthesised: the projection is linear, onto modes measured from the
        placements themselves -- the same discipline as ``extract``."""
        self._require_placed()
        self._require_common_order("resolution()")
        V = self.basis()
        if int(V.shape[1]) == 0:
            return None
        X = None
        for g in self._placed:
            Pc = self.balanced(g)
            X = Pc if X is None else X + Pc
        xp = _ns(X)
        Vx = _env.asdtype_of(X, np.asarray(V)) if _env.is_torch(xp) else np.asarray(V)
        return (X @ Vx) @ Vx.conj().T

    def render(self, lens: str, concept=None):
        """Exit to a side: inverse-convert ``concept`` (or, by default, the screen's own
        settled ``resolution()``) out through that lens.  With a concept entered from
        another side this is the cross-lens conversion ``A (x) B`` -- composition through
        the shared basis.  ``None`` out when nothing resolved."""
        g = self._lens_of(lens)
        if g.inverse is None:
            raise ValueError(f"lens {lens!r} is entry-only (registered without an inverse); "
                             f"it can be placed, read and coupled; rendering out takes an inverse")
        C = self.resolution() if concept is None else concept
        return None if C is None else g.inverse(C)

    def read(self) -> ScreenRead:
        """The aperture measurement of the screen's joint frame -- on the un-folded basis
        (``reads.spectral_optics`` on the unit-diagonal correlation, never the entropy-folded
        screen), read in balance (the connected frame).  Same estimators as everywhere else
        in the library: one measurement, viewed from the screen."""
        J = self.joint()
        Jc = self.balanced_joint()                      # everything is read at the screen, in balance
        ap = self._c("joint_ap", lambda: self._aperture_on(Jc))
        sp, dl = ap.spectral, ap.diffraction_limit
        D = int(next(iter(self._placed.values())).shape[1])
        return ScreenRead(
            n_lenses=len(self._placed), T=int(J.shape[0]), D=D,
            K_signal=int(sp.resolved_modes), contrast=float(sp.contrast),
            top_share=float(sp.top_share), noise_floor=float(sp.noise_floor),
            attenuation=float(sp.attenuation),
            coherence=float(_coherence(Jc, lag=1)),   # un-folded: ap.projection().screen would fold
            a_delta=float(dl.a_delta), correlation_length=float(dl.xi),
            basis_dim=int(self.basis().shape[1]))

    def coupling(self, lens_a: str, lens_b: str) -> Coupling:
        """The full measured coupling record between two placed sides (see
        ``reads.coupling``): the signed exact-permutation ``z``, its ``sign`` and
        ``strength``, the ``phase`` the sign is the real shadow of, and how ``tight`` the two
        sides are locked.  Derived at the screen from how the two balance against each
        other -- never a constant passed in."""
        for g in (lens_a, lens_b):
            if g not in self._placed:
                raise KeyError(f"lens {g!r} has nothing placed; call place({g!r}, surface) first")
        if int(self._placed[lens_a].shape[0]) != int(self._placed[lens_b].shape[0]):
            self._require_common_order("coupling()")
        return coupling(self._placed[lens_a], self._placed[lens_b], far=self._far)

    def couple(self, lens_a: str, lens_b: str) -> float:
        """The signed coupling strength between two sides: ``+`` attract, ``-`` detract,
        magnitude = strength, and exactly ``0.0`` when nothing resolves above the level.
        The scalar decision; ``coupling()`` returns the evidence behind it."""
        return float(self.coupling(lens_a, lens_b).strength)

    def balance(self) -> Balance:
        """The self-balancing zero: each side at ITS OWN zero, and what that zero left.

        Per side, ``offsets[g]`` is the DC removed to reach the zero and ``residual[g]`` is the
        column mean remaining after it.  ``pvalue[g]`` scores that residual against the exact
        no-drift moments (``E||r||^2 = tr(S)/T``, ``Var||r||^2 = 2 tr(S^2)/T^2``, which fix a
        scaled chi-square with ``nu = (tr S)^2/tr(S^2)``, the participation ratio of the
        covariance spectrum), and ``closed[g]`` is that score at the reader's level.

        ``total`` is the largest column sum left in the balanced frame, so it reports whether
        the frame as a whole closes.

        Under the derived default zero the residual is 0 and every side reads closed.  A side
        that declares its OWN zero is tested against it: a zero placed where the system
        balances reads closed, and one placed elsewhere shows in the residual."""
        self._require_placed()
        offsets, residual, pvalue, closed, sums = {}, {}, {}, {}, []
        for g in self._placed:
            # Per side, so a beam on its own order balances on its own order.  When every side
            # shares one, these column sums ARE the joint's, so `total` is unchanged.
            Xc = self.balanced(g)
            xp = _ns(Xc)
            T = int(Xc.shape[0])
            Z = self._zero_of(g)                                        # the zero this side removed
            offsets[g] = float(_env.vnorm(xp, Z)) / float(np.sqrt(T)) if T > 0 else 0.0
            r = _env.mean0(xp, Xc)                                      # what that zero left behind
            Q = float(_env.vnorm(xp, r)) ** 2
            residual[g] = float(np.sqrt(Q))
            pvalue[g], closed[g] = self._drift_pvalue(Xc, r, Q, T)
            sums.append(float(np.abs(np.asarray(_env.to_numpy(_env.sum_ax(xp, Xc, 0)))).max()))
        return Balance(total=(max(sums) if sums else 0.0), offsets=offsets, residual=residual,
                       pvalue=pvalue, closed=closed,
                       frame=(self.balanced_joint() if self.shares_order else None))

    def realise(self, lens_from: str, lens_to: str) -> Realisation:
        """What the crossing actually delivers, pushed through the receiving lens's real
        conversion -- the third reading beside ``certify`` (imaging fidelity) and ``transfer``
        (the etendue ceiling).

        The energy that should condense is rendered out through the receiver's ``inverse``,
        entered again through its ``entry``, and re-projected onto what that side resolves.
        What survives is what the lens genuinely delivers; the gap to ``Transfer.delivered``
        is the conversion's loss, cleanly separated from the phase-space loss ``tau`` already
        accounts for.  Both energies are measured through the sender's own law, so the ratio
        is a fraction of the quantity it is reported against."""
        t = self.transfer(lens_from, lens_to)
        g = self._lens_of(lens_to)
        if g.inverse is None:
            raise ValueError(f"lens {lens_to!r} is entry-only (registered without an inverse); "
                             f"what it REALISES is a property of the round trip, which takes both")
        Vt = self.directions(lens_to)
        if int(Vt.shape[1]) == 0 or t.pertinent <= 0.0:
            return Realisation(ideal=t.delivered, realised=0.0, efficiency=0.0,
                               shortfall=t.delivered, passive=True)
        Xf = self.balanced(lens_from)
        xp = _ns(Xf)
        Vx = _env.asdtype_of(Xf, np.asarray(Vt)) if _env.is_torch(xp) else np.asarray(Vt)
        matched = (Xf @ Vx) @ Vx.conj().T                # what should condense
        back = g.entry(g.inverse(matched))               # out and back, through this lens
        if not hasattr(back, "shape"):
            back = np.asarray(back)
        if tuple(int(v) for v in back.shape) != tuple(int(v) for v in matched.shape):
            raise ValueError(f"lens {lens_to!r}: entry(inverse(.)) returned shape "
                             f"{tuple(int(v) for v in back.shape)} for a concept frame of shape "
                             f"{tuple(int(v) for v in matched.shape)} -- expected the shared basis")
        arrived = (back @ Vx) @ Vx.conj().T              # what the receiver still resolves
        e_matched = self._total_energy(lens_from, matched)
        e_arrived = self._total_energy(lens_from, arrived)
        eff = (e_arrived / e_matched) if e_matched > 0 else 0.0
        # "delivered no more than it was given" is exact in exact arithmetic; both energies are
        # sums over the frame, so the comparison is taken at that sum's own backward error.
        slack = max(int(v) for v in arrived.shape) * macheps(np, np.asarray(eff, float))
        realised = t.delivered * eff
        return Realisation(ideal=t.delivered, realised=realised, efficiency=eff,
                           shortfall=t.delivered - realised, passive=bool(eff <= 1.0 + slack))

    def _drift_pvalue(self, Xc, r, Q: float, T: int):
        """Is the residual ``r`` what a side balanced at its declared zero would show anyway?

        Under no drift the rows are exchangeable about that zero, so the sample mean has
        ``E||r||^2 = tr(S)/T`` and ``Var||r||^2 = 2 tr(S^2)/T^2``.  Those two moments fix a
        scaled chi-square exactly: ``||r||^2 ~ g * chi^2_nu`` with ``g = tr(S^2)/(T tr S)`` and
        ``nu = (tr S)^2 / tr(S^2)`` -- the participation ratio of the covariance spectrum, the
        same effective-count construction as ``2^H`` elsewhere, derived from the data rather
        than chosen.  The tail is the regularized incomplete gamma the floor machinery already
        carries.  Calibrated to the nominal level for tall frames and conservative for
        wide-short ones.

        The statistic is ``||r||^2``, so its power is the same in every direction.  A
        direction-aware form (Hotelling's ``T^2``, weighting by ``S^{-1}``) is sharper where
        the covariance is small, and it takes ``T > D`` with ``S`` invertible; this one holds
        at every shape."""
        xp = _ns(Xc)
        if T < 2 or Q <= 0.0:
            return 1.0, True                       # a zero the frame already sits at
        Xr = Xc - r
        S = xp.conj(Xr).T @ Xr / (T - 1)
        trS = float(xp.real(_env.sum_ax(xp, _env.diagonal(xp, S))))
        trS2 = float(_env.vnorm(xp, S)) ** 2       # tr(S^2) = ||S||_F^2 for Hermitian S
        if trS <= 0.0 or trS2 <= 0.0:
            return 1.0, True
        g = trS2 / (T * trS)
        nu = trS * trS / trS2
        pv = float(_reg_gamma_upper(nu / 2.0, Q / (2.0 * g)))
        return pv, bool(pv >= self._far)

    def linear(self, lens: str, surface) -> Linearity:
        """Does this lens pass the side's modes independently?

        A lens converts the whole frame; a beam carries many modes; and one lens serves them
        all precisely because it is linear.  Measured on the side's own signal: converting its
        modes together against converting them one at a time (``additivity``), and converting a
        scaled frame against scaling the conversion (``homogeneity``).  A departure means the
        modes interact inside the conversion, so splitting a mode off, converting it, and
        recombining stops agreeing with converting the beam whole.

        Takes ``surface`` for the reason ``certify`` does: linearity is a property of the
        conversion, so it is measured on what the conversion takes IN.  ``place`` keeps only
        ``entry(surface)`` -- the frame as it sits on the screen -- so the surface is not
        recoverable from state, and a screen-space frame pushed back through ``entry`` would
        measure ``entry . entry`` on a side whose width happens to match the basis."""
        g = self._lens_of(lens)
        Xn = np.asarray(_env.to_numpy(surface if hasattr(surface, "shape") else np.asarray(surface)))
        if Xn.ndim != 2:
            raise ValueError(f"linearity needs a 2-D surface to measure a conversion on; got "
                             f"shape {getattr(Xn, 'shape', None)}")

        def _rel(res, ref):
            # Measured in balance, as every screen read is: a side's zero absorbs a constant
            # offset, so an affine conversion departs by a constant and the screen sees a
            # linear one.  What survives balancing is departure the modes actually feel.
            r = np.asarray(res) - np.asarray(res).mean(axis=0, keepdims=True)
            d = float(np.linalg.norm(np.asarray(ref) - np.asarray(ref).mean(axis=0, keepdims=True)))
            return float(np.linalg.norm(r) / d) if d > 0 else 0.0

        # homogeneity: a scale through the conversion is the conversion of the scale
        c = 2.0
        e1 = np.asarray(_env.to_numpy(g.entry(Xn)))
        hom = _rel(np.asarray(_env.to_numpy(g.entry(c * Xn))) - c * e1, c * e1)

        # additivity: the modes together against the modes apart.  The split is the surface's
        # OWN rank-1 components -- the modes as the side carries them, before conversion --
        # taken to the depth of the beam the screen reads off this lens.
        n_modes = len(self.beam(lens).modes) if lens in self._placed else 0
        Xc = Xn - Xn.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(max(int(n_modes), 0), int(np.count_nonzero(S > 0)))
        if k >= 2:
            frames = [np.outer(U[:, i] * S[i], Vt[i]) for i in range(k)]
            whole = np.asarray(_env.to_numpy(g.entry(sum(frames))))
            parts = sum(np.asarray(_env.to_numpy(g.entry(f))) for f in frames)
            add = _rel(whole - parts, whole)
            d_res = whole - parts
            residuals = [(d_res - d_res.mean(axis=0, keepdims=True), add)]
        else:
            add = float("nan")
            residuals = []
        h_res = np.asarray(_env.to_numpy(g.entry(c * Xn))) - c * e1
        residuals.append((h_res - h_res.mean(axis=0, keepdims=True), hom))
        return Linearity(additivity=add, homogeneity=hom, modes=int(n_modes),
                         linear=all(self._resolves_nothing(r, v) for r, v in residuals))

    def _resolves_nothing(self, residual, rel: float) -> bool:
        """True where a residual carries no structure the instrument could see -- the same
        derived decision ``certify`` makes, so the two agree on what counts as absent.

        ``rel`` is that residual measured against the frame it departs from.  A conversion run in
        floating point leaves a residual of order n*eps whatever it does, and that residual is the arithmetic -- see ``certify`` for the bound.  It has to be ruled out
        before the screen reads it, because whitening scales each channel by its own spread and
        would raise pure round-off to unit amplitude."""
        r = np.asarray(residual)
        if r.ndim != 2 or float(np.linalg.norm(r)) == 0.0:
            return True
        if float(rel) <= max(int(v) for v in r.shape) * macheps(np, r):
            return True
        if int(r.shape[0]) < 3 or int(r.shape[1]) < 2:
            return True
        return int(Projection(r, far=self._far, null=self._null, seed=self._seed).K_signal) == 0

    # ── the conversion certificate ────────────────────────────────────────────
    def lossless(self, lens: str, surface) -> float:
        """The residual of ``inverse(entry(surface))`` against ``surface``, relative:
        ``||R||_F / ||surface||_F``.  ``0.0`` is a perfect conversion; anything else is the
        conversion leaking.  ``certify()`` turns this into a decision against the surface's
        own resolution."""
        return float(self._residual(lens, surface)[0])

    def certify(self, lens: str, surface) -> Losslessness:
        """The losslessness certificate of a lens: the round-trip residual, and whether the
        conversion lost anything the instrument could have resolved.  The residual is read
        onto its own screen -- if it resolves no mode above the derived floor it carries no
        structure, so nothing measurable leaked; if it resolves one, the lens dropped
        something the aperture can see.  See :class:`Losslessness` for the two derived
        checks the decision conjoins (a residual that resolves nothing but exceeds the null
        conversion is a leak too)."""
        rel, R, _ = self._residual(lens, surface)
        xp = _ns(R)
        # A round trip through floating-point arithmetic does not return the surface exactly, and
        # the residual it leaves is not a loss -- it is the arithmetic.  The classical backward-error
        # bound for a computed product contracted over an inner dimension n is n*eps [Higham 2002,
        # Thm 3.5], so a relative residual at or below n*eps carries nothing to read.  It must be
        # caught here: the screen whitens each channel by its own scale, which would lift pure
        # round-off to unit amplitude and resolve "modes" in it.  Both sides are derived -- n is the
        # surface's own extent, eps the working dtype's.
        roundoff = max(int(v) for v in R.shape) * macheps(xp, R)
        if float(rel) <= roundoff:
            return Losslessness(residual=float(rel), sigma_top=0.0, noise_floor=float("inf"),
                                K_signal=0, lossless=True)
        sc = Projection(R, far=self._far, null=self._null_of(lens), seed=self._seed)
        return Losslessness(residual=float(rel), sigma_top=float(sc.sigma_top),
                            noise_floor=float(sc.noise_floor), K_signal=int(sc.K_signal),
                            lossless=bool(sc.K_signal == 0 and rel < 1.0))

    def _residual(self, lens: str, surface):
        g = self._lens_of(lens)
        if g.inverse is None:
            raise ValueError(f"lens {lens!r} is entry-only (registered without an inverse); "
                             f"losslessness is a property of the round trip, which takes both")
        back = g.inverse(g.entry(surface))
        X = surface if hasattr(surface, "shape") else np.asarray(surface)
        B = back if hasattr(back, "shape") else np.asarray(back)
        if len(X.shape) != 2:
            raise ValueError(f"losslessness needs a 2-D surface to measure a residual on; got "
                             f"shape {getattr(X, 'shape', None)}")
        if tuple(int(s) for s in B.shape) != tuple(int(s) for s in X.shape):
            raise ValueError(f"lens {lens!r}: the round trip returned shape "
                             f"{tuple(int(s) for s in B.shape)} for a surface of shape "
                             f"{tuple(int(s) for s in X.shape)} -- expected the same surface")
        xp = _ns(X)
        R = _env.asnum(B) - _env.asnum(X)
        nX = float(_env.vnorm(xp, _env.asnum(X)))
        nR = float(_env.vnorm(xp, R))
        return (nR / nX if nX > 0 else (0.0 if nR == 0 else float("inf"))), R, X

    def __repr__(self) -> str:
        if not self._placed:
            return f"Screen(registered={sorted(self._lenses)}, nothing placed)"
        J = self.joint()
        return (f"Screen(sides={list(self._placed)}, T={int(J.shape[0])}, "
                f"D={int(next(iter(self._placed.values())).shape[1])})")


__all__ = ["Screen", "Lens", "Beam", "ScreenRead", "Balance", "Transfer", "Linearity",
           "Realisation", "Losslessness"]
