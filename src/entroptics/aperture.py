"""
aperture.py -- the Entroptics front door (the OPTICS side: information ABOUT a structure).

ENTROPTICS (entropy + optics): read the structure of any signal as an
optical aperture whose resolution is fixed by the signal's OWN Shannon entropy.
The observer does not choose the aperture; the signal sets its own diffraction
limit.  Every read is then an optical-aperture quantity, and the same reads apply
to any 2-D field structured along an ordered axis and a feature axis.

``Aperture`` is the single front door.  Use it BATCH or STREAMING; it accepts
numpy arrays or torch tensors (converted at the boundary -- no separate torch
wrapper needed):

    from entroptics import Aperture

    # BATCH: the whole signal at once
    ap = Aperture(W)                       # W: (T, F), real or complex
    ap.etendue, ap.strehl, ap.a_delta      # the optics (about the structure)
    ap.rates()                             # EXACT per-mode decay rates (dynamics)
    ap.projection()                            # the companion projection (projection.py)

    # STREAMING: feed frames from the first one and propagate
    ap = Aperture(window=512)              # optional window bounds the optics snapshot
    for frame in signal:                   # numpy or torch, real or complex
        ap.update(frame)                   # O(F^2) per frame; from frame 0
    ap.rates()                             # exact decay rates so far (long/short range)
    ap.optics()                            # optics snapshot on the current window

    # SPLICE / RESUME across sessions (exact at forgetting=1)
    s = ap.state();  ap2 = Aperture.from_state(s)     # resume exactly
    whole = a.splice(b)                                # a, b: two streams -> concatenated-stream operator

The companion :class:`projection.Projection` holds the PROJECTION (the information *within*
the structure): ``ap.projection()`` and ``Projection(W).aperture()`` cross between the views.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from . import environment as _env
from .entropy import geometry, live_view
from .projection import (Projection, ProjectionRead, read, footprints,
                     mode_significance, ModeSignificance)
from .extract import filter_projection
from .dynamics import Dynamics, DecayRates, DynamicsState, dynamics as _dynamics
from . import reads
from .reads import (
    phi, magnification, scale_duality, duality_of, occupied_modes, OccupiedModes,
    level_edge, LevelEdge,
    phi_T, phi_F, sigma_T, sigma_F, AxisRead, axis_read, axis_spectrum,
    etendue, strehl, space_bandwidth,
    spectral_optics, SpectralOptics, principal_directions, attenuation_interval, CertifiedInterval,
    resolved_dimension_interval, CertifiedCount, SpectralAccumulator,
    concentration_band, concentration, Concentration,
    Coupling,
    decay, diffraction_limit, DiffractionLimit, decay_scatter, DecayScatter,
    mercer_certificate, MercerCertificate,
    rayleigh_shape_factor, fresnel_number, shape_factor, optics,
    scale_profile, ScaleProfile,
)
from .beam import Beam
from .screen import (Screen, Lens, ScreenRead, Balance, Transfer,
                     Realisation, Linearity, Losslessness)
from .tensor import tensor_read, tensor_embed, tensor_reconstruct, tensor_fidelity
from .fields import slabs, over_planes, pool
from .sweep import sweep as _sweep
from . import null_providers
from .null_providers import (FloorContext, KINDS, by_kind, floor_from_null_sampler,
                             permutation, reference_null, ReferenceNull, self_calibrating_null,
                             top_spectrum_value, shuffle_in_time)


def _to_np(x):
    """Materialise a numpy array or torch.Tensor to numpy (preserving complex),
    None-safe -- delegates to the shared ``environment.to_numpy`` shim."""
    return None if x is None else _env.to_numpy(x)


#: The minimum window, in frames, when the caller states none.
#:
#: A resource envelope, not a threshold in a measurement: how much history the aperture retains
#: once the signal has stopped being coherent. The same category as a null's draw count -- it buys
#: memory and latency and decides nothing about what a number means -- so it is a legitimate caller
#: input, named here, so it is reached by name.
#:
#: The frame carries no row count to read it off. ``axis_spectrum(W, 0)`` takes the T rows as
#: variables and the F columns as samples, so its rank is ``min(T, F) - 1`` and saturates at
#: ``F - 1`` once ``T >= F``; past that, extra rows add no rank and only grow the denominator of
#: ``phi_T = 2^H / T``. On white noise at F = 16, rows 17 / 32 / 64 / 128 / 256 all resolve exactly
#: 15 modes while phi_T falls 0.61 / 0.38 / 0.22 / 0.11 / 0.06 -- the read never switches from ill
#: posed to well posed, so the choice is the caller's.
MIN_WINDOW = 128

#: `window` was not given, so the constructor decides it from what it was handed.  Distinct
#: from an explicit ``window=None``, which is a caller saying "no windowing" outright.
_WINDOW_UNSET = object()


class Aperture:
    """An optical aperture whose resolution is set by a signal's own Shannon
    entropy -- the single STREAMING-FIRST front door.

    The aperture is LOCAL and FORGETTING: it accumulates into a fixed-size dynamical
    operator (O(F^2), all frames -> the GLOBAL rates / forgetting / feature spectrum) and
    keeps an ADAPTIVE local window of recent frames for the frame-level reads (phi_T, the
    optics snapshot).  ``window`` is a MINIMUM, not a clock: the aperture keeps at least that
    many frames and MORE while the signal is still coherent -- its own correlation length, read
    off the operator's forgetting margin -- and forgets only the decorrelated tail.  A persistent
    active signal is never truncated; pure noise forgets to the minimum.  There is no unbounded
    batch underlay; memory tracks the signal's coherence.

    ``window`` follows what the aperture is handed.  ``Aperture(W)`` is given a finite record and
    reads all of it (``window=None``).  ``Aperture()`` + :meth:`update` has frames still arriving,
    so it keeps the coherent window and at least :data:`MIN_WINDOW` frames, which is what bounds
    memory on an unbounded stream.  Pass ``window=`` to state your own either way.

    NOTE when a window IS set (a stream, or an explicit ``window=``): ``phi_T`` and the
    other frame-level reads are taken over the coherent window, whose length is a per-signal
    measurement, so their denominators differ.  ``2^H`` on identical (961, 64) frames of fGn /
    trend / shift / noise reads 45.9 / 61.4 / 49.5 / 54.3 while the windows read 186 / 961 / 157 /
    128 -- so ``Aperture(W).phi_T`` mostly reports the window.  That is the right behaviour for a
    streaming aperture and the wrong statistic for a cross-signal comparison; build with
    ``window=None`` (or call :func:`reads.phi_T` on the frame) when the frames must be commensurate.

    Batch:      ``Aperture(W)`` ingests the rows of W (T, F) -- axis-0 ORDERED (_T), axis-1
                FEATURE (_F) -- as a stream: the operator sees ALL of W, the frame reads see
                the coherent window (>= ``window``).
    Streaming:  ``Aperture(window=..., forgetting=...)`` then ``.update(frame)`` per frame.
                The operator streams incrementally; the optics snapshot is the current
                coherent window (recomputed on demand, cache cleared each update).

    The GLOBAL streaming reads come off the operator: ``.rates()`` (exact per-mode decay),
    ``.margin`` / ``.dynamics().forgetting()`` (the forgetting axiom), ``.dynamics()
    .resolved()`` (K_signal from Pxx), ``.dynamics().significance()``.  ``.state()`` /
    ``.from_state`` / ``.splice`` resume and splice streams.  Accepts numpy or torch (on-device).
    """

    def __init__(self, W=None, mask=None, *, window=_WINDOW_UNSET,
                 forgetting: float = 1.0, rank: int | None = None,
                 null=None, reference=None, seed: int = 0, far: float = 0.05):
        # A finite record is read whole; a stream is bounded.  One rule, applied to what the
        # caller actually handed over: `Aperture(W)` was given the record and reads all of it,
        # while `Aperture()` + `update(frame)` has frames still arriving and keeps the coherent
        # window (>= MIN_WINDOW) so memory stays bounded.  An explicit `window=` wins either way.
        self.window = (None if W is not None else MIN_WINDOW) if window is _WINDOW_UNSET else window
        self.far = float(far)       # the reader's false-alarm level for EVERY read on this

        self.forgetting = float(forgetting)
        self.rank = rank
        self._mask = mask
        self._null = null           # noise-floor null PROVIDER (None -> reference/mp, below);
        self._reference = reference  # signal-free reference realisations -> PREFERRED calibrated null
        self._seed = int(seed)      # a stateful provider's update(frame) runs in the stream.
        # Floor resolution (per cut point): explicit ``null`` > ``reference_null`` from
        # ``reference`` (the calibrated null the library PREFERS when a signal-free reference
        # is available) > the derived ``mp`` default.  ``null`` may also be a by_kind(...) /
        # {kind: provider} mapping to route the screen and spectral floors apart.
        self._buf: deque = deque()   # streaming frame buffer -- ADAPTIVELY trimmed (below)
        self._dyn: Dynamics | None = None         # streaming dynamical operator (persists)
        self._batch = None                        # fixed batch array, if constructed from W
        self._nfeat: int | None = None
        self._cache: dict = {}
        if W is not None:
            if len(getattr(W, "shape", ())) != 2:
                raise ValueError(f"Aperture expects a 2-D array (T, F); got {getattr(W, 'shape', None)}")
            if min(int(v) for v in W.shape) == 0:
                raise ValueError(f"Aperture expects a screen with both axes non-empty; got "
                                 f"{tuple(int(v) for v in W.shape)}. An axis of length 0 carries no "
                                 f"cell to read, which is not the same as an axis whose cells are "
                                 f"all missing -- pass those as NaN or behind a mask.")
            self._batch = W                     # raw (numpy OR torch) -- dynamics can stay on-device
            self._nfeat = int(W.shape[1])

    # ── streaming ─────────────────────────────────────────────────────────────
    def update(self, frame) -> "Aperture":
        """Feed one frame (an F-vector, numpy or torch) -- streams the dynamical
        operator (from frame 0, backend-agnostic: torch frames stay on-device) and
        appends to the optics window.  Invalidates the optics snapshot."""
        if self._dyn is None:
            self._nfeat = int(np.asarray(_to_np(frame)).reshape(-1).shape[0])
            self._dyn = Dynamics(self._nfeat, forgetting=self.forgetting, rank=self.rank)
        self._dyn.update(frame)     # RAW frame -> stays numpy/torch on its device
        self._buf.append(frame)     # keep raw frames for the optics window
        if self._null is not None and hasattr(self._null, "update"):
            self._null.update(frame)   # a STATEFUL null provider tracks the stream (runs in the dynamical)
        # ADAPTIVE FORGETTING: keep >= `window` frames, and MORE while a coherent signal is
        # still active; forget only the decorrelated tail (the signal decides, not a clock).
        # Checked every `window` frames so the horizon eig is amortised.
        _min = self._min_window()
        if _min is not None and self._dyn.n_frames % _min == 0:
            keep = max(_min, self._coherence_horizon())
            while len(self._buf) > keep:
                self._buf.popleft()
        self._batch = None
        self._mask = None
        self._cache.clear()         # window changed -> optics snapshot is stale
        return self

    @property
    def W(self):
        """The current data window in its NATIVE backend (numpy or torch) -- the
        optics reads run on it directly (on-GPU when it is a torch tensor); the
        dynamics core keeps the original backend too.  ``Projection`` materialises numpy."""
        return self._c("W", self._materialize)

    @property
    def mask(self):
        """The absence mask for :attr:`W` -- ``True`` marks a cell that was NOT observed, either
        because nothing was recorded there or because the reading is known bad.  It is trimmed to
        the same window as ``W``: the two are always read as a pair, so they are always cut as a
        pair, and a mask given for the whole record annotates whatever slice of it the reads see."""
        m = self._mask
        if m is None:
            return None
        n = int(self.W.shape[0])
        return m if int(m.shape[0]) <= n else m[-n:]

    @mask.setter
    def mask(self, m):
        self._mask = m
        self._cache.clear()                 # the reads all pair W with the mask -- they are stale

    def _min_window(self) -> int | None:
        """The minimum window in frames, or ``None`` for no windowing at all.

        One accessor so the three call sites cannot drift, and so the resource envelope
        (:data:`MIN_WINDOW`) is reached by name."""
        return None if self.window is None else int(self.window)

    def _coherence_horizon(self) -> int:
        """Frames of still-coherent history to keep BEYOND the minimum window, set by the
        signal itself (not a clock): the memory an active mode implies.  Read off the
        operator's forgetting margin ``m = max_k|mu_k|`` -- the correlation length to decay
        to ``eps`` is ``ln(1/eps)/(-ln m)``.  ``0`` when no signal is resolved (forget to the
        minimum); the whole stream when a mode is persistent (``m -> 1``: never truncate an
        active signal).  Incremental, from the O(F^2)/frame operator; no O(T^2).

        ``eps`` is DERIVED, not chosen: it is the operator's own noise floor expressed as a
        fraction of its dominant mode, ``eps = edge / lambda_1 = 1 / contrast``, off the same
        feature spectrum and the same ``apply_floor`` call that ``_signal_rank`` resolves the rank
        with.  A mode has been forgotten when its amplitude reaches the level the operator cannot
        tell from noise, and that level is a property of this spectrum.  It replaces a fixed ``eps = 0.05``, which set the window -- and therefore every
        frame-level read taken on it -- from a constant.  The identical derivation is already
        relied on downstream in ``lumen/reasoning._operator_horizon``.

        Falls back to the MINIMUM window when the contrast cannot be read (a spectrum with nothing
        above its own floor), joining the two failure paths already here -- ``resolved() < 1`` and
        the ``except``.  Keeping the whole stream would be the other reading of "no measured reason
        to forget", but the window exists to bound memory, and an unbounded buffer on an unreadable
        floor trades a guarantee the caller relies on for a number nobody measured."""
        try:
            core = self._core()
            if core.resolved() < 1:
                return 0                                   # no active signal -> minimum window
            m = float(core.forgetting()["margin"])
            eps = core.floor_contrast()
            eps = (1.0 / eps) if (eps is not None and math.isfinite(eps) and eps > 1.0) else None
        except Exception:
            return 0            # coherence undetermined (e.g. missing/degenerate data) -> minimum
        if not math.isfinite(m) or m <= 0.0:
            return 0
        if m >= 1.0 - 1e-9:
            return 1 << 60          # persistent active mode -> keep all.  This margin bounds the
                                    # frame WINDOW (a locality/resource bound, PAPER §13.1) and
                                    # enters no read, unlike the operator's own `forgets`, which is
                                    # reported and so is taken at the arithmetic's resolution.
        if eps is None:
            return 0            # no readable floor -> minimum window, and memory stays bounded
        return int(math.ceil(math.log(1.0 / eps) / (-math.log(m))))

    # ── why _materialize short-circuits the horizon ───────────────────────────────────────────
    #
    # `keep = max(_min, _coherence_horizon())` is >= `_min` by construction, so a frame already no
    # longer than `_min` cannot be truncated whatever the horizon says.  Computing it anyway costs a
    # full eigendecomposition of the F x F accumulator -- O(F^3), and F is the pooled spatial volume,
    # so it is 4096 at L=16 and 32768 at L=32.
    #
    # Measured, one 32-row frame at F=4096, output bit-identical either way:
    #     .W with the default window=128 : 44.57 s
    #     .W with the short-circuit      :  0.00 s
    # A caller reading many short frames (an ensemble of configurations, one aperture each) paid
    # that per frame, which is where a nine-hour certification run came from.
    #
    # This changes no read: it skips a computation whose result provably cannot affect the outcome.
    _horizon_is_moot = None                 # anchor for the comment above; see the two guards below

    def _materialize(self):
        # Streaming-only and adaptive: the frame reads see a local window -- at least ``window``
        # frames (the minimum boundary), and MORE while the signal is still coherent (its own
        # correlation length, from the operator).  Never the whole signal unless it stays
        # active; the ordered-axis cost is O(kept^2), the operator (``_core``) is global.
        if self._batch is not None:
            B = self._batch
            _min = self._min_window()
            if _min is None:
                return B
            if int(B.shape[0]) <= _min:
                return B                    # see _horizon_is_moot
            keep = max(_min, self._coherence_horizon())
            return B if int(B.shape[0]) <= keep else B[-keep:]
        if self._buf:
            frames = list(self._buf)
            _min = self._min_window()
            if _min is not None and len(frames) > _min:     # see _horizon_is_moot
                keep = max(_min, self._coherence_horizon())
                if len(frames) > keep:
                    frames = frames[-keep:]
            xp = _env.ns(frames[0])
            return xp.stack(frames)
        raise ValueError("Aperture has no data yet; pass W or call update(frame)")

    def _frames(self) -> list:
        """The window's rows as individual frames -- from the streaming buffer OR,
        for a batch aperture, the rows of the batch (so splice carries either)."""
        if self._buf:
            return list(self._buf)
        if self._batch is not None:
            return [self._batch[i] for i in range(int(self._batch.shape[0]))]
        return []

    def _c(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ── shared primitives (computed once, reused by every read) ───────────────
    def _geom(self) -> dict:
        """The geometry of what was measured, cached -- the per-axis reads derive from it.
        The aperture's own mask travels with it: a cell nothing was observed in is absent,
        not zero, and must not widen the axis the signal is scored against."""
        return self._c("geom", lambda: geometry(self.W, self.mask))

    def _ev(self, axis: int):
        """The per-axis correlation eigenspectrum, cached (the up-to-O(len^3)
        eigendecomposition every axis read shares)."""
        return self._c(f"ev{axis}", lambda: axis_spectrum(live_view(self.W, self.mask), axis))

    # ── per-axis bundles (see each axis clearly) ──────────────────────────────
    @property
    def T(self) -> AxisRead:
        """The ORDERED axis: AxisRead(H, n, delta, phi, sigma)."""
        return self._c("axis_T", lambda: axis_read(self.W, 0, self.mask, geom=self._geom(),
                                                    evals=self._ev(0)))

    @property
    def F(self) -> AxisRead:
        """The FEATURE axis: AxisRead(H, n, delta, phi, sigma)."""
        return self._c("axis_F", lambda: axis_read(self.W, 1, self.mask, geom=self._geom(),
                                                    evals=self._ev(1)))

    # per-axis flat accessors (delegate to the cached AxisRead bundles)
    @property
    def H_T(self) -> float:
        """Ordered-axis power-marginal entropy (bits)."""
        return self.T.H
    @property
    def H_F(self) -> float:
        """Feature-axis power-marginal entropy (bits)."""
        return self.F.H
    @property
    def n_T(self) -> int:
        """Ordered-axis effective mode count, round(2^{H_T})."""
        return self.T.n
    @property
    def n_F(self) -> int:
        """Feature-axis effective mode count, round(2^{H_F}) (noise-guarded)."""
        return self.F.n
    @property
    def delta_T(self) -> float:
        """Ordered-axis matched cell scale (always 1; the ordered axis never folds)."""
        return self.T.delta
    @property
    def delta_F(self) -> float:
        """Feature-axis matched cell scale, L_F/2^{H_F} (noise-guarded)."""
        return self.F.delta
    @property
    def phi_T(self) -> float:
        """Ordered-axis fill fraction phi_T."""
        return self.T.phi
    @property
    def phi_F(self) -> float:
        """Feature-axis fill fraction ``phi_F = 2^H / F`` of THIS FRAME's feature power marginal.

        ``Dynamics.phi_F`` is the same quantity on a different subject -- the accumulated
        operator's feature-correlation spectrum -- so the two read close but are not the same
        number, and which one you have is set by the receiver: ``ap.phi_F`` is the frame,
        ``ap.dynamics().phi_F`` is the operator."""
        return self.F.phi
    @property
    def sigma_T(self) -> float:
        """Ordered-axis leading correlation singular value, sqrt(lambda_1)."""
        return self.T.sigma
    @property
    def sigma_F(self) -> float:
        """Feature-axis leading correlation singular value, sqrt(lambda_1)."""
        return self.F.sigma

    @property
    def geometry(self) -> dict:
        """The matched scale read from the signal's own Shannon entropy
        (H_T, n_T, delta_T, H_F, n_F, delta_F)."""
        return dict(self._geom())

    # ── combined scale ────────────────────────────────────────────────────────
    @property
    def phi(self) -> float:
        """phi in (0,1]: the screen fill fraction (2^{H_sv}/N over the whole block)."""
        return self._c("phi", lambda: phi(self.W, self.mask))

    @property
    def magnification(self) -> float:
        """1/phi in [1, inf): the reciprocal magnification (reuses the cached phi)."""
        return duality_of(self.phi)["magnification"]

    @property
    def duality(self) -> dict:
        """{phi, magnification, at_diffraction_limit} -- both faces of the scale
        (reuses the cached phi -- no second SVD).  One arithmetic with the free
        :func:`reads.scale_duality`, which reads phi off a frame instead."""
        return duality_of(self.phi)

    @property
    def at_diffraction_limit(self) -> bool:
        return self.duality["at_diffraction_limit"]

    # ── combined screen area ──────────────────────────────────────────────────
    @property
    def etendue(self) -> float:
        """phi_F * phi_T: the conserved 2-D aperture area."""
        return self._c("et", lambda: self.F.phi * self.T.phi)

    @property
    def space_bandwidth(self) -> int:
        """n_F * n_T: the number of resolvable spots."""
        return self._c("sb", lambda: int(self.F.n * self.T.n))

    @property
    def strehl(self) -> float:
        """Strehl ratio: coherence of the dominant ordered-axis mode (reuses the
        cached ordered-axis spectrum -- no second eigendecomposition)."""
        return self._c("st", lambda: strehl(self.W, self.mask, evals=self._ev(0)))

    # ── mode spectrum / propagation constant ──────────────────────────────────
    @property
    def spectral(self) -> SpectralOptics:
        """The optics of the feature-correlation eigenspectrum: contrast,
        top_share, resolved modes, noise floor, attenuation constant, phase
        constant, dispersion."""
        return self._c("spec", lambda: spectral_optics(self.W, self.mask,
                                                        null=self._effective_null("spectral"),
                                                        far=self.far, seed=self._seed))

    @property
    def contrast(self) -> float:
        """Peak feature-correlation eigenvalue over the noise-floor edge, lambda1/edge (>1 => structure)."""
        return self.spectral.contrast
    @property
    def top_share(self) -> float:
        """Dominant-mode power fraction, lambda1/sum(lambda) (Strehl-like)."""
        return self.spectral.top_share
    @property
    def resolved_modes(self) -> int:
        """Count of feature-correlation eigenvalues above the noise floor."""
        return self.spectral.resolved_modes
    @property
    def noise_floor(self) -> float:
        """Finite-size Tracy-Widom (Johnstone) correlation edge."""
        return self.spectral.noise_floor
    @property
    def attenuation(self) -> float:
        """The attenuation constant alpha (Re gamma) of the dominant mode."""
        return self.spectral.attenuation
    @property
    def phase(self) -> float:
        """The phase constant beta (Im gamma) of the dominant mode."""
        return self.spectral.phase
    @property
    def dispersion(self) -> float:
        """Standard deviation of the per-mode attenuation across the resolved modes."""
        return self.spectral.dispersion
    @property
    def resolved_power(self) -> float:
        """Summed eigenvalue excess (lambda_k - edge) above the noise floor -- the resolved-mode power."""
        return self.spectral.resolved_power
    @property
    def dominance(self) -> float:
        """(lambda1 - 1)/(N - 1) in [0,1]: the leading feature mode's normalized excess."""
        return self.spectral.dominance

    @property
    def principal_directions(self) -> np.ndarray:
        """The ``(N, k)`` feature-space DIRECTIONS of the resolved correlation modes -- WHICH
        directions stand above the floor, where ``spectral.resolved_modes`` reports how many
        (``k`` is exactly that count).  Read off the same unit-diagonal correlation as
        ``spectral``, so the basis is scale-invariant and a sparse carrier is never folded
        away; cached beside it.  See ``reads.principal_directions``."""
        return self._c("pdir", lambda: principal_directions(
            self.W, self.mask, null=self._effective_null("spectral"),
            far=self.far, seed=self._seed))

    def attenuation_interval(self, band: float) -> CertifiedInterval:
        """A certified interval for the attenuation constant given an input
        spectral-norm band (see reads.attenuation_interval).  Reuses the cached
        spectral read."""
        return attenuation_interval(self.W, self.mask, band=band, sg=self.spectral)

    # ── concentration / focus ─────────────────────────────────────────────────
    @property
    def concentration(self) -> Concentration:
        """Focus of the rows on their dominant axis (Fisher-information
        concentration): intensity, focus, resultant."""
        return self._c("conc", lambda: concentration(self.W, self.mask))

    @property
    def focus(self) -> float:
        """Axial concentration sigma1^2/M of the rows on their leading principal axis."""
        return self.concentration.focus
    @property
    def intensity(self) -> float:
        """Top eigenvalue sigma1^2 of the row second-moment matrix."""
        return self.concentration.intensity

    # ── decay (the OTF) + diffraction limit (intrinsic: the signal's own decay) ──
    @property
    def decay(self) -> np.ndarray:
        """The signal's OWN ordered-axis autocorrelation C(tau) -- its OTF
        (Wiener-Khinchin; coherent for signed/complex inputs, incoherent |W|^2 for
        non-negative ones).  This is where the diffraction limit comes from."""
        return self._c("decay", lambda: decay(self.W, self.mask))

    @property
    def diffraction_limit(self) -> DiffractionLimit:
        """The diffraction limit a_delta from the signal's own decay -- the entropy
        width (Entroptics, primary) plus the classical Abbe integral length
        (secondary).  For the EXACT per-mode rates use ``rates()`` (the operator)."""
        return self._c("dl", lambda: diffraction_limit(self.decay))

    @property
    def a_delta(self) -> float:
        """The (entropy-width) diffraction limit a_delta = 1/2^{H}, H = entropy of
        C^2 (approximate; ``rates().dominant`` is the exact dominant-mode rate).

        ``decay_scatter`` says how much of that width this record's own channels disagree about,
        which is what tells you whether to read it as the signal's."""
        return self.diffraction_limit.a_delta

    @property
    def decay_scatter(self) -> DecayScatter:
        """How much of :attr:`decay` is this record's own sampling scatter (see
        :class:`reads.DecayScatter`).  The decay is a sum over per-channel autocovariances, so the
        channels are replicates of it and their disagreement measures the read's uncertainty --
        nothing is assumed and nothing is removed.  When ``noise_share`` approaches ``tail_share``
        the width away from zero lag is scatter, and :attr:`a_delta` reads a longer correlation
        than the record supports."""
        return self._c("dscat", lambda: decay_scatter(self.W, self.mask))

    @property
    def correlation_length(self) -> float:
        """xi -- the integral correlation length (reciprocal of the Abbe limit)."""
        return self.diffraction_limit.xi

    @property
    def dominant_decay_rate(self) -> float:
        """The decay rate of the DOMINANT (slowest, |mu|-largest) mode, alpha_1 = -log|mu_1|, from
        the exact Koopman/DMD operator spectrum (``rates().dominant``).  A forward operator read:
        identify the linear propagator from the trajectory and read its dominant rate, the slowest
        mode of a multi-mode signal.  Deterministic in the operator eigenvalues."""
        return float(self._core().rates().dominant)

    @property
    def connected_decay_rate(self) -> float:
        """The decay rate of the DOMINANT (slowest) mode, alpha_1 = -log|mu_1|, from the exact
        Koopman/DMD operator spectrum on the CONNECTED (mean-subtracted) dynamics -- the
        fluctuation dynamics (see ``Dynamics.connected_decay_rate``).  A forward operator read;
        deterministic in the operator eigenvalues."""
        return self._core().connected_decay_rate()

    @property
    def mercer(self) -> MercerCertificate:
        """The Mercer certificate: a_delta read the temporal way (decay entropy)
        AND the spectral way (stationary eigenspectrum) -- they must coincide.
        ``.ratio`` ~ const validates the read; a departure flags non-stationarity.
        (O(T^2) memory + O(T^3) eig -- the validation, not the hot path.)"""
        return self._c("mercer", lambda: mercer_certificate(self.W, self.mask))

    @property
    def rayleigh_shape_factor(self) -> float:
        """xi * a_delta -- the Rayleigh shape factor g (a dimensionless shape
        functional of the decay, not a conjugate-domain width product; entropy-native,
        no transform)."""
        return self._c("rsf", lambda: rayleigh_shape_factor(self.decay))

    @property
    def shape_factor(self) -> float:
        """a_delta / phi_F (Rayleigh / Abbe shape factor)."""
        return self._c("shape", lambda: (self.a_delta / self.F.phi
                                         if self.F.phi > 0 else float("nan")))

    def fresnel_number(self, window) -> float:
        """The Fresnel number ~ window * phi_T (near/far-field coordinate).  Reuses
        the cached ordered-axis fill fraction (no extra eigendecomposition)."""
        return float(window * self.T.phi)

    # ── the dynamical operator (exact decay rates; streaming; splice-able) ─────
    def _core(self) -> Dynamics:
        """The persistent streaming dynamical operator (lazily seeded from a batch)."""
        if self._dyn is None:
            if self._batch is not None:
                self._dyn = _dynamics(self._batch, forgetting=self.forgetting, rank=self.rank)
            elif self._nfeat is not None:
                self._dyn = Dynamics(self._nfeat, forgetting=self.forgetting, rank=self.rank)
            else:
                raise ValueError("Aperture has no data; pass W or call update(frame)")
        return self._dyn

    def dynamics(self) -> Dynamics:
        """The signal's streaming DYNAMICAL OPERATOR (online DMD / Koopman): the
        one-step propagator whose eigenvalues give the EXACT per-mode decay rates
        and frequencies (see dynamics.Dynamics)."""
        return self._core()


    @property
    def margin(self) -> float:
        """The aperture-FORGETTING margin ``max_k|mu_k|`` from the operator spectrum
        (``< 1`` iff the screen forgets) -- the aperture form of the extraction-bound axiom,
        read incrementally with no O(T^2) autocorrelation.  The full read (``margin``,
        ``forgets``, ``n_modes``) is ``dynamics().forgetting()``.  (Distinct from the
        ``forgetting`` constructor arg ``lambda``, the memory horizon.)"""
        return self._core().forgetting()["margin"]



    def state(self) -> DynamicsState:
        """Export the full dynamical-operator state (all long-range params) -- resume
        or splice a stream exactly (see Dynamics.state)."""
        return self._core().state()

    @classmethod
    def from_state(cls, s: DynamicsState, *, window: int | None = None,
                   rank: int | None = None) -> "Aperture":
        """Resume a streaming Aperture from an exported operator state -- continue
        feeding frames exactly where the prior stream left off."""
        ap = cls(window=window, forgetting=s.forgetting, rank=rank)
        ap._dyn = Dynamics.from_state(s, rank=rank)
        ap._nfeat = ap._dyn.F
        return ap

    def splice(self, other: "Aperture", *, adjacent: bool = True) -> "Aperture":
        """Splice two apertures into one (exact at forgetting=1): merges the
        dynamical operators (the concatenated-stream operator) and concatenates the
        optics windows.  Works for BATCH or STREAMING sources (each contributes its
        window's rows).  ``adjacent``: ``other`` immediately follows ``self``."""
        out = Aperture(window=self.window, forgetting=self.forgetting, rank=self.rank)
        out._dyn = self._core().merge(other._core(), adjacent=adjacent)
        out._nfeat = out._dyn.F
        for f in self._frames() + other._frames():
            out._buf.append(f)
        return out

    # ── the full read + the companion projection ──────────────────────────────
    def optics(self) -> dict:
        """The full, fully-INTRINSIC optical read (the decay is derived from the
        signal itself -- no external input).  Uses the SAME canonical schema as
        ``reads.optics`` via ``reads.assemble_optics`` (they cannot drift)."""
        return reads.assemble_optics(
            self.T, self.F, self.spectral, self.diffraction_limit,
            phi_val=self.phi, strehl_val=self.strehl,
            focus=self.focus, intensity=self.intensity,
            at_diffraction_limit=self.at_diffraction_limit)

    def _effective_null(self, kind: str):
        """The floor provider the aperture uses for a cut point: an explicit ``null`` wins;
        else ``reference_null`` calibrated on the ``reference`` (the PREFERRED calibrated null
        when a signal-free reference is available); else ``None`` (the derived ``mp`` default,
        which is optimal for an i.i.d. bulk, where a reference adds nothing)."""
        if self._null is not None:
            return self._null
        if self._reference is None:
            return None                                        # -> derived mp default
        key = f"refnull_{kind}"
        if key not in self._cache:
            vals = [top_spectrum_value(np.asarray(_to_np(r)), kind) for r in self._reference]
            self._cache[key] = reference_null(vals)
        return self._cache[key]

    def projection(self, *, far: float | None = None, null=None, seed: int | None = None) -> Projection:
        """The companion :class:`projection.Projection` for the current window -- the PROJECTION.
        The ``K_signal`` noise floor comes from the null PROVIDER: ``null`` overrides, else
        the aperture's ``reference_null`` (if a signal-free ``reference`` was given -- the
        preferred calibrated null), else the derived ``mp`` default (see
        :func:`projection.noise_floor` / :mod:`null_providers`).  The provider is evaluated on
        THIS window's screen (LOCAL).  Deterministic per ``seed`` for a resampling provider."""
        far = self.far if far is None else float(far)
        return Projection(self.W, mask=self.mask, far=far,
                      null=null if null is not None else self._effective_null("projection"),
                      seed=self._seed if seed is None else int(seed))   # native backend

    def extract(self, *, far: float | None = None, reject_persistent: bool = True, shrink: bool = True):
        """The read-side FILTER, through the front door: pull the resolved signal out of the
        aperture's window at NATIVE resolution.  Projects the data onto the projection's modes above
        the derived floor whose footprint is transient-like -- ``reject_persistent`` drops the
        phi_F <= phi_T persistent modes (narrowband RFI); ``shrink`` applies Gavish & Donoho (2017)
        optimal singular-value shrinkage against the derived floor (else a hard floor cut).

        Nothing is synthesised: ``clean`` is a linear PROJECTION of the MEASURED data onto its own
        resolved modes (``clean = U diag(Sd) Vt``, U/Vt from the data's own projection).  Returns
        ``(clean, info)`` -- ``info`` carries K_signal, contrast, and the kept/dropped mode indices
        with their phi_T/phi_F.  The filter itself is :func:`extract.filter_projection`, which takes
        the projection this aperture already holds."""
        return filter_projection(self.projection(far=far),           # `far=None` -> self.far
                                 reject_persistent=reject_persistent, shrink=shrink)






    def rates(self) -> DecayRates:

        """EXACT per-mode decay rates alpha_k = -log|mu_k| and frequencies

        beta_k = arg(mu_k) from the dynamical operator's eigenvalues -- long_range

        (slowest) and short_range (fastest) in one read: the exact counterpart to

        the entropy-width ``a_delta`` approximation."""

        return self._core().rates()


    def propagator_full(self, *, rcond: float | None = None):

        """The full-space one-step propagator A (F x F), x_{t+1} ~= A x_t

        (see Dynamics.propagator_full)."""

        return self._core().propagator_full(rcond=rcond)


    def predict(self, x, *, rcond: float | None = None):

        """One-step forecast A x for a state ``x`` using the streaming operator

        (see Dynamics.predict)."""

        return self._core().predict(x, rcond=rcond)


    def resolved(self, *, far: float | None = None, k: int | None = None) -> int:

        """Streaming ``K_signal`` from the operator (``P_{xx}``, all frames), scored by the

        aperture's effective floor (``reference_null`` if a ``reference`` was given, else

        ``mp``).  The global, O(F^3)-once form of the window ``Projection``'s ``K_signal``

        (see :meth:`dynamics.Dynamics.resolved`)."""

        return self._core().resolved(null=self._effective_null("bulk"),

                                     far=(self.far if far is None else float(far)),

                                     seed=self._seed, k=k)


    def has_signal(self, *, far: float | None = None, null=None) -> bool:

        """CHEAP signal GATE for the current window: ``True`` iff the screen resolves at

        least one mode above the noise floor (``K_signal > 0``).  Builds only the

        lightweight monitor :class:`Projection` (fold + singular VALUES + floor -- **not** the

        SVD basis, coherence, embedding, or the Koopman/DMD operator), so a caller can gate

        the HEAVY reads and never form the DMD operator or the full optics on a structureless

        window::



            if ap.has_signal():        # fold + svdvals + floor (the monitor Projection)

                rates = ap.rates()     # expensive DMD -- only when there IS signal



        NOTE on when this pays off: the gate itself builds the monitor ``Projection`` (the fold is

        the dominant cost), so gating is a net win only when the gated chain is MUCH heavier

        than one Projection (e.g. ``rates`` + full ``optics`` at large ``F``).  There is no gate

        cheaper than the fold -- every read forms the screen first; the ensemble lever is

        instead to BATCH the fold across frames (:func:`projection.read_batch`).

        :func:`projection.probe_signal` is an SVD-free variant (still folds; conservative)."""

        return bool(self.projection(far=far, null=null).has_signal)   # `far=None` -> self.far


    @property

    def footprints(self) -> list[Beam]:

        """Per-mode localization fingerprints of the resolved screen modes (see

        :class:`beam.Beam`): the SHAPE of each mode above the noise floor

        -- broadband signal vs a localized (narrowband / compact) blob at the same

        singular value.  Reads what the scalar resolved-dimension count cannot."""

        return self._c("footprints", lambda: self.projection().footprints)


    @property

    def significance(self) -> ModeSignificance:

        """Per-mode evidence against the noise null (see :class:`projection.ModeSignificance`):

        the standardized Tracy-Widom deviate and tail probability of every singular value.

        ``K_signal == #(pvalue < far)`` -- the read reports the evidence, the false-alarm

        level is the reader's."""

        return self._c("significance", lambda: self.projection().significance)


    def tensor(self, d: int | None = None, *, rank: tuple | None = None) -> dict:

        """Delay-embedded Tucker (HOSVD) of the current window at NATIVE resolution

        -- exposes HOW the feature spectrum evolves WITHIN a window (the fine

        structure the averaged screen SVD loses).  ``d`` is the delay-window width.

        Backend-agnostic (on-GPU for a torch window).  See tensor.tensor_embed."""

        return tensor_read(self.W, self.mask, d, rank=rank)   # native backend


    def sweep(self, *, patch: int = 1024, step: int | None = None,
              coherence: float | None = None, null="local", local_window: int = 3):
        """Sweep a bounded aperture across this signal's FEATURE axis and read where it is coherent.

        One signal, wider than one aperture: entroptics reads a finite aperture, not an infinite
        field, so a field too wide to resolve in one read is swept by a fixed-capacity patch and the
        coherence of each patch is the gate.  Returns per-band dicts with the column ``span`` and
        the on-pulse ``width`` / ``tau_decay`` in samples.

        ``coherence`` defaults to ``far`` corrected for the sweep's own multiplicity; ``null``
        chooses how each coherent patch's floor is calibrated (``"local"`` = region-dynamic).
        See :func:`sweep.sweep` for both in full."""
        return _sweep(self.W, self.mask, patch=patch, step=step, coherence=coherence,
                      null=null, far=self.far, local_window=local_window)

    def scale_profile(self, windows=None, *, far: float | None = None, null=None,
                      seed: int | None = None) -> ScaleProfile:
        """Structure vs observation window (resolution vs aperture size): sweep
        trailing ordered-axis windows and read the structure at each.  ``windows``
        are ordered-axis CELLS, not seconds (see reads.scale_profile).

        Carries the aperture's own floor into every window, like every other read here:
        ``K_signal`` counts against it and ``contrast`` divides by it, so a profile read
        on the default floor while the aperture was given another would be a different
        measurement wearing the same name."""
        return scale_profile(
            self.W, windows, mask=self.mask,
            far=self.far if far is None else float(far),
            null=null if null is not None else self._effective_null("projection"),
            seed=self._seed if seed is None else int(seed))

    def __repr__(self) -> str:
        if self._batch is None and not self._buf:
            return (f"Aperture(streaming, frames={0 if self._dyn is None else self._dyn.n_frames}, "
                    f"window={self.window})")
        return (f"Aperture(shape={self.W.shape}, phi={self.phi:.3f}, "
                f"etendue={self.etendue:.3f}, strehl={self.strehl:.3f})")


__all__ = [
    # the front door + the projection companion + the two-way screen
    "Aperture", "Projection", "Screen", "Lens", "Beam", "ScreenRead", "Balance", "Transfer",
    "Realisation", "Linearity", "Losslessness",
    # per-axis
    "AxisRead", "axis_read", "axis_spectrum",
    # the rank edge: occupancy read from the profile's own step, not from a floor
    "occupied_modes", "OccupiedModes",
    # the level edge: the best two-population split, and how much of the spread it explains
    "level_edge", "LevelEdge",
    # spectrum / propagation
    "SpectralOptics",
    "attenuation_interval", "CertifiedInterval", "concentration_band",
    "resolved_dimension_interval", "CertifiedCount", "SpectralAccumulator",
    "Concentration",
    "Coupling",
    # decay (OTF) + diffraction limit + Mercer certificate
    "diffraction_limit", "DiffractionLimit",
    # `decay_scatter` is NOT exported: it takes a frame, so a top-level copy would be a second
    # path to `Aperture.decay_scatter`.  The primitive stays reachable as `reads.decay_scatter`.
    "DecayScatter",
    "MercerCertificate",
    "rayleigh_shape_factor", "fresnel_number", "shape_factor",
    # dynamical operator (exact decay rates, streaming, splice-able)
    "Dynamics", "DecayRates", "DynamicsState",
    # per-mode localization footprints (the shape of each resolved mode)
    "footprints",
    # per-mode significance (the evidence the floor thresholds; alpha is the reader's)
    "mode_significance", "ModeSignificance",
    # a {kind: provider} mapping for a different provider per cut point -- to any read/Aperture)
    "null_providers", "FloorContext", "KINDS", "by_kind", "floor_from_null_sampler",
    "permutation", "reference_null", "ReferenceNull", "self_calibrating_null",
    "top_spectrum_value", "shuffle_in_time",
    # delay-embedded Tucker (HOSVD) -- within-window fine structure
    "tensor_read", "tensor_embed", "tensor_reconstruct", "tensor_fidelity",
    # N-D field reduction (geometry-preserving)
    "slabs", "over_planes", "pool",
    # structure vs observation window
    "ScaleProfile",
    # full read + geometry + projection
    "ProjectionRead",]
