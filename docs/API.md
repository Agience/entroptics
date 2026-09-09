# API reference

The public surface of `entroptics`, generated from the library itself by
[`docs/generate_api.py`](generate_api.py) -- every name, signature and summary is read from the
code, so this cannot drift from what is installed.

Everything listed here is importable directly from the package:

```python
from entroptics import Aperture, Projection, Screen
```

A name absent from this page is not public API, whatever its visibility in Python: it may change
without notice. The construction each read implements is defined in
[`research/PAPER.md`](../research/PAPER.md), which is the reference for what the numbers mean.


## The front door

One object over a 2-D record; every read below is reachable from it.

| name | kind | summary |
| --- | --- | --- |
| `Aperture(W=None, mask=None, *, window=<unset>, forgetting: 'float' = 1.0, rank: 'int \| None' = None, null=None, reference=None, seed: 'int' = 0, far: 'float' = 0.05)` | class | An optical aperture whose resolution is set by a signal's own Shannon entropy |

## Optical reads

The aperture quantities: fill fractions, etendue, Strehl, the decay and its diffraction limit, coherence, coupling, concentration.

| name | kind | summary |
| --- | --- | --- |
| `AxisRead(H: 'float', n: 'int', delta: 'float', phi: 'float', sigma: 'float') -> None` | class | Everything about ONE axis of a screen (see axis_read()). |
| `Carriage(z: 'object', carried: 'float', resolved: 'int', cutoff: 'float', n: 'int', effective_n: 'float') -> None` | class | What the weights of a weighted read carry (see :func:`carriage`). |
| `CertifiedCount(resolved_modes: 'int', resolved_lo: 'int', resolved_hi: 'int', band: 'float') -> None` | class | A certified interval for the resolved-mode count (see resolved_dimension_interval()). |
| `CertifiedInterval(attenuation: 'float', attenuation_lo: 'float', attenuation_hi: 'float', band: 'float', certified: 'bool') -> None` | class | A certified interval for the attenuation constant alpha (see attenuation_interval()). |
| `Concentration(intensity: 'float', focus: 'float', resultant: 'float', n: 'int', dim: 'int') -> None` | class | The concentration / focus of a stack of row-vectors (see concentration()). |
| `Coupling(z: 'float', sign: 'int', strength: 'float', phase: 'float', tightness: 'float', resolved: 'bool', cutoff: 'float', n: 'int') -> None` | class | The MEASURED coupling between two sides that meet on a shared basis (see :func:`coupling`). |
| `DecayScatter(noise_share: 'float', tail_share: 'float', channels: 'int') -> None` | class | How much of a decay is the record's OWN sampling scatter, measured from the record. |
| `DiffractionLimit(a_delta: 'float', xi: 'float', a_delta_abbe: 'float', H: 'float') -> None` | class | The diffraction limit a_delta from a decay profile (the temporal read). |
| `LevelEdge(k: 'int', separability: 'float') -> None` | class | Where an ordered profile separates into two populations, and how much of its spread that separation accounts for (see :func:`level_edge`). |
| `MercerCertificate(a_delta_temporal: 'float', a_delta_spectral: 'float', ratio: 'float', n_dof: 'float') -> None` | class | The model-free Mercer certificate: a_delta read TWO independent ways |
| `OccupiedModes(k: 'int', margin: 'float', step: 'float') -> None` | class | Where an aperture's occupied modes end and its empty ones begin, with the evidence for it (see :func:`occupied_modes`). |
| `ScaleProfile(windows: 'np.ndarray', K_signal: 'np.ndarray', contrast: 'np.ndarray', coherence: 'np.ndarray', a_delta: 'np.ndarray', phi_T: 'np.ndarray', resolved_window: 'int', dominant_window: 'int', transitions: 'np.ndarray') -> None` | class | Structure as a function of observation window (see scale_profile()) -- the zoom-by-density read. |
| `SpectralAccumulator(n_features: 'int', *, whiten: 'bool' = False) -> 'None'` | class | Pool the feature correlation over intact 2-D planes and/or an ensemble into ONE spectrum. |
| `SpectralOptics(contrast: 'float', top_share: 'float', resolved_modes: 'int', noise_floor: 'float', attenuation: 'float', phase: 'float', dispersion: 'float', resolved_power: 'float', dominance: 'float', eigenvalues: 'np.ndarray') -> None` | class | The optics of the correlation eigenspectrum (see spectral_optics()). |
| `attenuation_interval(data: 'np.ndarray', mask: 'np.ndarray \| None' = None, *, band: 'float', sg: "'SpectralOptics \| None'" = None) -> 'CertifiedInterval'` | function | Certified interval for the attenuation constant alpha given an input spectral-norm band ``band`` (an upper bound on \|\|C_input - C_true\|\|_2). |
| `axis_read(W, axis: 'int', mask=None, *, geom: 'dict \| None' = None, evals=None) -> 'AxisRead'` | function | Bundle the per-axis reads (H, n, delta, phi, sigma) for ``axis`` (0 = ordered/T, 1 = feature/F). |
| `axis_spectrum(W, axis: 'int', mask=None)` | function | Correlation eigenvalues (descending, non-negative) along ``axis`` (0 = ordered/T, 1 = feature/F) -- the per-axis correlation eigenspectrum every axis read is derived from. |
| `carriage(X, w, *, far: 'float' = 0.05) -> 'Carriage'` | function | How much of a WEIGHTED aggregation the weights actually carry, against the exact null that the weights are re-paired with the frames at random. |
| `concentration_band(n_rows: 'int', n_cols: 'int', *, spec_norm: 'float' = 1.0, c_conc: 'float' = 2.0) -> 'float'` | function | A-priori spectral-norm band for the EMPIRICAL correlation matrix from ``n_rows`` iid samples of an ``n_cols``-dim vector. |
| `diffraction_limit(profile) -> 'DiffractionLimit'` | function | The diffraction limit a_delta from a 1-D decay ``profile`` C(tau), read from the ENTROPY WIDTH of the decay (see the module notes for the Wiener-Khinchin -> OTF -> Abbe chain): |
| `fresnel_number(W, window, mask=None) -> 'float'` | function | FRESNEL number N_F ~ window * phi_T -- the near/far-field (UV/IR) coordinate. |
| `level_edge(weights) -> 'LevelEdge'` | function | The LEVEL EDGE of an ordered profile: the split that best separates it into a high group and a low one, by maximum between-class variance. |
| `occupied_modes(weights) -> 'OccupiedModes'` | function | The RANK EDGE of an ordered spectrum: how many modes carry power, read from the profile own step rather than from a noise floor. |
| `rayleigh_shape_factor(profile) -> 'float'` | function | The Rayleigh SHAPE FACTOR g = xi * a_delta (the paper's "shape factor g", Prop 4.5) -- the integral correlation length times the entropy-width diffraction limit. |
| `resolved_dimension_interval(data: 'np.ndarray', mask: 'np.ndarray \| None' = None, *, band: 'float', sg: "'SpectralOptics \| None'" = None) -> 'CertifiedCount'` | function | Certified interval for the resolved-mode count ``K = #{eigenvalue > edge}`` given an input spectral-norm band ``band`` (an upper bound on ``\|\|C_input - C_true\|\|_2``). |
| `shape_factor(W, profile, mask=None) -> 'float'` | function | The Abbe RESOLUTION FACTOR c = a_delta / phi_F (Rayleigh / Abbe: resolution = factor / aperture) -- the screen's own "1.22", read per-signal, not a universal constant. |
| `spectral_batch(frames, *, null=None, far: 'float' = 0.05, seed: 'int' = 0) -> 'list'` | function | Read :func:`spectral_optics` for a BATCH of same-shape 2-D frames in one pass -- the column de-mean and the ``(N, N)`` covariances are formed batched (``(B, T, N)`` -> ``(B, N, N)``), then EACH frame's eigenspectrum + floor + optics is assembled by the SAME :func:`_spectral_from_cov` the per-frame ``spectral_optics`` calls. |

## The screen and its noise floor

Whiten, fold to the entropy-matched grid, and count what stands above the derived Tracy-Widom floor.

| name | kind | summary |
| --- | --- | --- |
| `BatchRead(K_signal: 'int', sigma_top: 'float', noise_floor: 'float', S: 'np.ndarray') -> None` | class | One frame's lightweight monitor read from :func:`read_batch` |
| `ModeSignificance(deviate: 'np.ndarray', pvalue: 'np.ndarray') -> None` | class | Per-singular-value evidence against the noise null, carrying no threshold: the standardized Tracy-Widom deviate and the tail probability of every singular value. |
| `Projection(W, mask=None, *, far: 'float' = 0.05, null=None, seed: 'int' = 0)` | class | The projection of a signal onto its entropy-matched screen. |
| `ProjectionRead(delta_T: 'float', delta_F: 'float', coherence: 'float', K_signal: 'int', H_screen: 'float', sigma_top: 'float', noise_floor: 'float') -> None` | class | The full screen read as a plain record (output of ``read``). |
| `footprints(U: 'np.ndarray', S: 'np.ndarray', Vt: 'np.ndarray', k_signal: 'int') -> 'list[Beam]'` | function | Per-mode read: each of the ``k_signal`` modes standing above the noise floor, as a leaf :class:`beam.Beam`. |
| `mode_significance(screen: 'np.ndarray', s: 'np.ndarray \| None' = None) -> 'ModeSignificance'` | function | Per-mode significance of the screen's singular spectrum against the derived noise null, free of any threshold: for each singular value ``s_k`` the standardized Tracy-Widom deviate ``g_k = (s_k^2/sigma^2 - mu)/sigma_J`` and its tail probability ``p_k = P(TW1 > g_k)``. |
| `read_batch(frames, *, far: 'float' = 0.05, null=None, seed: 'int' = 0) -> 'list[BatchRead]'` | function | Read a batch of same-shape frames onto their screens in one vectorized pass and return a per-frame :class:`BatchRead` (``K_signal`` / ``sigma_top`` / ``noise_floor`` / ``S``), **bit-identical** to ``[Projection(f).read()-equivalent for f in frames]`` but amortizing the per-frame fold + object overhead (the ensemble throughput lever at small ``F``). |

## Null providers

The threshold a detection is taken against. The derived edge is the default; a caller may supply their own.

| name | kind | summary |
| --- | --- | --- |
| `FloorContext(spectrum: 'np.ndarray \| None', data: 'np.ndarray \| None', shape: 'tuple', far: 'float', kind: 'str', rng: "'np.random.Generator'") -> None` | class | Everything a null provider may need for one screen, so a provider is a pure ``FloorContext -> float`` returning a scalar in the units of ``spectrum``. |
| `ReferenceNull(reference_top_values=None, *, far: 'float \| None' = None, forgetting: 'float' = 1.0)` | class | Stateful, O(1) sharpening reference null (see :func:`reference_null`): maintains the running mean/variance of a signal-free reference's top-mode values by Welford's method (O(1) per sample, O(1) memory) and floors at ``mean + z(far)*std``, sharpening analytically to any ``far``. |
| `by_kind(**providers) -> 'Callable[[FloorContext], float]'` | function | Compose per-cut-point providers into one provider that dispatches on ``ctx.kind``: ``by_kind(projection=P, spectral=Q, bulk=R)`` sends the screen floor (``K_signal``), the single-screen correlation floor (``resolved_modes``), and the pooled ensemble floor (``SpectralAccumulator``) to P, Q, R respectively. |
| `floor_from_null_sampler(surrogate: "Callable[[np.ndarray, 'np.random.Generator'], np.ndarray]", *, draws: 'int' = 200, far: 'float \| None' = None) -> 'Callable[[FloorContext], float]'` | function | Turn any surrogate into a null provider: ``floor = (1 - far)`` quantile of the top spectrum value over ``draws`` draws of ``surrogate(data, rng)``. |
| `permutation(*, draws: 'int' = 200, far: 'float \| None' = None) -> 'Callable[[FloorContext], float]'` | function | One built-in example of a caller-style sampled null: ``floor_from_null_sampler(shuffle_in_time, draws=draws, far=far)`` -- the distribution- free permutation floor (destroys cross-channel structure, keeps each marginal). |
| `reference_null(reference_top_values, *, far: 'float \| None' = None) -> 'Callable[[FloorContext], float]'` | function | A deterministic O(1) null calibrated on a signal-free reference: the floor is ``center + z(far)*scale`` with ``center, scale`` the mean and std of the reference's top-mode values (``top_spectrum_value`` of each signal-free realisation) and ``z(far)`` the standard-normal inverse survival function. |
| `self_calibrating_null(noise, kind: 'str' = 'projection', *, block_rows: 'int', stride: 'int \| None' = None, far: 'float \| None' = None)` | function | A :func:`reference_null` calibrated locally on a region's own signal-free noise -- the self- contained, region-dynamic form. |
| `shuffle_in_time(X: 'np.ndarray', rng) -> 'np.ndarray'` | function | An example surrogate: shuffle each column independently along the ordered axis (rows), destroying ordered and cross-channel structure while preserving every channel's own marginal. |
| `top_spectrum_value(X: 'np.ndarray', kind: 'str') -> 'float'` | function | Score a surrogate ``X`` in the same units the floor thresholds: the top singular value (``kind="projection"``) or the top unit-diagonal correlation eigenvalue (any other kind -- ``"spectral"`` / ``"bulk"``). |

## The streaming operator

Online DMD / Koopman: per-mode decay rates from a fixed-size sufficient statistic, splice-exact across segments.

| name | kind | summary |
| --- | --- | --- |
| `DecayRates(mu: 'object', alpha: 'object', beta: 'object', long_range: 'float', short_range: 'float', dominant: 'float', n_modes: 'int', n_frames: 'int') -> None` | class | Exact per-mode decay rates from the dynamical operator's eigenvalues. |
| `Dynamics(n_features: 'int', *, forgetting: 'float' = 1.0, rank: 'int \| None' = None, far: 'float' = 0.05, null=None)` | class | Streaming dynamical operator (online DMD / Koopman) on a sequence of feature vectors x_t in R^F (or C^F). |
| `DynamicsState(Pxx: 'object', Pyx: 'object', first: 'object \| None', prev: 'object \| None', forgetting: 'float', n_frames: 'int', n_pairs: 'int', Px: 'object \| None' = None) -> None` | class | The full state of a Dynamics operator -- the complete tensors and counts, sufficient to resume, splice, or reconstruct the operator exactly. |
| `HankelSpectrum(evals: 'object', isolation: 'float', psd: 'float', n: 'int') -> None` | class | Transfer/Koopman eigenvalues read from a scalar correlation sequence's own moments. |
| `hankel_spectrum(c, n: 'int', *, rcond: 'float' = 1e-06) -> 'HankelSpectrum'` | function | The transfer-operator spectrum of a real correlation sequence via the reflection-positive moment pencil (a.k.a. |
| `jackknife(samples, read, *, n_bins: 'int \| None' = None)` | function | Delete-one(-bin) jackknife point estimate and standard error of a scalar ``read``. |

## The observable lift

Delay embedding, for trajectories with no linear one-step map on their own coordinates.

| name | kind | summary |
| --- | --- | --- |
| `delay_embed(W, d: 'int')` | function | Delay-embed a trajectory ``W`` ``(T, F)`` into Hankel observables ``(T-d+1, d*F)``: row ``t`` is the window ``[w_t, w_{t+1}, ..., w_{t+d-1}]`` flattened -- the Takens / Hankel coordinates in which nonlinear dynamics become approximately linear (Koopman). |
| `koopman_lift(W, d: 'int', *, forgetting: 'float' = 1.0, rank: 'int \| None' = None, far: 'float' = 0.05, null=None) -> 'Dynamics'` | function | Fit a :class:`Dynamics` operator on the delay-embedded trajectory -- the blessed lift from a nonlinear ``(T, F)`` trajectory to a linear Koopman operator. |

## The two-way screen

Two or more sides meeting on one shared basis: lenses, beams, crossings, coupling.

| name | kind | summary |
| --- | --- | --- |
| `Balance(total: 'float', offsets: 'dict', residual: 'dict', pvalue: 'dict', closed: 'dict', frame: 'object') -> None` | class | The screen's self-balancing zero (see ``Screen.balance``). |
| `Lens(name: 'str', entry: 'Callable[[Any], Any]', inverse: 'Callable[[Any], Any] \| None' = None, energy: 'Callable[[Any], Any] \| None' = None, zero: 'Callable[[Any], Any] \| None' = None, null: 'Any' = None) -> None` | class | One side of a screen: its name, its own conversions, and its own laws. |
| `Linearity(additivity: 'float', homogeneity: 'float', modes: 'int', linear: 'bool') -> None` | class | Whether a lens passes a beam's modes independently (see ``Screen.linear``). |
| `Losslessness(residual: 'float', sigma_top: 'float', noise_floor: 'float', K_signal: 'int', lossless: 'bool') -> None` | class | The conversion certificate of one lens (see ``Screen.certify``): is ``inverse(entry(surface))`` the surface again, within the resolution the aperture reports? The decision is the conjunction of two derived checks, neither of them a chosen tolerance: |
| `Realisation(ideal: 'float', realised: 'float', efficiency: 'float', shortfall: 'float', passive: 'bool') -> None` | class | What a crossing actually delivers through the receiving lens's own conversion, against what the etendue bound permits (see ``Screen.realise``). |
| `Screen(*, far: 'float' = 0.05, null=None, seed: 'int' = 0) -> 'None'` | class | The two-way screen: lenses meeting on a shared basis, read from either side. |
| `ScreenRead(n_lenses: 'int', T: 'int', D: 'int', K_signal: 'int', contrast: 'float', top_share: 'float', noise_floor: 'float', attenuation: 'float', coherence: 'float', a_delta: 'float', correlation_length: 'float', basis_dim: 'int') -> None` | class | The aperture measurement of the screen's joint frame (see ``Screen.read``). |
| `Transfer(absorbed: 'float', transmitted: 'float', flux: 'float', energy: 'float', participation: 'float', pertinent: 'float', tau: 'float', delivered: 'float', reflected: 'float', bystanding: 'float', etendue_from: 'float', etendue_to: 'float', match: 'float', concentration: 'float', radiance_from: 'float', radiance_to: 'float', modes_from: 'int', modes_to: 'int', condensation: 'list') -> None` | class | What crosses the screen from one side to the other (see ``Screen.transfer``). |

## Beams

A placed frame read as an aperture.

| name | kind | summary |
| --- | --- | --- |
| `Beam(lens: 'str', index: 'int', energy: 'float', flow: 'np.ndarray', basis: 'np.ndarray', profile: 'np.ndarray', _fills: 'object' = None, _modes: 'object' = None) -> None` | class | What one side carries |

## Batched reads

The same read over a stack of frames, on CPU or GPU, equal to the per-frame result.

| name | kind | summary |
| --- | --- | --- |
| `ResolvedBatch(K_signal: "'np.ndarray'", sigma_top: "'np.ndarray'", noise_floor: "'np.ndarray'", energy: "'np.ndarray'" = None, projector: "'object'" = None) -> None` | class | One batched resolved-screen read (output of :func:`resolved_batch`). |
| `ResolvedScreen(F, *, far: 'float' = 0.05, null=None, seed: 'int' = 0, refresh_every: 'int' = 32, warmup: 'int' = 64, whiten: 'bool' = True, forgetting: 'float' = 1.0)` | class | A stateful, resumable resolved screen for a revisited screen (e.g. |
| `ResolvedScreenBatch(B, F, *, far: 'float' = 0.05, null=None, seed: 'int' = 0, refresh_every: 'int' = 32, warmup: 'int' = 64, whiten: 'bool' = True, forgetting: 'float' = 1.0)` | class | The scalable stateful resolved screen -- ``B`` revisited screens (e.g. |
| `ResourceLimits(threads: 'int \| None' = None, memory_gb: 'float \| None' = None, gpu: 'bool' = True) -> None` | class | A caller's resource envelope for a :func:`resolved_batch` read |
| `recommend_backend(B: 'int', T: 'int', F: 'int', *, data_on_gpu: 'bool' = False, cuda_available: 'bool \| None' = None) -> 'tuple[str, str]'` | function | Recommend ``"cpu"`` or ``"gpu"`` for a :func:`resolved_batch` read of a ``(B, T, F)`` stack (``F`` the folded feature width), with a one-line reason -- guidance for placing the tensor (``resolved_batch`` itself dispatches on where the data already lives). |
| `resolved_batch(X, *, fold='auto', far: 'float' = 0.05, null=None, seed: 'int' = 0, energy: 'bool' = False, basis: 'bool' = False, subset=None, device='auto', limits=None, _allow_chunk: 'bool' = True) -> 'ResolvedBatch'` | function | Read a stack ``X: (B, T, F)`` onto the resolved screen |

## N-D fields

Reducing a higher-dimensional field to two axes: pool for ordered reads, plane-fold for feature reads.

| name | kind | summary |
| --- | --- | --- |
| `tensor_embed(data, d: 'int' = 16, rank: 'tuple \| None' = None) -> 'dict'` | function | Delay-embedded Tucker (HOSVD) decomposition of a normalized array. |
| `tensor_fidelity(data, te: 'dict') -> 'float'` | function | Round-trip fidelity for the tensor path: 1 - RMSE(data_norm, data_hat_norm), in [0, 1] (1.0 = perfect). |
| `tensor_read(W, mask=None, d: 'int \| None' = None, *, rank: 'tuple \| None' = None) -> 'dict'` | function | Convenience front door: whiten a raw (T, F) waterfall to native resolution (entropy.normalize) and return its delay-embedded Tucker (HOSVD). |
| `tensor_reconstruct(te: 'dict', T_out: 'int \| None' = None)` | function | Reconstruct normalized data from Tucker factors (inverse of tensor_embed). |

## Field helpers

Constructing and shaping the input record.

| name | kind | summary |
| --- | --- | --- |
| `over_planes(field, plane_axes, read=None, reduce: 'str' = 'mean') -> 'float'` | function | Apply a scalar 2-D ``read`` to each plane of ``field`` (default ``phi``) and aggregate over the slabs. |
| `pool(field, ordered_axis: 'int')` | function | Flatten all non-ordered axes of an N-D ``field`` into the feature axis: return a 2-D ``(L_ordered, F')`` array whose rows are the ordered index and whose columns pool every off-axis site as a sample of the same ordered-axis process. |
| `slabs(field, plane_axes)` | function | Yield each 2-D ``(plane_axes[0], plane_axes[1])`` slice of an N-D ``field``, iterating over all other axes -- geometry-preserving (each plane stays intact). |

## Backend and precision

numpy or torch, chosen by the input; determinism and working precision.

| name | kind | summary |
| --- | --- | --- |
| `precision() -> 'int'` | function | The current compute precision (64 or 32). |
| `set_precision(bits: 'int') -> 'None'` | function | Set the ENVIRONMENTAL compute precision: 64 (default, bit-perfect) or 32 (fast, GPU). |

## Other

| name | kind | summary |
| --- | --- | --- |
| `KINDS` | value | Built-in immutable sequence. |
| `null_providers` | value | null_providers.py |

---

78 public names. Generated by `python docs/generate_api.py` from `entroptics.__all__`; re-run it when the API changes.
