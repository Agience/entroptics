# Changelog

All notable changes to Entroptics are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While the version is below 1.0, a minor or patch
release may change a public name or a returned number; each such change is listed under
**Changed** or **Removed** with what to use instead.

Released versions are archived on Zenodo under the concept DOI
[10.5281/zenodo.21273400](https://doi.org/10.5281/zenodo.21273400), which resolves to the latest
version.

## [0.2.5] - 2026-09-29

The write path, the operator's Fourier view, and reads that downstream analyses had been carrying by
hand.
`extract()` is corrected to the lossless projection it is defined as. The screen's null is
redesigned: the default read now holds its false-alarm level for Gaussian noise at equal and unequal
channel levels, and for heavy-tailed, skewed and count noise, where 0.2.3 read structure into most
of them. Heavy right tails (lognormal, Pareto) still exceed it, by less than 0.2.3 did.
Every screen read moves (the singular values, floor, `K_signal` and coherence); `decay` and the
operator reads are unchanged.

### Added
- **`decay(W, mask, *, periodic=False, disconnected=...)`** (`entroptics.reads`).
  - `periodic=True` reads the ordered axis as a ring, the lag taken modulo `T`, so every lag pairs
    all `T` rows and `C(tau) == C(T - tau)` exactly.
  - `disconnected` chooses whose level is removed: omitted, the record's own mean, as before; `None`,
    nothing; or a level the caller supplies (a scalar, or one per live channel), such as an ensemble
    mean.
  - Refused: a boolean or non-finite level, a complex level on a real record, and a periodic read
    of a record with a dropped row.
  - With neither argument, `decay` returns 0.2.3's numbers exactly.
- A `release-consistency` CI job: it fails when the version in `pyproject.toml` differs from
  `CITATION.cff`, or when the version has no `v<version>` tag.
- **The write path, `Aperture.basis()`** returns a `Basis`: the record's resolved modes as an exact
  encode / decode pair over its channels, the Karhunen-Loeve transform restricted to what the read
  resolved. `encode(frame)` gives each row's `K` mode coordinates and `decode(A)` the frame they
  describe; `split(frame)` returns `(resolved, residual)` with `resolved + residual == frame`. Each
  row is coded on its own, so a basis read once can be shared ahead of time and applied to later
  frames. The rows are orthonormal in the record's noise metric, at native channel resolution:
  each channel is scaled by its noise, read from what the resolved span leaves. A channel never measured decodes
  to NaN; a row with a gap is solved by least squares over its measured cells. `modes=` selects
  resolved modes by index.
  - `Basis.certify(frame)` returns `RoundTrip`, the round trip's certificate: the rows'
    orthonormality defect, which bounds the other two; the inverse error of `encode(decode(A))`;
    the projection's idempotence error; the lossless identity; and the share of the frame the span
    captures. All four error figures are at round-off (1.5e-15 at most over 80 records,
    `research/benchmarks/write_path.py`).
  - `Basis.drift(frame)` returns `Drift`: what a later record holds that the basis does not span,
    read against the existing noise floor.
    - The record is read in its own noise metric. Each channel's noise is read from what the span
      leaves through the method-of-moments system m = |I - P|^2 x, which holds in any metric and is
      solved exactly wherever it has rank -- by Woodbury in O(F K^4) when every channel clears
      the resolution bound, otherwise by an n x n eigendecomposition. Where it has no rank (a
      channel lying in the span, or more channels than the residual can tell apart), the metric
      the span was read in stands.
    - A record with no room outside the span returns `K = 0` and no projection.
    - `Drift.identified` is True only when every channel's noise is fixed to within 1/z of itself
      at `far`: each channel's standard error, from the moments' covariance evaluated at the solved
      noise and carried through the system, is below 1/z of it (on the Woodbury path, a rigorous
      sufficient bound). Few channels outside the span, a short record, or a channel (nearly)
      inside the span leave it False, and the floor's level is then not held. Measured
      (`write_path.py`): at most 5.9% false alarms among identified records on eight
      configurations (8 of 135, within the binomial spread of the level), and no identified channel off by more than 0.1% over 4000 records
      with exact moments.
    - The residual is rotated into the complement of the span, built from the span's Householder
      reflectors.
    - Measured over 80 records at `far = 0.05`: false alarms on later records of the same process
      0/80 with unequal channel noise, 2/80 with equal noise, and 0/80 complex; detections 80/80 for
      a planted extra mode and 75/80 for a local transient.
- **`entroptics.experimental.operator_read(W, *, far)`** (EXPERIMENTAL: not exported at the top
  level; the name, signature and numbers may change) returns `OperatorRead`: the record as an
  operator plus an exact residual. `far` is the only input.
  - The order `K` is the count of past-against-future canonical correlations above the Jacobi-ensemble
    Tracy-Widom edge, on non-overlapping windows, at every dyadic depth and zoom level, with the
    level corrected for every look.
  - The depth is the window span at the coarsest level that still counts.
  - The modes come from the matrix pencil on lags 1 ... 2d.
  - `modes` is a `ModePowers`, so `spectrum(f)` is the same Fourier view. `split(W)` projects the
    record onto its modes' signals: lossless, and NaN where unmeasured.
  - Measured (`research/benchmarks/operator_vs_fft.py`): 3.5% claims on white noise at
    `far = 0.05`; AR(1) decays of 0.104 and 0.675 for 0.105 and 0.693. `research/benchmarks/README.md`
    lists every case, including its weak regimes (an odd count on a damped pair, a 64-sample record) and its
    cost.
- **`research/benchmarks/`**, each output committed and stamped with a SHA-256 of the library
  source that produced it:
  - `operator_vs_fft.py`: read A, the lift and the FFT, rectangular and Hann, on each case, with
    timings;
  - `write_path.py`: the basis's drift rates and certificate, and extract's noiseless recovery;
  - `screen_null.py`: the screen's false alarms over noise families and shapes, and its detection
    and exact rank count on planted signals, with 0.2.3's run beside it;
  - `default_read.py`: flat channels, count noise and the pencil cut, on 0.2.5 and 0.2.3;
  - `readme_examples.py`: every README example, run as printed.
- `research/validation/RESULTS.md` is stamped with the SHA-256 of the source that produced it.
- **`ModePowers.spectrum(f)`**: the Fourier view rendered from the operator,
  S(f) = sum_k P_k (L_k(f) + L_k(-f)) / 2 with L_k the Lorentzian of mode k, at any frequencies.
  It is the transform of the decay `reconstruct_decay` rebuilds, integrates to C(0), and each
  line's width is its decay rate. A mode on or outside the unit circle is
  drawn at its reflected radius, so the view is continuous across the circle.
- `Dynamics.modes()` returns `ModePowers`: each connected mode's pole, decay rate, frequency, power
  P_k and share of the power, largest power first. It is the same computation as
  `reconstruct_decay`, which now uses it, so C(tau) = sum_k P_k mu_k^tau can be rebuilt from it
  exactly. `ModePowers` is exported beside `DecayRates`.
- `bootstrap(samples, read, *, draws=None, rng=0, indices=None)`, exported top-level: the replicates
  of a read on resamples drawn with replacement. Exactly one of `draws` or `indices` is required.
  Replicate `b` draws
  `rng.integers(0, N, N)`, in order, so a hand loop over the same stream gives the same replicates
  bit for bit; a passed `Generator` continues its stream; prebuilt `indices` share one set of draws
  across several reads. A read that raises propagates.
- `jackknife(..., groups=...)`: delete-one-group over explicit index arrays (independent streams,
  chains, blocks; unequal and non-contiguous groups; a sample in no group is always kept). A read
  that returns an array is resampled element-wise, and the estimate and SE come back with its shape.
  A non-finite replicate gives a non-finite SE. The scalar path is unchanged.
- `empirical_bernstein(samples, delta, *, span)` returns `EmpiricalBernstein`: the
  Maurer-Pontil interval on a bounded mean, at confidence `1 - delta`. `span` (the a-priori width of
  the samples' range) is required, since the theorem needs it; a span narrower than the observed
  range is refused.
- `effective_rates(profile)`: the local decay rate log(c[t]/c[t+1]) of a profile, NaN where it is
  not positive at both lags. `crossing_lag(profile, level)`: the lag where a profile first falls to
  `level`, interpolated log-linearly (linearly where a value is not positive), NaN if it never does.
- `matrix_pencil(C0, C1)` returns `MatrixPencil`: the generalised eigenvalues and vectors of a pair
  of measured correlator matrices, whitened on `C0`'s positive spectrum, with `psd` reporting lost
  positivity. `hankel_spectrum` is now built on it, bit for bit.
- `cross_covariance(X, Y, lags, *, periodic=False, disconnected=...)`: the lagged covariance of two
  finite records, summed over channels, negative lags allowed, with `decay`'s conventions and levels;
  `cross_covariance(X, X, range(T))` is `decay(X)`.
- `integrated_autocorrelation(x, *, window="first_nonpositive")` returns
  `IntegratedAutocorrelation`: a chain's integrated autocorrelation time over the positive lobe of
  its own decay, the standard error of its mean, and the statistical error of both, so a drift or a
  chain shorter than its memory shows as `tau_se` comparable to `tau_int`. Only the lags summed are
  formed (O(T W)). `spread_over_chains(values)`: the standard error of a mean over
  independent chains.

### Changed
- **The documentation is reorganised:**
  - the README says what the library does, why and when to use it, with examples and its limits;
  - [`docs/GUIDE.md`](docs/GUIDE.md) holds every read, the input conventions, the glossary and what
    the library guarantees;
  - [`research/benchmarks/README.md`](research/benchmarks/README.md) holds the measurements: the
    screen's false-alarm rates, the comparison with the FFT (rectangular and Hann, including where the
    FFT wins), and costs.
- The version metadata reads 0.2.5 in `pyproject.toml`, `CITATION.cff` and `.zenodo.json`.
  `CITATION.cff` cites the concept DOI until Zenodo mints the 0.2.5 version DOI; the stale 0.2.3
  version DOI is removed.
- The paper is published as Markdown, HTML and PDF (`research/PAPER.md`, `.html`, `.pdf`); the README
  links all three and carries a paper badge.
- Line endings are normalised across the repository, five library files included.
- **`matrix_pencil` / `hankel_spectrum` cut `C0` where the record's own noise shows,** where they
  used a fixed `rcond = 1e-6`. `C0` is positive semidefinite in the limit, so its most negative eigenvalue is the
  size of the noise the record shows; directions at or below it, or below round-off when `C0` is
  positive, are dropped. An explicit `rcond` is still honoured. Measured
  (`research/benchmarks/default_read.py`): about the same at low moment order (AR(0.9) at n = 2,
  0.0079 against 0.0080 over 4000 samples and 0.0154 against 0.0150 over 500), and more accurate at
  high order -- AR(0.9) over 4000 samples at n = 12, leading-eigenvalue error 0.032 against 0.311;
  two modes at n = 12, 0.011 against 0.199. `hankel_spectrum`'s values move where the old cut
  dropped or kept a different set of directions.
- `decay_scatter` takes `periodic` and `disconnected`, meaning what they mean in `decay` (which
  validates them), and returns the per-lag standard error `se` of C(tau). The channel terms are
  formed as `decay`'s own lag products; the shares move only at round-off.
- **`Aperture.extract()` is lossless.** It is the hard projection onto the resolved modes
  (Def 8.4): every channel projected onto the resolved modes' time profiles, read on the screen and
  carried back to `W`'s grid by the fold's adjoint. The singular values are kept as read, so the
  filter is an orthogonal projection (its projector idempotent). The new `info["residual"]` is `W - clean`, and `clean + info["residual"] == W`
  wherever `W` was measured.
  `clean` now always has `W`'s shape at the recorded channel resolution: a channel or row never
  measured is NaN, and `info["centre"]` / `info["scale"]` are per channel of
  `W`. Missing cells are missing in both parts. Returned numbers change: on the calibration burst
  the read beats the raw frame at every noisy S/N (relative error 0.178 / 0.027 / 0.001 at S/N
  10 / 50 / 1000, against 0.109 / 0.028 / 0.012 on 0.2.3, with the screen's new null), and a
  resolved mode is recovered exactly at any contrast above the floor (a burst of any width from half
  a sample to 128 samples, 1.8e-16 to 1.1e-15, and a sine filling the record, 6.9e-16, at contrast
  5.3; `research/benchmarks/write_path.py`).
  `research/figures/calibration.csv` and `.png`, and the paper's §12 numbers, are regenerated.
- The ordered-axis correlation spectrum behind `optics()` is read on the smaller of its two dual
  Gram matrices: O(T F^2) where it was O(T^3). Their nonzero eigenvalues are identical, so every
  returned number is the same up to round-off. What remains of `optics()`'s cost is the decay's
  direct lag sum, O(T^2 F): 12 ms at T = 1024, 272 ms at 4096 and 5.2 s at 16384, F = 16
  (`research/benchmarks/operator_vs_fft.py`).
- **The screen's null is redesigned.** The default read counts the modes above the edge that
  independent channels would reach, at that null's exact variance:
  - each channel is centred on its mean and scaled by its RMS, so the screen's Gram is a sample
    correlation matrix (it was the median and a shrunk median absolute deviation);
  - the feature axis folds by partition: each cell is a contiguous run of whole channels, summed
    and divided by the square root of its length, the runs differing in length by at most one. The
    fold's columns are orthonormal by construction, so independent channels fold to independent
    cells of one variance, in O(T F) (the area mean gave cells of unequal variance, and a cell that
    took a fraction of a channel correlated its neighbours);
  - the floor's per-cell variance is the screen's mean cell energy with the degree of freedom the
    centring took. On an unfolded, fully measured screen this is the null's variance, N / (N - 1),
    exactly, for noise of any marginal; on a folded one, channels that share structure raise it. (It
    was the median row energy with a Gaussian chi-square correction.)
  - a channel is flat when every measured value is the same, and its centre is then that value
    exactly.

  Measured (`research/benchmarks/screen_null.py`, `far = 0.05`, 200 noise-only records per case,
  on 0.2.5 and on 0.2.3), at shapes from 256 x 8 to 20 x 1000: false alarms 0 to 0.020 for
  Gaussian noise at equal and unequal channel levels, real and complex, Student t, exponential,
  chi-square and Poisson. Heavy right tails exceed the level, most at 1024 x 64: lognormal
  (sigma 1) 0.085, lognormal (sigma 1.5) 0.36, Pareto (3) 0.16. The same cases read 1.00 on 0.2.3
  for every exponential, lognormal, chi-square and Pareto case, up to 0.96 for Student t, up to 1.00
  for Poisson, and up to 0.735 for Gaussian noise at unequal levels. Planted
  persistent, burst, edge and two-mode signals are found in 90-100% of records and their rank is
  counted exactly in 72-100% (0.2.3: 21-100%).

  What a correlation read resolves follows from its construction:
  - A whitened channel carries energy `T`, so a mode confined to `k` channels clears the edge only
    when `k > (1 + sqrt(F / T))^2`, about: two channels when `T >> F`, nine at 64 x 256. A tone in
    one channel is not resolved at any strength. A two-channel line near the edge at 256 x 32 is
    found in 4% of records (0.2.3: 58%).
  - A mode is read against the whole record's energy, so a weak mode beside a dominant one can fall
    under the edge.
- **`probe_signal` reads the singular values.** The floor's variance comes from the spectrum, so
  the gate compares the exact top singular value with the floor. It equals `K_signal > 0`, where it
  was a conservative bound, and it still saves the basis, embedding and coherence.
- **`ResolvedScreen` and `ResolvedScreenBatch`** keep each screen's row count, row sum and raw
  Gram, and each refresh forms from them exactly the whitened Gram `resolved_batch` forms from the
  same rows. A channel that never moved leaves the live width, and the floor is read from the
  whitened Gram's eigenvalues. `energy()` whitens its rows the way the basis was read. `forgetting`
  fades per row, so the memory is the same however the stream is blocked, and the floor's row
  count is the weight the Gram carries. Complex rows take the Hermitian Gram. A screen with no live
  channel reads `K = 0`. `state()` carries the three statistics; a state saved by 0.2.3 does not
  load.
- **`null_providers.weighted_effective`** reads the edge of a matrix whose columns carry unequal
  noise: variance `sum v^2 / sum v` at width `(sum v)^2 / sum v^2`, with `v` the per-channel
  noise measured across the ensemble and carried through the folds. The aggregate's channel scale
  carries its own signal, so its noise lands at a different level in each channel.

### Fixed
- **A stream of noise reads as the batch read of the same rows.** `ResolvedScreen` and
  `ResolvedScreenBatch` whitened with statistics frozen from their first `warmup` rows, and put the
  rows before that into the Gram raw. The frozen centre left each channel an offset that grows into
  a rank-one mode over the stream, and a constant channel counted in the floor's width. So both
  read structure in almost every stream of pure noise appended a few rows at a time, as they did
  in 0.2.3. Mean and RMS are sums, so the statistics now accumulate.
- **`weighted_effective` holds its level on a small ensemble.** A sample's deviation from the
  weighted mean it is part of has expected square `Var * (1 - 2/M + 1/eff)`: its own weight enters
  the mean. The factor was missing, so a small ensemble (M = 8, unequal channel levels) read
  structure in pure noise.
- **The read is invariant to the recording level and the memory layout.** The geometry squares
  the raw frame, so a frame at 1e155 (1e19 in float32) overflowed, and the matched scale collapsed
  to no fold. The frame is now carried by the power of two of its largest measured cell, which is
  exact, and every read of it is a ratio. The frame is also given one memory layout first, since a
  column-major frame summed in a different order and read differently in the last bits.
- **Complex records through the batched read.** `resolved_batch(..., energy=True)` squared the left
  singular vectors where it needed their modulus, and its projector paired `V` with its transpose
  where it needed the conjugate transpose, so both were wrong for complex rows; the sign iteration's
  trace is now taken real. A frame with a constant channel returns its projector on the batch's
  channel grid, so equal shapes stack.
- **A channel far above its noise is centred exactly.** Each channel's mean is taken relative to
  its first measured value, whose differences from nearby values are exact, so a level 1e14 times
  the noise keeps its centre to the last bit, on numpy and torch alike.
- **A channel that never moved no longer manufactures structure.** A constant channel, such as a
  stuck sensor, carried nothing to the screen, yet its level steered the entropy fold and diluted
  the columns it was folded into. It now leaves the screen the way a dead channel does: out of the
  geometry, the fold and the floor. `Aperture.extract` and `Aperture.basis` keep it at its value,
  and the residual holds the rest, so both stay lossless. `read_batch` and `resolved_batch` read
  such a frame exactly as `Projection` does. Measured (`research/benchmarks/default_read.py`, on
  0.2.3 and on 0.2.5): false alarms at `far = 0.05` with 2 constant channels beside 8 of noise went
  from 52% to 0.5%, with 8 constant channels from 96% to 0.5%, and beside 64 of noise from 65% to
  1%. Eight sparse count channels, which 0.2.3 could not scale (38.5%), are now whitened like any
  other: 0.5%.
- **Count noise reads as noise.** Poisson noise over 8 channels read structure in 40% (mean 1) and
  10% (mean 3) of noise-only records on 0.2.3, and reads it in 0.8% and 0% now, with a planted mode
  still found in every record (`default_read.py`).
- **The whitening takes a channel at any level.** `normalize` carries each channel by the power of
  two of its own peak, exactly, so a channel at 1e-300 or 1e300 in float64, or at 1e20 in float32,
  whitens to what it would at 1 (to round-off); and its sums run in one fixed order, so a record
  whitens the same whatever its memory layout.

### Removed
- The `shrink` argument of `Aperture.extract()` and `extract.gavish_donoho`. The singular-value
  shrinkage altered the singular values, so the filter was neither lossless nor idempotent. Passing
  `shrink` raises `TypeError`.
- `entropy.mad_stats` and `entropy.resolution_floor`, with the median-absolute-deviation whitening
  they served, and the constant `entropy.MAD_LOGVAR`. Use `entropy.whiten_stats(xp, data, bad)`,
  which returns the whitening's `(centre, scale)`.
- The `warmup` argument of `ResolvedScreen` and `ResolvedScreenBatch`: the statistics accumulate from
  the first row.

## [0.2.3] - 2026-09-09

Version DOI: [10.5281/zenodo.22687899](https://doi.org/10.5281/zenodo.22687899). The public
surface is unchanged from 0.2.2.

### Fixed
- **Reads at very small magnitudes.** The default floor's noise variance carried an absolute
  `1e-30` guard, which inflated it 2.79x at a screen scale of 1e-15 and 1.8e6x at 1e-18. The guard
  is now relative. `mode_significance` returns a deviate of 0 and a p-value of 1 on a screen with
  no energy.
- **`projection.coherence`** no longer overflows to NaN above a magnitude of about 1e38. Its
  cancellation guard is relative: a change of units by 10x no longer moves the z-score.
- **Correlation reads** (`spectral_optics`, `principal_directions`) normalise with a floor relative
  to the largest variance. `resolved_modes` used to read 0 on a screen scaled to 1e-30.

### Changed
- Docstrings state that `resolved_modes` is not a rank or model-order selector, and that `coupling`
  at one shared coordinate is Pearson's r.

### Documentation
- `docs/API.md`, generated from the library by `docs/generate_api.py`.
- `CODE_OF_CONDUCT.md`; `CONTRIBUTING.md` gains a section on reporting an issue.
- The README gains a statement of need, documentation and help sections, a list of what runs
  without a download, and a declaration of generative AI use.
- The paper moves the two-way screen and symbol sequences to appendices. Supplemental papers
  (coupling, FRB, JOSS) are added with reproduce and verify scripts.

## [0.2.2] - 2026-09-08

### Added
- **`carriage(X, w, *, far=0.05)` and `Carriage`:** what the weights of a weighted aggregation
  carry, against an exact permutation null, including Kish's `effective_n`.
- **`null_providers.weighted_effective(stack, weights)`:** a closed-form floor for a weighted mean.
- **Complex noise model (TW2).** Adds:
  - `tw2_sf`, `tw2_quantile` and `tw_quantile(far, *, complex_=False)`;
  - a `complex_` keyword on `johnstone`, `debias_denominator`, `noise_sigma2` and
    `screen_floor_sq`;
  - `entropy.MAD_SCALE_C`.
- `normalize(..., return_stats=True)` returns the whitening map `(whitened, centre, scale)`, and a
  `Projection` exposes `centre` and `scale`.

### Changed
- **`Aperture.extract()` returns `clean` in the caller's units.** The whitening is undone, and
  `info` reports `centre` and `scale`.
- **A complex frame gets the complex noise model:** the TW2 quantile, complex Johnstone centring
  and complex whitening. `K_signal` and `noise_floor` change on complex input; real input is
  unchanged.
- `phi` and `magnification` are read on the block as recorded, with no centring, as in 0.1.5.
- `entroptics.__version__` comes from the installed distribution. An uninstalled source tree
  reports `0.0.0+source`.
- `entroptics.entropy` no longer imports scipy.

### Fixed
- `coupling` no longer raises when one side is real and the other complex.

## [0.2.1] - 2026-09-02

### Changed
- **`Screen` became `Projection`.** The module `screen.py` became `projection.py`, `ScreenRead`
  became `ProjectionRead`, and `Aperture.screen()` became `Aperture.projection()`. The null
  providers' cut point `"screen"` became `"projection"`.
  - `Screen`, `ScreenRead` and `entroptics.screen` still import, but they are now the shared screen
    between systems (below). `Screen(W)` raises `TypeError`.
- **`Aperture(W)` reads all of `W`.** A finite record is read whole; a streaming `Aperture()` keeps
  a window. An explicit `window=` still wins.
- **`Aperture(..., far=0.05)`** sets the aperture's false-alarm level, which its reads use unless
  given their own.
- **The feature fold requires continuity as well as concentration.** The band is
  `fold_band(T, F, far=)`: a Dirichlet/Cantelli significance term and a Tracy-Widom sufficiency
  term. Where the fold decision changes, `K_signal` and the reads that follow it change.
- **The ordered-axis reads** (`phi_T`, `sigma_T`, `etendue`, `strehl`) are read on the centred
  block.
- **`decay` reads the record as given.** It no longer squares non-negative input; pass `W**2` for
  the incoherent read.
- **`sweep(W, mask=None, *, coherence=None, ...)`:**
  - the coherence gate is derived from `far` and the number of patches;
  - it accepts a mask;
  - it keeps complex input complex.
- **`Dynamics.connected_decay_rate`** returns NaN for a growing fit instead of a negative rate.
- **Round-off tolerances are relative to the data's own scale:** dead-channel detection,
  `forgetting(tol=None)`, `shannon_bits` and `surprisal_bits`.
- **Reads accept `mask=`,** and `Aperture` passes its own mask to them.
- Importing `entroptics` sets `OPENBLAS_NUM_THREADS=1` if it is not already set.

### Added
- **The shared screen between systems,** all top-level: `Screen` (`register`, `place`, `update`,
  `transfer`, `realise`, `couple`, `coupling`, `certify`, `linear`, `lossless`, `balance`,
  `uncondensed`), `Lens`, `Beam`, `Transfer`, `Realisation`, `Balance`, `Linearity` and
  `Losslessness`.
- **`reads.coupling(a, b, *, far)` and `Coupling`:** a signed coupling on a shared basis, against an
  exact permutation null.
- **`entroptics.batch`:** `resolved_batch`, `ResolvedBatch`, `ResolvedScreen`,
  `ResolvedScreenBatch`, `ResourceLimits` and `recommend_backend`, on numpy or on a torch tensor's
  device.
- **Dynamics:** multi-step `predict(steps=h)`, `rollout`, `floor_contrast` and `carry_over_gaps`.
- **`entroptics.lift`:** `delay_embed` and `koopman_lift`.
- **`entroptics.proximity`:** the spectral-proximity digest and `SpectrumProbe`.
- **`entroptics.sequence`:** entropy rate, block entropies, the Lempel-Ziv rate, redundancy,
  effective length and `surrogate_test`.
- **Reads:**
  - `occupied_modes`, `level_edge`, `decay_scatter` and `principal_directions`;
  - `Projection.refloor(null)`;
  - the fold primitives in `entropy`, including `fold_band`, `fold_width` and
    `feature_axis_is_continuous`.
- **Extras:** `figures` (matplotlib, h5py); `threadpoolctl` in `test` and `dev`.

### Removed
- **21 names from the top-level namespace:** `phi`, `magnification`, `scale_duality`, `phi_T`,
  `phi_F`, `sigma_T`, `sigma_F`, `etendue`, `strehl`, `space_bandwidth`, `spectral_optics`,
  `concentration`, `decay`, `mercer_certificate`, `scale_profile`, `optics`, `geometry`, `read`,
  `sweep`, `extract` and `ModeFootprint`.
  - Reach each through `Aperture(W, window=None)` or through its module (`entroptics.reads`,
    `entroptics.entropy`, `entroptics.projection`, `entroptics.sweep`).
- **`ModeFootprint`:** footprints are `Beam`s, and `Beam.energy` is the squared singular value.
- **Other names:** `extract.extract(W)`, `screen.embed`, `Screen.K`, `Screen.embeddings` and
  `Screen.vocabulary`. Use `Projection.beam` in place of the last three.

### Fixed
- A masked channel is no longer read as a measured zero, in the fill reads, in `coupling` and in
  `sweep`.
- An empty spectrum reads NaN instead of `1/n`.
- A zero-length axis raises `ValueError` instead of an internal error.

[0.2.5]: https://github.com/Agience/entroptics/compare/v0.2.3...v0.2.5
[0.2.3]: https://github.com/Agience/entroptics/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Agience/entroptics/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Agience/entroptics/releases/tag/v0.2.1
