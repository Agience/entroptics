# Changelog

All notable changes to Entroptics are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). While the version is below 1.0, a minor or patch
release may change a public name or a returned number; each such change is listed under
**Changed** or **Removed** with what to use instead.

Released versions are archived on Zenodo under the concept DOI
[10.5281/zenodo.21273400](https://doi.org/10.5281/zenodo.21273400), which resolves to the latest
version.

## [Unreleased]

### Planned
- **A heavy-tail guard that reads the noise, not the modes.** The row bound behind the switched
  floor (`null_providers.row_influence`) measures how much any single row lifts the *top*
  eigenvalue. On a record with a strong mode the top eigenvalue is that mode, and its own rows
  carry a large share of it, so the guard sends the record to the exact permutation test although
  no noise eigenvalue is in question. That costs time, not accuracy: `rates.2000x32` reads at
  2.1× and `resolved_screen.400x32` at 1.24× 0.2.7's instructions for this reason. The guard will
  bound the lift of the largest eigenvalue the closed form would leave unresolved, so a record
  whose only large row contributions belong to a resolved mode keeps the closed form. It is held
  to the release gate: no accuracy metric may move the worse way, and both reads must come back to
  0.2.7's cost.

## [0.2.8] - 2026-10-08

Every read holds the rows it thresholds, and the floor chooses, per read, between the exact
permutation test and the closed form from the data itself: heavy right tails no longer read as
modes anywhere, the operator's window reads light tails at the closed form's cost, and streamed rows
expire as the aperture forgets. The release gate now also catches a loss spread thinly over many
metrics, and its accuracy baseline covers all three reads.

### Changed
- **One floor where samples are held, chosen per read from the data: the closed form where no row
  can carry a noise eigenvalue over it, the exact permutation test where one can.**
  - The closed-form edge alone over-read heavy right tails: on lognormal noise (σ = 1.5) up to
    41% of noise records claimed a mode at `far = 0.05`, Pareto (3) up to 17%. On light tails it
    under-read (0–2%). The excess is made of single rows: a channel's largest cell carries most of
    its energy, and channels whose largest cells share a row read as a mode. One such row carries
    30–53% of the top eigenvalue.
  - A screen (`Projection`, `spectral_optics`) takes the exact test wherever `far * n >= 1` for
    the `n` rows read, as it did in 0.2.7. There the exact test is the more powerful floor: on light
    tails the closed form sits above it (the edge of a sample covariance, where a sample
    correlation's top eigenvalue fluctuates less).
  - The operator's window and a pool (cut point `"bulk"`) take the closed form at every level where
    no row can cross it: 0.2.7's floor and cost on light tails, which keeps the streaming read
    cheaper than an FFT pipeline on long multichannel records. Taking the exact test there instead
    cost 13–15× the pipeline's instructions on 16 channels: each of its 19 draws is a pass over every
    row.
  - The closed form answers where its room above the null edge's centre, `q(far)` times the
    Tracy–Widom scale, clears two things:
    - the most any row lifts the top eigenvalue as it stands, `max_t |v . y_t|^2`
      (Courant–Fischer; `null_providers.row_influence`);
    - the largest lift a coincidence of two channels' cells reaches under the null with probability
      above `far`: the `(floor(far T) + 1)`-th largest cross-channel product `|y_tj| |y_t'k|`
      (each cell pair shares a row with probability exactly `1/T`;
      `null_providers.coincidence_influence`).
  - The coincidence bound is decided by a count: the `k`-th largest product is within the room
    exactly when fewer than `k` products exceed it, and only a cell above the room over the largest
    cell can be in such a product.
  - `closed_form_holds` states the switch, and `closed_form_far` gives the level where a record
    switches.
  - `default_provider` returns `switched` wherever data is held. It applies to `Projection`,
    `spectral_optics` and every pool. `Projection.significance` and `Dynamics.significance` report
    the evidence of the branch taken.
  - No cap or tuned constant enters. Light tails reach the closed form without the `1/far - 1` draws
    wherever it answers, and heavy tails keep the exact test wherever a coincidence could cross the
    floor.
  - The row bounds cost about three passes over the record. The largest cell is bounded first from
    the whole record's extremes, two contiguous passes, and read column by column only where that
    bound does not settle it.
- **The streams hold windows that expire as the aperture forgets. A record handed in at once is
  read whole.**
  - `Dynamics` keeps every frame of its latest ingest (a block is a record). Of what streamed in
    before it, it keeps the frames an aperture stream keeps: at least `F + 1`, and more while a mode
    it resolves is still coherent (`Dynamics.horizon`, the derivation `Aperture._coherence_horizon`
    used, now shared).
  - `resolved`, the DMD truncation, `floor_contrast` and `significance` read that window at the
    library floor. `Dynamics.window` returns it, and `DynamicsState.held` carries it.
  - The aperture's frame window and the operator's window are the same frames.
  - The operator itself (rates, modes, prediction) still accumulates every pair.
- **`ResolvedScreenBatch` is one `ResolvedScreen` per screen** behind the batched interface.
  `.windows` and `.screens` expose them. `ResolvedScreen` takes its window from its own `Dynamics`:
  an append is read whole, and earlier streamed rows expire. The `(B, F, F)` Gram read at the
  closed form is removed: its level did not hold (above).
  - Its screens' exact floors are taken together (`null_providers.apply_floors`), for its reads and
    for the horizons its operators read at an expiry point. Each screen draws from its own
    generator, as it would alone, and the screens' `j`-th draws share one stacked eigensolve. Every
    read is the screen's own, bit for bit. Sixteen screens of 400 rows, appended 16 at a time and
    read after each append, take 10.6% fewer instructions. The release baseline gains that read,
    `resolved_screen_batch.16x400x32`.
  - An expiry that cannot drop a frame (the window holds no more than `F + 1` rows and the latest
    block) no longer reads the horizon. The horizon depends on the held frames alone and is read
    when next asked for.
- **A correlation floor scores its draws from the smaller Gram.** `top_spectrum_value` at the
  `"spectral"` and `"bulk"` cut points reads the top eigenvalue of the column-scaled frame's
  `min(T, F)`-square Gram, as the projection cut point already did, not an SVD of the `T x F` frame
  (0.15 ms against 0.70 ms a draw at 2000 x 32). The round-off margin the rank test adds was
  already a Gram's. The floors move by round-off only; the level is unchanged.
- **An exact floor reads each draw's column energies once.** A correlation draw's score and its
  round-off margin each took a pass over the draw for the same energies; they now share one, and
  so does the observed record's floor. Every floor is bit for bit what it was (1056 floors across
  the three reads, four noise laws, six shapes, real and complex), at 4% fewer instructions on a
  long read. The draws themselves, numpy's per-column shuffle, are about three quarters of an exact
  floor on a long record: a sort-based uniform permutation measured 3.6× more instructions.
- **`research/benchmarks/count_vs_fft.py` counts the FFT comparison in instructions.** The like-for-
  like reads of `operator_vs_fft.py` -- the bare FFT, the FFT pipeline and the library's read, one
  channel and 16, T = 1024 to 1048576 -- costed as `cost.py` costs a read (valgrind, the difference
  between runs with and without three calls, each run first warmed by one, taken twice). The
  benchmark README's cost section and `fig_cost.png` are drawn from it. An instruction count does
  not carry the host's load or clock, so two releases and the FFT are compared on what each
  computes.
- **The release gate tests the direction of the metrics that moved.** `entroptics.gate.compare`
  also fails a candidate when the `higher` / `lower` metrics that changed moved the worse way more
  often than a change that is no worse would. That count is at most Binomial(n, 1/2), and the test
  takes its exact tail. A loss spread thinly over many cells passes every per-metric test: taking
  the closed form at every level where its bounds hold for the screen reads as well (`Projection`)
  moved 21 sensitivity cells down and 4 up, none beyond its own noise, and a narrowband line at
  1024 × 64 was found 9% of the time instead of 20% (P = 0.0005). The per-metric test and the sign test share the level, each at `far / 2`, so
  `z` is the normal quantile at `far / (2 M)`. A `bound` metric is held to its bound alone. The
  gate's report prints the count and its chance.
- **The accuracy baseline covers every read whose floor is a different cut point.** `baseline.py`
  measured `Projection` alone, and 0.2.8's change of floor left all 457 of its metrics
  bit-identical. It now measures the correlation read (`spectral_optics`, keys `spectral.`) and the
  operator's resolved count on its window (`Dynamics.resolved`, keys `operator.`) on the same
  signals, noise laws, record shapes, strength ladder and record counts.
- **The release baselines (`baseline_metrics.json`, `cost_metrics.json`) are 0.2.8's.**
  - Accuracy: 1345 metrics over the three reads, each detection rate tagged with its level. Against
    0.2.7 on the same harness no metric is worse, and the metrics that moved went 51 the worse way
    and 65 the better. 144 of 0.2.7's rates are set aside: its operator read claimed a mode in 16–36%
    of lognormal noise records and 20% of complex 32 × 512 ones at `far = 0.05`, and 0.2.8 holds
    5% there.
  - Cost, against 0.2.7 (instructions, valgrind): `spectral.1024x64` 280 M → 186 M (the smaller
    Gram, above); `stream.65536x16`, the streaming read of two tones in 16 channels, 201 M → 233 M;
    `resolved_screen.400x32` 328 M → 405 M; `rates.2000x32` 138 M → 290 M; every other read level.
    New reads: `resolved_screen_batch.16x400x32` and `stream.65536x16`.
  - On the FFT benchmark's records from T = 16384 (`count_vs_fft.py`) the streaming read costs
    1.3–1.4× 0.2.7's, with the same counts, and 1.2–1.6× fewer instructions than the FFT pipeline
    on 16 channels (0.2.7: 1.6–2.3× fewer). The excess is the row bounds' passes over the window.
  - `rates.2000x32` and `resolved_screen.400x32` read a record with one strong mode. The row bound
    reads the top eigenvalue, there the mode itself, whose rows carry a large share of it, so the
    exact test answers: their excess is its draws. A guard that reads the noise eigenvalues is
    planned (Unreleased, above).
- **`SpectralAccumulator` holds every plane** (an ensemble does not age) and reads them at the
  library floor. `spectral()` takes `far` and `seed`, and each draw shuffles every channel within its
  own plane, gaps held. Memory is the pooled planes.
- **A burst that has streamed out of the window is no longer read.** A 12-row burst streamed into
  noise frame by frame is found in 100% of streams at its end and in 3% once `F + 1` further frames
  have arrived. A burst inside a record handed in at once is read with the record.
- `test_dynamics`'s strict truncation level is `1e-12`: there the closed form answers. The
  ensemble-of-runs and operator-significance tests take the exact test's identity, `#(p <= far)`.
  The blocked-stream forgetting test compares blocks no longer than the minimum window.

### Removed
- `ResolvedScreenBatch`'s Gram state (`_Moments`) and its batched closed-form refresh.

### Fixed
- **A pool of planes reads the closed-form edge at its own degrees of freedom.** Each plane is
  centred on its own mean, so `M` planes of `T` rows in all carry `T - M` degrees of freedom; the
  edge was read at `T - 1` and sat too low: on Gaussian noise, 4 planes of 16 × 128 claimed a mode
  in 10.5% of records (complex: 17.5%) at `far = 0.05`. `SpectralAccumulator.dof` is `T - M`, and
  `FloorContext.dof` / `apply_floor(dof=)` carry it to `mp`. One plane reads as before.

## [0.2.7] - 2026-09-30

Every floor that holds its samples is now an exact test: the screen and correlation reads take a
permutation floor whose level is `far` for any law of the noise, and the reference nulls take the
exact rank over their realisations. The Tracy–Widom laws are computed rather than tabulated.
`Basis.drift` compares a record with the source of its basis. `ResolvedScreen` reads its stream's
window. A release baseline (sensitivity, accuracy, instructions and memory per read) and a gate
that holds each release to it. Values move: every read at the default floor, and the counts on
heavy-tailed noise, where the closed-form edge over-read.

### Added
- **A release baseline and its gate.**
  - `research/benchmarks/baseline.py` measures sensitivity and accuracy, each with its own sampling
    error: 457 metrics.
    - Sensitivity: detection on a ladder of planted strengths relative to the noise edge, over
      four signal shapes, four noise laws and three record shapes.
    - Accuracy: exact-count accuracy, the false-alarm level as a bound, decay-rate and frequency
      error against a known linear system, and the extract filter's error against a known burst.
  - `research/benchmarks/cost.py` measures resources and throughput for eight reads: retired
    instructions per read, counted by valgrind (reproducible to about 0.1%), and peak allocated
    bytes.
  - The committed `baseline_metrics.json` and `cost_metrics.json` are the reference. A release is
    held to them by `entroptics.gate.compare` (command line: `research/benchmarks/gate.py`). A
    candidate fails when any metric is worse than its baseline beyond the two measurements'
    combined noise (z at `far / M`, so the family errs at most `far` of the time on an unchanged
    library), or when a metric goes missing.
- **`reference_null` and `ReferenceNull` are exact rank floors.** They were `mean + z(far) std` of
  the reference's top values, a normal model of a right-skewed law, which claimed structure in 10%
  of reference-like records at `far = 0.05` (8 × 8, 40 realisations). The floor is now the exact
  rank over the realisations: 0.043 (`reference_null`) and 0.057 (`ReferenceNull`) averaged over
  reference sets.
  - `ReferenceNull` keeps its values rather than running moments. With `forgetting < 1` it holds
    the `(1 + f) / (1 - f)` most recent, the effective count of the weights it stands for; with
    `forgetting == 1` it holds every value pushed. With `n` realisations it claims nothing below
  `far = 1/(n + 1)`. So `sweep(null="local")` needs at least `1/far - 1` quiet neighbours to
  resolve a patch.

### Changed
- **`ResolvedScreen` reads its window, at the exact floor.** It kept a cumulative Gram of every row
  appended and could only take the closed-form floor. It now keeps the rows of the window an
  `Aperture` stream keeps -- at least `F + 1` rows, and more while a mode the stream's own
  operator resolves is still coherent -- and reads them at the exact permutation floor, bit for
  bit `resolved_batch(window, fold=False)`. A channel that never moved in the window leaves the
  live width. `ResolvedScreen.window` returns the rows read. `state()` carries the window and
  the operator. `ResolvedScreenBatch` keeps its batched cumulative Gram and the closed-form
  floor, for thousands of screens at once.
- **The screen's noise floor is an exact permutation test by default.** With no `null=`, the
  projection floor behind `K_signal` (`Projection`, `Aperture.projection`, `probe_signal`, the
  batched frame read and `resolved_batch`) is now the exact Monte Carlo test of independent
  channels: each channel shuffled in time with its missing cells held, the fold decided again on
  the shuffled record, and the top value scored as its own screen's standardized Tracy–Widom
  deviate (a draw's screen need not have the observed width, and a singular value is comparable
  only at one width; the level still comes from the ranks alone) and taken at the exact rank over
  the fewest draws the level allows (19 at `far=0.05`). Its false-alarm rate is `far` for any law of the noise. The fold is
  decided per draw because its continuity test reads how neighbouring channels move together,
  which the shuffle destroys: a fold carried over from the observed record ran at 0.085 on
  lognormal noise, given a fold.
  - The closed-form Tracy–Widom edge (`null_providers.mp`) held its level only for light tails.
    On 13 noise families × 6 shapes (`research/benchmarks/screen_null.py`, 200 records per cell),
    the default pools to 0.050 false alarms, and `mp` to 0.025 with 0.36 on lognormal noise
    (σ = 1.5) and 0.16 on Pareto noise.
  - The default finds at least as much as `mp` on every planted signal: a narrowband line
    12–35% against 3–15%, a narrow burst 95–97% against 90–94%.
  - A draw re-reads only what the shuffle moves: the fold's concentration verdict is the
    observed record's, and only its continuity test runs per draw, and only when the record is
    concentrated. Channels are shuffled channel-major, and the top value comes from the smaller
    Gram. At `far=0.05` a default read costs about 3–9× an `mp` read. The draw count grows as
    `1/far`, so `far=0.01` costs about five times that.
  - `Projection` decides its fold at the read's own `far`, as `resolved_batch` and the paper's
    Definition 2.2 do; it took the fold gate at 0.05 whatever `far` was.
  - The correlation reads (`spectral_optics`, `spectral_batch`, `Aperture.spectral`) take the same
    exact default on their centred samples, gaps held. On lognormal noise (σ = 1.5, 256 × 16) the
    correlation floor's false alarms go from 0.147 to 0.057. Their contrast, attenuation and
    resolved power move with the lower floor; no count in the goldens moved.
  - Pass `null=null_providers.mp` for the previous floor. Where only a covariance is held
    (`SpectralAccumulator`, the Dynamics truncation) the default stays `mp`: a covariance carries
    no samples to shuffle. `null_providers.default_provider(kind, data)` states the rule.
  - `Projection.significance` reports the evidence of the floor in force. Under any exact
    permutation floor (the default, `permutation()`, or either through a mapping or `by_kind`)
    this is the exact Monte Carlo p-value from the floor's own draws, and
    `K_signal == #(pvalue <= far)`. `mode_significance` remains the closed-form Tracy–Widom
    evidence, with `mp`'s count `#(pvalue < far)`.
  - `permutation()` and `floor_from_null_sampler()` take `draws=None` by default, meaning
    `null_providers.fewest_draws(far)`; they took 200.
  - The committed goldens, the validation results, the calibration table, the FRB tables, the
    benchmark page and its figures, and the FRB and JOSS companions are regenerated. The benchmark
    page's noise table pools each noise law over its record shapes, since the worst of six shapes
    of an exact test sits above its level by chance. The paper's §8 states the exact floor, with the edge as its closed form, and
    every number in §12 is re-read (`research/validation/check_paper.py`: all 780 trace to an
    artifact).

### Removed
- **`aperture.MIN_WINDOW`** (a fixed 128). A batch `Aperture(W)` never windows: every record is
  read whole, so a guard that compared a plane's height against it can go. A stream's minimum
  window is read off its width, `F + 1`, and `Aperture.window` returns it.

### Fixed
- **`Basis.drift` asks whether a record is the source's process.** It read the residual off the
  span against independent channels. A process with more correlated modes than the basis resolves
  leaves a correlated bulk in every record's residual, so independent records of one process read
  drift in 70% of cases (93% on 0.2.6), and a planted extra mode was found less often as it
  strengthened. The basis now keeps its source record (`Basis.source`). Drift carries the source's
  rows into the same complement coordinates and takes the exact rank over reads of random subsets
  of the pooled rows: 5% false alarms on the same process, and an extra mode found in 3%, 97%, 100%
  of records at strengths 1, 2, 4. A caller's `null` still replaces the floor.
- **`reference_null` takes signal-free planes, read at each screen's own width.** A reference of
  top values was pinned to one folded width, and a screen's fold is decided on its own record, so a
  plane of the reference's shape that folded narrower was refused (the U(1) Coulomb-phase 8 × 8
  planes fold to 8 × 5). Given planes, the provider reads each at the width of the screen it
  thresholds, once per width, and `shape` is the planes' shape, which the caller controls. The
  `Aperture(reference=...)` path passes its realisations as planes. It had scored the raw records,
  not their screens, which are in different units.
- **A statistic the shuffle keeps is never evidence.** An observed value and a surrogate draw
  that differ only in round-off (one live channel, whose norm a time shuffle keeps exactly) were
  compared bit for bit, so round-off alone resolved a mode in a fraction of records. Every draw
  and the observed read now carry the round-off bound of their Gram eigenvalue,
  `(m + p) eps ||X||_F^2` in the data's own arithmetic (float32 rounds 5e8 times coarser), and a
  tie is a tie: a single surviving channel resolves nothing.
- **A masked record's null is drawn with its mask held.** Shuffling a finished screen scatters
  the zeros that stand in for missing cells, so a record with missing runs read structure in up
  to 80.5% of noise records (`research/benchmarks/screen_null.py`, masked rows). The draw now shuffles each channel's measured values among its own
  measured cells, pooled over 72 masked cells at 0.0436.
- **A fold cell's correlated channels are within the null's reach.** A shuffle of the finished
  screen keeps the energy a fold cell gathers from channels that move together. Four aligned
  channels of 200, folded five to a cell, read below that floor. The null is drawn on the
  channels before the fold.
- **`sequence.surrogate_test`'s onset is an exact Monte Carlo test.** Its default draw count is
  `fewest_draws(level)` and the onset fires at `p <= level`. With a fixed 40 draws, the smallest
  attainable p-value exceeded the Bonferroni level, so no onset could fire.
- **The Tracy–Widom laws are computed, not approximated.** The edge quantile and every per-mode
  p-value now come from the laws themselves, in `entroptics.tracy_widom`:
  - TW1 and TW2 are the Fredholm determinants det(I − B) and det(I − B²) of the Hankel operator
    B(x, y) = Ai(x + y + s);
  - the Airy function is built from its differential equation;
  - accuracy is about 1e-13, relative in the upper tail.
  The TW1 quantile table and the Chiani Gamma approximation (about 7e-3 CDF error) are removed.
  - The table's 0.025 entry was 1.3675; the TW1 97.5% quantile is 1.4538. Every read at
    `far=0.025` used the wrong edge.
  - At the default `far=0.05` the edge quantile moves from 0.9793 to 0.97931605. Floors, and the
    contrast and attenuation read against them, move by about 6e-7 relative. No count changes in
    the suite.
  - `K_signal == #(p_k < far)` now holds at every level, since the floor and the p-values share one
    law.
- **Complex data is scored against the complex law everywhere.** `mode_significance`,
  `Dynamics.significance`, `Dynamics.resolved`, the DMD truncation, the streaming and batched
  resolved screens (`resolved_batch`), `SpectralAccumulator` and proximity's `bulk_edge` scored a
  complex stream against the real law (TW1 and the real centring). A covariance-only floor now carries the data's ensemble (`FloorContext.complex_`,
  `apply_floor(..., complex_=)`).
- **The incomplete gamma behind the screen's balance test** sums to round-off with `math.lgamma`,
  in place of chosen tolerances and iteration caps.
- **The normal quantile and digamma / trigamma are evaluated to float precision.**
  - The normal quantile (behind `reference_null`, the sweep, the coupling and carriage cut-offs,
    and the basis) is Newton's method on the log-survival, from a start the tail bound places
    right of the root. It replaces Acklam's fitted rational (1.2e-9 error).
  - Digamma and trigamma (behind the fold's Dirichlet null) recur to where their asymptotic series
    reaches round-off, then sum it with exact Bernoulli numbers. They were a five-term series
    (8.8e-12 and 1.9e-10 error).
- **The block gap fill runs to its fixed point.** `carry_over_gaps` iterated at most 64 times,
  and at heavy dropout returned a fill that was still moving (70% of cells dropped needs 145
  steps). It now runs until a step moves no cell by more than the arithmetic's own resolution. The
  `iters` keyword is removed. It costs 5–22 ms on a 2000 × 12 record with 5–50% of cells dropped,
  and nothing when no cell is missing.
- **No read depends on the absolute size of the data.** Absolute floors and tolerances are now
  relative to the data (its largest value times eps) or to the arithmetic's own round-off:
  - the streaming feature correlation (`Dynamics` resolved count, significance, `phi_F`);
  - `top_spectrum_value`, the statistic behind a reference null;
  - the operator's numerical rank (the Higham `F eps`, from `1e-10`);
  - the log guards (the float format's smallest normal, from `1e-300`).
  A record scaled by 1e-20 used to read a different streaming count.
- **A constant reference gives its value as the floor**: `reference_null` and `ReferenceNull` no
  longer add `1e-30` to the scale.
- **`far` is honoured, not clipped.** `fold_band` and the Cantelli multiplier clipped `far` to
  [1e-9, 0.5], so `far = 0.9` was read as 0.5. Any `far` in (0, 1) is now used as given, and one
  outside it is refused.
- **A stream's minimum window is read off its width: `F + 1` frames**, the fewest whose centred
  frame carries every feature direction. It was a fixed 128 (`aperture.MIN_WINDOW`, now removed).
  An explicit `window=` still wins.
- **The coherence horizon no longer swallows errors.** An `except Exception` returned the minimum
  window on any failure. An aperture with no data yet is now checked explicitly, and any other
  error surfaces.
- **The streaming frame window's persistence test** is the operator's own (`forgetting()["forgets"]`,
  at its round-off), in place of a fixed `1 - 1e-9`.
- **`optics()` and `Aperture.optics()` share one diffraction-limit test** (`duality_of`); the free
  function used `|1/phi - 1| < 1e-9`.
- **The streaming operator's null is sized by the pairs its covariance sums**, not by frames. The
  affected reads are `resolved`, the DMD truncation, `floor_contrast` and `significance`. An
  ensemble of short independent runs (`adjacent=False`) has far fewer pairs than frames: on pure
  noise, 60 runs of 3 frames each reported structure in 26.5% of records at `far = 0.05`.
- **The rank-capped eigensolve's constants are derived.**
  - The sketch is as wide as the exact rank bound.
  - It is taken while its leading flop count beats the full eigensolve's (k < F/3).
  - Exactness is certified to the F eps round-off of the trace, not 1e-10.
- **`Screen.coupling` reads each side where it balances**, like every other screen read. It read
  the raw placed frames, so a side with a declared `zero` was coupled on structure its own zero
  removes.
- **`Screen.linear` scores a side with that side's own null**, as `certify` does. It used the
  screen's null, so the two disagreed on what counts as absent.
- **`lempel_ziv_rate` counts the LZ-76 parsing it documents.** It counted an LZ-78 parsing (a
  dictionary of earlier phrases), in quadratic time. The Lempel–Ziv 1976 example now parses into 6
  words, not 7. The count is O(N), by an online suffix automaton.
- **`surrogate_test`'s onset holds the caller's level.**
  - It took `p < 0.05` at each order in turn, so a structureless sequence reported an onset in
    12.5% of records.
  - It now takes `far` (new keyword), Bonferroni over the orders tested.
  - `n_max` defaults to the longest block the sequence can sample (`sampled_order`, new). `draws`
    defaults to the fewest shuffles that can resolve the level.
  - `entropy_rate`'s `n_max` defaults the same way.
  - The two `1e-12` tolerances are the entropy sum's round-off.
- **`Dynamics.merge` with an empty side keeps the caller's `far` and `null`.** It rebuilt the result
  at the default operating point.
- **`concentration_band` holds at its level for every shape.** It multiplied `sqrt(N/T) + N/T`
  by a chosen constant (`c_conc=2.0`) and did not cover when N ≪ T: at T = 4096, N = 4 a whitened
  Gaussian record left the band 22% of the time. The band is now the Gaussian theorem's own,
  `||C|| (2 delta + delta^2)` with `delta = sqrt(N/T) + sqrt(2 ln(2/far) / T)` (Davidson–Szarek), at
  level `far`. `c_conc` is removed; pass `far=` (default 0.05) to set the level.
- **`ResolvedScreen` and `ResolvedScreenBatch` read at every read.** They refreshed their basis
  every `refresh_every=32` appends and read a stale basis in between, so a mode arriving in fewer
  rows than that stayed invisible to `K_signal` and `energy` (12 burst rows after a read: 0 modes
  where the batch read finds 1). An append now marks the read stale, and the next read refreshes
  it. `refresh_every` is removed: a refresh runs as often as the caller reads.
- **The sampled floor is an exact Monte Carlo test.** `floor_from_null_sampler` (and
  `permutation()`) took an interpolated `1 - far` quantile of its draws, which sits below the rank
  an exact test needs, so the observed record exceeded it more often than `far`. The floor is now
  the `n + 1 - floor(far (n + 1))`-th smallest of the `n` draws: the false-alarm rate is at most
  `far` for any `n`, whenever the record is exchangeable with its surrogates. Below `1/far - 1`
  draws the floor is infinite (no draw count that small can claim anything at `far`).
- `tensor_embed`'s error states the bound it enforces (`d <= T-1`). Stale docstrings are corrected:
  - the axis fills use the smaller Gram, not a T x T eigendecomposition;
  - proximity refers to the screen's mean / RMS whitening;
  - `weighted_effective` no longer cites a removed function.

### Performance
- **`decay` holds O(T F) memory, not O(T^2).** It formed the full T x T ordered Gram plus a T x T
  lag-index array (4.6 GB at T = 16384). On numpy it now takes each channel's lag sums by direct
  correlation: 10x faster at T = 4096. The torch path keeps the Gram on its device. The periodic
  read folds the same sums, A(tau) + A(T - tau), and stays exactly symmetric.
- **`coherence` takes its moments on the cheaper side.** The z-score's moments are sums over the
  N x N row Gram, and each has an exact form on the feature side (O(N F^4), no N x N array). The
  read takes whichever costs less, so the value is the same either way. A 100 000-row screen no
  longer needs an 80 GB Gram: 266 ms at N = 65536.
- `extract`'s adjoint lift returns the profiles directly when the fold is the identity, in place
  of multiplying identity blocks (O(T^2 K)).

## [0.2.6] - 2026-09-29

The reference null's shape guard, the pencil's kept order, and the benchmark page with figures.
No read's value changes.

### Added
- **`reference_null(..., shape=(N, F))` and `ReferenceNull(..., shape=(N, F))`**: the shape of the
  screens the reference was read from. A screen of any other shape is refused with a `ValueError`,
  since the floor is an absolute top singular value and a different shape has a different null.
  Without `shape`, nothing changes.
- **`HankelSpectrum.kept` and `MatrixPencil.kept`**: how many of the pencil's directions the cut
  kept, which is the order the read actually has. Where it is below `n + 1`, the directions left
  out were within the record's own noise.
- `research/benchmarks/figures.py`: the benchmark page's figures and winner tables, drawn from the
  committed outputs; the page opens each measurement with its question and answer.
- A regression test that a reference-calibrated floor counts structure and not the noise's
  marginal: a skewed marginal alone stays at the level, and a correlated mode is still found. Both
  checks fail on the whitening before 0.2.5.

### Changed
- A `reference_null` floor pinned on screens read by 0.2.3 or earlier is in that whitening's units.
  Under 0.2.5 the screen's energy no longer carries the marginal's (RMS / MAD)^2, so a pinned
  reference has to be re-read with the library that reads the screens.

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

[Unreleased]: https://github.com/Agience/entroptics/compare/v0.2.8...HEAD
[0.2.8]: https://github.com/Agience/entroptics/compare/v0.2.7...v0.2.8
[0.2.7]: https://github.com/Agience/entroptics/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/Agience/entroptics/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/Agience/entroptics/compare/v0.2.3...v0.2.5
[0.2.3]: https://github.com/Agience/entroptics/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Agience/entroptics/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Agience/entroptics/releases/tag/v0.2.1
