# Entroptics — the guide

The reference behind the [README](../README.md): what each read is, the conventions every input
follows, and what the library guarantees. The derivations are in the paper ([Markdown](../research/PAPER.md), [HTML](../research/PAPER.html), [PDF](../research/PAPER.pdf));
every public name is in the [API reference](API.md); the measurements are in
[`research/benchmarks/`](../research/benchmarks/README.md).

## The operator read (experimental)

`entroptics.experimental.operator_read(x)` reads a 1-D record's operator with nothing supplied but
`far`: the number of modes, the delay depth, and each mode's frequency, decay and power. It is in
`entroptics.experimental` because its name, signature and numbers may change in a later release;
the stable routes to the same modes are `Aperture(W).dynamics().modes()` for a multichannel record
and `koopman_lift(x, d).modes()` for a 1-D one at a depth you choose.

**How it reads.** The only input is `far`.
- **The order `K`** is the count of past-against-future canonical correlations above the
  Tracy–Widom edge of the Jacobi ensemble (Johnstone 2008), taken on non-overlapping windows, where
  that edge is exact for i.i.d. noise. The count runs at every dyadic window depth and at every zoom
  level (block means over 1, 2, 4, … samples), with the level corrected for every look.
- **The depth `d`** is the window span at the coarsest level that still counts: the objects' own
  coherence.
- **The modes** come from the matrix pencil on the lags 1 … 2d; white noise sits at lag 0.

## The write path: the metric, the certificate, drift

- **The metric.** The rows are orthonormal in the record's noise metric, at native channel
  resolution. Each channel's noise is read from what the resolved span leaves, through a moment
  equation that holds in any metric: the exact moment solution wherever the system has rank. Where it has none -- a channel inside the span, or more channels
  than the residual can tell apart -- the metric the span was read in stands.
- **The certificate.** `defect` is the rows' orthonormality, and it bounds `inverse` and
  `idempotent`.
- **Drift** reads a later record in that record's own noise metric. The residual is rotated into
  the dimensions the basis leaves, where the existing floor's null applies. Measured over 80 records
  by [`research/benchmarks/write_path.py`](../research/benchmarks/write_path.py), at `far = 0.05`:
  - false alarms on later records from the same process: 0 of 80 with unequal channel noise, 2 of
    80 with equal noise, and 0 of 80 on a complex process;
  - detections: a planted extra mode in 80 of 80, and a local transient in 75 of 80.
  - `drift` reports `identified=True` only when every channel's noise is fixed to within 1/z of
    itself at the reader's level `far`: each channel's standard error -- from the moments' own
    covariance, evaluated at the solved noise -- is below 1/z of it. Few channels outside the span,
    a short record, or a channel lying (nearly) inside the span leave it `False`, and then the
    level is not held. Where it is `True`, drift held its level (`write_path.py`): at most 5.9% of
    identified records alarmed (8 of 135 at worst, within the binomial spread of the 6.75 the level
    allows) on each of eight real and complex configurations, and over 4000 records with exact moments no identified
    channel was off by more than 0.1%.

## How it compares

| | FFT | PCA / KLT | DMD | Entroptics |
|---|---|---|---|---|
| basis | fixed sinusoids on the record's bins | the data's own principal directions | the data's own modes | the data's own modes, and their principal directions (`basis()`) |
| how many components | all of them, one per bin | the analyst's choice (a variance fraction, or Minka's MLE) | a truncation rank or tolerance the analyst supplies | counted against the Tracy–Widom edge at `far` |
| decay | not read | not read | per mode | per mode |
| missing samples | filled first | filled or dropped first | filled first | marked absent: every read divides by what was measured |
| round trip | exact at full length | exact at full rank | an approximate reconstruction | exact at any count: `resolved + residual == W` |

On planted signals the derived count finds the true number of modes more often than the standard
selectors ([`research/validation/RESULTS.md`](../research/validation/RESULTS.md), experiment 17):
exact-count accuracy 1.000 against Gavish–Donoho's 0.935 over all 36 cells. AIC (0.946) and MDL
(0.889) are defined on only 27 of them, where the snapshot count exceeds the variable count. For DMD as
such, [PyDMD](https://github.com/PyDMD/PyDMD) and [PyKoopman](https://github.com/dynamicslab/pykoopman)
are more complete. Entroptics differs by keeping the operator as a fixed sufficient statistic whose
read cost stays fixed as the stream grows, and by truncating at its own derived floor.

## The apparatus

The optical chain: **a beam passes through an aperture, a lens converts it, and a screen receives it
as a projection.**

| | |
|---|---|
| **`Aperture`** | What **bounds** a beam, and the read of everything *about* it. The single front door: batch `Aperture(W)` or streaming `Aperture(window=…).update(frame)`. Per-axis and screen-area reads, the mode spectrum, the diffraction limit, the operator, and the write path. |
| **`Projection`** | The screen **as one side sees it**: one signal on its own entropy-matched grid, with its SVD, coherence, and the modes above the noise floor. |
| **`Lens`** | A system's **conversion**: `entry` (surface → the screen's coordinates) and `inverse` (back out), plus that system's own laws — `energy`, `zero`, `null`. All domain code lives here. |
| **`Screen`** | Where beams **land**: the surface two or more systems share, and the crossing measurements between them. |
| **`Beam`** | What a side **carries**: energy, étendue, and the directions it occupies. |

| you have | you want | use |
|---|---|---|
| one signal | its structure, resolution, decay rates, étendue | **`Aperture`** |
| one signal | to code it on its own modes and back, or share that basis | **`Aperture.basis()`** |
| one signal | its factorization or embedding | **`Projection`** |
| two or more systems | them to meet, convert, couple, or trade energy | **`Screen`** |
| a surface to convert | it mapped onto a screen's coordinates and back | **`Lens`** |

A `Screen` is for questions *between* signals:

```python
from entroptics import Screen

s = Screen()                                                  # far= sets the reader's level
s.register("A", entry=to_concept_a, inverse=from_concept_a)   # a lens IS its conversion
s.register("B", entry=to_concept_b, inverse=from_concept_b,
           energy=my_energy_law, zero=my_zero, null=my_floor)  # ...and its own laws

concept   = s.place("A", surface_a)      # A.entry   : signal -> concept
surface_b = s.render("B", concept)       # B.inverse : concept -> B's signal

s.couple("A", "B")        # the MEASURED signed coupling (exact permutation null); 0 unresolved
s.transfer("A", "B")      # absorbed vs transmitted, the signed flux, and where it condensed
s.certify("A", surface)   # imaging: does inverse . entry return the surface?
s.balance()               # each side at its own zero, and whether that zero closes
```

Only the row-paired reads (`couple`, `joint`, `read`, `resolution`) need the sides to share an
ordered axis. A lens declares its own noise (`null`), the
false-alarm level `far` is the reader's (Neyman–Pearson), and the caller owns the loop.

## What it reads

All reads are intrinsic — derived from `W` alone — and each is tied to a standard theorem.

### One screen, in focus

| | reads | what it says |
|---|---|---|
| **Scale** | `ap.phi`, `ap.magnification` = `1/phi` | the fill fraction and its reciprocal reach |
| | `ap.H_T`/`H_F`, `n_T`/`n_F`, `delta_T`/`delta_F` | per-axis entropy, matched grid width, cell scale |
| **Aperture area** | `ap.etendue` = `phi_F · phi_T` | the bounded 2-D area the screen carries |
| | `ap.space_bandwidth` = `n_F · n_T` | degrees of freedom it *can* carry — a capacity |
| **Coherence** | `ap.strehl`, `ap.phi_T`/`phi_F`, `ap.sigma_T`/`sigma_F` | dominant-mode power fraction, per-axis fills, leading singular values |
| | `Projection.coherence` | closed-form z against the exact row-permutation null |
| **Concentration** | `ap.concentration` | `intensity` (σ₁²), `focus` (axial), `resultant` (directional) |

`space_bandwidth` is a capacity. What the screen fills is `ap.etendue * ap.space_bandwidth`: 1 for a
single mode on an unfolded screen, rising with the modes present. When the feature axis folds, the
product reads `F_eff / F` of that, with `ap.projection().F_eff` the folded width. Measured on a noiseless rank-1
record over 64 channels (`readme_examples.py`): 1.0000 at T = 256 and 1024; at T = 512 that
record's feature axis folds, and the product reads 0.406 = 26/64.

### The screen's null

`K` counts the modes that stand above the edge that independent channels would reach.
- Every channel is centred on its mean and scaled by its RMS. The screen's Gram is then a sample
  correlation matrix.
- The floor is the Tracy–Widom edge of that null, whose law is proved for i.i.d. entries with a
  finite fourth moment (Bao, Pan & Zhou 2012). Its per-cell variance is the screen's mean cell
  energy. On an unfolded, fully measured screen that is the null's own, `N / (N − 1)`, exactly. On a
  folded screen, channels that share structure raise it, which only raises the floor.
- A channel that never moved (every value the same) leaves the screen the way a dead one does. A
  filter keeps it at its value.

Measured by [`research/benchmarks/screen_null.py`](../research/benchmarks/screen_null.py) at
`far = 0.05`, over 200 noise-only records per case, at shapes from 256 × 8 to 20 × 1000:
- **The level holds:** false alarms of 0 to 0.020 for Gaussian noise at equal and unequal channel
  levels (real and complex), Student t, exponential, χ² and Poisson noise. That is below the level
  with room: at these sizes the correlation null's top eigenvalue sits below the Tracy–Widom law of
  a covariance.
- **Heavy right tails exceed it,** most at the longest record (1024 × 64): lognormal (σ = 1) 0.085,
  lognormal (σ = 1.5) 0.36, Pareto (3) 0.16. A finite record of such noise sits far from the edge
  law.
- **The same cases on 0.2.3** ([`screen_null_0.2.3.jsonl`](../research/benchmarks/screen_null_0.2.3.jsonl)):
  1.00 on every exponential, lognormal, χ² and Pareto case, up to 0.96 on Student t and 1.00 on
  Poisson, and up to 0.735 on Gaussian noise at unequal channel levels.

What a correlation read can resolve follows from its construction:
- **A mode has to span channels.** A whitened channel carries energy `T`, so a mode confined to
  `k` channels can clear the edge only when `k T` exceeds it, about `k > (1 + √(F/T))²`: two
  channels when `T ≫ F`, and nine at 64 × 256. A tone in one channel is not resolved at any
  strength. A two-channel line near the edge at 256 × 32 is found in 4% of records (0.2.3: 58%).
- **A mode is read against the whole record's energy.** A weak mode beside a dominant one can
  therefore fall under the edge.
- **In exchange:** every planted persistent, burst, edge and two-mode signal in the harness is
  found in 90–100% of records. Its rank is counted exactly in 72–100% of them, against 21–100% on
  0.2.3.

### The mode spectrum

`ap.spectral` reads the correlation spectrum against a derived noise floor:
- `contrast`, `top_share`, `resolved_modes`, `noise_floor`, `resolved_power` (summed eigenvalue
  excess above the floor), and `dominance` = `(λ₁−1)/(F−1)`;
- the propagation constant `γ = α + iβ`, and `dispersion`;
- `ap.attenuation_interval(band)`, a Weyl-certified interval for `α`.

The floor comes from a `null=` provider (`entroptics.null_providers`). Unset, it is the derived
Marchenko–Pastur / Johnstone default `mp`. `null=null_providers.robust` gives the deterministic
data-derived Tukey fence.

### The diffraction limit

- `ap.decay`: the optical transfer function, a direct-sum autocorrelation.
- `ap.a_delta` (the entropy width `1/2^{H(C²)}`) and `ap.correlation_length` (the decay length ξ).
- `ap.mercer`: a model-free temporal-vs-spectral cross-check.
- `ap.rayleigh_shape_factor`, `ap.fresnel_number(window)`, `ap.shape_factor`.
- `ap.decay_scatter`: the channels are replicates of the decay, so their disagreement is the read's
  own uncertainty, per lag (`se`). `noise_share` far below `tail_share` means the correlation is
  structure the channels agree on.

### How it moves, and at what scale

- **The operator**, `Aperture.dynamics()` / `Aperture.rates()`, is a streaming online-DMD / Koopman
  operator that is spliceable and resumable. It gives exact per-mode decay rates `α_k = −log|μ_k|`
  and frequencies `β_k = arg(μ_k)`.
  - `modes()` adds each mode's power `P_k` and share, strongest first.
  - `modes().spectrum(f)` draws the Fourier view from them.
  - `reconstruct_decay` rebuilds `C(τ) = Σ P_k μ_k^τ` at any lag.
- **Multi-scale**, `ap.scale_profile()`: structure as a function of observation window.
- **Sweep**, `Aperture.sweep()`: fix the aperture to a bounded capacity and sweep it where the
  coherence gate finds structure.

### Recovering the signal

- **The write path**, `Aperture.basis()`: above.
- **The filter**, `Aperture.extract()`, is the hard projection onto the resolved modes (paper
  Def 8.4).
  - Every channel is projected onto the resolved modes' own time profiles, with persistent
    narrowband (`φ_F ≤ φ_T`) modes moved to the residual.
  - It returns `(clean, info)` in `W`'s own units, with `clean + info["residual"] == W`: an
    orthogonal projection, with the singular values kept as read.
  - On an exactly rank-1 record with no noise (`write_path.py`), it recovers a burst of any width,
    from half a sample to 128 samples, to between 1.8e-16 and 1.1e-15 at `σ_top / floor` = 5.3. A
    sine filling the record is recovered to 6.9e-16.
  - A noise-free record with several modes is recovered whole only where each mode clears the edge
    of the screen's null (above).
- **Tensor**, `Aperture.tensor()`: a delay-embedded Tucker/HOSVD of the within-window fine structure.

### Other shapes of input

- **N-D fields**, `entroptics.fields`: `slabs`, `over_planes` and `pool` reduce a higher-D field to
  the 2-D screen.
- **A stack at once**, `resolved_batch(X)` for `X: (B, T, F)`: the same resolved read over many
  frames. It runs on numpy (bit-identical to a per-frame `Projection`) or on a torch tensor on its
  device. `ResolvedScreen` / `ResolvedScreenBatch` are its stateful siblings, and `ResourceLimits`
  bounds threads, memory and GPU.
- **Records that are not a 2-D field:**
  - `entroptics.proximity` gives a magnitude-carrying spectral digest and a probe over a set of them.
  - `entroptics.sequence` gives the ordered-axis reads on a symbol stream.
- **Chains and resampling:**
  - `integrated_autocorrelation` (τ_int and a chain mean's error, each with its own error),
    `spread_over_chains`, `bootstrap`, `jackknife(groups=)`;
  - `empirical_bernstein`, `cross_covariance`, `matrix_pencil`, `effective_rates`, `crossing_lag`.

## Real data: fast radio bursts

`research/figures/frb_panel.py` applies the plain library to the public CHIME/FRB Catalog 1
waterfalls at native 16384-channel resolution, untuned. It supplies only observer facts: dead
channels are dropped.

```python
live = np.isfinite(W).all(axis=0) & (np.nanstd(W, axis=0) > 0)   # observer fact: the RFI mask
clean, info = Aperture(W[:, live], window=None).extract()        # everything downstream is the library
```

The per-burst reads are in [`research/figures/frb_panel.csv`](../research/figures/frb_panel.csv), and the
method is §12.1 of the paper.

## Glossary

Each read carries an optical name. Each entry points to the result in the paper that governs it
(paper §10).

| read | what it is | governed by |
|---|---|---|
| **operator** | the record's modes: poles `μ_k`, decay rates `α_k = −log|μ_k|`, frequencies `β_k = arg μ_k`, powers `P_k` | Theorem 9.2 (exact recovery for a noise-free linear map) |
| **residual** | what the modes do not account for; `resolved + residual` is the record exactly | Definition 8.4 |
| **Fourier view** | `S(f) = Σ P_k (L_k(f) + L_k(−f))/2`, drawn from the modes | the transform of `C(τ) = Re Σ P_k μ_k^τ` |
| **axis fill** `φ_F`, `φ_T` | `2^{H}/L` of an axis's correlation spectrum | Lemma 3.2, Prop 3.5 |
| **screen fill** `φ` | the exponential entropy of the singular spectrum over its length | Lemma 3.2 |
| **étendue** | `φ_F φ_T`, the joint aperture area | Prop 3.5 |
| **space–bandwidth** | `n_F n_T`, the resolvable-cell count | Definition 3.3 |
| **Strehl** | `λ₁ / Σλ`, the dominant coherent-mode fraction | Lemma 3.4 |
| **coherence** | a z-score against the exact row-permutation null | Theorem 5.2 |
| **decay** `C(τ)` | the biased autocovariance: the optical transfer function | Lemma 4.2, Cor 4.3 |
| **diffraction limit** `a_δ` | `2^{−H(C²)}`, the inverse resolvable spacing | Definition 4.4 |
| **Mercer ratio** | the temporal over the spectral width | Prop 4.7 |
| **propagation constant** | `α + iβ`: mode contrast and carrier | Definition 6.1, Lemma 6.2 |
| **noise floor** | the Tracy–Widom edge at `far` | §8 |
| **K** | the modes standing above the floor | §8 |
| **basis** (KLT) | the resolved modes' principal directions over the channels, orthonormal in the noise metric | `Aperture.basis()` |
| **drift** | what a later record holds that a basis does not span, read against the floor | `Basis.drift` |

## Why it's principled

Every read is a classical result specialised to finite, discrete data. A signal's autocorrelation *is*
its optical transfer function (**Wiener–Khinchin** → **Fourier optics**). The diffraction limit is
therefore one over that transfer function's bandwidth (**Abbe/Rayleigh**). The count is taken
against the **Tracy–Widom** edge, the universal law of the largest eigenvalue of noise. The full
derivation is in the paper, and its governing lemmas are machine-checked (below).

## Backends & determinism

- **One code path, numpy or torch.** A numpy array runs on the CPU; a torch tensor runs on its
  device and stays there. Torch is imported only when a tensor appears.
- **Deterministic.** Seeded reads are reproducible, and numpy and torch agree to floating-point
  round-off.
- **Complex-safe** end to end. Reads take the record as given; for the incoherent read of an
  amplitude record, pass the intensity (`decay(W ** 2)`). `operator_read` takes real records.

## Axis convention

Every input `W` has shape `(T, F)`:
- axis 0 (rows, `_T`) is the **ordered** / evolution axis; "time" names a role;
- axis 1 (columns, `_F`) is the **feature** / channel axis; "frequency" names a role.

## Absent data

Mark what was never observed, either as `NaN` in `W` or with a boolean `mask` where `True` means
*not observed*:

```python
ap = Aperture(W, mask=flags)        # flags[t, f] True  ->  that cell was not observed
```

Every read then divides by the **measured extent**. A read taken through a mask equals the same
frame with those channels deleted, to floating point. A zero-filled channel is a different frame:
zero is a reading of no power, and it widens the axis the signal is scored against.

## Per-channel structure

Multiplying the whole record by a constant:
- every dimensionless read is identical from `1e+30` down to `1e-140`;
- every dimensioned read scales exactly as its own dimension.

Adding a constant baseline leaves every read unchanged.

[`src/tests/test_scale_invariance.py`](../src/tests/test_scale_invariance.py) sweeps both halves, with
a control that each read still separates structure from noise.

*Per-channel* gain is different: to the instrument, a channel with ten times the gain carries ten
times the power. If that gain is instrumental, remove it before you read:

```python
from entroptics.entropy import normalize
ap = Aperture(normalize(W))          # per-channel mean removed, RMS equalised
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite (`src/tests/`) covers:
- the full optics read, pinned as a golden contract;
- numpy↔torch parity;
- the mathematical invariants;
- the write path's round trip and drift, in both directions;
- the operator read against planted tones and against noise;
- determinism, and degenerate inputs.

Four files pin how the reads behave, each with a control that lets it fail:

| file | what it holds |
|---|---|
| `test_scale_invariance.py` | dimensionless reads identical across `1e+30`–`1e-140`, and every read still separates structure from noise |
| `test_fold_band_calibration.py` | `fold_band`'s false-fold rate on pure noise, and the power that must survive it |
| `test_coupling_reduces_to_pearson.py` | at one shared coordinate, `coupling.strength` **is** Pearson's r |
| `test_resolved_modes_is_not_a_rank.py` | `resolved_modes` counts modes above a noise floor; it is not a matrix rank |

## Formal certification

The governing lemmas are **machine-checked in Lean 4 / Mathlib**
([`research/lean/`](https://github.com/Agience/entroptics/tree/main/research/lean), 44 theorems):
- the fill-fraction and Strehl bounds;
- positive semidefiniteness of the biased autocovariance;
- the exact permutation-null mean of the coherence;
- the Weyl-certified attenuation interval;
- axial≠directional concentration;
- exact decay-rate recovery and additive splicing.

`lake build` compiles with **no `sorry`**.
