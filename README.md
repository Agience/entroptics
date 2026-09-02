# Entroptics

[![PyPI](https://img.shields.io/pypi/v/entroptics)](https://pypi.org/project/entroptics/)
[![Python](https://img.shields.io/pypi/pyversions/entroptics)](https://pypi.org/project/entroptics/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/Agience/entroptics/blob/main/LICENSE.md)
[![CI](https://github.com/Agience/entroptics/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/entroptics/actions/workflows/ci.yml)
[![Proofs](https://img.shields.io/badge/proofs-Lean%204%20%2F%20Mathlib-4B0082)](https://github.com/Agience/entroptics/tree/main/research/lean)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21273400-blue)](https://doi.org/10.5281/zenodo.21273400)
[![Sponsor](https://img.shields.io/badge/Sponsor-ikailo-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/ikailo)

## *The universe, in focus.*

**Read any 2-D signal as a finite optical aperture whose resolution is fixed by the signal's own entropy.**

Entroptics (entropy + optics) treats a 2-D array `W` of shape `(T, F)`, one **ordered** axis (time / evolution) and one **feature** axis (channels / frequency), as a finite optical aperture. The signal sets its own focus from the entropy of its power marginals. Every quantity Entroptics reports is then a standard optical / wave measurement (étendue, Strehl, OTF, diffraction limit, propagation constant), and the same reads apply to any structured 2-D field: a spectrogram, a waterfall, an embedding stack, a market panel, an image.

It is a small, standalone library, **numpy only** at the core (scipy and torch optional), built entirely from geometry and standard theorems. Parameter-free and domain-agnostic.

## Install

```bash
pip install -e .              # core (numpy only)
pip install -e ".[torch]"     # + GPU / torch tensors
pip install -e ".[scipy]"     # + exact MAD constant
pip install -e ".[dev]"       # + pytest, torch, scipy (to run the tests)
```

Requires Python ≥ 3.10 and numpy ≥ 2.0.

## The apparatus

The optical chain: **a beam passes through an aperture, a lens converts it, and a screen receives it as a projection.**

A screen is viewed differently from each side, and `Projection` is that view made concrete — one signal on its own entropy-matched grid. 

| | |
|---|---|
| **`Aperture`** | What **bounds** a beam, and the read of everything *about* it (read-only). The single front door: batch `Aperture(W)` or streaming `Aperture(window=…).update(frame)`. Per-axis and screen-area reads, the mode spectrum, the diffraction limit, and a streaming dynamical operator for exact decay rates. Its size *is* the beam's étendue. |
| **`Projection`** | The screen **as one side sees it** — one signal on its own entropy-matched grid (read-only). Folds the signal onto its entropy-matched grid and reads its SVD factorization, coherence, and the modes above the noise floor. A denoised view is the `extract` filter — a projection onto those resolved modes. |
| **`Lens`** | A system's **conversion**: `entry` (surface → the screen's coordinates) and `inverse` (back out), plus that system's own laws — `energy`, `zero`, `null`. All domain code lives here. |
| **`Screen`** | Where beams **land**: the surface two or more systems share, and the crossing measurements between them. |
| **`Beam`** | What a side **carries**: energy, étendue, and the directions it occupies. A bundle of beams — each mode is itself a `Beam`, down to a leaf spanning one direction. |

`ap.projection()` and `Projection(W).aperture()` cross between the single-signal views.

### Which to reach for

| you have | you want | use |
|---|---|---|
| one signal | its structure, resolution, decay rates, étendue | **`Aperture`** |
| one signal | its factorization, embedding, or a denoised view | **`Projection`** (or `Aperture.extract`) |
| two or more systems | them to meet, convert, couple, or trade energy | **`Screen`** |
| a surface to convert | it mapped onto a screen's coordinates and back | **`Lens`** |

A `Screen` earns its place only when the question is *between* signals. For one signal it adds a
lens registration and a placement for a reading `Aperture` or `Projection` gives directly.

### The screen

`N` lenses carry `N` conversions and no pairwise table exists. Every screen read runs on the native, un-folded frame.

```python
from entroptics import Screen

s = Screen()                                                  # far= sets the reader's level
s.register("A", entry=to_concept_a, inverse=from_concept_a)   # a lens IS its conversion
s.register("B", entry=to_concept_b, inverse=from_concept_b,
           energy=my_energy_law, zero=my_zero, null=my_floor)  # ...and its own laws

concept   = s.place("A", surface_a)      # A.entry   : signal -> concept
surface_b = s.render("B", concept)       # B.inverse : concept -> B's signal
```

**Reads between sides**

```python
s.couple("A", "B")        # the MEASURED signed coupling (exact permutation null); 0 unresolved
s.coupling("A", "B")      # the evidence behind it: z, sign, strength, phase, tightness
s.transfer("A", "B")      # absorbed vs transmitted, the signed flux, and which concepts
                          # the energy condensed into
s.uncondensed("A", "B")   # what did NOT condense, as a frame ready for another screen
```

**Reads about one side**

```python
s.beam("A")               # what it carries: energy, flow, etendue, modes, basis, profile
s.aperture("A")           # the full optics of that side
s.directions("A")         # the directions it resolves, against ITS OWN floor
s.energy("A")             # its energy flow, by ITS OWN law
```

**Certificates on a lens**

```python
s.certify("A", surface)   # imaging: does inverse . entry return the surface?
s.lossless("A", surface)  # the same as a bare relative residual
s.realise("A", "B")       # what the crossing actually delivers vs the étendue bound
s.linear("A")             # does the lens pass the beam's modes independently?
```

**The screen as a whole**

```python
s.read()                  # the aperture measurement of the joint frame
s.basis()                 # the shared surface's coordinates
s.balance()               # each side at its own zero, and whether that zero closes
s.resolution()            # the settled state; None when nothing clears the floor
```

The sides need not share an ordered axis — bank transactions run on real time, language on information flow, and they still meet on the shared basis. Only the row-paired reads (`couple`, `joint`, `read`, `resolution`) need a common order, and they say so.

**Who decides what.** A lens declares its own noise (`null`) — a detector and a market count signal differently. The false-alarm level `far` is the reader's, by Neyman–Pearson. The caller owns the loop: these reads are pull, and the handoff in either direction is a plain `(T, D)` frame.

## Quickstart

### Batch

```python
import numpy as np
from entroptics import Aperture

W = np.random.default_rng(0).standard_normal((256, 64))

ap = Aperture(W)

o = ap.optics()               # the full intrinsic read, as a dict (33 fields)
ap.phi, ap.magnification      # fill fraction and its reciprocal (scale duality)
ap.etendue, ap.space_bandwidth
ap.strehl                     # dominant-mode coherence
ap.T, ap.F                    # per-axis AxisRead(H, n, delta, phi, sigma)
ap.spectral                   # contrast, attenuation α, phase β, dispersion, resolved_power, dominance
ap.a_delta, ap.correlation_length, ap.mercer   # diffraction limit + certificate
ap.scale_profile()            # structure vs observation window (resolution vs aperture size)
```

### The projection

```python
sc = ap.projection()
sc.K_signal        # SVD modes standing above the noise floor
sc.coherence       # ordered-axis structure z-score (deterministic)
sc.footprints      # per-mode (phi_T, phi_F) localization of each resolved mode
```

### The filter

Pull the resolved signal out of the field — a projection onto its own screen modes.

```python
clean, info = ap.extract()          # the whole frame, since `Aperture(W)` reads all of it
info["K_signal"]                    # resolved modes above the floor
info["contrast"]                    # leading singular value over the floor (σ₁ / Φ)
info["n_kept"], info["n_dropped"]   # transient modes kept, persistent (RFI) modes dropped
```

`clean = U · diag(S̃) · Vᴴ` uses the data's own screen modes `U, Vᴴ` and the Gavish–Donoho optimal singular-value shrinkage `S̃` against the derived floor: exact recovery in the noise-free limit, idempotent, with the persistent narrowband (`φ_F ≤ φ_T`) modes dropped.

### Streaming, resume, splice

Feed frames from the first one and propagate; the exact-rate operator updates online.

```python
ap = Aperture(window=512)          # window bounds the optics snapshot
for frame in signal:               # each frame an F-vector (numpy or torch)
    ap.update(frame)

ap.rates()                         # long_range (slowest) & short_range (fastest) decay
ap.predict(frame)                  # one-step forecast A·x (A = ap.propagator_full())

s = ap.state()                     # export the full operator state …
ap2 = Aperture.from_state(s)       # … and resume exactly (bit-for-bit)

whole = a.splice(b)                # a, b: two Aperture streams → the concatenated-stream operator (exact at forgetting=1)
```

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

`space_bandwidth` is a capacity, not a content: an unfolded screen reports `T·F` whatever sits on
it. What the screen actually fills is `ap.etendue * ap.space_bandwidth` — exactly 1 for a single
mode, rising with the modes present.

### The mode spectrum

`ap.spectral` reads the correlation spectrum against a derived noise floor:

- `contrast`, `top_share`, `resolved_modes`, `noise_floor`, `resolved_power` (summed eigenvalue
  excess above the floor), `dominance` = `(λ₁−1)/(F−1)`.
- the propagation constant `γ = α + iβ` — attenuation `α`, phase `β` — and `dispersion`.
- `ap.attenuation_interval(band)` returns a Weyl-certified interval for `α`.

The floor comes from a `null=` provider (`entroptics.null_providers`): the derived
Marchenko–Pastur / Johnstone default `mp` when `null` is unset, or the deterministic
data-derived Tukey fence via `null=null_providers.robust`.

### The diffraction limit

- `ap.decay` — the OTF, an FFT-free autocorrelation.
- `ap.a_delta` (entropy width `1/2^{H(C²)}`) and `ap.correlation_length` (the decay length ξ).
- `ap.mercer` — a model-free temporal-vs-spectral cross-check.
- `ap.rayleigh_shape_factor` (`g = xi * a_delta`), `ap.fresnel_number(window)`, `ap.shape_factor`
  (the Abbe factor `a_delta / phi_F`).
- `ap.decay_scatter` — the decay is a sum over per-channel autocovariances, so the channels are
  replicates of it and their disagreement measures the read's own uncertainty. No null, nothing
  subtracted. `noise_share` far below `tail_share` means the correlation is structure the channels
  agree on; the two converging means the width is scatter, and `a_delta` overstates the correlation
  length. The cure is more channels.

### How it moves, and at what scale

- **Dynamics**, `Aperture.rates()`: exact per-mode decay rates `α_k = −log|μ_k|` and frequencies
  `β_k = arg(μ_k)` from a streaming online-DMD / Koopman operator; splice-able and resumable.
  `Aperture.dominant_decay_rate` reads the slowest mode's rate `α_1 = −log|μ_1|` straight off that
  spectrum, isolating that one mode. A lag-window read returns a blend of all of them. `Aperture.propagator_full()` and
  `predict(x)` expose and apply the full one-step operator `A = P_yx · P_xx⁺`.
- **Multi-scale**, `ap.scale_profile()`: structure as a function of observation window —
  `K_signal`, `coherence`, `a_delta`, `phi_T` per window, plus `resolved_window` and
  `dominant_window` (in ordered-axis cells).
- **Sweep**, `Aperture.sweep()`: fix the aperture to a bounded capacity and sweep it where the
  coherence gate finds structure; noise-only patches are skipped. Returns per-band `span`, entropy
  `width` and tail decay `tau`, in dimensionless window samples.

### Recovering the signal

- **Projection**, `Projection` (`footprints`, `significance`, `read`), and `Aperture.tensor()` — a
  delay-embedded Tucker/HOSVD exposing within-window fine structure the averaged screen loses.
- **Filter**, `Aperture.extract()`: project the field onto its own resolved screen modes with
  Gavish–Donoho optimal shrinkage against the derived floor, dropping persistent narrowband
  (`phi_F <= phi_T`) interference. Returns `(clean, info)`; `clean` is a linear projection of the
  measured data — it synthesises nothing — and `info` carries `K_signal`, `contrast`, `coherence`
  and the kept/dropped modes.

### Other shapes of input

- **N-D fields**, `entroptics.fields`: `slabs(field, plane_axes)` and
  `over_planes(field, plane_axes, read=, reduce=)` reduce a higher-D field to the 2-D screen
  **while keeping each plane intact** (what feature reads need, since they use within-plane
  correlation); `pool(field, ordered_axis)` flattens sites as samples, which is what ordered reads
  need.
- **A stack at once**, `resolved_batch(X)` for `X: (B, T, F)`: the same resolved read over many
  frames, backend-optimal — numpy on the CPU (bit-identical to a per-frame `Projection`), a torch
  tensor on its device. Two cost tiers: the cheap survey gate (`K_signal`, `sigma_top`,
  `noise_floor`) always, the resolved `basis` and per-row `energy` on demand and only for frames
  that cleared the gate. `ResolvedScreen` / `ResolvedScreenBatch` are the stateful siblings for a
  screen you revisit (an LLM KV head across turns): append rows, refresh the basis lazily, resume
  from `state()`. `ResourceLimits` bounds threads, memory and GPU use; chunking is
  output-transparent.
- **Koopman lift**, `entroptics.lift`: `delay_embed(W, d)` builds Takens/Hankel coordinates
  `(T,F) → (T-d+1, d·F)`, and `koopman_lift(W, d)` fits the operator there — the path from a
  nonlinear or near-orthogonal trajectory to a linear one. An oscillator that resolves no modes
  raw resolves them after the lift.

### Records that are not a 2-D field

- **Spectral proximity**, `entroptics.proximity`: a magnitude-carrying, width-free digest of a
  frame's spectrum (`mp_spectrum`, `spectral_distance`, `bulk_edge`, `effective_width`) and a
  probe over a set of them (`SpectrumProbe`).
- **Symbol sequences**, `entroptics.sequence`: the ordered-axis reads on a symbol stream —
  `entropy_rate`, `block_entropies`, `lempel_ziv_rate`, `redundancy_rate`, `effective_length`,
  `surrogate_test`.

## Real data: fast radio bursts

`research/figures/frb_panel.py` applies the plain library, with zero tuning, to the public CHIME/FRB Catalog 1 waterfalls at native 16384-channel resolution. It supplies only observer facts — dead channels are **dropped, never zero-filled** — and passes the surviving channels to the aperture front door. The catalog waterfalls arrive already dedispersed; Entroptics adds no dedispersion, no derotation, and no domain physics of its own.

```python
from entroptics import Aperture

live = np.isfinite(W).all(axis=0) & (np.nanstd(W, axis=0) > 0)   # observer fact: the RFI mask
clean, info = Aperture(W[:, live], window=None).extract()        # everything downstream is the library
info["K_signal"], info["contrast"], info["coherence"]
```

`extract` is the Gavish-Donoho projection onto the resolved modes with the `φ_F > φ_T` persistent-structure cut: the noise sea is attenuated and persistent narrowband interference removed, with the burst morphology intact. The per-burst reads behind the figure are in [`research/figures/frb_panel.csv`](research/figures/frb_panel.csv), and the method is §14.1 of [the paper](research/PAPER.md).

## Why it's principled

Every read is a classical result specialised to finite, discrete data. The backbone: a signal's autocorrelation *is* its optical transfer function (**Wiener–Khinchin** → **Fourier optics**), so the diffraction limit is one over that OTF's bandwidth (**Abbe/Rayleigh**). The full derivation, and the Lean-checked lemmas, are in the [paper](https://github.com/Agience/entroptics/blob/main/research/PAPER.pdf).

## Backends & determinism

- **One code path, numpy or torch.** Feed a numpy array → runs on CPU; feed a torch tensor → runs on its device (GPU), staying on-device; torch is imported lazily only when a tensor appears.
- **Deterministic.** The coherence is a closed-form permutation-null z-score, so results are reproducible and bit-identical across numpy and torch (to floating-point round-off).
- **Complex-safe** end to end. Reads take the record as given and infer nothing from it — whether you hold a field or an intensity is a fact about your instrument, not about the sign of your samples. For the incoherent read of an amplitude record, pass the intensity (`decay(W ** 2)`). `Aperture.sweep()` finds and reads its bands on a field like any other record — `span`, `coherence`, `contrast`, `K`, `noise_floor`, `phi_T/F` — and the span takes you back to the original vectors, complex intact. Its `peak`, `width` and `tau_decay` come off a per-sample brightness, which a field does not carry, so those three return `-1`/`NaN`; pass `abs(W)` or `abs(W)**2` when you want them.

## Axis convention

Every input `W` has shape `(T, F)`:

- axis-0 (rows, `_T`) is the **ordered** / evolution axis, "time" is a *role*, not literal physics;
- axis-1 (cols, `_F`) is the **feature** / channel axis, "frequency" is likewise a role.

Any 2-D array with one ordered axis works.

## Absent data

Mark what was never observed — either as `NaN` in `W`, or with a boolean `mask` where `True` means
*not observed*. The two say the same thing, and a mask works over a value that is finite and wrong
(a saturated sample, an RFI-flagged channel that still holds a reading):

```python
ap = Aperture(W, mask=flags)        # flags[t, f] True  ->  that cell was not observed
```

Every read then divides by the **measured extent** — the rows and columns that carry at least one
observation — not by the array's shape. A read taken through a mask equals the same frame with
those channels deleted, to floating point.

Absence is not an observation of zero. Substituting `0` for an unobserved channel is a different
frame: zero is a real reading of no power, it belongs in the extent, and it widens the axis the
signal is scored against. Mark it absent and the reads are unmoved; fill it with zeros and they move, correctly.

## Per-channel structure

Multiplying the whole record by a constant changes nothing — every read is identical at `1e-21`
and at `1`. A constant baseline changes nothing either; the reads are taken on the centred block.

*Per-channel* structure is different. A channel with ten times another channel's gain is, to the
instrument, a channel carrying ten times the power — there is nothing in the data that says whether
that is the sky or the amplifier. So the reads describe it, because it is part of the frame they
were handed. `phi_F` is the exception: the feature-axis correlation has unit diagonal, so it
rescales every channel by construction.

If per-channel gain or baseline is instrumental, remove it before you read:

```python
from entroptics.entropy import normalize
ap = Aperture(normalize(W))          # per-channel median removed, robust scale equalised
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite (`src/tests/`) pins the full optics read as a golden contract, checks numpy↔torch parity, the mathematical invariants (étendue = φ_F·φ_T, exact decay-rate recovery, PSD autocovariance, axial-vs-directional concentration, the read-side filter's exact projection and idempotence), round-trips (tensor reconstruct, factor pack/unpack), determinism, and degenerate-input robustness.

## Formal certification

The governing lemmas of the theory ([`research/PAPER.pdf`](https://github.com/Agience/entroptics/blob/main/research/PAPER.pdf)) are **machine-checked in Lean 4 / Mathlib** ([`research/lean/`](https://github.com/Agience/entroptics/tree/main/research/lean), 44 theorems): the fill-fraction and Strehl bounds, positive-semidefiniteness of the biased autocovariance (peak-at-zero-lag OTF), the exact permutation-null mean of the coherence, the Weyl-certified attenuation interval, axial≠directional concentration, and exact decay-rate recovery + additive splicing. `lake build` compiles with **no `sorry`**, resting only on Mathlib's standard axioms.

## One system, three instruments

Entroptics is the **lens** of the Agience system — the accuracy instrument. No observer sees the whole: every signal arrives through a finite aperture, and entropy is the exact measure of the gap between the aperture and the world. Entroptics reads that gap honestly — every constant derived, the observer's one input (decision risk) declared out loud, the governing lemmas machine-checked.

The lens composes with two siblings: **[Mantle](https://github.com/Agience/agience-mantle)**, the memory — an encrypted-by-default artifact store where provenance lives inside every artifact, and where reachability across the grant graph decides *which keys are issued*; and **[Agience](https://github.com/Agience)**, the seat — the platform where people and agents work, and where nothing a model produced carries weight until an observer stakes something on it. Entroptics carries information through space, Mantle carries it through time, and Agience is the one who is looking.

## Contributing

Bug reports, validation experiments, backend-parity fixes, and new reads (with their theorems) are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The short version: the library is parameter-free, deterministic, and numpy-only at the core, and contributions must keep it that way — every constant needs a provenance, every claim needs a theorem, and the golden-contract tests must stay bit-identical across backends.

## License

**Apache License 2.0** - see [`LICENSE.md`](https://github.com/Agience/entroptics/blob/main/LICENSE.md), [`NOTICE`](https://github.com/Agience/entroptics/blob/main/NOTICE), [`PATENTS.md`](https://github.com/Agience/entroptics/blob/main/PATENTS.md) and [`PLEDGE.md`](https://github.com/Agience/entroptics/blob/main/PLEDGE.md).

## Star History

<a href="https://www.star-history.com/?repos=Agience%2Fentroptics&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Agience/entroptics&type=date&theme=dark&legend=top-left&sealed_token=DRxmKEUu-jgYDGrGi5K7vVFwrww1YJiMFU2_nv85yGjwPbsvhmTkOSlVv2aQQGkVDHXd2jlGQnjZDHbYYOXwfObR6iE9wTeV5jyplb30xZ3GFdD1ebDZIAonKgIvYBZ5vH8Z7T-2lSgsWrktUeeoPUdPPRELXBa4LY0ILQatLXzOLeWpq4dU5eVFXTcH" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Agience/entroptics&type=date&legend=top-left&sealed_token=DRxmKEUu-jgYDGrGi5K7vVFwrww1YJiMFU2_nv85yGjwPbsvhmTkOSlVv2aQQGkVDHXd2jlGQnjZDHbYYOXwfObR6iE9wTeV5jyplb30xZ3GFdD1ebDZIAonKgIvYBZ5vH8Z7T-2lSgsWrktUeeoPUdPPRELXBa4LY0ILQatLXzOLeWpq4dU5eVFXTcH" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Agience/entroptics&type=date&legend=top-left&sealed_token=DRxmKEUu-jgYDGrGi5K7vVFwrww1YJiMFU2_nv85yGjwPbsvhmTkOSlVv2aQQGkVDHXd2jlGQnjZDHbYYOXwfObR6iE9wTeV5jyplb30xZ3GFdD1ebDZIAonKgIvYBZ5vH8Z7T-2lSgsWrktUeeoPUdPPRELXBa4LY0ILQatLXzOLeWpq4dU5eVFXTcH" />
 </picture>
</a>
