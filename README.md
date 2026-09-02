# Entroptics

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

## Two objects

| | |
|---|---|
| **`Aperture`** | The **optics**, information *about* a structure (read-only). The single front door: batch `Aperture(W)` or streaming `Aperture(window=…).update(frame)`. Exposes the per-axis and screen-area reads, the mode spectrum, the diffraction limit, and a streaming dynamical operator for exact decay rates. |
| **`Screen`** | The **projection**, information *within* the structure (read-only). Folds the signal onto its entropy-matched grid and reads its SVD structure, coherence, and the modes above the noise floor. A denoised view is the `extract` filter — a projection onto those resolved modes. |

`ap.screen()` and `Screen(W).aperture()` cross between the two views.

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
sc = ap.screen()
sc.K_signal        # SVD modes standing above the noise floor
sc.coherence       # ordered-axis structure z-score (deterministic)
sc.footprints      # per-mode (phi_T, phi_F) localization of each resolved mode
```

### The filter

Pull the resolved signal out of the field — a projection onto its own screen modes.

```python
clean, info = ap.extract()          # or: entroptics.extract(W) for the full frame
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

All reads are intrinsic (derived from `W` alone) and tied to a standard theorem.

- **Scale**, `phi` (fill fraction) and `magnification = 1/phi`; per-axis `H_T/H_F`, `n_T/n_F`, `delta_T/delta_F` from the entropy of the power marginals.
- **Aperture area**, `etendue = phi_F · phi_T` (a bounded 2-D area), `space_bandwidth = n_F · n_T` (resolvable spots).
- **Coherence**, `strehl` (dominant-mode power fraction), `phi_T/phi_F`, `sigma_T/sigma_F`; `Screen.coherence` (a closed-form z-score against the exact row-permutation null).
- **Mode spectrum**, `spectral_optics`: contrast, top-share, resolved modes, noise floor from a `null=` provider (see `entroptics.null_providers`) — the derived Marchenko–Pastur / Johnstone default (`mp`, used when `null` is unset) or the deterministic data-derived Tukey fence (pass `null=null_providers.robust`), the propagation constant γ = α + iβ (attenuation α, phase β), dispersion, `resolved_power` (the summed eigenvalue excess above the floor), and `dominance = (λ₁−1)/(F−1)` (F = correlation variables). `attenuation_interval` gives a Weyl-certified interval for α.
- **Concentration**, `concentration`: `intensity` (σ₁²), `focus` (axial), `resultant` (directional, von Mises–Fisher).
- **Diffraction limit**, `decay` (the OTF, an FFT-free autocorrelation), `diffraction_limit` (entropy width `a_delta = 1/2^{H(C²)}` + classical Abbe length), `mercer_certificate` (a model-free temporal-vs-spectral cross-check), `rayleigh_shape_factor` (the shape factor `g = xi * a_delta`), `fresnel_number`, `shape_factor` (the Abbe factor `a_delta / phi_F`), `decay_rate` (log-linear fit of a supplied 1-D curve).
- **N-D fields**, `entroptics.fields`: `slabs(field, plane_axes)` / `over_planes(field, plane_axes, read=, reduce=)` reduce a higher-D field to the 2-D screen **while keeping each plane intact** (the correct reduction for feature reads, which need within-plane correlation); `pool(field, ordered_axis)` flattens sites as samples (the correct reduction for ordered reads).
- **Dynamics**, `Aperture.rates()`: exact per-mode decay rates `α_k = −log|μ_k|` and frequencies `β_k = arg(μ_k)` from a streaming online-DMD / Koopman operator; splice-able and resumable. `Aperture.propagator_full()` / `predict(x)` expose and apply the full one-step operator `A = P_yx · P_xx⁺`.
- **Multi-scale**, `scale_profile(W)`: structure as a function of observation window (resolution vs aperture size), `K_signal` / `coherence` / `a_delta` / `phi_T` per window, plus `resolved_window` / `dominant_window` (in ordered-axis cells).
- **Projection**, `Screen` (`embeddings`, `footprints`, `read`) and `Aperture.tensor()` (a delay-embedded Tucker/HOSVD that exposes within-window fine structure the averaged screen loses).
- **Filter**, `Aperture.extract()` / `entroptics.extract(W)`: the read-side denoiser — project the field onto its own resolved screen modes with Gavish–Donoho optimal singular-value shrinkage against the derived floor, dropping persistent narrowband (`phi_F <= phi_T`) interference. Returns `(clean, info)` where `clean` is a linear projection of the measured data (it synthesises nothing) and `info` carries `K_signal`, `contrast`, `coherence`, and the kept/dropped modes.
- **Sweep**, `entroptics.sweep(W)`: fix the aperture to a bounded capacity and sweep it where the coherence gate finds structure (noise-only patches are skipped); returns per-coherent-band reads — column `span`, entropy `width`, and tail decay `tau` — in abstract, dimensionless window samples.

## Unwinding a propagation channel

`research/applications/cosmology.py` drives Entroptics to invert a burst's interstellar propagation and return it to the source frame. Each transform in the medium's stack carries one free parameter, fixed by **maximizing an entroptics read**:

| transform | recovers | read maximized |
|---|---|---|
| dispersion `τ(f) = K·DM·f⁻²` | `DM` | leading-mode contrast |
| Faraday rotation `Δψ = RM·λ²` | `RM` | derotated polarized amplitude |
| scattering `⊛ e^{−t/τ}` | `τ_scatter` | per-band tail decay rate |
| instrument / RFI | — | the `φ_F > φ_T` geometric filter |

```python
from cosmology import unwind

r = unwind(I, freqs, dt)               # intensity → DM + scattering + RFI-cleaned source frame
r = unwind(I, freqs, dt, Q=Q, U=U)     # + Faraday derotation and source-frame polarization
r.source                               # dedispersed, derotated, RFI-cleaned burst
r.dm, r.rm, r.tau_scatter, r.pol_fraction
```

The library supplies the objectives (contrast, polarized amplitude, decay rate) and `unwind` applies the known-physics inverse transforms out of band. Pass `skip={"dispersion"}` for data a stage has already been applied to (e.g. coherently-dedispersed baseband).

## The theory in one line

Everything hangs off one chain of standard results:

> **Wiener–Khinchin** (autocorrelation ↔ power spectrum) provides the lag average → the **Fourier-optics autocorrelation theorem** (OTF = pupil autocorrelation) → **Abbe/Rayleigh** (resolution = 1 / OTF-bandwidth = 1 / correlation length).

The entropy width `1/2^{H}` (with `H` the Shannon entropy of the squared decay `C²`) is the noise-robust estimator of that reciprocal correlation length. By **Mercer**, the temporal read (decay entropy) and the spectral read (the stationary correlation operator's eigenspectrum) track each other, a built-in, model-free certificate.

## Backends & determinism

- **One code path, numpy or torch.** Feed a numpy array → runs on CPU; feed a torch tensor → runs on its device (GPU), staying on-device; torch is imported lazily only when a tensor appears.
- **Deterministic.** The coherence is a closed-form permutation-null z-score, so results are reproducible and bit-identical across numpy and torch (to floating-point round-off).
- **Complex-safe** end to end (coherent field vs. incoherent intensity reads are dispatched automatically).

## Axis convention

Every input `W` has shape `(T, F)`:

- axis-0 (rows, `_T`) is the **ordered** / evolution axis, "time" is a *role*, not literal physics;
- axis-1 (cols, `_F`) is the **feature** / channel axis, "frequency" is likewise a role.

Any 2-D array with one ordered axis works.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite (`src/tests/`) pins the full optics read as a golden contract, checks numpy↔torch parity, the mathematical invariants (étendue = φ_F·φ_T, exact decay-rate recovery, PSD autocovariance, axial-vs-directional concentration, the read-side filter's exact projection and idempotence), round-trips (tensor reconstruct, factor pack/unpack), determinism, and degenerate-input robustness.

## Formal certification

The governing lemmas of the theory (`research/PAPER.pdf`) are **machine-checked in Lean 4 / Mathlib** (`research/lean/`, 30 theorems): the fill-fraction and Strehl bounds, positive-semidefiniteness of the biased autocovariance (peak-at-zero-lag OTF), the exact permutation-null mean of the coherence, the Weyl-certified attenuation interval, axial≠directional concentration, and exact decay-rate recovery + additive splicing. `lake build` compiles with **no `sorry`**, resting only on Mathlib's standard axioms.

## License

Apache 2.0 for the source code (copyright only). Patent rights are reserved, see [`LICENSE.md`](LICENSE.md) and [`PATENTS.md`](PATENTS.md). You are free to read, run, validate, reproduce, cite, and build research on this code and its proofs.
