# Entroptics

[![PyPI](https://img.shields.io/pypi/v/entroptics)](https://pypi.org/project/entroptics/)
[![Python](https://img.shields.io/pypi/pyversions/entroptics)](https://pypi.org/project/entroptics/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/Agience/entroptics/blob/main/LICENSE.md)
[![CI](https://github.com/Agience/entroptics/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/entroptics/actions/workflows/ci.yml)
[![Proofs](https://img.shields.io/badge/proofs-Lean%204%20%2F%20Mathlib-4B0082)](https://github.com/Agience/entroptics/tree/main/research/lean)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b)](https://github.com/Agience/entroptics/blob/main/research/PAPER.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21273400-blue)](https://doi.org/10.5281/zenodo.21273400)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**Find the real structure in noisy multichannel data — how many components it holds, what they are,
and how fast they decay and oscillate — without choosing a rank or tuning a threshold.**

Give Entroptics any 2-D array: time × sensors, a spectrogram, a radio waterfall, a stack of
embeddings. It tells you how many independent components stand above the noise, separates them from
the noise exactly, and reads each one's frequency and decay rate. The only setting is the
false-alarm rate you are willing to accept (5% by default).

## Install

```bash
pip install entroptics              # numpy only
pip install "entroptics[torch]"     # + torch tensors and the GPU
```

Python ≥ 3.10 and numpy ≥ 2.0. From a checkout, `pip install -e ".[dev]"` adds the test and
figure dependencies.

## Why use it

- **No knobs.** The number of components is decided against the level pure noise of the same shape
  would reach, at the false-alarm rate you set.
- **It finds the right count.** On planted components above the noise, it recovers the true number
  in all 36 cases tested, where Gavish–Donoho, AIC and MDL get 89–95% right
  ([validation](research/validation/RESULTS.md), experiment 17).
- **Calibrated on noise.** On pure noise it reports a component at or below the rate you ask for:
  Gaussian, Student-t, skewed and count noise, at equal or unequal channel levels
  ([benchmarks](research/benchmarks/README.md)).
- **Lossless.** The signal it keeps plus the residual it returns is your input, to floating-point
  round-off.
- **Frequency and decay together.** It reads how fast each component dies away as well as its
  frequency, between the FFT's bins, across gaps in the record, and next to a much stronger tone
  ([vs the FFT](research/benchmarks/README.md#3-how-does-it-compare-with-the-fft)).
- **Messy data welcome.** Missing cells, dead or stuck channels, channels at very different levels,
  few samples across many channels.
- **Light and fast.** numpy only; torch (and a GPU) when you pass a tensor. Streams at O(F²) per
  frame, with state you can save, resume and splice.

## When to reach for it

| you have | you want | call |
|---|---|---|
| a multichannel record | how many real components it holds | `Aperture(W).projection().K_signal` |
| a noisy record | the signal apart from the noise | `Aperture(W).extract()` |
| one signal, or a few channels | its frequencies and decay rates | `koopman_lift(x, d).modes()`, `Aperture(W).dynamics().modes()` |
| a reference recording | a basis to compress later data with, and an alarm when new structure appears | `Aperture(W).basis()`, `.encode`, `.drift` |
| a live stream | the same reads, updated frame by frame | `Aperture(window=...).update(frame)` |
| many frames at once | the count for each, in one call (numpy or GPU) | `resolved_batch(X)` |

## Examples

### How many components, and the signal apart from the noise

```python
import numpy as np
from entroptics import Aperture

rng = np.random.default_rng(1)
signal = rng.standard_normal((600, 3)) @ rng.standard_normal((3, 40))   # 3 sources over 40 sensors
W = signal + rng.standard_normal((600, 40))                             # plus noise

ap = Aperture(W)
ap.projection().K_signal                    # 3
Aperture(rng.standard_normal((600, 40))).projection().K_signal         # 0 on pure noise

clean, info = ap.extract()                  # clean + info["residual"] == W, to 4e-16
# distance from the true signal: 0.562 for the raw record, 0.184 for clean
```

### Frequencies and decay rates of a single signal

```python
from entroptics import koopman_lift

t = np.arange(1024)
y = np.exp(-0.01 * t) * np.cos(2 * np.pi * 0.0402 * t) + 0.02 * np.random.default_rng(2).standard_normal(1024)
m = koopman_lift(y[:, None], 16).modes()    # modes of the signal's 16-step delay coordinates
# the strongest mode: frequency 0.04022, decay 0.0102 (true 0.0402, 0.010)
```

### Learn a basis once, apply it to later data

```python
rng = np.random.default_rng(1)
mix = np.linalg.qr(rng.standard_normal((40, 3)))[0].T     # 3 hidden sources over 40 channels
gain = np.exp(rng.uniform(-0.5, 0.5, 40))                   # unequal channel noise

def frames(T):
    return (rng.standard_normal((T, 3)) * [6, 4, 3]) @ mix + gain * rng.standard_normal((T, 40))

b = Aperture(frames(600)).basis()   # b.K, b.F == 3, 40
later = frames(600)

A = b.encode(later)                  # (600, 3): three numbers per row instead of forty
resolved, residual = b.split(later)  # resolved + residual == later
b.drift(later).K                     # 0: the basis still spans what arrives
new = np.linalg.qr(rng.standard_normal((40, 1)))[0][:, 0]
b.drift(frames(600) + np.outer(1.5 * rng.standard_normal(600), new)).K   # 1: a new component
```

### Streaming

```python
ap = Aperture(window=512)
for frame in stream:                 # each frame a vector of F channels (numpy or torch)
    ap.update(frame)

ap.rates()                           # the slowest and fastest decay in the stream
s = ap.state()                       # save, and later resume exactly with Aperture.from_state(s)
```

Every output shown above is what [`readme_examples.py`](research/benchmarks/readme_examples.py)
prints ([output](research/benchmarks/readme_examples.txt)).

## Statement of need

Choosing how many components a record holds is the first step of a large class of analyses. It is
usually done with a rank or threshold the analyst picks, or with a criterion that needs a known
noise level, so a pipeline carries a constant tuned on one instrument and silently wrong on another.
Entroptics is for records whose instrument you do not control, or many substrates at once:
radio-astronomy waterfalls, spectrograms, sensor panels, embedding stacks. It is measured against
the standard selectors (the Gavish–Donoho optimal threshold, Wax–Kailath AIC and MDL) in the
[validation](research/validation/RESULTS.md). For dynamic mode
decomposition as such, [PyDMD](https://github.com/PyDMD/PyDMD) and
[PyKoopman](https://github.com/dynamicslab/pykoopman) are more complete.

## Know its limits

- **Noise correlated across channels or along time** (a common-mode drift, 1/f noise) is structure
  as far as the count is concerned. Remove it, or read a noise reference, first.
- **A component has to span channels.** A component confined to about `(1 + √(F/T))²` channels or
  fewer is not counted, however strong: two channels when records are long, nine for 64 samples
  across 256 channels.
- **Very heavy right tails** (lognormal, Pareto noise) exceed the false-alarm rate you set, most on
  long records.
- **A weak component beside a dominant one** is counted against the whole record's energy, and can
  fall under the line.

## How it works

Entroptics treats the array as a finite optical aperture. The signal's own entropy sets the
resolution it is read at; the count of components is taken against the Tracy–Widom edge — the
universal law for the largest eigenvalue of noise — at your false-alarm rate; and the dynamics are
read as an operator whose modes carry each component's frequency, decay and power. Every read is a
classical result (Wiener–Khinchin, Abbe/Rayleigh, Tracy–Widom, Koopman) specialised to finite data,
and the governing lemmas are machine-checked in Lean 4.

## Reproducing the numbers

The tests, the validation suite and every benchmark generate their inputs from a fixed seed, with no
download: `pytest`, `python research/validation/run_all.py`, and the scripts in
[`research/benchmarks/`](research/benchmarks/README.md). The two FRB figure routines read the public
CHIME/FRB Catalog 1 waterfalls ([how to fetch them](research/supplemental/frb/README.md)).

## Documentation

- **[The guide](docs/GUIDE.md):** every read and what it means, the conventions for inputs and
  missing data, and what the library guarantees.
- **[Benchmarks](research/benchmarks/README.md):** false-alarm rates, the comparison with the FFT,
  and costs, each from a committed script.
- **The paper** — [Markdown](research/PAPER.md) · [HTML](research/PAPER.html) ·
  [PDF](https://github.com/Agience/entroptics/blob/main/research/PAPER.pdf): the derivations, and
  where every constant comes from.
- **[API reference](docs/API.md)** and the **[CHANGELOG](CHANGELOG.md)**.
- **[Validation](research/validation/RESULTS.md):** what each read recovers against a planted truth.

## Getting help

- **Questions and bug reports:** open an issue at
  [github.com/Agience/entroptics/issues](https://github.com/Agience/entroptics/issues), with the array
  shape, the backend and the read you called.
- **Security issues:** email **connect@agience.ai**.
- **Contributions:** see [CONTRIBUTING.md](CONTRIBUTING.md); participation is governed by the
  [Code of Conduct](CODE_OF_CONDUCT.md).

## Star history

<a href="https://www.star-history.com/?repos=Agience%2Fentroptics&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Agience/entroptics&type=date&theme=dark&legend=top-left&sealed_token=DRxmKEUu-jgYDGrGi5K7vVFwrww1YJiMFU2_nv85yGjwPbsvhmTkOSlVv2aQQGkVDHXd2jlGQnjZDHbYYOXwfObR6iE9wTeV5jyplb30xZ3GFdD1ebDZIAonKgIvYBZ5vH8Z7T-2lSgsWrktUeeoPUdPPRELXBa4LY0ILQatLXzOLeWpq4dU5eVFXTcH" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Agience/entroptics&type=date&legend=top-left&sealed_token=DRxmKEUu-jgYDGrGi5K7vVFwrww1YJiMFU2_nv85yGjwPbsvhmTkOSlVv2aQQGkVDHXd2jlGQnjZDHbYYOXwfObR6iE9wTeV5jyplb30xZ3GFdD1ebDZIAonKgIvYBZ5vH8Z7T-2lSgsWrktUeeoPUdPPRELXBa4LY0ILQatLXzOLeWpq4dU5eVFXTcH" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Agience/entroptics&type=date&legend=top-left&sealed_token=DRxmKEUu-jgYDGrGi5K7vVFwrww1YJiMFU2_nv85yGjwPbsvhmTkOSlVv2aQQGkVDHXd2jlGQnjZDHbYYOXwfObR6iE9wTeV5jyplb30xZ3GFdD1ebDZIAonKgIvYBZ5vH8Z7T-2lSgsWrktUeeoPUdPPRELXBa4LY0ILQatLXzOLeWpq4dU5eVFXTcH" />
 </picture>
</a>

Licensed under Apache-2.0 — see [`LICENSE.md`](https://github.com/Agience/entroptics/blob/main/LICENSE.md),
[`NOTICE`](https://github.com/Agience/entroptics/blob/main/NOTICE), [`PATENTS.md`](https://github.com/Agience/entroptics/blob/main/PATENTS.md)
and [`PLEDGE.md`](https://github.com/Agience/entroptics/blob/main/PLEDGE.md).

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.
