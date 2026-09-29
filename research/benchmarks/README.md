# Benchmarks

Entroptics against the standard tools: the standard rank selectors for counting components, and
the FFT for cost, decay rates and tones. Every number on this page comes from a script whose output
is committed beside it. The figures and tables are drawn from those outputs by
[`figures.py`](figures.py).

## The results

| question | Entroptics | best alternative | winner |
|---|---|---|---|
| [Counting components exactly](#1-counting-components) (27 planted cases) | **1.000** | AIC 0.946, Gavish–Donoho 0.935, MDL 0.889 | **Entroptics** |
| [Components reported in correlated noise](#1-counting-components), nothing planted (lower is better) | **9.0** | Gavish–Donoho 17.9, MDL 18.9, AIC 44.7 | **Entroptics** |
| [Cost of a full read](#2-cost), 16 channels, T = 16384 to 1M | **1.85–207 ms** (counts 3 of 4 at T = 262144) | FFT pipeline, 3.5–516 ms | **Entroptics**, 1.6–2.6× faster |
| [Decay rates of damped tones](#3-against-the-fft-on-tones) (8 tones) | **best on 5** (fixed depth) | FFT pipeline, best on 3 | **Entroptics** |
| [Placing tone frequencies](#3-against-the-fft-on-tones) (15 tones) | **best on 8** | FFT pipeline, best on 7 | **Entroptics** |
| [Counting tones](#3-against-the-fft-on-tones) (7 records) | exact on 3, closer on 2 | FFT pipeline: exact on 4, closer on 1 | even |
| [Cost of a full read](#2-cost), 16 channels, T up to 4096 | 0.63–0.89 ms | **FFT pipeline, 0.30–0.79 ms** | FFT |
| [Cost of a full read](#2-cost), one channel | 0.64–328 ms | **FFT pipeline, 0.11–26 ms** | FFT, 5.6–16× faster |

Two more checks of the count:
- **[Pure noise](#4-pure-noise).** Entroptics reports a component in at most 2% of noise records
  when you allow 5%, for every kind of noise tested except very heavy right tails (lognormal,
  Pareto: 8.5–36%).
- **[Planted signals](#5-planted-signals).** A component spread across the channels is found in
  90–100% of records.

### The methods

| name on this page | in Entroptics? | what it is |
|---|---|---|
| **Entroptics** | yes | the component count, `Projection(W).K_signal` |
| **Entroptics, streaming** | yes | `Aperture(W).dynamics()`, then `.resolved()` (the count) and `.modes()` (frequencies, decay rates, powers) |
| **Entroptics, fixed depth** | yes | `koopman_lift(x, 16)`, then `.resolved()` and `.modes()`: one channel read at a delay depth of 16 |
| **Entroptics, automatic depth** | yes (`entroptics.experimental`) | `operator_read(x)`: the count, the depth and the modes all read from the record |
| **FFT alone** | no | `scipy.fft.rfft`: a spectrum only, with no count or decay rates; shown for scale |
| **FFT pipeline** | no | what it takes to get Entroptics' outputs from an FFT (`fft_pipeline` in [`operator_vs_fft.py`](operator_vs_fft.py)): one FFT per channel; peaks counted above the noise at the same 5% false-alarm rate; each peak's frequency interpolated and its decay rate read from its line width. Hann-windowed; section 3 also runs it unwindowed |
| **Gavish–Donoho**, **AIC**, **MDL** | no | the standard rank selectors: the optimal hard threshold, and Wax–Kailath's two information criteria |

## 1. Counting components

**The test.** Plant 1, 3 or 5 components in noise, at three strengths and several record shapes
(30 records each), and ask each method how many there are.

![Share of cases each method counts exactly](fig_rank.png)

| method | counted exactly | components reported in correlated noise, nothing planted |
|---|---|---|
| **Entroptics** | **1.000** | **9.0** |
| Gavish–Donoho | 0.935 | 17.9 |
| AIC | 0.946 | 44.7 |
| MDL | 0.889 | 18.9 |

Both columns average over the cases all four methods cover: AIC and MDL do not apply to square
records, which leaves 27 planted cases and two record shapes of correlated noise. Entroptics also
counts the 9 square cases exactly. It needs no tuning and no known noise level. Gavish–Donoho misses
weak components when there are five of them (0.40–0.53 at the weakest strength).

The right-hand column is serially correlated noise with nothing planted. Correlation in time
genuinely lifts the noise above the level independent noise would reach, so the methods report
components there; Entroptics reports the fewest on average.

Scripts: [`exp17_rank_baselines.py`](../validation/exp17_rank_baselines.py) and
[`exp18_coloured_null.py`](../validation/exp18_coloured_null.py), results in sections 17 and 18 of
[`RESULTS.md`](../validation/RESULTS.md).

## 2. Cost

**The test.** Two tones in white noise, each channel with its own phases. Every method returns the
same things: how many components there are at a 5% false-alarm rate, and each one's frequency,
decay rate and power. Each time is the best of three to five calls on one thread, and the tables
show the median over three full sweeps.

![Time per call against record length, one channel and 16 channels](fig_cost.png)

K is the count, and the truth is 4 in every row:
- **The FFT pipeline** counts peaks. A tone counts as two components, its positive- and
  negative-frequency pair.
- **The streaming read** counts independent patterns across the channels. Two tones, each at its own
  phase in every channel, make four.

The fastest full read is in bold.

**16 channels.**

| T | FFT alone (a spectrum only) | FFT pipeline | FFT pipeline, 16× padded | Entroptics, streaming |
|---|---|---|---|---|
| 1024 | 0.030 ms | **0.298 ms** (K 4) | 2.15 ms (K 4) | 0.628 ms (K 4) |
| 4096 | 0.154 ms | **0.791 ms** (K 4) | 12.3 ms (K 4) | 0.890 ms (K 3) |
| 16384 | 0.961 ms | 3.50 ms (K 4) | 76.6 ms (K 4) | **1.85 ms** (K 4) |
| 65536 | 5.81 ms | 20.6 ms (K 4) | 341 ms (K 4) | **8.62 ms** (K 4) |
| 262144 | 41.7 ms | 115 ms (K 4) | 1.52 s (K 4) | **44.2 ms** (K 3) |
| 1048576 | 201 ms | 516 ms (K 6) | — | **207 ms** (K 4) |

- **Speed.** From T = 16384 the streaming read is faster than the FFT pipeline: 1.6–1.9× at
  16384, and 2.3–2.6× from 65536 in every sweep. From T = 262144 it costs about what the FFT alone
  does, and the FFT alone returns no count and no decay rates.
- **Why it scales.** It folds each frame into a fixed-size summary, so its cost grows in step with
  T. The FFT's cost grows as T log T, once per channel, and the pipeline then works through the
  spectrum.
- **Where it misses.** At T = 4096 and 262144 it counts 3, missing one of the weaker tone's two
  patterns.

**One channel.**

| T | FFT alone (a spectrum only) | FFT pipeline | FFT pipeline, 16× padded | Entroptics, fixed depth | Entroptics, automatic depth |
|---|---|---|---|---|---|
| 1024 | 0.0049 ms | **0.115 ms** (K 4) | 0.252 ms (K 4) | 0.641 ms (K 4) | 6.03 ms (K 4) |
| 4096 | 0.0124 ms | **0.155 ms** (K 4) | 0.754 ms (K 4) | 0.940 ms (K 4) | 19.6 ms (K 4) |
| 16384 | 0.0519 ms | **0.321 ms** (K 4) | 3.10 ms (K 4) | 2.26 ms (K 4) | 400 ms (K 4) |
| 65536 | 0.300 ms | **1.06 ms** (K 6) | 17.3 ms (K 6) | 11.2 ms (K 4) | 30.7 s (K 4) |
| 262144 | 1.30 ms | **4.64 ms** (K 4) | 124 ms (K 4) | 70.4 ms (K 4) | — |
| 1048576 | 8.66 ms | **26.1 ms** (K 4) | 600 ms (K 4) | 328 ms (K 4) | — |

- **The FFT pipeline wins on one channel,** 5.6–16× faster than the fixed-depth read.
- **Against the padded pipeline** (the 16×-padded one, whose accuracy section 3 reports), the
  fixed-depth read is faster from T = 16384. It also counts right at every length.
- **The automatic depth** reads a deeper window the longer a tone persists. Its cost grows much
  faster than T: from 400 ms at T = 16384 to 30.7 s at 65536, where it is timed with one call per
  sweep.

Script: [`operator_vs_fft.py`](operator_vs_fft.py) → [`operator_vs_fft.jsonl`](operator_vs_fft.jsonl)
(the `like for like` rows).

## 3. Against the FFT on tones

**The test.** Seven records built to be hard in different ways: tones that decay, a gap in the
record, a very short record, tones close together, a weak tone beside a strong one. For decay rates
and counts, the FFT pipeline runs both Hann-windowed and unwindowed, and each row credits it with
whichever reads closer to the truth. That gives the FFT the benefit of knowing the answer.

**Decay rates.** How fast each damped tone dies away; the FFT reads it from how wide its peak is.
Closest to the truth in bold.

| record | tone | true decay | Entroptics, automatic depth | Entroptics, fixed depth | FFT pipeline |
|---|---|---|---|---|---|
| two damped tones between the FFT's bins, T = 1024, σ = 0.02 | 0.0402 | 0.010 | 0.0090 | **0.0105** | 0.0092 |
| | 0.0926 | 0.030 | lost | **0.0315** | 0.0275 |
| the same, σ = 0.1 | 0.0402 | 0.010 | 0.0089 | 0.0152 | **0.0092** |
| | 0.0926 | 0.030 | lost | lost | **0.0057** |
| the same, T = 64 | 0.0402 | 0.010 | 0.1093 | **0.0098** | 0.0000 |
| | 0.0926 | 0.030 | lost | **0.0301** | 0.0362 |
| the same across a 30-sample gap | 0.0402 | 0.010 | 0.0086 | **0.0102** | 0.0097 |
| | 0.0926 | 0.030 | lost | 0.0326 | **0.0287** |

The fixed-depth read is closest on 5 of the 8 tones, and within 9% of the truth on every tone it
places except in the noisiest record. The FFT pipeline is closest on 3. One of those is the noisy
0.030 tone, which only the FFT places, at 0.0057.

**Counting tones.** Components counted on each record, a tone counting as two. Closest to the truth
in bold.

| record | true | Entroptics, automatic depth | FFT pipeline |
|---|---|---|---|
| two damped tones between the FFT's bins | 4 | **3** | 2 |
| the same, σ = 0.1 | 4 | **2** | **2** |
| the same, T = 64 | 4 | 2 | **4** |
| the same across a 30-sample gap | 4 | **3** | 2 |
| two tones 0.006 apart | 4 | **4** | **4** |
| a −50 dB tone 10 bins from a strong one | 4 | **4** | **4** |
| three tones | 6 | **6** | **6** |

**Frequencies.** Each tone's frequency error, in cycles per sample, from each method's own counted
peaks or modes. Lower is better; the best in each row is in bold. "Lost" means the method has no
estimate within half the gap to the nearest other tone.

![Frequency error of each method on each tone](fig_fft.png)

| record | tone | Entroptics, automatic depth | Entroptics, fixed depth | FFT pipeline, unwindowed | FFT pipeline, Hann | best |
|---|---|---|---|---|---|---|
| two damped tones between the FFT's bins, T = 1024, σ = 0.02 | 0.0402 | **5.0e-06** | 4.8e-05 | 1.5e-05 | 1.1e-04 | Entroptics |
| | 0.0926 | lost | **3.9e-06** | 5.3e-04 | lost | Entroptics |
| the same, σ = 0.1 | 0.0402 | **3.2e-05** | 1.9e-04 | 5.2e-04 | 6.6e-04 | Entroptics |
| | 0.0926 | lost | lost | **2.0e-03** | lost | FFT |
| the same, T = 64 | 0.0402 | 1.0e-02 | 1.8e-04 | 6.3e-04 | **1.5e-04** | FFT |
| | 0.0926 | lost | 7.5e-04 | 3.7e-04 | **8.5e-05** | FFT |
| the same across a 30-sample gap | 0.0402 | **1.4e-05** | 2.5e-05 | 7.3e-05 | 9.9e-05 | Entroptics |
| | 0.0926 | lost | **6.1e-05** | 5.6e-04 | lost | Entroptics |
| two tones 0.006 apart, T = 2048, σ = 0.05 | 0.040 | 1.5e-06 | 2.3e-03 | 3.4e-06 | **1.5e-07** | FFT |
| | 0.046 | 3.2e-07 | 9.9e-04 | 1.6e-06 | **1.5e-07** | FFT |
| a −50 dB tone 10 bins from a strong one, σ = 1e-4 | 0.1 | 1.0e-08 | 6.5e-08 | 6.9e-08 | **1.0e-09** | FFT |
| | 0.1049 (−50 dB) | **1.4e-05** | 9.1e-05 | 2.3e-04 | 6.9e-05 | Entroptics |
| three tones, T = 2048, σ = 0.05 | 0.0402 | **2.6e-07** | 9.3e-06 | 6.8e-07 | 4.8e-07 | Entroptics |
| | 0.0926 | **1.2e-08** | 3.7e-06 | 3.3e-07 | 1.3e-06 | Entroptics |
| | 0.15 | 7.0e-07 | 3.3e-05 | 6.2e-07 | **1.0e-07** | FFT |

Where each method wins:
- **Entroptics:** both tones of the clean damped record, the stronger tone of the noisier one, both
  tones across the gap, the weak tone beside the strong one, and two of the three tones on the
  three-tone record.
- **The FFT pipeline:** the shortest record, the two close tones, the strong tone of the −50 dB
  pair, the third tone, and one noisy decaying tone.

Script: [`operator_vs_fft.py`](operator_vs_fft.py) → [`operator_vs_fft.jsonl`](operator_vs_fft.jsonl).

## 4. Pure noise

**The test.** Feed Entroptics records with nothing in them and count how often it reports a
component anyway. You choose how often that is acceptable: 5% by default.

**The result.** At most 2% for every kind of noise tested except very heavy right tails (lognormal,
Pareto).

![How often a component is reported in pure noise](fig_null.png)

Over 200 records per case, taking the worst of six record shapes (256 × 8 to 20 × 1000).

| noise | a component reported | within the 5% you allowed? |
|---|---|---|
| Gaussian | 1.5% | yes |
| Gaussian, channels at different levels (e^-1 .. e^1) | 2.0% | yes |
| Gaussian, channels at different levels (e^-2 .. e^2) | 2.0% | yes |
| complex Gaussian, channels at different levels | 2.0% | yes |
| Student t (5), occasional large values | 1.5% | yes |
| exponential, lopsided | 2.0% | yes |
| χ² (1), lopsided | 1.5% | yes |
| Poisson (1), counts | 1.0% | yes |
| Poisson (5), counts | 1.0% | yes |
| Poisson (5), counts at different levels | 1.5% | yes |
| lognormal (σ = 1), very heavy right tail | 8.5% | no |
| lognormal (σ = 1.5), very heavy right tail | 36% | no |
| Pareto (3), very heavy right tail | 16% | no |

**Why it holds.** Every channel is scaled to the same size before the count, so the count asks only
whether channels move together. In noise, the largest pattern that can appear by chance follows a
known law (Tracy–Widom), and Entroptics draws its line where pure noise crosses it 5% of the time.

Very heavy right tails reach that law slowly (lognormal) or not at all (Pareto (3), whose fourth
moment is infinite). Their extreme values cross the line more often the longer the record: at 1024
samples × 64 channels, 36% for lognormal (σ = 1.5) and 16% for Pareto (3).

Script: [`screen_null.py`](screen_null.py) → [`screen_null.jsonl`](screen_null.jsonl).

## 5. Planted signals

**The test.** Plant a known signal in noise and check that Entroptics finds it and counts it
exactly, over three record shapes and two channel-level profiles.

| planted signal | found | counted exactly |
|---|---|---|
| a persistent mode across all channels | 100% | 100% |
| a narrow broadband burst | 90–100% | 90–100% |
| three modes just above the noise | 100% | 72–100% |
| two strong modes | 100% | 99–100% |
| a narrowband line in two channels | 0–15% | 0–15% |

A component has to span enough channels to stand above the noise: more than about `(1 + √(F/T))²`.
That is two channels on a long record, and nine for 64 samples across 256 channels. A line confined
to two channels near the noise is below that; read such a record channel by channel.

Script: [`screen_null.py`](screen_null.py) (the `detect` records).

## Every script

| script | output | what it measures |
|---|---|---|
| [`operator_vs_fft.py`](operator_vs_fft.py) | `operator_vs_fft.jsonl` | sections 2 and 3 |
| [`screen_null.py`](screen_null.py) | `screen_null.jsonl` | sections 4 and 5 |
| [`../validation/exp17_rank_baselines.py`](../validation/exp17_rank_baselines.py), [`exp18_coloured_null.py`](../validation/exp18_coloured_null.py) | [`RESULTS.md`](../validation/RESULTS.md) | section 1 |
| [`write_path.py`](write_path.py) | `write_path.jsonl` | the basis's round trip, drift's false alarms and detections, noise-free recovery |
| [`default_read.py`](default_read.py) | `default_read.jsonl` | constant and sparse channels, count noise, the pencil cut |
| [`operator_read.py`](operator_read.py) | printed | the research prototype of the automatic-depth read |
| [`readme_examples.py`](readme_examples.py) | `readme_examples.txt` | every README example, run as printed |
| [`figures.py`](figures.py) | `fig_*.png` | the figures and tables on this page, from the outputs above |

```bash
python research/benchmarks/operator_vs_fft.py        # any script; needs scipy for the FFT pipeline
python research/benchmarks/figures.py                # redraw the figures (needs the [figures] extra)
```

Seeded and single-threaded, with every input generated in the script. Timings move between runs on
a shared machine. The cost section therefore reports the median of three sweeps, with the
run-to-run range in the text.
