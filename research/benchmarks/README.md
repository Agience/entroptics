# Benchmarks

Three questions, each measured by a seeded script whose output is committed beside it and stamped
with the SHA-256 of the library source that produced it:

1. **[Does it see structure in pure noise?](#1-does-it-see-structure-in-pure-noise)** It should
   report structure no more often than the false-alarm rate you ask for, whatever the noise looks
   like.
2. **[Does it find a planted signal and count it right?](#2-does-it-find-a-planted-signal-and-count-it-right)**
3. **[How does it compare with the FFT?](#3-how-does-it-compare-with-the-fft)** On placing tones,
   and on cost.

The figures and the tables below are drawn from the committed outputs by
[`figures.py`](figures.py), so they change exactly when a benchmark is re-run.

## 1. Does it see structure in pure noise?

**Short answer:** since 0.2.5, at a requested rate of 5%, it claims structure in at most 2% of
pure-noise records for every noise family below except heavy right tails; on 0.2.3 it claimed
structure in most of them.

![False-alarm rate on pure noise, 0.2.3 against 0.2.6](fig_null.png)

The rate at which each version reports at least one component in noise that has none, at
`far = 0.05`, over 200 records per case. Each figure is the worst over six shapes, from 256 × 8 to
20 × 1000. Lower is better; the better version is in bold.

| noise | 0.2.3 | 0.2.6 | holds the 5% level on 0.2.6 |
|---|---|---|---|
| Gaussian | 0.040 | **0.015** | yes |
| Gaussian, channel levels e^-1 .. e^1 | 0.735 | **0.020** | yes |
| Gaussian, channel levels e^-2 .. e^2 | 0.500 | **0.020** | yes |
| complex Gaussian, channel levels e^-1 .. e^1 | 0.730 | **0.020** | yes |
| Student t (5) | 0.960 | **0.015** | yes |
| exponential | 1.000 | **0.020** | yes |
| χ² (1) | 1.000 | **0.015** | yes |
| Poisson (1) | 0.525 | **0.010** | yes |
| Poisson (5) | 1.000 | **0.010** | yes |
| Poisson (5), channel levels 1 .. 20 | 1.000 | **0.015** | yes |
| lognormal (σ = 1) | 1.000 | **0.085** | no |
| lognormal (σ = 1.5) | 1.000 | **0.360** | no |
| Pareto (3) | 1.000 | **0.160** | no |

**Why it holds since 0.2.5:** every channel is centred on its mean and scaled by its RMS, so the read is
against a sample correlation matrix, whose largest eigenvalue follows the Tracy–Widom law for noise
of any distribution with a finite fourth moment (Bao, Pan & Zhou 2012). The floor is that law's
edge at the requested rate. The heavy-tailed families are where a record of this length is still
far from that law; they exceed the level most on the longest record (1024 × 64).

Script: [`screen_null.py`](screen_null.py) → [`screen_null.jsonl`](screen_null.jsonl); the same
script run on 0.2.3 → [`screen_null_0.2.3.jsonl`](screen_null_0.2.3.jsonl).

## 2. Does it find a planted signal and count it right?

**Short answer:** it finds a component that spans the channels in 90–100% of records, and counts
exactly how many are there in 72–100% (0.2.3: 21–100%). A component confined to one or two channels
is the exception, by design (below).

The share of records in which the read returns exactly the planted number of components, over
three shapes and two channel-level profiles. The better version, on average, is named.

| planted signal | counted exactly, 0.2.3 | counted exactly, 0.2.6 | better on average |
|---|---|---|---|
| a persistent mode across all channels | 0.90–1.00 (mean 0.98) | 1.00–1.00 (mean 1.00) | 0.2.6 |
| a narrow broadband burst | 0.21–1.00 (mean 0.82) | 0.90–1.00 (mean 0.97) | 0.2.6 |
| three modes just above the noise edge | 0.83–1.00 (mean 0.96) | 0.72–1.00 (mean 0.95) | 0.2.3 |
| two strong modes | 1.00–1.00 (mean 1.00) | 0.99–1.00 (mean 1.00) | same |
| a narrowband line in two channels | 0.01–0.96 (mean 0.35) | 0.00–0.15 (mean 0.05) | 0.2.3 |

**The narrowband line is the trade.** Every channel now carries the same energy, so a component
confined to `k` channels can stand above the noise only when `k` exceeds about `(1 + √(F/T))²`:
two channels on a long record, nine for 64 samples across 256 channels. A two-channel line near the
edge is below that. This is what buys the calibration in section 1; a record whose structure lives
in one or two channels is better read channel by channel.

Script: [`screen_null.py`](screen_null.py) (the `detect` records in the same outputs).

## 3. How does it compare with the FFT?

**Short answer:** of the 15 tones below, Entroptics places 8 best (`operator_read` 6,
`koopman_lift` 2) and the Hann-windowed FFT 7. Entroptics wins where a tone decays, crosses a gap
in the record, or sits beside a much stronger one; the Hann-windowed FFT wins on long, clean,
persistent tones and on the shortest record. Only Entroptics reads decay rates at all. The FFT is
much faster.

![Frequency error of each method on each tone](fig_fft.png)

Each cell is the frequency error of the tone, in cycles per sample; lower is better, and the best
in each row is in bold. "Lost" means the method's nearest estimate lies closer to a different tone
than to this one.

| record | tone | `operator_read` | `koopman_lift` | FFT | FFT (Hann) | best |
|---|---|---|---|---|---|---|
| two damped tones between the FFT's bins, T = 1024, σ = 0.02 | 0.0402 | **5.0e-06** | 4.8e-05 | 1.5e-05 | 1.1e-04 | Entroptics |
| | 0.0926 | lost | **3.9e-06** | 5.3e-04 | lost | Entroptics |
| the same, σ = 0.1 | 0.0402 | **3.2e-05** | 1.9e-04 | 5.2e-04 | 6.6e-04 | Entroptics |
| | 0.0926 | lost | lost | lost | **4.0e-03** | FFT (Hann) |
| the same, T = 64 | 0.0402 | 1.0e-02 | 1.8e-04 | 6.3e-04 | **1.5e-04** | FFT (Hann) |
| | 0.0926 | lost | 7.5e-04 | 3.7e-04 | **1.3e-04** | FFT (Hann) |
| the same across a 30-sample gap | 0.0402 | **1.4e-05** | 2.5e-05 | 7.3e-05 | 9.8e-05 | Entroptics |
| | 0.0926 | lost | **6.1e-05** | lost | lost | Entroptics |
| two tones 0.006 apart, T = 2048, σ = 0.05 | 0.040 | 1.5e-06 | 2.3e-03 | 3.4e-06 | **1.5e-07** | FFT (Hann) |
| | 0.046 | 3.2e-07 | 9.9e-04 | 1.6e-06 | **1.5e-07** | FFT (Hann) |
| a −50 dB tone 10 bins from a strong one, σ = 1e-4 | 0.1 | 1.0e-08 | 6.5e-08 | 6.9e-08 | **1.0e-09** | FFT (Hann) |
| | 0.1049 (−50 dB) | **1.4e-05** | 9.1e-05 | lost | lost | Entroptics |
| three tones, T = 2048, σ = 0.05 | 0.0402 | **2.6e-07** | 9.3e-06 | 6.8e-07 | 4.8e-07 | Entroptics |
| | 0.0926 | **1.2e-08** | 3.7e-06 | 3.3e-07 | 1.3e-06 | Entroptics |
| | 0.15 | 7.0e-07 | 3.3e-05 | 6.2e-07 | **1.0e-07** | FFT (Hann) |

The methods:
- **`operator_read`** (`entroptics.experimental`) reads the number of modes and the delay depth from
  the record itself.
- **`koopman_lift`** (stable) reads the modes at one delay depth you choose, here 16 for every case.
- **FFT** is a 16× zero-padded periodogram with log-parabolic interpolation of its strongest peaks,
  rectangular or Hann windowed, with a gap filled with the mean.

What the table does not show:
- **Decay rates.** Both Entroptics reads report each tone's decay (0.0090 and 0.0105 for a true
  0.010, 0.0315 for a true 0.030); the FFT reports none. On AR(1) noise `operator_read` recovers
  decays of 0.104 and 0.675 against true 0.105 and 0.693.
- **The −50 dB tone, seen as a spectrum.** Midway between the two tones, the Fourier view drawn from
  the operator is 99 dB below the strong tone, the Hann periodogram 82 dB and the rectangular one
  51 dB: the rectangular FFT's leakage buries the weak tone, and the Hann FFT shows it but picks a
  sidelobe of the strong one.
- **Noise.** On white noise, `operator_read` claims a tone in 3.5% of records at `far = 0.05`
  (200 records, T = 2048, F = 16).

`operator_read`'s weak cases, all in the table: an odd mode count on a damped pair loses the more
damped tone, and a 64-sample record is too short for its depth. `koopman_lift` at a fixed depth
covers both.

### Cost

![Time per call against record length](fig_cost.png)

Best of several calls, one thread, on a shared machine (timings move between runs). The FFT is the
fastest at every length.

| T | `np.fft.rfft`, F = 16 | `Aperture(W).dynamics().modes()`, F = 16 | `koopman_lift(x, 16).modes()` | `operator_read(x)` | `Aperture(W).optics()`, F = 16 | fastest |
|---|---|---|---|---|---|---|
| 1024 | **0.046 ms** | 0.63 ms | 0.61 ms | 6.1 ms | 11 ms | FFT |
| 4096 | **0.32 ms** | 1.4 ms | 0.98 ms | 22 ms | 258 ms | FFT |
| 16384 | **1.4 ms** | 2.5 ms | 3.2 ms | 552 ms | 4.8 s | FFT |

- **The streaming operator** (`dynamics()`) costs O(F²) per frame and O(F³) per read, independent
  of the stream's length; it is 14× the FFT's time at T = 1024 and 1.7× at 16384.
- **`operator_read`** costs O(T d + d³), and its depth `d` follows how long the tones persist, so a
  long persistent record is its expensive case.
- **`optics()`**, the full aperture read, costs O(T² F) for its direct-sum decay; use the operator
  when you only need the modes.

Script: [`operator_vs_fft.py`](operator_vs_fft.py) → [`operator_vs_fft.jsonl`](operator_vs_fft.jsonl).

## Every script

| script | output | what it measures |
|---|---|---|
| [`screen_null.py`](screen_null.py) | `screen_null.jsonl`, `screen_null_0.2.3.jsonl` | sections 1 and 2 |
| [`operator_vs_fft.py`](operator_vs_fft.py) | `operator_vs_fft.jsonl` | section 3 |
| [`write_path.py`](write_path.py) | `write_path.jsonl` | the basis's round trip, drift's false alarms and detections, noise-free recovery |
| [`default_read.py`](default_read.py) | `default_read.jsonl`, `default_read_0.2.3.jsonl` | constant and sparse channels, count noise, the pencil cut |
| [`readme_examples.py`](readme_examples.py) | `readme_examples.txt` | every README example, run as printed |
| [`figures.py`](figures.py) | `fig_*.png` | the figures and tables on this page, from the outputs above |

```bash
python research/benchmarks/screen_null.py            # any script; a path to another src/ reads that copy
python research/benchmarks/figures.py                # redraw the figures (needs the [figures] extra)
```

Seeded and single-threaded, with every input generated in the script.
