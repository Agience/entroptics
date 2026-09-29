# Benchmarks

Every number the README, the CHANGELOG and the guide quote comes from a script in this folder, and
each script's output is committed beside it, stamped with the SHA-256 of the library source that
produced it. Seeded and single-threaded, with every input generated in the script.

| script | output | what it measures |
|---|---|---|
| `screen_null.py` | `screen_null.jsonl` (and `screen_null_0.2.3.jsonl`, the same run on 0.2.3) | false alarms on pure noise across noise families and shapes; detection and exact rank counts on planted signals |
| `operator_vs_fft.py` | `operator_vs_fft.jsonl` | the operator reads against the FFT, rectangular and Hann; timings |
| `write_path.py` | `write_path.jsonl` | the basis's round trip, drift's false alarms and detections, noise-free recovery |
| `default_read.py` | `default_read.jsonl` (and `default_read_0.2.3.jsonl`) | constant and sparse channels, count noise, the pencil cut |
| `readme_examples.py` | `readme_examples.txt` | every README example, run as printed |

```bash
python research/benchmarks/screen_null.py        # or any script above; a path to another src/ reads that copy
```

## The screen's null

`K` counts the modes that stand above the edge that independent channels would reach.
- Every channel is centred on its mean and scaled by its RMS. The screen's Gram is then a sample
  correlation matrix.
- The floor is the Tracy–Widom edge of that null, whose law is proved for i.i.d. entries with a
  finite fourth moment (Bao, Pan & Zhou 2012). Its per-cell variance is the screen's mean cell
  energy. On an unfolded, fully measured screen that is the null's own, `N / (N − 1)`, exactly. On a
  folded screen, channels that share structure raise it, which only raises the floor.
- A channel that never moved (every value the same) leaves the screen the way a dead one does. A
  filter keeps it at its value.

Measured by [`research/benchmarks/screen_null.py`](screen_null.py) at
`far = 0.05`, over 200 noise-only records per case, at shapes from 256 × 8 to 20 × 1000:
- **The level holds:** false alarms of 0 to 0.020 for Gaussian noise at equal and unequal channel
  levels (real and complex), Student t, exponential, χ² and Poisson noise. That is below the level
  with room: at these sizes the correlation null's top eigenvalue sits below the Tracy–Widom law of
  a covariance.
- **Heavy right tails exceed it,** most at the longest record (1024 × 64): lognormal (σ = 1) 0.085,
  lognormal (σ = 1.5) 0.36, Pareto (3) 0.16. A finite record of such noise sits far from the edge
  law.
- **The same cases on 0.2.3** ([`screen_null_0.2.3.jsonl`](screen_null_0.2.3.jsonl)):
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

## Against the FFT

Measured by [`research/benchmarks/operator_vs_fft.py`](operator_vs_fft.py) (output:
[`operator_vs_fft.jsonl`](operator_vs_fft.jsonl)), one seeded record per case.
Each cell is the frequency error of each tone, in cycles per sample, with the decay rate read in
brackets.
- **Read A** is `operator_read`.
- **Lift** is the stable `koopman_lift` at one delay depth, `d = 16`, the same for every case.
- **FFT** is a 16× zero-padded periodogram with log-parabolic interpolation of its strongest peaks,
  rectangular and Hann windowed, with a gap filled with the mean.
- **"Lost"** means the tone is not among the strongest peaks or modes, so the nearest one found is
  another tone.

| record | tone (true decay) | read A | lift, d = 16 | FFT | FFT, Hann |
|---|---|---|---|---|---|
| two damped tones between bins, T = 1024, σ = 0.02 | 0.0402 (0.010) | **5.0e-6** (0.0090) | 4.8e-5 (0.0105) | 1.5e-5 | 1.1e-4 |
| | 0.0926 (0.030) | lost | **3.9e-6** (0.0315) | 5.3e-4 | lost |
| the same, σ = 0.1 | 0.0402 (0.010) | **3.2e-5** (0.0089) | 1.9e-4 (0.0152) | 5.2e-4 | 6.6e-4 |
| | 0.0926 (0.030) | lost | lost | lost | **4.0e-3** |
| the same across a 30-sample gap | 0.0402 (0.010) | **1.4e-5** (0.0086) | 2.5e-5 (0.0102) | 7.3e-5 | 9.8e-5 |
| | 0.0926 (0.030) | lost | **6.1e-5** (0.0326) | lost | lost |
| the same, T = 64 | 0.0402 (0.010) | lost | 1.8e-4 (0.0098) | 6.3e-4 | **1.5e-4** |
| | 0.0926 (0.030) | lost | 7.5e-4 (0.0301) | 3.7e-4 | **1.3e-4** |
| two tones 0.006 apart, T = 2048, σ = 0.05 | 0.040 | 1.5e-6 | 2.3e-3 | 3.4e-6 | **1.5e-7** |
| | 0.046 | 3.2e-7 | 9.9e-4 | 1.6e-6 | **1.5e-7** |
| a −50 dB tone 10 bins from a strong one, σ = 1e-4 | 0.1 | 1.0e-8 | 6.5e-8 | 6.9e-8 | **1.0e-9** |
| | 0.1049 (−50 dB) | **1.4e-5** | 9.1e-5 | lost | lost |
| three tones, T = 2048, σ = 0.05 | 0.0402 | **2.6e-7** | 9.3e-6 | 6.8e-7 | 4.8e-7 |
| | 0.0926 | **1.2e-8** | 3.7e-6 | 3.3e-7 | 1.3e-6 |
| | 0.15 | 7.0e-7 | 3.3e-5 | 6.2e-7 | **1.0e-7** |

What the table shows:

- **The FFT reads no decay, and each Entroptics read does.** Read A recovers AR(1) decay rates of
  0.104 and 0.675 against true values of 0.105 and 0.693, and the lift reads 0.0315 for a tone
  decaying at 0.030.
- **A gap costs the FFT a tone that the lift keeps.** With either window, the damped tone across a
  30-sample gap is lost; the lift reads it at 6.1e-5.
- **A weak line beside a strong one.**
  - Read A finds the −50 dB tone 10 bins away and places it to 1.4e-5.
  - The rectangular FFT buries it in leakage. With a Hann window it is visible, at −50 dB, but the
    peak-picking step selects a sidelobe of the strong tone instead.
  - Midway between the two tones, the operator's view reads −99 dB below the strong tone, the Hann
    periodogram −82 dB and the rectangular one −51 dB.
- **On long, clean, persistent tones the two trade places.** The Hann-windowed FFT reads the close
  pair to 1.5e-7 and the strong tone to 1.0e-9; read A is ahead on two of the three separated
  tones (2.6e-7 and 1.2e-8 against 4.8e-7 and 1.3e-6).
- **At σ = 0.1 the weaker damped tone is found only by the Hann FFT,** at 4.0e-3.
- **Read A has three weak regimes, measured here:**
  - An odd count on a damped pair, K = 3, loses the more strongly damped tone, on the clean record
    and across the gap.
  - A 64-sample record is too short for its depth read.
  - Its cost grows with depth (see below).
  
  The lift at a fixed depth covers the first two.
- **On white noise**, read A claims an object in 3.5% of records at `far = 0.05` (200 records,
  T = 2048, F = 16).

## Cost

Best of several calls, one thread, from one run of the same benchmark, on a shared machine (timings
move between runs):

| T | `Aperture(W).dynamics().modes()`, F = 16 | `koopman_lift(x, 16).modes()` | `operator_read(x)` (depth) | `np.fft.rfft`, F = 16 |
|---|---|---|---|---|
| 1024 | 0.62 ms | 0.68 ms | 6.2 ms (64) | 0.05 ms |
| 4096 | 0.98 ms | 1.1 ms | 21 ms (256) | 0.21 ms |
| 16384 | 2.4 ms | 4.0 ms | 611 ms (1024) | 1.2 ms |

- **The streaming operator** costs O(F²) per frame and O(F³) per read, independent of the stream's
  length (paper §9).
- **The full optics read**, `Aperture(W).optics()`, costs O(T² F): its decay is a direct lag sum
  over the T×T Gram matrix. Measured at F = 16: 12 ms at T = 1024, 272 ms at 4096, and 5.2 s at
  16384.
- **The FFT** costs O(T log T).
- **Read A** costs O(T d + d³), and `d` follows the objects' coherence, so a persistent object at
  large T is its expensive case.
- **The operator** (`dynamics()`, the lift) is the cheap path at large T; reach for `optics()`
  when you want the aperture reads themselves.
