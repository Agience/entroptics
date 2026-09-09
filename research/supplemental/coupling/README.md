# PAPER_coupling.md — the coupling statistic paper

Exact permutation-null moments for a signed bilinear coupling statistic on a shared basis, with
the real-embedding construction that makes the standardisation exact for complex frames.

## Venue

Formatted for **Computational Statistics and Data Analysis** (CSDA). CSDA is the primary target
for a specific reason: both of the works this paper positions itself against — Kazi-Aoual et al.
(1995) on the exact permutation moments of `tr(W_x W_y)`, and Josse et al. (2008) on the RV
coefficient's significance test — appeared there, so the readership is exactly the one that will
recognise what is and is not new here.

The secondary target is the **Journal of Multivariate Analysis**.

## Contents

| file | what it is |
|---|---|
| `PAPER_coupling.md` | the paper |
| `reproduce.py` | regenerates every number in it |
| `verify.py` | checks the paper's prose against the tables |
| `tables/brute_force.csv` | closed form against brute-force re-pairing (Table 1) |
| `tables/sign.csv` | planted-sign recovery (Table 2) |
| `tables/null.csv` | null calibration over 1600 pairs (§§4, 5.3) |
| `tables/embedding.csv` | the T=2, D=1 counterexample (§2.5) |

## Reproducing

```
python research/supplemental/coupling/reproduce.py     # a few minutes
python research/supplemental/coupling/verify.py        # checks the prose against the tables
```

The brute-force permutation check of §5.1 is the paper's main empirical claim, and it is one
command. The full run draws 200,000 re-pairings at each of five shapes.

```
python research/supplemental/coupling/reproduce.py --quick    # 20,000 draws, smoke test only
```

`--quick` is not what the paper reports; after it, `verify.py` reports the 100k and 200k checks
as skipped rather than passed.

No external data is needed — every frame is generated from a fixed seed. The tables are
byte-reproducible across runs.

## How the statistic is computed

Through the public API of the `entroptics` library, which is where this construction is
implemented:

```python
from entroptics import Screen

s = Screen(far=0.05)
s.register("a", entry=lambda x: x)
s.register("b", entry=lambda x: x)
s.place("a", A)
s.place("b", B)
c = s.coupling("a", "b")     # c.z, c.sign, c.strength, c.phase, c.resolved
```

`Screen.coupling` forms the real embedding and evaluates Definition 2.3. Nothing in the
reproduction reaches past the public API.

The one thing computed directly in `reproduce.py` is the brute-force permutation sample of §5.1,
and that is deliberate: it is the reference the library's closed form is being checked against,
so drawing it from the library would check nothing.

## Relation to the instrument paper

`research/PAPER.md` is the full reference for the construction this statistic is one read of. It
carries the two-way screen the coupling is defined on, the Lean 4 / Mathlib certification of the
combinatorial core of Theorem 2.2, and the rest of the library's reads. This paper is
self-contained and makes no use of that setting beyond citing it.
