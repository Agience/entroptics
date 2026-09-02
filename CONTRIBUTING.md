# Contributing to Entroptics

Thank you for considering a contribution. Entroptics is a research instrument as much as a library: its value is that every read is intrinsic, every constant has a provenance, and every governing claim is a named theorem — several of them machine-checked. Contributions are judged against that contract first and code style second.

---

## The Contract

Six properties define the library. A contribution that violates one will not be merged, however useful it looks:

1. **Parameter-free, verifiably.** Every fixed number in the library is a *derived* mathematical quantity (a χ² median, a Tracy–Widom quantile, an influence-function variance) or the canonical realization of a *stated requirement*. The only decision input external to the instrument is the reader's false-alarm level `α` — which by Neyman–Pearson cannot come from the data. In the code that level is spelled **`far`** (`Aperture(far=)`, `Projection(far=)`, `ctx.far`; default `0.05`), and it travels with the null provider rather than beside it. If your change introduces a constant, its provenance (derived / forced / declared) must be documented alongside the existing constants table (paper §13.1). **Tuned thresholds and per-substrate calibration are categorically out of scope.**
2. **One code path, backend-agnostic.** Every kernel is written once against the array namespace of its input: numpy runs on CPU, torch stays on its device. Torch is imported lazily, only when a tensor appears. No backend-conditional math.
3. **Deterministic, and reproducible for a stated reason.** The default read path is closed-form — no draws anywhere in it: the default null is the analytic `mp` edge, the coherence null is the closed-form permutation *z*-score, and `test_determinism.py` pins the coherence and screen reads to signatures that accept no `rng` at all. Where the library does sample it is opt-in and always seeded — the randomized range-finder behind `Dynamics.resolved(k=...)` (exact `eigvalsh` when `k is None`), the batched power-iteration start vector in `batch._spectral_norm`, `sequence.surrogate_test`, and any caller-supplied resampling provider (`permutation()`, `floor_from_null_sampler`). So the guarantee is **deterministic per `seed`**, not absence of RNG; a new read may sample only if it is seeded and its result is reproducible from that seed. Reproducibility also rests on the **BLAS thread pin** (`OPENBLAS_NUM_THREADS`, set at package scope in `__init__.py` and repeated in `conftest.py` and `run_all.py`): OpenBLAS varies its reduction order with pool size, so seeding alone does not get to a bit-identical result. Do not move, remove, or "strengthen" that pin — `test_blas_thread_pin.py` measures it by its effect, with a negative control. numpy and torch evaluations must agree to floating-point round-off; repeated evaluations must be bit-identical.
4. **Reads synthesize nothing.** `Aperture.extract()` is a projection of the measured data onto its own resolved modes — idempotent, exact in the noise-free limit (`test_extract.py`). Any proposed read must be a function of `W` alone, plus declared decision inputs only (`far` / the null provider / `forgetting`, and for `extract` its `reject_persistent` and `shrink` switches), and must not generate content the field does not contain.
5. **numpy-only at the core.** The one hard dependency is `numpy>=2.0` (for `numpy.linalg.svdvals`), on Python `>=3.10`; scipy and torch are optional extras. New hard dependencies require prior discussion in an issue, and the bar is high.
6. **One public path per reading.** A measurement is reached through the front door that owns it — never through both a front door and a free function of the same name. `entroptics.<name>` is *always* the module (`dynamics`, `extract` and `sweep` each once resolved to a function and were removed for this reason); the frame-level functions in `entroptics.reads` / `entroptics.entropy` are the implementation, not a second API. `test_public_surface.py` enforces this, so a convenience alias will fail the suite.

## Claims Need Theorems

Every read maps to a standard optical quantity and cites the theorem that governs it — the dictionary is paper §12, and §15 records which of those governing facts are certified. A new read must arrive with: the definition, the governing result it specializes (named, cited), its bounds or invariances, and a validation experiment on planted ground truth. If the read carries a new governing lemma, the gold standard is a Lean 4 / Mathlib proof in `research/lean/` (`lake build`, pinned to `leanprover/lean4:v4.31.0` and Mathlib `v4.31.0`; the development compiles with no `sorry`); where a proof is deferred, say so explicitly in the PR and paper text — certified and cited-but-not-reproved statements are kept clearly distinct.

## Tests Are the Regression Contract

```bash
pip install -e ".[dev]"     # numpy + scipy + threadpoolctl + torch — runs everything
pip install -e ".[test]"    # torch-free; the parity tests skip themselves
pytest
```

Both extras carry `threadpoolctl`: the BLAS pin is measured as an effect (pool size), not as a string, and a missing `threadpoolctl` is a failure rather than a skip, so nothing can go green without having been measured.

The suite pins the full optics read as a **golden contract** — the exact field values of `Aperture.optics()` on fixed seeds (7 and 3), in `src/tests/golden/optics.json`. If your change legitimately moves a golden value, the PR must explain *why* the mathematics changed, not just update the number; regenerate with `python src/tests/golden/_generate.py` only once that reasoning exists. Additions must cover: the mathematical invariants they claim (bounds, invariances, exactness), numpy↔torch parity, determinism, and degenerate-input robustness (empty, constant, single-row/column, all-masked — an entirely unmeasured frame must not read as an observation of zero). Validation experiments live in `research/validation/` and are regenerated deterministically by `run_all.py`, which rewrites the committed `RESULTS.md`; new reads add an experiment that recovers a planted truth.

---

## How to Contribute

**Bugs:** open an issue with a minimal reproducing array (seeded), expected vs. actual values, and backend/version info. **Numerical discrepancies between backends are bugs** — report them even when tiny.

**New reads / features:** open an issue first with the definition and the governing theorem. Implementation without an acknowledged issue is likely wasted work.

**Research applications** (the worked example is the CHIME/FRB read in `research/figures/frb_panel.py`): welcome — the pattern is that the library supplies objectives (contrast, coherence, decay rates) and the application applies known-physics transforms out of band, each free parameter fixed by maximizing an entroptics read rather than by fitting. Domain adapters that would bake domain assumptions *into the read path* belong in your own package on top of Entroptics, not here (see also the patent notice in [PATENTS.md](PATENTS.md)).

**Code:** fork, branch, sign off every commit (`git commit -s`, certifying the [Developer Certificate of Origin](https://developercertificate.org/)), open a PR. Commit format: `fix:` / `feat:` / `docs:` / `test:` / `chore:` with a short body when intent isn't obvious from the diff.

**Security or responsible-disclosure concerns:** email **connect@agience.ai**; keep it out of the public issue tracker.

## Licensing of Contributions

Entroptics is Apache 2.0 and only Apache 2.0 — [`LICENSE.md`](LICENSE.md) is the verbatim upstream text with no carve-out, so its Section 3 patent grant applies in full to what is implemented here. [PATENTS.md](PATENTS.md) is the patent notice, and [PLEDGE.md](PLEDGE.md) the promise not to assert Ikailo Inc.'s patents against non-commercial and research use; inventions not implemented here are a patent posture and do not make this repository dual-licensed. By submitting a contribution you agree it is licensed under Apache 2.0 (per §5 of the license), including the Section 3 patent grant for your contribution. For a substantial contribution, Ikailo Inc. may additionally ask you to sign a contributor licence agreement.

## Pull Request Checklist

- [ ] No new constants without documented provenance; no tuned thresholds
- [ ] One code path; numpy↔torch parity test included or extended
- [ ] One public path per reading — no free function shadowing a front door
- [ ] Deterministic per `seed`; the BLAS pin untouched; golden-contract changes justified mathematically in the PR
- [ ] New reads cite their governing theorem and include a planted-truth validation
- [ ] Degenerate inputs handled and tested
- [ ] Commits signed off; conventional commit format

## Code of Conduct

Be respectful. Contributions, issues, and discussions must remain professional and constructive. This is a research instrument; disagreements are settled by mathematics and measurement, not volume.

---

*Entroptics is the lens of the Agience system — see [Mantle](https://github.com/Agience/agience-mantle) (the encrypted memory) and [Agience](https://github.com/Agience) (the platform). Ikailo Inc. Questions: connect@agience.ai*
