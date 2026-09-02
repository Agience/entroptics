# Contributing to Entroptics

Thank you for considering a contribution. Entroptics is a research instrument as much as a library: its value is that every read is intrinsic, every constant has a provenance, and every governing claim is a named theorem — several of them machine-checked. Contributions are judged against that contract first and code style second.

---

## The Contract

Five properties define the library. A contribution that violates one will not be merged, however useful it looks:

1. **Parameter-free, verifiably.** Every fixed number in the library is a *derived* mathematical quantity (a χ² median, a Tracy–Widom quantile, an influence-function variance) or the canonical realization of a *stated requirement*. The only decision input external to the instrument is the reader's false-alarm level `α` — which by Neyman–Pearson cannot come from the data. If your change introduces a constant, its provenance (derived / forced / declared) must be documented alongside the existing constants table (paper §11.1). **Tuned thresholds and per-substrate calibration are categorically out of scope.**
2. **One code path, backend-agnostic.** Every kernel is written once against the array namespace of its input: numpy runs on CPU, torch stays on its device. Torch is imported lazily, only when a tensor appears. No backend-conditional math.
3. **Deterministic and bit-identical.** No sampling, no RNG in any read path (the coherence null is closed-form for exactly this reason). numpy and torch evaluations must agree to floating-point round-off; repeated evaluations must be bit-identical.
4. **Reads synthesize nothing.** `extract` is a projection of the measured data onto its own resolved modes — idempotent, exact in the noise-free limit. Any proposed read must be a function of `W` alone (plus the declared `α` / null provider / forgetting inputs) and must not generate content the field does not contain.
5. **numpy-only at the core.** scipy and torch are optional extras. New hard dependencies require prior discussion in an issue, and the bar is high.

## Claims Need Theorems

Every read maps to a standard optical quantity and cites the theorem that governs it (paper §10). A new read must arrive with: the definition, the governing result it specializes (named, cited), its bounds or invariances, and a validation experiment on planted ground truth. If the read carries a new governing lemma, the gold standard is a Lean 4 / Mathlib proof in `research/lean/` (the development compiles with no `sorry`); where a proof is deferred, say so explicitly in the PR and paper text — certified and cited-but-not-reproved statements are kept clearly distinct.

## Tests Are the Regression Contract

```bash
pip install -e ".[dev]"
pytest
```

The suite pins the full optics read as a **golden contract** — the exact field values of `optics()` on fixed seeds. If your change legitimately moves a golden value, the PR must explain *why* the mathematics changed, not just update the number. Additions must cover: the mathematical invariants they claim (bounds, invariances, exactness), numpy↔torch parity, determinism, and degenerate-input robustness (empty, constant, single-row/column, all-masked). Validation experiments live in `research/validation/` and are regenerated deterministically by `run_all.py`; new reads add an experiment that recovers a planted truth.

---

## How to Contribute

**Bugs:** open an issue with a minimal reproducing array (seeded), expected vs. actual values, and backend/version info. **Numerical discrepancies between backends are bugs** — report them even when tiny.

**New reads / features:** open an issue first with the definition and the governing theorem. Implementation without an acknowledged issue is likely wasted work.

**Research applications** (like `research/applications/cosmology.py`): welcome — the pattern is that the library supplies objectives (contrast, coherence, decay rates) and the application applies known-physics transforms out of band. Domain adapters that would bake domain assumptions *into the read path* belong in your own package on top of Entroptics, not here (see also the reserved-inventions note in [PATENTS.md](PATENTS.md)).

**Code:** fork, branch, sign off every commit (`git commit -s`), open a PR. Commit format: `fix:` / `feat:` / `docs:` / `test:` / `chore:` with a short body when intent isn't obvious from the diff.

**Security or responsible-disclosure concerns:** email **connect@ikailo.com** rather than opening a public issue.

## Licensing of Contributions

The source is Apache 2.0 (copyright); patent posture is described in [PATENTS.md](PATENTS.md) and [PLEDGE.md](PLEDGE.md). By submitting a contribution you agree it is licensed under Apache 2.0 (per §5 of the license). For substantial contributions, Ikailo Inc. may additionally ask you to sign the [Ikailo CLA](https://github.com/Agience/agience-core/blob/main/CLA.md) so the project's dual-licensing posture stays intact — the bot will tell you if it applies.

## Pull Request Checklist

- [ ] No new constants without documented provenance; no tuned thresholds
- [ ] One code path; numpy↔torch parity test included or extended
- [ ] Deterministic; golden-contract changes justified mathematically in the PR
- [ ] New reads cite their governing theorem and include a planted-truth validation
- [ ] Degenerate inputs handled and tested
- [ ] Commits signed off; conventional commit format

## Code of Conduct

Be respectful. Contributions, issues, and discussions must remain professional and constructive. This is a research instrument; disagreements are settled by mathematics and measurement, not volume.

---

*Entroptics is the lens of the Agience system — see [Mantle](https://github.com/Agience/agience-mantle) (the encrypted memory) and [Agience](https://github.com/Agience/agience-core) (the platform). Ikailo Inc., Canada. Questions: connect@ikailo.com*
