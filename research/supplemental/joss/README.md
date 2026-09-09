# PAPER_joss.md — Journal of Open Source Software submission

JOSS reviews **software**, not results. This paper describes what the library does, who would use
it and how it relates to existing tools. It does not argue for the optical framing, which is the
instrument preprint's job.

## Contents

| file | what it is |
|---|---|
| `PAPER_joss.md` | the paper, with JOSS YAML front matter |
| `paper.bib` | the bibliography (17 entries, all cited) |

Body length: **758 words**, against JOSS's 1000-word guideline.

## Before submitting

**Rename the file to `paper.md`.** JOSS's build looks for `paper.md` and `paper.bib`, conventionally
at the repository root or in a `paper/` directory. The name here is `PAPER_joss.md` only to sit
alongside the other papers in this tree without colliding.

Compile locally to check the front matter and bibliography render:

```
docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) \
  --env JOURNAL=joss openjournals/inara
```

---

# JOSS review checklist — where this repository stands

Checked against the [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html)
on 2026-09-08. **This is an audit, not a fix list that has been applied** — the gaps below are
reported and left in place.

## Meets the criteria

| item | evidence |
|---|---|
| Public repository, version control | `https://github.com/Agience/entroptics` |
| OSI-approved licence, in a `LICENSE` file | Apache-2.0 in `LICENSE.md`, declared in `pyproject.toml` and its trove classifier |
| Installation instructions | `README.md` § Install — PyPI, extras, and an editable checkout |
| Example usage | `README.md` § Quickstart, with worked examples for the batch read, projection, filter and streaming paths |
| Automated tests | 675 tests under `src/tests`, all passing; CI on `ubuntu`/`macos`/`windows` × Python 3.10–3.12, plus a numpy/torch parity job (`.github/workflows/ci.yml`) |
| Contribution guidelines | `CONTRIBUTING.md` — build, test, DCO sign-off, commit format, licensing |
| Substantial scholarly effort | 19-experiment seeded validation suite, a Lean 4 / Mathlib certification, and three preprints |
| Reproducibility of reported results | `research/validation/run_all.py` regenerates `RESULTS.md` byte-identically; each paper directory carries a `reproduce.py` and a `verify.py` |
| Archive with a DOI | Zenodo `10.5281/zenodo.21273400` |
| Author ORCID | `0009-0002-0150-4027`, in the paper front matter, `CITATION.cff` and `.zenodo.json` |
| Version consistency | `0.2.2` in `pyproject.toml`, `CITATION.cff` and `.zenodo.json`; the library reports it from installed distribution metadata rather than a literal |

## Gaps

Checked again after the 2026-09-08 pass. **All five gaps previously reported here have been
closed**; what follows records what was done, so a reviewer can see the evidence rather than a
claim.

| gap as reported | closed by |
|---|---|
| No labelled statement of need in the README | `README.md` § **Statement of need** — the problem, the alternatives it is measured against (`scikit-learn` PCA / Minka, `optht`, Wax–Kailath AIC/MDL), the audience, and an explicit pointer to PyDMD and PyKoopman as the better choice when DMD is the problem |
| No API documentation beyond docstrings | [`docs/API.md`](../../../docs/API.md) — all 78 public names with signatures and summaries, **generated from `entroptics.__all__`** by `docs/generate_api.py`, so it cannot drift from the code; output is deterministic and committed, so staleness shows as a diff |
| Community guidelines incomplete | `README.md` § **Getting help** and `CONTRIBUTING.md` § **Reporting an issue** — where to file, what a useful report contains, the security address, and the conduct link |
| No `CODE_OF_CONDUCT.md` | [`CODE_OF_CONDUCT.md`](../../../CODE_OF_CONDUCT.md) — Contributor Covenant 2.1, enforcement address `connect@agience.ai` |
| Data-dependence not stated up front | `README.md` § **What runs without a download** — the test suite and all 19 experiments need no external data; the two FRB figure routines are named as the only exception, with a pointer to the fetch |

## Remaining

Nothing from the review criteria. Two items are submission mechanics rather than gaps:

1. **Rename `PAPER_joss.md` to `paper.md`** before submitting (see *Before submitting* above).
2. **The two companion preprints have no DOIs yet.** `paper.bib` cites them as pre-prints; if
   they are posted to arXiv or Zenodo before submission, add the DOIs there and in
   `research/PAPER.md`'s reference list.

## Not applicable

Human or animal subjects; data-sharing requirements beyond the external CHIME/FRB release, which is
public, separately citable, and documented rather than vendored.
