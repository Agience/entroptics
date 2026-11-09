# joss/ — Journal of Open Source Software submission

JOSS reviews software, not results. This paper describes what the library does, who would use it
and how it relates to existing tools. It does not argue for the optical framing, which is the
instrument paper's job (`research/PAPER.md`).

| file | what it is |
|---|---|
| `paper.md` | the paper, with JOSS YAML front matter; the body is under JOSS's 1000-word guideline |
| `paper.bib` | the bibliography |

JOSS's build reads `paper.md` and `paper.bib` from this directory. To check that the front matter
and bibliography render:

```
docker run --rm --volume $PWD:/data --user $(id -u):$(id -g) \
  --env JOURNAL=joss openjournals/inara
```

## Against the review criteria

Checked against the [JOSS review criteria](https://joss.readthedocs.io/en/latest/review_criteria.html).

| item | evidence |
|---|---|
| Public repository, version control | `https://github.com/Agience/entroptics` |
| OSI-approved licence, in a `LICENSE` file | Apache-2.0 in `LICENSE.md`, declared in `pyproject.toml` as the SPDX expression `Apache-2.0` |
| Installation instructions | `README.md` § Install — PyPI, extras, and an editable checkout |
| Example usage | `README.md` § Examples; [`docs/GUIDE.md`](../docs/GUIDE.md) for every read |
| API documentation | [`docs/API.md`](../docs/API.md), generated from `entroptics.__all__` by `docs/generate_api.py` |
| Statement of need | `README.md` § Statement of need, with the alternatives it is measured against and a pointer to PyDMD and PyKoopman where DMD is the problem |
| Automated tests | `src/tests`, passing with the `[dev]` extra; CI on ubuntu, macOS and Windows × Python 3.10–3.12, plus a numpy/torch parity job (`.github/workflows/ci.yml`) |
| Community guidelines | `CONTRIBUTING.md` (build, test, DCO sign-off, reporting an issue), `README.md` § Getting help, and [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) |
| Substantial scholarly effort | a seeded validation suite, a Lean 4 / Mathlib certification, and companion papers |
| Reproducibility | `research/validation/run_all.py` regenerates `RESULTS.md` byte-identically. The tests, the experiments and every benchmark need no external data; the real-data figure is regenerated in [entroptics-frb](https://github.com/Agience/entroptics-frb), which documents the CHIME/FRB fetch |
| Archive with a DOI | Zenodo concept DOI `10.5281/zenodo.21273400`; each release gets its own version DOI |
| Author ORCID | `0009-0002-0150-4027`, in the paper front matter, `CITATION.cff` and `.zenodo.json` |
| Version consistency | one version in `pyproject.toml`, `CITATION.cff` and `.zenodo.json`; the library reports it from the installed distribution metadata |

The two companion pre-prints in `paper.bib` carry no DOI; when either is archived, its DOI goes in
`paper.bib` and in `research/PAPER.md`'s reference list.

Not applicable: human or animal subjects. The CHIME/FRB release is public and separately citable,
and is documented rather than vendored.
