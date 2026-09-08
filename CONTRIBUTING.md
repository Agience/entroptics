# Contributing to Entroptics

## Build and test

```bash
pip install -e ".[dev]"     # numpy + scipy + threadpoolctl + torch
pip install -e ".[test]"    # torch-free; the parity tests skip themselves
pytest
```

The library is parameter-free, deterministic and numpy-only at the core. Every constant needs a
provenance, every claim needs a theorem, and the golden-contract tests must stay bit-identical
across backends.

## Contributing

Fork, branch from `main`, sign off every commit (`git commit -s`) to certify the
[DCO](https://developercertificate.org/), open a PR. Commit format: `fix:` · `feat(scope):` ·
`docs:` · `test:` · `chore:`.

By contributing you agree your contribution is Apache-2.0 (per section 5), including its
section 3 patent grant.

Licensed under Apache-2.0 — see [`LICENSE.md`](LICENSE.md), [`NOTICE`](NOTICE),
[`PATENTS.md`](PATENTS.md) and [`PLEDGE.md`](PLEDGE.md).
