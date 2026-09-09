---
title: 'Entroptics: parameter-free rank selection and mode reads for 2-D signals'
tags:
  - Python
  - signal processing
  - random matrix theory
  - rank selection
  - dynamic mode decomposition
authors:
  - name: Ikailo John Sessford
    orcid: 0009-0002-0150-4027
    affiliation: 1
affiliations:
  - name: Ikailo Inc., Canada
    index: 1
date: 8 September 2026
bibliography: paper.bib
---

# Summary

`entroptics` is a Python library that reads structure out of a two-dimensional array with one
ordered axis (time, depth, evolution) and one feature axis (channels, frequency bins,
coordinates). It answers three questions that recur across signal-processing pipelines: at what
resolution should this record be represented, how many of its degrees of freedom are signal
rather than noise, and what does the signal look like once the noise is removed.

The library selects a rank against a derived Tracy–Widom noise floor [@tracy1996; @johnstone2001],
reconstructs the field on the surviving modes with Gavish–Donoho shrinkage [@gavish2017], reads
per-mode decay rates from a streaming dynamic mode decomposition operator [@tu2014], and supplies
closed-form permutation nulls for two association statistics. It is deterministic, has a single
backend-agnostic code path that runs on `numpy` arrays or `torch` tensors without change, and
handles missing data as absent rather than as zero.

Its distinguishing property is that **no constant in it is fitted to data or calibrated to a
substrate**. Every fixed number is a derived mathematical quantity — a $\chi^2$ median, an
influence-function variance, a universal Tracy–Widom quantile — or a criterion stated in the
documentation. What a user supplies is an operating point: a false-alarm level $\alpha$, and the
null it is taken against.

# Statement of need

Rank selection is a prerequisite for a large class of analyses and is usually done with a
threshold the analyst chooses, or with a criterion that requires a known noise level. In practice
this means a pipeline carries a constant that was tuned on one instrument and is silently wrong on
another. Existing tools reflect this: `scikit-learn`'s PCA [@sklearn] offers a variance-explained
fraction or Minka's MLE [@minka2000]; `optht` [@optht] implements the Gavish–Donoho optimal hard
threshold, whose unknown-noise form estimates its scale from the median singular value; and the
Wax–Kailath AIC and MDL criteria [@wax1985], standard in array processing, are derived for $n$
snapshots of $p$ variables with $n > p$ and are undefined otherwise.

`entroptics` targets users who need a rank, a resolution and a reconstruction from records whose
instrument they do not control, or across many substrates at once — radio astronomy waterfalls,
spectrograms, sensor panels, embedding stacks — where a per-substrate constant is exactly what
cannot be supplied. It is aimed at researchers and practitioners in signal processing, radio
astronomy and applied statistics.

Two further needs it addresses are more specific. It handles the wide, short regime ($F \gg T$)
that a dedispersed burst cutout or a short multichannel record presents, where the finite-size
Tracy–Widom edge holds the top eigenvalue but the asymptotic Marchenko–Pastur edge does not. And
it treats a masked or never-measured cell as absent throughout, so that every read divides by the
extent actually observed rather than by the array's nominal shape.

# Functionality and comparison

The dynamic-mode-decomposition component overlaps with `PyDMD` [@pydmd] and `pykoopman`
[@pykoopman], which are more complete DMD libraries: they carry many more DMD variants, and a user
whose problem is DMD should use them. `entroptics` differs in scope and in state. Its operator is
maintained as a fixed $F \times F$ sufficient statistic updated in one online pass, so reads cost
$O(F^3)$ independent of stream length and a stream can be split, spliced and resumed exactly; and
its rank truncation in the under-sampled regime is taken at the library's own derived floor rather
than at a supplied tolerance.

The association statistics are closed-form permutation tests. One standardises adjacent-row
similarity against the exact Cliff–Ord/Mantel second moment [@cliffords1981]; the other is a
signed bilinear coupling between two frames on a shared basis, whose exact permutation moments are
derived in a companion preprint [@coupling2026]. Both avoid resampling. They are related to, but
distinct from, the RV coefficient [@robert1976] and linear CKA [@kornblith2019], which are
rotation-invariant and therefore non-negative.

The library ships a validation suite of 19 seeded experiments that plant a known ground truth and
check the corresponding read recovers it, a test suite of 675 tests pinning the identities and
numpy/torch parity, and a Lean 4 / Mathlib development machine-checking the governing lemmas. An
application to public CHIME/FRB Catalog 1 data is reported separately [@frb2026], and the full
construction, with the provenance of every constant, is given in an accompanying preprint
[@entroptics2026].

# Acknowledgements

The CHIME/FRB Collaboration's public Catalog 1 release [@chimefrb2021] made the real-data
validation possible.

# Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.

# References
