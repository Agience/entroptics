# PAPER_frb.md — FRB applications paper

An untuned rank-selection and reconstruction procedure applied to public CHIME/FRB Catalog 1
waterfalls, compared against CHIME's own *fitburst* forward model.

## Venue

Formatted for **RAS Techniques and Instruments** (RASTI). RASTI is the primary target: the paper
reports an instrument-side procedure applied to a public dataset, with the emphasis on what the
procedure does and what it cannot do, which is what RASTI's scope covers.

The secondary target is **Astronomy and Computing**. If it goes there instead, the changes are
presentational: A&C wants a software-availability statement as a numbered section rather than the
reproduction section's prose, and its structured abstract splits into Context / Aims / Methods /
Results. No content moves.

## Contents

| file | what it is |
|---|---|
| `PAPER_frb.md` | the paper |
| `reproduce.py` | regenerates every number in it |
| `tables/events.csv` | per-burst reads for the four events (Table 1) |
| `tables/spotcheck.csv` | the 12-waterfall random draw (§3.2) |
| `tables/agreement.csv` | every correlation, including both reference points (Tables 2, 3) |

The figure is `research/figures/frb_panel.png`, written by `research/figures/frb_panel.py`.

## Reproducing

```
python research/supplemental/frb/reproduce.py
```

Every seed is fixed; the tables are byte-reproducible across runs. The script imports `entroptics`
and calls its public front door — it reimplements no part of the instrument.

Requires the `figures` extra, for `h5py` and `matplotlib`:

```
pip install -e ".[figures]"
```

## The data

The waterfalls are **not in this repository**. They are a separate public download:

> CHIME/FRB Catalog 1 waterfalls
> CHIME/FRB Collaboration 2021, ApJS 257, 59 (arXiv:2106.04352)
> Public release: CANFAR **CISTI.CANFAR/21.0007** — https://www.canfar.net/

### Fetching it

The release is published through the CANFAR archive at the Canadian Astronomy Data Centre. It is
browsable at https://www.canfar.net/ under the collection DOI above, and can be pulled with the
CADC client tools (`pip install vos`, then `vcp`) or over HTTP from the archive's file listing.
Download the per-event waterfall tree.

**No script in this repository downloads it.** That is deliberate. The release is far larger than
anything a reproduction script should pull unasked, and these scripts write committed tables, so a
partial or silently-failing download would overwrite a good table with an empty one while still
exiting zero. The scripts refuse, and say what is missing, rather than guessing.

### Pointing the scripts at it

Once you have the tree, name it in any one of three ways — most specific first:

```
# 1. an explicit argument, per run
python research/supplemental/frb/reproduce.py /path/to/chime-frb-catalog1

# 2. the environment variable, per run
FRB_WATERFALLS=/path/to/chime-frb-catalog1 python research/supplemental/frb/reproduce.py

# 3. per machine: copy the example to research.local.env at the repository root and edit it
cp research.local.env.example research.local.env
```

`research.local.env` is git-ignored: it names a path on one machine, which is the thing this
repository must not carry.

The root is the directory that **contains** the per-event directories, so that
`<root>/*/*_waterfall.h5` matches. Verify with:

```
ls /path/to/chime-frb-catalog1/*/*_waterfall.h5 | head
```

The four events of Figure 1 are `FRB20190425A`, `FRB20190106B`, `FRB20190227A`, `FRB20190323B`.
Any event not found is skipped, so check the printed table has four rows.

## Relation to the instrument paper

`research/PAPER.md` is the full reference for the construction: the definitions, the proofs, the
provenance of every constant, and the synthetic validation with a planted ground truth. This paper
carries a self-contained account of the read path in §2 and cites the instrument paper for
everything else.
