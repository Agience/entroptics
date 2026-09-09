"""Regenerate every number in PAPER_frb.md.

    python research/supplemental/frb/reproduce.py            # everything
    python research/supplemental/frb/reproduce.py <root>     # with an explicit waterfall tree

Writes, into ``tables/`` beside this file:

    events.csv       per-burst reads for the four Figure-1 events (from frb_panel.py)
    spotcheck.csv    the 12-waterfall random draw            (from frb_spotcheck.py)
    agreement.csv    every correlation the paper reports, including the two references

Nothing here reimplements the instrument.  ``entroptics`` is imported and its public front
door called; the two figure routines under ``research/figures`` are imported and run.  The
only thing computed here that is not already in the repository is the pair of reference
points of section 4 -- the noise-limited reference and the fold-resolution ceiling -- which
exist to anchor the correlations rather than to leave them unscaled.

DATA.  The CHIME/FRB Catalog 1 waterfalls are a SEPARATE PUBLIC DOWNLOAD from the CANFAR
archive (CISTI.CANFAR/21.0007) and are not part of this repository -- see README.md for the
fetch.  This script does not download them: the release is far larger than anything a
reproduction script should pull unasked, and a partial or silently-failing download would
overwrite good tables with empty ones.  It resolves the tree the way every figure here does
(argument, then FRB_WATERFALLS, then research.local.env) and refuses with instructions if it
finds nothing.
"""
from __future__ import annotations

import csv
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# The BLAS pool is pinned before numpy is reached by anything downstream: OpenBLAS splits a
# reduction across its pool, so the summation order -- and the last bit of every SVD read --
# depends on how many threads it happens to have.  The committed tables are diffed, so this
# is the difference between reproducible and nearly reproducible.  setdefault, so an operator
# who chose a value keeps it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLES = HERE / "tables"

# Run against the working tree's src/, as the validation suite and the figures do, rather than
# whatever wheel happens to be installed -- the paper reports what this checkout produces.
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "research" / "figures"))

import h5py  # noqa: E402

from entroptics import Aperture  # noqa: E402

import data_path  # noqa: E402

#: The four events of Figure 1, as chosen in ``research/figures/frb_panel.py``.
EVENTS = ["FRB20190425A", "FRB20190106B", "FRB20190227A", "FRB20190323B"]

#: Seed for the noise draw of the noise-limited reference.  Fixed, so the reference is a
#: number and not a range that moves between runs.
NOISE_SEED = 20260908

#: Fraction of the forward model's band-summed power the on-burst window must contain.  A
#: stated criterion, declared here rather than tuned: it is the only free choice in section 4
#: and the paper says so.
ON_BURST_CONTAINMENT = 0.99


# --------------------------------------------------------------------------------------
# the scoring, identical to research/figures/frb_panel.py._agreement
# --------------------------------------------------------------------------------------

def _demedian(A):
    """Remove each channel's own standing level.  ``model_wfall`` carries no per-channel
    baseline: its per-channel median is numerically zero (at most 3.9e-06 over these four
    events, against model maxima of 0.108-0.506), because it is a statement about the burst and
    not about each channel's standing level.  A series that carried a baseline would therefore
    be scored against one that does not, and the baseline would count as disagreement.  Applied
    to all series or to none."""
    return A - np.nanmedian(A, axis=1, keepdims=True)


def _corr(a, b, m):
    x, y = _demedian(a)[m], _demedian(b)[m]
    x = x - x.mean()
    y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 0 else np.nan


def _on_burst(mod, keep=ON_BURST_CONTAINMENT):
    """The smallest contiguous run of time samples holding ``keep`` of the model's band-summed
    power.  The model is non-negative and smooth, so this is its own support and no threshold
    on the noisy record is involved."""
    p = np.nansum(np.clip(mod, 0.0, None), axis=0)
    total = p.sum()
    T = p.size
    best = (T + 1, 0, T)
    for i in range(T):
        s = 0.0
        for j in range(i, T):
            s += p[j]
            if s >= keep * total:
                if (j - i + 1) < best[0]:
                    best = (j - i + 1, i, j + 1)
                break
    return best[1], best[2]


def _read(wf):
    """The read-side path of frb_panel.py, unchanged: drop dead channels, front door, map the
    folded screen back onto the recorded frequency axis."""
    W = wf.T
    live = np.isfinite(W).all(axis=0) & (np.nanstd(W, axis=0) > 0)
    n_live = int(live.sum())
    clean, info = Aperture(W[:, live], window=None).extract()
    n_F = int(clean.shape[1])
    idx = (np.arange(n_live) * n_F) // n_live
    if n_F != n_live:
        clean = clean[:, idx]
    full = np.full_like(wf, np.nan)
    full[live, :] = clean.T
    return full, info, live, idx, n_F, n_live


def _fold(A_live, idx, n_F, n_live):
    """Box-average across each folded group and re-expand -- everything the fold does, applied
    to whatever series is handed in."""
    if n_F == n_live:
        return A_live
    B = np.stack([A_live[:, idx == j].mean(1) for j in range(n_F)], axis=1)
    return B[:, idx]


def agreement(root):
    """Every correlation the paper reports, per burst.

    Three series against the forward model (raw, fold-only, the read) and two reference
    points.  The references answer the question the three columns raise on their own -- a
    correlation of 0.56 against a model that is not ground truth means nothing without a
    scale, and there is no ground truth here to supply one:

      noise-limited   the forward model corrupted by this record's OWN measured per-channel
                      noise, scored against the model.  What a method that recovered the
                      model EXACTLY would score if it were measured through this waterfall.
      fold ceiling    the forward model degraded by exactly the fold the read is taken at,
                      scored against the model.  The most any read at that width could score,
                      from resolution alone.

    They bracket the read from below and above, and neither is a claim about the burst."""
    rows = []
    # the same quantities unrounded, so the means the paper quotes are means of the
    # correlations and not means of their printed 3-decimal forms
    _full = {k: [] for k in ("corr_raw_model", "corr_rebinned_model", "corr_read_model",
                             "ref_noise_limited", "ref_fold_ceiling",
                             "corr_raw_model_onburst", "corr_read_model_onburst",
                             "model_max", "model_baseline_max")}
    g = np.random.default_rng(NOISE_SEED)
    for event in EVENTS:
        hits = glob.glob(os.path.join(root, "*", f"{event}_waterfall.h5"))
        if not hits:
            print(f"  {event}: not found under the tree -- SKIPPED")
            continue
        with h5py.File(hits[0], "r") as f:
            wf = np.array(f["frb/wfall"], dtype=np.float64)
            mod = np.array(f["frb/model_wfall"], dtype=np.float64)

        read, _info, live, idx, n_F, n_live = _read(wf)

        W = wf.T
        rebin = np.full_like(wf, np.nan)
        rebin[live, :] = _fold(W[:, live], idx, n_F, n_live).T

        # the fold-resolution ceiling: the MODEL through the same fold
        modfold = np.full_like(mod, np.nan)
        modfold[live, :] = _fold(mod[live, :].T, idx, n_F, n_live).T

        t0, t1 = _on_burst(mod)

        # this record's own per-channel noise scale, read robustly OFF the burst
        off = np.ones(wf.shape[1], bool)
        off[t0:t1] = False
        R = wf[:, off]
        sigma = 1.4826 * np.nanmedian(np.abs(R - np.nanmedian(R, axis=1, keepdims=True)), axis=1)
        noisy = mod + g.standard_normal(mod.shape) * sigma[:, None]

        # every series scored on the SAME cells -- the read is undefined on dead channels, and
        # scoring each "wherever it is finite" scores them on different pixels
        m = (np.isfinite(read) & np.isfinite(rebin) & np.isfinite(wf)
             & np.isfinite(mod) & np.isfinite(modfold))
        m_on = np.zeros_like(m)
        m_on[:, t0:t1] = True
        m_on &= m

        # The premise of the baseline removal, measured rather than asserted: model_wfall carries
        # no per-channel standing level.  It is numerically zero and not exactly zero -- the model
        # is a strictly positive smooth surface -- so the paper states the bound.
        mod_med = np.nanmedian(mod, axis=1)[live]
        baseline_max = float(np.nanmax(np.abs(mod_med)))

        # Computed once, unrounded, then formatted.  The mean row below is the mean of THESE,
        # not of their printed forms -- averaging four 3-decimal strings is a different number,
        # and the difference lands in the third decimal of what the paper quotes.
        vals = {
            "model_max": float(np.nanmax(mod)),
            "model_baseline_max": baseline_max,
            "corr_raw_model": _corr(wf, mod, m),
            "corr_rebinned_model": _corr(rebin, mod, m),
            "corr_read_model": _corr(read, mod, m),
            "ref_noise_limited": _corr(noisy, mod, m),
            "ref_fold_ceiling": _corr(modfold, mod, m),
            "corr_raw_model_onburst": _corr(wf, mod, m_on),
            "corr_read_model_onburst": _corr(read, mod, m_on),
        }
        for _k in _full:
            _full[_k].append(vals[_k])

        rows.append({
            "event": event,
            "T": wf.shape[1],
            "on_burst_lo": t0,
            "on_burst_hi": t1,
            "noise_sigma_median": f"{np.nanmedian(sigma[live]):.4f}",
            "model_max": f"{vals['model_max']:.3f}",
            "model_baseline_max": f"{vals['model_baseline_max']:.3e}",
            "corr_raw_model": f"{vals['corr_raw_model']:.3f}",
            "corr_rebinned_model": f"{vals['corr_rebinned_model']:.3f}",
            "corr_read_model": f"{vals['corr_read_model']:.3f}",
            "ref_noise_limited": f"{vals['ref_noise_limited']:.3f}",
            "ref_fold_ceiling": f"{vals['ref_fold_ceiling']:.4f}",
            "corr_raw_model_onburst": f"{vals['corr_raw_model_onburst']:.3f}",
            "corr_read_model_onburst": f"{vals['corr_read_model_onburst']:.3f}",
        })
        print(f"  {event:14s} raw {rows[-1]['corr_raw_model']:>6}  "
              f"read {rows[-1]['corr_read_model']:>6}  "
              f"noise-limited {rows[-1]['ref_noise_limited']:>6}  "
              f"fold ceiling {rows[-1]['ref_fold_ceiling']:>7}")
    return rows, _full


# --------------------------------------------------------------------------------------

def _copy_table(src, dst):
    dst.write_bytes(src.read_bytes())
    print(f"  wrote {dst.relative_to(REPO)}")


def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        root = data_path.waterfall_root(explicit)
    except data_path.WaterfallsNotConfigured as exc:
        raise SystemExit(f"{exc}\n\nSee {HERE / 'README.md'} for the CANFAR fetch.") from None
    if not glob.glob(os.path.join(root, "*", "*_waterfall.h5")):
        raise SystemExit(
            f"No '*_waterfall.h5' under {root}.\n"
            f"That path resolves, but holds no waterfalls -- a partial or wrongly-rooted "
            f"download.\nThe root is the directory CONTAINING the per-event directories.\n"
            f"See {HERE / 'README.md'}.")

    TABLES.mkdir(exist_ok=True)
    print(f"waterfall tree: {root}\n")

    # The two committed figure routines, run as they stand.  They write their own PNG and CSV
    # under research/figures; the tables this paper cites are copied in beside it so the paper
    # directory is self-contained for a referee.
    print("running research/figures/frb_panel.py")
    import frb_panel
    frb_panel.ROOT = root
    frb_panel.main()

    print("\nrunning research/figures/frb_spotcheck.py")
    import frb_spotcheck
    sys.argv = [sys.argv[0], root]
    frb_spotcheck.main()

    print("\ncomputing the agreement table and its two reference points")
    rows, _full = agreement(root)

    print("\ncollecting tables")
    figs = REPO / "research" / "figures"
    _copy_table(figs / "frb_panel.csv", TABLES / "events.csv")
    _copy_table(figs / "frb_spotcheck.csv", TABLES / "spotcheck.csv")

    # The MEANS the paper quotes, at full precision.  Emitted as a row rather than left to the
    # reader: averaging the four printed 3-decimal values is not the same number as averaging the
    # correlations, and the difference lands in the third decimal -- 0.214 against 0.215 for the
    # box-average control.  The row is what the paper quotes and what verify.py checks.
    means = {k: float(np.mean(_full[k])) for k in _full}
    # six decimals, not four: the mean of the box-average column is 0.214465, and printing it
    # as 0.2145 puts it exactly on a rounding boundary where a checker cannot tell 0.214
    # from 0.215 -- which is the ambiguity this row exists to remove.
    mean_row = {k: (f"{means[k]:.6f}" if k in means else "") for k in rows[0]}
    mean_row["event"] = "mean"

    out = TABLES / "agreement.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        w.writerow(mean_row)
    print(f"  wrote {out.relative_to(REPO)}")

    print("\nmeans over the four events (full precision):")
    for k, v in means.items():
        print(f"  {k:28s} {v:.4f}")
    gain = means["corr_read_model"] - means["corr_raw_model"]
    fold = means["corr_rebinned_model"] - means["corr_raw_model"]
    print(f"\n  gain (read - raw)        {gain:.4f}")
    print(f"  fold's share of the gain {100 * fold / gain:.2f}%")


if __name__ == "__main__":
    main()
