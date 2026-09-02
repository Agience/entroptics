"""frb_panel.png -- CHIME model | raw | Entroptics, on real CHIME/FRB Catalog 1 bursts.

For each bright burst, three panels at native 16384-channel resolution (left to right):

  1. CHIME model    -- frb/model_wfall, CHIME's fitburst forward model (a smooth parametric
                       fit: Gaussian(s) in time x running-power-law in frequency). 
                       Retreived from the CHIME/FRB Catalog 1 HDF5 files
  2. raw            -- frb/wfall, the dedispersed waterfall cutout (the noisy measurement).
  3. Entroptics     -- Aperture(raw).extract(): Gavish-Donoho shrinkage against the derived noise
                       floor + a geometric persistent-structure cut (phi_F > phi_T), a parameter-free
                       filter of the real data at native resolution.  Keeps the true burst
                       morphology; removes only the noise / RFI modes.

    python frb_panel.py <root>   ->  ./frb_panel.png, ./frb_panel.csv

DATA.  The four events are CHIME/FRB Catalog 1 waterfall cutouts, a separate public download
from the CANFAR archive (CISTI.CANFAR/21.0007) -- they are NOT in this repository.  <root> is
a directory of subdirectories holding ``<EVENT>_waterfall.h5``; the glob is ``<root>/*/``, one
level deep.  The four this figure reads are:

    FRB20190425A  FRB20190106B  FRB20190227A  FRB20190323B

Any event not found is SKIPPED, so a partial download silently produces a figure with fewer
rows -- check the printed table has four.  Requires the ``figures`` extra
(``pip install -e ".[figures]"``) for h5py and matplotlib.
"""
import csv
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import h5py

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from entroptics import Aperture

if len(sys.argv) > 1:
    ROOT = sys.argv[1]
else:
    raise ValueError("usage: python frb_panel.py <path-to-CHIME-FRB-Catalog1>")
EVENTS = ["FRB20190425A", "FRB20190106B", "FRB20190227A", "FRB20190323B"]

COLS = ["RAW",
        "Fitburst",
        "Entroptics",
        "Removed"]


def _find(event):
    hits = glob.glob(os.path.join(ROOT, "*", f"{event}_waterfall.h5"))
    return hits[0] if hits else None


def _load(path):
    with h5py.File(path, "r") as f:
        wf = np.array(f["frb/wfall"], dtype=np.float64)          # (freq, time)
        mod = np.array(f["frb/model_wfall"], dtype=np.float64)   # (freq, time)
        ext = np.array(f["frb/extent"], dtype=np.float64)        # [t0,t1,f0,f1]
        dm = float(f["frb"].attrs.get("dm", np.nan))
    return wf, mod, ext, dm


def _entroptics(wf):
    """Aperture front door on the live channels, mapped back to the full (freq,time) axis.

    The read is taken on the folded screen (Def 2.2), which for these frames is narrower than the
    live width.  It is mapped back through the fold's own index so the panels share the recorded
    frequency axis; the mapping invents nothing, since the read is constant across each folded
    group.  ``n_F`` is returned so the committed table records the width it was read at."""
    W = wf.T                                                     # (time, freq)
    live = np.isfinite(W).all(axis=0) & (np.nanstd(W, axis=0) > 0)
    n_live = int(live.sum())
    clean, info = Aperture(W[:, live], window=None).extract()    # (time, n_F) on the folded screen
    n_F = int(clean.shape[1])
    idx = (np.arange(n_live) * n_F) // n_live
    if n_F != n_live:                                            # undo the fold for display only
        clean = clean[:, idx]
    full = np.full_like(wf, np.nan)                              # (freq, time)
    full[live, :] = clean.T
    return full, info, n_live, n_F, live, idx


def _agreement(wf, read, live, idx, n_F, mod):
    """Agreement with CHIME's forward model, for the read and for two references.

    This is a CONSISTENCY check and not an accuracy one: ``model_wfall`` is a parametric fit to
    the same waterfall, not ground truth, so what it can show is whether the read moves toward
    CHIME's own account of the burst -- not whether either is right.

    Two things would make the comparison unfair, and both are controlled.  All three are scored
    on the SAME cells (the live channels), because the read is undefined on dead ones and scoring
    each "wherever it is finite" scores them on different pixels.  And the read arrives smoothed
    along frequency -- it is piecewise-constant across each folded group -- while the model is
    smooth too, so a plain box-average of the RAW to the same width is included: everything the
    fold does and nothing the read does."""
    def _c(a, b, m):
        x, y = a[m], b[m]
        x = x - x.mean(); y = y - y.mean()
        d = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / d) if d > 0 else np.nan

    W = wf.T
    X = W[:, live]
    n_live = int(live.sum())
    Xb = np.stack([X[:, idx == j].mean(1) for j in range(n_F)], axis=1) if n_F != n_live else X
    rebin = np.full_like(wf, np.nan)
    rebin[live, :] = (Xb[:, idx] if n_F != n_live else Xb).T

    m = np.isfinite(read) & np.isfinite(rebin) & np.isfinite(wf) & np.isfinite(mod)
    return _c(wf, mod, m), _c(rebin, mod, m), _c(read, mod, m)


def _show(ax, arr, ext, title=None, xlabel=None):
    finite = arr[np.isfinite(arr)]
    vlo, vhi = np.percentile(finite, [50, 99.7]) if finite.size else (0, 1)
    cmap = plt.cm.magma.copy(); cmap.set_bad("#1a1a1a")
    ax.imshow(arr, extent=ext, origin="lower", aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi)
    if title:  ax.set_title(title, fontsize=10, fontweight="bold")
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(labelsize=7.5)


def main():
    rows = [(e, _find(e)) for e in EVENTS]
    rows = [(e, p) for e, p in rows if p]
    n = len(rows)
    fig = plt.figure(figsize=(16.4, 3.15 * n))
    gs = gridspec.GridSpec(n, 4, hspace=0.24, wspace=0.11,
                           left=0.128, right=0.99, top=0.90, bottom=0.072)

    for i, (event, path) in enumerate(rows):
        wf, mod, ext, dm = _load(path)
        clean, info, nlive, n_F, _live, _idx = _entroptics(wf)
        # What the filter took out.  The read carries the burst's MORPHOLOGY, not the input's
        # amplitude scale -- it is taken on the scale-equalised screen -- so differencing the two
        # directly would show a scaled copy of the burst and read as signal loss that is not
        # there.  The scale is matched by least squares over the live cells first; what remains
        # is what the read genuinely did not keep.
        ok = np.isfinite(clean) & np.isfinite(wf)
        den = float(np.sum(clean[ok] ** 2))
        scale = float(np.sum(wf[ok] * clean[ok]) / den) if den > 0 else 1.0
        removed = wf - scale * clean

        xlabel = "time (ms)" if i == n - 1 else None
        ax0 = fig.add_subplot(gs[i, 0])
        _show(ax0, wf, ext, title=(COLS[0] if i == 0 else None), xlabel=xlabel)
        _show(fig.add_subplot(gs[i, 1]), mod, ext, title=(COLS[1] if i == 0 else None), xlabel=xlabel)
        _show(fig.add_subplot(gs[i, 2]), clean, ext, title=(COLS[2] if i == 0 else None), xlabel=xlabel)
        _show(fig.add_subplot(gs[i, 3]), removed, ext, title=(COLS[3] if i == 0 else None), xlabel=xlabel)

        # all per-row labels live outside the panels, in the left margin
        pos = ax0.get_position()
        yc = 0.5 * (pos.y0 + pos.y1)
        block = (f"{event}\nDM {dm:.1f}\n\n"
                 f"$K$ = {info['K_signal']}\ncontrast = {info['contrast']:.1f}$\\times$\n"
                 f"$z$ = {info['coherence']:.0f}\nkept {info['n_kept']} / dropped {info['n_dropped']}\n"
                 f"read at {n_F} of {nlive}")
        fig.text(0.010, yc, block, va="center", ha="left", fontsize=8.4)
        ax0.set_ylabel("frequency (MHz)", fontsize=9)

    fig.suptitle("Real CHIME/FRB bursts, read untuned at the width the instrument folds to",
                 fontsize=11, y=0.972)
    fig.text(0.5, 0.008,
             "Data: CHIME/FRB Catalog 1 (CHIME/FRB Collab. 2021, ApJS 257, 59; arXiv:2106.04352), public release CANFAR CISTI.CANFAR/21.0007.",
             ha="center", va="bottom", fontsize=7.4, color="0.45")
    out = Path(__file__).resolve().parent / "frb_panel.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # committed per-burst reads behind the figure
    dat = Path(__file__).resolve().parent / "frb_panel.csv"
    with open(dat, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "dm", "T", "F_recorded", "n_live", "n_F_read", "K_signal",
                    "contrast", "coherence_z", "n_kept", "n_dropped",
                    "corr_raw_model", "corr_rebinned_model", "corr_read_model"])
        for event, path in rows:
            wf, mod, ext, dm = _load(path)
            read, info, nlive, n_F, live, idx = _entroptics(wf)
            c_raw, c_reb, c_read = _agreement(wf, read, live, idx, n_F, mod)
            w.writerow([event, f"{dm:.1f}", wf.shape[1], wf.shape[0], nlive, n_F,
                        info["K_signal"], f"{info['contrast']:.2f}",
                        f"{info['coherence']:.1f}", info["n_kept"], info["n_dropped"],
                        f"{c_raw:.3f}", f"{c_reb:.3f}", f"{c_read:.3f}"])
            print(f"  {event:14s} live={nlive:5d} read={n_F:5d}  K={info['K_signal']:2d}  "
                  f"contrast={info['contrast']:6.1f}x  z={info['coherence']:6.1f}  "
                  f"kept={info['n_kept']} dropped={info['n_dropped']}  "
                  f"corr raw/rebin/read = {c_raw:.3f}/{c_reb:.3f}/{c_read:.3f}")
    print(f"wrote {dat}")


if __name__ == "__main__":
    main()
