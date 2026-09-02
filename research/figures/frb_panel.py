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

    python frb_panel.py   ->  ./frb_panel.png
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

COLS = ["Fitburst",
        "RAW",
        "Entroptics"]


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
    """Aperture front door on the live channels, mapped back to the full (freq,time) axis."""
    W = wf.T                                                     # (time, freq)
    live = np.isfinite(W).all(axis=0) & (np.nanstd(W, axis=0) > 0)
    clean, info = Aperture(W[:, live], window=None).extract()   # (time, n_live), full frame through the front door
    full = np.full_like(wf, np.nan)                              # (freq, time)
    full[live, :] = clean.T
    return full, info, int(live.sum())


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
    fig = plt.figure(figsize=(12.6, 3.15 * n))
    gs = gridspec.GridSpec(n, 3, hspace=0.24, wspace=0.11,
                           left=0.165, right=0.99, top=0.90, bottom=0.072)

    for i, (event, path) in enumerate(rows):
        wf, mod, ext, dm = _load(path)
        clean, info, nlive = _entroptics(wf)

        xlabel = "time (ms)" if i == n - 1 else None
        ax0 = fig.add_subplot(gs[i, 0])
        _show(ax0, mod, ext, title=(COLS[0] if i == 0 else None), xlabel=xlabel)
        _show(fig.add_subplot(gs[i, 1]), wf, ext, title=(COLS[1] if i == 0 else None), xlabel=xlabel)
        _show(fig.add_subplot(gs[i, 2]), clean, ext, title=(COLS[2] if i == 0 else None), xlabel=xlabel)

        # all per-row labels live OUTSIDE the panels, in the left margin
        pos = ax0.get_position()
        yc = 0.5 * (pos.y0 + pos.y1)
        block = (f"{event}\nDM {dm:.1f}\n\n"
                 f"$K$ = {info['K_signal']}\ncontrast = {info['contrast']:.1f}$\\times$\n"
                 f"$z$ = {info['coherence']:.0f}\nkept {info['n_kept']} / RFI {info['n_dropped']}")
        fig.text(0.013, yc, block, va="center", ha="left", fontsize=8.6)
        ax0.set_ylabel("frequency (MHz)", fontsize=9)

    fig.suptitle("Real CHIME/FRB bursts at native 16384-channel resolution",
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
        w.writerow(["event", "dm", "n_live", "K_signal", "contrast", "coherence_z",
                    "n_kept", "n_dropped"])
        for event, path in rows:
            wf, mod, ext, dm = _load(path)
            _, info, nlive = _entroptics(wf)
            w.writerow([event, f"{dm:.1f}", nlive, info["K_signal"],
                        f"{info['contrast']:.2f}", f"{info['coherence']:.1f}",
                        info["n_kept"], info["n_dropped"]])
            print(f"  {event:14s} live={nlive:5d}  K={info['K_signal']:2d}  "
                  f"contrast={info['contrast']:6.1f}x  z={info['coherence']:6.1f}  "
                  f"kept={info['n_kept']} dropped={info['n_dropped']}")
    print(f"wrote {dat}")


if __name__ == "__main__":
    main()
