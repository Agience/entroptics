"""frb_spotcheck.csv -- does the instrument handle CHIME/FRB data at all, on bursts nobody chose?

Figure 2 shows four bursts drawn at random.  This is the same read on a further random draw from
the same public release, and it answers one question and no other: does the untuned instrument
RUN on real waterfalls it was not selected against, and resolve something above its own floor.

It is not a performance measurement.  There is no ground truth for a fast radio burst -- the
source is unknown, and CHIME's `model_wfall` is a parametric fit, not truth -- so nothing here is
scored against a reference.  What is reported is what the instrument says about each record.

    python frb_spotcheck.py <root> [n] [seed]   ->  ./frb_spotcheck.csv

<root> is the CANFAR waterfall tree, as for frb_panel.py (see its docstring for the download).
Events are drawn uniformly without replacement from every `*_waterfall.h5` under it, using a
fixed seed so the draw is reproducible and was not steered.  Requires the ``figures`` extra.
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

from entroptics import Aperture

N_DEFAULT = 12
SEED_DEFAULT = 20260901


def _entroptics(wf):
    """The read-side path of frb_panel.py, unchanged: drop dead channels, front door, map back."""
    W = wf.T                                                     # (time, freq)
    live = np.isfinite(W).all(axis=0) & (np.nanstd(W, axis=0) > 0)
    clean, info = Aperture(W[:, live], window=None).extract()
    return info, int(live.sum()), W.shape[0]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: python frb_spotcheck.py <path-to-CHIME-FRB-Catalog1> [n] [seed]")
    root = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else N_DEFAULT
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else SEED_DEFAULT

    paths = sorted(glob.glob(os.path.join(root, "*", "*_waterfall.h5")))
    if not paths:
        raise SystemExit(f"no *_waterfall.h5 under {root}")
    rng = np.random.default_rng(seed)
    pick = [paths[i] for i in rng.choice(len(paths), size=min(n, len(paths)), replace=False)]

    out = Path(__file__).resolve().parent / "frb_spotcheck.csv"
    resolved = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# random draw of {len(pick)} of {len(paths)} waterfalls, seed={seed}"])
        w.writerow(["event", "T", "F_recorded", "F_live", "K_signal", "contrast",
                    "coherence_z", "n_kept", "n_dropped"])
        for p in pick:
            event = os.path.basename(p).replace("_waterfall.h5", "")
            with h5py.File(p, "r") as f:
                wf = np.array(f["frb/wfall"], dtype=np.float64)
            info, nlive, T = _entroptics(wf)
            resolved += int(info["K_signal"] >= 1)
            w.writerow([event, T, wf.shape[0], nlive, info["K_signal"],
                        f"{info['contrast']:.2f}", f"{info['coherence']:.1f}",
                        info["n_kept"], info["n_dropped"]])
            print(f"  {event:14s} T={T:5d} F={wf.shape[0]:6d} live={nlive:6d}  "
                  f"K={info['K_signal']:2d} contrast={info['contrast']:6.2f} "
                  f"z={info['coherence']:7.1f} kept={info['n_kept']} dropped={info['n_dropped']}")
    print(f"\nresolved K_signal >= 1 on {resolved}/{len(pick)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
