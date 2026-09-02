"""calibration.png -- the entroptics ``extract`` filter, calibrated by eye against the numbers.

Six panels, empirical:
  1 injected burst        2 + noise               3 recovery
  4 recovery vs dropout   5 + noise + dropout     6 recovery of surviving channels

The PASS/FAIL guarantees live in src/tests/test_extract.py; this only draws them.

    python calibration.py   ->  ./calibration.png
"""
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from entroptics import Aperture

SNR = 12                   # injected peak signal-to-noise of the "capture"
DROP = 0.32               # fraction of channels randomly dropped in panels 5/6


def make_burst(T=64, F=256, peak=1.0):
    t = np.arange(T)[:, None]; f = np.arange(F)[None, :]
    b1 = np.exp(-0.5 * ((t - 0.46 * T) / 3.0) ** 2) * np.exp(-0.5 * ((f - 0.50 * F) / (0.22 * F)) ** 2)
    b2 = np.exp(-0.5 * ((t - 0.54 * T) / 4.0) ** 2) * np.exp(-0.5 * ((f - 0.66 * F) / (0.16 * F)) ** 2)
    B = b1 + 0.7 * b2
    return peak * B / B.max()


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else np.nan


def dropout_recovery(B, frac, snr, seed):
    """extract on a noisy burst with ``frac`` of its channels zeroed; return the recovery over the
    SURVIVING channels, the recovered image (dropped -> NaN) and the drop mask."""
    F = B.shape[1]
    rng = np.random.default_rng(seed)
    W = B + rng.standard_normal(B.shape) * (1.0 / snr)
    drop = np.zeros(F, bool)
    n = int(round(frac * F))
    if n:
        drop[rng.choice(F, n, replace=False)] = True
    Wz = W.copy(); Wz[:, drop] = 0.0
    rec, _ = Aperture(Wz, window=None).extract()
    surv = ~drop
    if rec.shape != B.shape:            # read collapsed under heavy dropout -> nothing resolved
        rec = np.zeros_like(B)
    rec_disp = rec.copy(); rec_disp[:, drop] = np.nan
    W_disp = W.copy(); W_disp[:, drop] = np.nan
    return corr(rec[:, surv], B[:, surv]), rec_disp, W_disp, drop


def main():
    B = make_burst()
    F = B.shape[1]

    # panels 2/3: noise only
    W2 = B + np.random.default_rng(0).standard_normal(B.shape) * (1.0 / SNR)
    rec3, _ = Aperture(W2, window=None).extract()
    p3 = corr(rec3, B) * 100

    # panels 5/6: noise + channel dropout
    p6, rec6_disp, W5_disp, drop = dropout_recovery(B, DROP, SNR, seed=0)
    p6 *= 100; pdrop = drop.mean() * 100

    # panel 4: recovery of surviving channels vs dropout fraction (a collapsed read -> 0% recovery)
    fracs = np.arange(0.0, 0.85, 0.03)
    curve = []
    for fr in fracs:
        vals = [dropout_recovery(B, fr, SNR, seed=100 + s)[0] for s in range(10)]
        vals = [0.0 if not np.isfinite(v) else v for v in vals]
        curve.append(np.mean(vals) * 100)
    fx = fracs * 100; cy = np.array(curve)

    # committed data behind the figure: the recovery-vs-dropout curve and the two scalar recoveries
    dat = Path(__file__).resolve().parent / "calibration.csv"
    with open(dat, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# entroptics extract() calibration (synthetic burst)"])
        w.writerow([f"# SNR={SNR}", f"noise_only_recovery_pct={p3:.2f}",
                    f"dropout_recovery_pct={p6:.2f}", f"drop_fraction_pct={pdrop:.1f}"])
        w.writerow(["channels_dropped_pct", "recovery_pct"])
        for x, y in zip(fx, cy):
            w.writerow([f"{x:.1f}", f"{y:.2f}"])
    print(f"wrote {dat}")

    fig = plt.figure(figsize=(12.5, 6.7))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28, left=0.065, right=0.985, top=0.90, bottom=0.10)

    def wf(ax, im, ttl):
        cmap = plt.cm.magma.copy(); cmap.set_bad("#1a1a1a")
        fin = im[np.isfinite(im)]
        vlo, vhi = np.percentile(fin, [50, 99.7]) if fin.size else (0, 1)
        ax.imshow(im.T, origin="lower", aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi)
        ax.set_title(ttl, fontsize=10, fontweight="bold"); ax.set_xlabel("time", fontsize=8.5)
        ax.tick_params(labelsize=7.5)

    wf(fig.add_subplot(gs[0, 0]), B, "injected burst")
    wf(fig.add_subplot(gs[0, 1]), W2, f"+ noise   (S/N {SNR})")
    wf(fig.add_subplot(gs[0, 2]), rec3, f"recovery\n{p3:.1f}%")

    axc = fig.add_subplot(gs[1, 0])
    axc.plot(fx, cy, "o-", color="#1f7a3d", ms=4)
    axc.axvline(pdrop, ls=":", color="0.5", lw=1)
    axc.set_xlabel("channels dropped (%)", fontsize=9); axc.set_ylabel("recovery (%)", fontsize=9)
    axc.set_ylim(0, 103); axc.set_xlim(0, 85)
    axc.set_title("recovery vs channels dropped", fontsize=10); axc.grid(alpha=0.3)
    axc.tick_params(labelsize=8)

    wf(fig.add_subplot(gs[1, 1]), W5_disp, f"+ noise (S/N {SNR}) + {pdrop:.0f}% random channel dropout")
    wf(fig.add_subplot(gs[1, 2]), rec6_disp, f"recovery of surviving channels\n{p6:.1f}%")

    fig.suptitle("Entroptics calibration", fontsize=14, fontweight="bold", y=0.975)
    out = Path(__file__).resolve().parent / "calibration.png"
    fig.savefig(out, dpi=150); print(f"wrote {out}   p3={p3:.2f}%  p6={p6:.2f}%  drop={pdrop:.1f}%")


if __name__ == "__main__":
    main()
