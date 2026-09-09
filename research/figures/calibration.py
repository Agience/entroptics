"""calibration.png -- the entroptics ``extract`` filter, calibrated by eye against the numbers.

Six panels, empirical:
  1 injected burst        2 + noise               3 recovery
  4 recovery vs dropout   5 + noise + dropout     6 recovery of surviving channels

The pass/fail guarantees live in src/tests/test_extract.py; this only draws them.

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
N_SEEDS = 10              # draws per dropout fraction in panel 4


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


def native(clean, F):
    """A read taken on the folded screen, mapped back onto the recorded feature axis.

    ``extract`` returns the field on the screen it was read on, and Def 2.2 folds that screen to
    the marginal's own width -- for this burst, 186 channels of 256.  Comparing the read to the
    input therefore has to undo the fold's index map first; ``clean`` is piecewise-constant across
    each folded group, so the inverse is the same map applied backwards and no interpolation is
    invented.  This is the same unfold ``frb_panel._entroptics`` does for display."""
    n = clean.shape[1]
    if n == F:
        return clean
    return clean[:, (np.arange(F) * n) // F]


def relerr(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def snr_band(B, snrs=(10, 50, 1000, None)):
    """The read against the truth across the noise-relevant band, and against the raw frame.

    ``test_extract.py`` computes exactly this and asserts BOUNDS on it -- correlation above 0.95,
    relative error below 0.2.  The paper quotes the VALUES, and nothing wrote them down, so a
    figure that drifted stayed in the prose until someone re-derived it by hand.  This emits them.

    ``snr=None`` is the noiseless limit.  The raw-field columns are the same comparison made
    against the unfiltered frame, which is what says where the filter stops helping: its own
    residual is about 1%, so above S/N ~ 200 the raw frame is already closer to the truth."""
    rows = []
    for snr in snrs:
        W = B if snr is None else B + np.random.default_rng(0).standard_normal(B.shape) / snr
        clean, _ = Aperture(W, window=None).extract()
        cn = native(clean, B.shape[1])
        rows.append({"snr": "none" if snr is None else snr,
                     "read_corr": f"{corr(cn, B):.3f}",
                     "raw_corr": f"{corr(W, B):.3f}",
                     "read_relerr": f"{relerr(cn, B):.3f}",
                     "raw_relerr": f"{relerr(W, B):.3f}"})
    return rows


def dropout_recovery(B, frac, snr, seed):
    """extract on a noisy burst with ``frac`` of its channels DROPPED; return the recovery over the
    surviving channels, the recovered image (dropped -> NaN) and the drop mask.

    The channels are dropped, not zeroed.  A mask is not a zero: zeroing hands the read a column
    of zero variance where nothing was observed, and the FRB path (``frb_panel._entroptics``)
    drops the dead channels before the front door.  Figure 1 calibrates the path Figure 2 runs.

    Returns NaN for the recovery when the read resolves nothing.  That is a different outcome
    from a poor recovery and the caller must not average the two together."""
    F = B.shape[1]
    rng = np.random.default_rng(seed)
    W = B + rng.standard_normal(B.shape) * (1.0 / snr)
    drop = np.zeros(F, bool)
    n = int(round(frac * F))
    if n:
        drop[rng.choice(F, n, replace=False)] = True
    surv = ~drop
    rec_surv, info = Aperture(W[:, surv], window=None).extract()
    rec_surv = native(rec_surv, int(surv.sum()))          # off the folded screen, onto the axis
    rec = np.full_like(B, np.nan)
    # "Resolved nothing" is a read that KEPT no mode -- every mode below the floor, or every one
    # above it cut as persistent.  Not a shape test: a folded screen comes back narrower than the
    # input on a read that resolved perfectly well, so shape answers a different question.
    if info["n_kept"] > 0:
        rec[:, surv] = rec_surv
    W_disp = W.copy(); W_disp[:, drop] = np.nan
    return corr(rec[:, surv], B[:, surv]), rec, W_disp, drop


def main():
    B = make_burst()
    F = B.shape[1]

    # panels 2/3: noise only
    W2 = B + np.random.default_rng(0).standard_normal(B.shape) * (1.0 / SNR)
    rec3, _ = Aperture(W2, window=None).extract()
    rec3 = native(rec3, F)                                # off the folded screen, onto the axis
    p3 = corr(rec3, B) * 100

    # panels 5/6: noise + channel dropout
    p6, rec6_disp, W5_disp, drop = dropout_recovery(B, DROP, SNR, seed=0)   # rec is NaN off-mask
    p6 *= 100; pdrop = drop.mean() * 100

    # panel 4: two separate series over dropout fraction.  A read that resolves nothing is a
    # FAILURE TO PRODUCE, not a 0% recovery -- averaging the two together plots the success rate
    # and calls it fidelity.
    #
    # The sweep runs to the LIMIT, which is one surviving channel: a grid that stops partway
    # leaves the collapse off the edge of the plot and reads as flat robustness everywhere.
    # It is sampled in the surviving COUNT rather than the dropped fraction, because that is the
    # quantity the read is limited by -- 84% of 256 channels dropped still leaves 41, which is
    # why nothing changes out there, and the last decade of the fraction axis holds every
    # surviving count from 41 down to 1.
    surv_grid = sorted({F - int(round(fr * F)) for fr in np.arange(0.0, 0.85, 0.03)}
                       | {41, 32, 26, 20, 16, 13, 10, 8, 6, 5, 4, 3, 2, 1})
    fracs = np.array([(F - s) / F for s in reversed(surv_grid)])
    survs = np.array(list(reversed(surv_grid)))
    succ, cond = [], []
    for fr in fracs:
        vals = [dropout_recovery(B, fr, SNR, seed=100 + s)[0] for s in range(N_SEEDS)]
        ok = [v for v in vals if np.isfinite(v)]
        succ.append(100.0 * len(ok) / len(vals))
        cond.append(np.mean(ok) * 100 if ok else np.nan)
    fx = fracs * 100
    sy = np.array(succ); cy = np.array(cond)

    # committed data behind the figure: the recovery-vs-dropout curve and the two scalar recoveries
    dat = Path(__file__).resolve().parent / "calibration.csv"
    with open(dat, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# entroptics extract() calibration (synthetic burst)"])
        w.writerow([f"# SNR={SNR}", f"noise_only_recovery_pct={p3:.2f}",
                    f"dropout_recovery_pct={p6:.2f}", f"drop_fraction_pct={pdrop:.1f}"])
        # the S/N band the paper quotes, emitted rather than left to a bounds assertion
        w.writerow(["# S/N band: the read and the raw frame, both against the truth"])
        w.writerow(["snr", "read_corr", "raw_corr", "read_relerr", "raw_relerr"])
        for r in snr_band(B):
            w.writerow([r["snr"], r["read_corr"], r["raw_corr"],
                        r["read_relerr"], r["raw_relerr"]])
        w.writerow([])
        w.writerow(["# recovery against channel dropout, swept to one surviving channel"])
        w.writerow(["channels_dropped_pct", "channels_surviving", "resolved_pct",
                    "recovery_pct_when_resolved"])
        for x, ns, sv, y in zip(fx, survs, sy, cy):
            w.writerow([f"{x:.2f}", int(ns), f"{sv:.1f}",
                        "" if not np.isfinite(y) else f"{y:.2f}"])
        # The terminal point, which is not a measurement: at 100% no channel survives and the
        # front door refuses the read rather than returning a recovery of anything.
        w.writerow(["100.00", 0, "0.0", ""])
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

    # Plotted against the SURVIVING count on a log axis, descending, rather than against the
    # dropped fraction: the surviving count is what limits the read, and on a linear fraction
    # axis every change is crushed into the last few percent -- 84% dropped still leaves 41 of
    # 256 channels, so a sweep drawn that way reads as flat robustness and hides its own limit.
    axc = fig.add_subplot(gs[1, 0])
    axc.plot(survs, cy, "o-", color="#1f7a3d", ms=3.5, label="recovery, when resolved")
    axc.plot(survs, sy, "s--", color="#8a4b9c", ms=3, lw=1.2, label=f"resolved at all ({N_SEEDS} draws)")
    axc.set_xscale("log")
    axc.invert_xaxis()
    axc.set_xticks([256, 64, 16, 4, 1])
    axc.set_xticklabels(["256", "64", "16", "4", "1"])
    axc.axvline(F - int(round(pdrop / 100 * F)), ls=":", color="0.5", lw=1)
    axc.set_xlabel("channels surviving  (of 256, log scale)", fontsize=9)
    axc.set_ylabel("percent", fontsize=9)
    axc.set_ylim(0, 103)
    axc.set_title("recovery vs channels dropped", fontsize=10); axc.grid(alpha=0.3)
    axc.tick_params(labelsize=8)
    # At 100% dropped no channel survives; there is no read to score, and the front door refuses
    # it rather than returning a recovery of nothing.  Stated on the panel, off the log axis.
    axc.annotate("0 surviving:\nread refused", xy=(0.985, 0.055), xycoords="axes fraction",
                 ha="right", va="bottom", fontsize=6.8, color="#b3261e",
                 bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#b3261e", lw=0.7))
    axc.legend(fontsize=6.8, loc="lower left", framealpha=0.9)
    # The dropped fraction the sweep is stated in, as the secondary scale.
    axt = axc.twiny()
    axt.set_xscale("log"); axt.set_xlim(axc.get_xlim())
    axt.set_xticks([256, 64, 16, 4, 1])
    axt.set_xticklabels([f"{100 * (1 - s / F):.0f}" for s in (256, 64, 16, 4, 1)], fontsize=7)
    axt.set_xlabel("channels dropped (%)", fontsize=8, labelpad=2)
    axt.tick_params(length=2)

    wf(fig.add_subplot(gs[1, 1]), W5_disp, f"+ noise (S/N {SNR}) + {pdrop:.0f}% random channel dropout")
    wf(fig.add_subplot(gs[1, 2]), rec6_disp, f"recovery of surviving channels\n{p6:.1f}%")

    fig.suptitle("Entroptics calibration", fontsize=14, fontweight="bold", y=0.975)
    out = Path(__file__).resolve().parent / "calibration.png"
    fig.savefig(out, dpi=150); print(f"wrote {out}   p3={p3:.2f}%  p6={p6:.2f}%  drop={pdrop:.1f}%")


if __name__ == "__main__":
    main()
