"""figures.py -- the benchmark figures and summary tables, drawn from the committed outputs.

    python figures.py            # writes fig_null.png, fig_fft.png, fig_cost.png; prints the tables

Reads only the committed ``*.jsonl`` beside it, so a figure changes exactly when its benchmark is
re-run.  Needs matplotlib (the ``[figures]`` extra).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FAR = 0.05
CURRENT = "0.2.6"          # the version the committed outputs were produced by


def rows(name):
    return [json.loads(line) for line in open(HERE / name)]


# ── the screen's null ──────────────────────────────────────────────────────────────────────────
def null_table():
    def worst(name):
        out = {}
        for r in rows(name):
            if "null" in r:
                out[r["null"]] = max(out.get(r["null"], 0.0), r["false_alarm_rate"])
        return out
    new, old = worst("screen_null.jsonl"), worst("screen_null_0.2.3.jsonl")
    return [(fam, old[fam], new[fam]) for fam in new]


def detect_table():
    """Per planted signal: the exact-rank rate over every shape and level, on 0.2.3 and CURRENT."""
    def load(name):
        out = {}
        for r in rows(name):
            if "detect" in r:
                out.setdefault(r["detect"], []).append(r["exact_rate"])
        return out
    new, old = load("screen_null.jsonl"), load("screen_null_0.2.3.jsonl")
    return [(k, old[k], new[k]) for k in new]


def fig_null(tab):
    fams = [t[0] for t in tab][::-1]
    old = np.array([t[1] for t in tab][::-1])
    new = np.array([t[2] for t in tab][::-1])
    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(fams) + 1.4))
    ax.barh(y + 0.2, old, 0.38, color="#c0c0c0", label="0.2.3")
    ax.barh(y - 0.2, new, 0.38, color="#2a6fdb", label=CURRENT)
    ax.axvline(FAR, color="#d62728", lw=1.2, ls="--", label=f"requested rate ({FAR:g})")
    ax.set_yticks(y, fams)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("false-alarm rate on pure noise (worst shape, 200 records each)")
    ax.set_title("Structure claimed in pure noise (lower is better)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_null.png", dpi=150)
    plt.close(fig)


# ── against the FFT ────────────────────────────────────────────────────────────────────────────
METHODS = (("a_f_err", "operator_read"), ("lift_f_err", "koopman_lift"), ("fft_f_err", "FFT"),
           ("hann_f_err", "FFT (Hann)"))


def fft_table():
    """One row per tone: each method's frequency error, or None where the method's nearest estimate
    lies closer to a different true tone than to this one (the tone is lost)."""
    out = []
    for r in rows("operator_vs_fft.jsonl"):
        if "tones" not in r:
            continue
        fs = [t["f_true"] for t in r["tones"]]
        for t in r["tones"]:
            others = [abs(t["f_true"] - g) for g in fs if g != t["f_true"]]
            half = min(others) / 2 if others else np.inf
            errs = {name: (t[key] if t[key] < half else None) for key, name in METHODS}
            found = {k: v for k, v in errs.items() if v is not None}
            best = min(found.values()) if found else None
            winners = [k for k, v in found.items() if best is not None and f"{v:.1e}" == f"{best:.1e}"]
            out.append(dict(case=r["case"], f=t["f_true"], decay=t.get("decay_true"),
                            errs=errs, winners=winners))
    return out


def fig_fft(tab):
    labels = [f"{t['case']}\n{t['f']:g}" for t in tab]
    y = np.arange(len(tab))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(tab) + 1.6))
    marks = {"operator_read": ("o", "#2a6fdb"), "koopman_lift": ("s", "#17becf"),
             "FFT": ("^", "#999999"), "FFT (Hann)": ("v", "#555555")}
    for name, (m, c) in marks.items():
        xs = [t["errs"][name] for t in tab]
        ax.scatter([x for x in xs if x is not None], [yy for yy, x in zip(y, xs) if x is not None],
                   marker=m, color=c, s=46, label=name, zorder=3)
        lost = [yy for yy, x in zip(y, xs) if x is None]
        off = {"operator_read": 0.19, "koopman_lift": 0.27, "FFT": 0.37, "FFT (Hann)": 0.5}[name]
        ax.scatter([off] * len(lost), lost, marker=m, facecolors="none", edgecolors=c, s=46, zorder=3)
    ax.axvline(1.5e-1, color="#bbbbbb", lw=0.8)
    ax.text(3e-1, len(tab) - 0.2, "lost", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.set_xlim(3e-10, 7e-1)
    ax.set_xscale("log")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("frequency error, cycles per sample (log scale; lower is better)")
    ax.set_title("Where each method places each tone")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.grid(axis="x", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(HERE / "fig_fft.png", dpi=150)
    plt.close(fig)


# ── cost ───────────────────────────────────────────────────────────────────────────────────────
def cost_table():
    t = {}
    for r in rows("operator_vs_fft.jsonl"):
        if "timing" not in r:
            continue
        T = r["T"]
        if r["timing"].startswith("dynamics") and r.get("F") == 16:
            t.setdefault("Aperture(W).dynamics().modes()", {})[T] = r["operator_ms"]
            t.setdefault("np.fft.rfft", {})[T] = r["fft_ms"]
        elif r["timing"].startswith("koopman_lift"):
            t.setdefault("koopman_lift(x, 16).modes()", {})[T] = r["operator_ms"]
        elif r["timing"].startswith("operator_read"):
            t.setdefault("operator_read(x)", {})[T] = r["operator_ms"]
        elif r["timing"].startswith("Aperture(W).optics()"):
            t.setdefault("Aperture(W).optics()", {})[T] = r["ms"]
    return t


def fig_cost(tab):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style = {"np.fft.rfft": ("#555555", "--"), "Aperture(W).dynamics().modes()": ("#2a6fdb", "-"),
             "koopman_lift(x, 16).modes()": ("#17becf", "-"), "operator_read(x)": ("#9467bd", "-"),
             "Aperture(W).optics()": ("#ff7f0e", ":")}
    for name, pts in tab.items():
        Ts = sorted(pts)
        c, ls = style[name]
        ax.plot(Ts, [pts[T] for T in Ts], ls, color=c, marker="o", label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("record length T (F = 16 where it applies)")
    ax.set_ylabel("time per call, ms (log scale)")
    ax.set_title("Cost against record length (lower is faster)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color="#eeeeee")
    fig.tight_layout()
    fig.savefig(HERE / "fig_cost.png", dpi=150)
    plt.close(fig)


def main():
    nt, dt, ft, ct = null_table(), detect_table(), fft_table(), cost_table()
    fig_null(nt)
    fig_fft(ft)
    fig_cost(ct)
    print(f"| noise | 0.2.3 | {CURRENT} | holds the 5% level on {CURRENT} |")
    print("|---|---|---|---|")
    for fam, o, n in nt:
        print(f"| {fam} | {o:.3f} | **{n:.3f}** | {'yes' if n <= FAR else 'no'} |" if n < o
              else f"| {fam} | **{o:.3f}** | {n:.3f} | {'yes' if n <= FAR else 'no'} |")
    print()
    print(f"| planted signal | rank counted exactly, 0.2.3 | rank counted exactly, {CURRENT} | better on average |")
    print("|---|---|---|---|")
    for k, eo, en in dt:
        mo, mn = round(float(np.mean(eo)), 2), round(float(np.mean(en)), 2)
        win = CURRENT if mn > mo else ("0.2.3" if mo > mn else "same")
        print(f"| {k} | {min(eo):.2f}–{max(eo):.2f} (mean {mo:.2f}) | {min(en):.2f}–{max(en):.2f} (mean {mn:.2f}) | {win} |")
    print()
    print("| record | tone | operator_read | koopman_lift | FFT | FFT (Hann) | best |")
    print("|---|---|---|---|---|---|---|")
    last = None
    for t in ft:
        cells = []
        for _, name in METHODS:
            v = t["errs"][name]
            s = "lost" if v is None else f"{v:.1e}"
            cells.append(f"**{s}**" if name in t["winners"] else s)
        rec = t["case"] if t["case"] != last else ""
        last = t["case"]
        print(f"| {rec} | {t['f']:g} | " + " | ".join(cells) + f" | {', '.join(t['winners']) or '—'} |")
    print()
    Ts = sorted({T for pts in ct.values() for T in pts})
    print("| T | " + " | ".join(ct) + " | fastest |")
    print("|---|" + "---|" * (len(ct) + 1))
    for T in Ts:
        vals = {k: ct[k].get(T) for k in ct}
        fast = min((v, k) for k, v in vals.items() if v is not None)[1]
        print(f"| {T} | " + " | ".join(f"{v:.2g} ms" if v is not None else "" for v in vals.values())
              + f" | {fast} |")


if __name__ == "__main__":
    main()
