"""figures.py -- the benchmark figures and summary tables, drawn from the committed outputs.

    python figures.py            # writes fig_*.png; prints the tables the benchmark page shows

Reads only committed outputs -- the ``*.jsonl`` beside it and ``research/validation/RESULTS.md`` --
so a figure changes exactly when its benchmark is re-run.  Needs matplotlib (the ``[figures]``
extra).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "validation" / "RESULTS.md"
FAR = 0.05
BLUE, TEAL, GREY, DARK = "#2a6fdb", "#17becf", "#999999", "#555555"


def rows(name):
    return [json.loads(line) for line in open(HERE / name)]


def md_table(section):
    """The first markdown table under ``## <section>`` in RESULTS.md, as a list of dicts."""
    text = RESULTS.read_text(encoding="utf-8")
    body = text[text.index(f"## {section}"):]
    lines = [ln for ln in body.split("\n") if ln.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    out = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(head):
            break
        out.append(dict(zip(head, cells)))
    return out


def num(x):
    return None if x == "n/a" else float(x)


# ── 1. against the FFT ─────────────────────────────────────────────────────────────────────────
AUTO, FIXED, STREAM = "Entroptics, automatic depth", "Entroptics, fixed depth", "Entroptics, streaming"
PIPE, PIPE_H = "FFT pipeline, unwindowed", "FFT pipeline, Hann"
METHODS = (("a_f_err", AUTO), ("lift_f_err", FIXED), ("rect_f_err", PIPE), ("pipe_f_err", PIPE_H))
MARKS = {AUTO: ("o", BLUE), FIXED: ("s", TEAL), PIPE: ("^", GREY), PIPE_H: ("v", DARK)}


def fft_table():
    """One row per tone: each method's frequency error, or None where the method's nearest estimate
    is off by at least half the gap to the nearest other true tone, or it has none (the tone is
    lost)."""
    out = []
    for r in rows("operator_vs_fft.jsonl"):
        if "tones" not in r:
            continue
        fs = [t["f_true"] for t in r["tones"]]
        for t in r["tones"]:
            others = [abs(t["f_true"] - g) for g in fs if g != t["f_true"]]
            half = min(others) / 2 if others else np.inf
            errs = {name: (t[key] if t[key] is not None and t[key] < half else None)
                    for key, name in METHODS}
            found = {k: v for k, v in errs.items() if v is not None}
            best = min(found.values()) if found else None
            winners = [k for k, v in found.items() if best is not None and f"{v:.1e}" == f"{best:.1e}"]
            out.append(dict(case=r["case"], f=t["f_true"], errs=errs, winners=winners))
    return out


def fig_fft(tab):
    labels = [f"{t['case']}\n{t['f']:g}" for t in tab]
    y = np.arange(len(tab))[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(tab) + 1.6))
    for name, (m, c) in MARKS.items():
        xs = [t["errs"][name] for t in tab]
        ax.scatter([x for x in xs if x is not None], [yy for yy, x in zip(y, xs) if x is not None],
                   marker=m, color=c, s=46, label=name, zorder=3)
        lost = [yy for yy, x in zip(y, xs) if x is None]
        off = {AUTO: 0.19, FIXED: 0.27, PIPE: 0.37, PIPE_H: 0.5}[name]
        ax.scatter([off] * len(lost), lost, marker=m, facecolors="none", edgecolors=c, s=46, zorder=3)
    ax.axvline(1.5e-1, color="#bbbbbb", lw=0.8)
    ax.text(3e-1, len(tab) - 0.2, "lost", ha="center", va="bottom", fontsize=8, color="#666666")
    ax.set_xscale("log")
    ax.set_xlim(3e-10, 7e-1)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("frequency error, cycles per sample (log scale; lower is better)")
    ax.set_title("Where each method places each tone")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.grid(axis="x", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(HERE / "fig_fft.png", dpi=150)
    plt.close(fig)


def lost(err, f, fs):
    """An estimate is lost when it lies nearer another true tone than half their separation."""
    others = [abs(f - g) for g in fs if g != f]
    return err is None or (others and err >= min(others) / 2)


DECAY = (("a", AUTO), ("lift", FIXED), ("fft", "FFT pipeline"))


def decay_table():
    """Each decaying tone's decay rate as each method reads it, or None where its frequency is lost."""
    out = []
    for r in rows("operator_vs_fft.jsonl"):
        if "tones" not in r:
            continue
        fs = [t["f_true"] for t in r["tones"]]
        for t in r["tones"]:
            if t["decay_true"] <= 0:
                continue
            got = {name: (None if lost(t[f"{k}_f_err"], t["f_true"], fs) else t[f"{k}_decay"])
                   for k, name in DECAY if k != "fft"}
            # the FFT pipeline under whichever window reads this tone's decay closer to the truth
            ffts = [t[f"{w}_decay"] for w in ("pipe", "rect") if not lost(t[f"{w}_f_err"], t["f_true"], fs)]
            got["FFT pipeline"] = min(ffts, key=lambda v: abs(v - t["decay_true"])) if ffts else None
            errs = {k: abs(v - t["decay_true"]) for k, v in got.items() if v is not None}
            best = min(errs, key=errs.get) if errs else None
            out.append(dict(case=r["case"], f=t["f_true"], true=t["decay_true"], got=got, best=best))
    return out


def count_table():
    """Components counted on each record against the truth, a real tone counting as two (its
    positive- and negative-frequency pair); the FFT pipeline's peaks are one-sided, so doubled, and
    it is credited with whichever window counts closer to the truth."""
    out = []
    for r in rows("operator_vs_fft.jsonl"):
        if "tones" in r:
            true = 2 * len(r["tones"])
            pipe = min((2 * r["pipe_K"], 2 * r["rect_K"]), key=lambda k: abs(k - true))
            out.append(dict(case=r["case"], true=true, auto=r["a_K"], pipe=pipe))
    return out


COST_LINES = {  # method in count_vs_fft.jsonl -> (label, colour, style)
    "fft": ("FFT alone (a spectrum only)", "#bbbbbb", ":"),
    "pipeline": ("FFT pipeline", GREY, "--"),
    "read": ("Entroptics", BLUE, "-"),
}


def cost_table():
    """{F: {method: {T: (instructions, se)}}} from ``count_vs_fft.jsonl`` -- each read's retired
    instructions per call (valgrind), which do not carry the host's load or clock -- and
    {F: {T: (K read, K pipeline)}}, the counts the same reads return, from the like-for-like rows of
    ``operator_vs_fft.jsonl`` (the pipeline's one-sided peaks taken as components, two each)."""
    t, k = {}, {}
    for r in rows("count_vs_fft.jsonl"):
        if "method" in r:
            t.setdefault(r["F"], {}).setdefault(r["method"], {})[r["T"]] = (r["instructions"], r["se"])
    for r in rows("operator_vs_fft.jsonl"):
        if r.get("timing") == "like for like" and r.get("rep") == 0:
            kr = r.get("streaming_K", r.get("fixed_depth_K"))
            k.setdefault(r["F"], {})[r["T"]] = (kr, 2 * r["pipeline_K"])
    return t, k


def _mi(c):
    """An instruction count in millions, with its uncertainty where that is more than a tenth of it
    (a call of a few hundred thousand instructions is at the measurement's own noise)."""
    v, se = c[0] / 1e6, c[1] / 1e6
    return f"{v:.3g} ± {se:.2g}" if se > 0.1 * abs(v) else f"{v:.3g}"


def fig_cost(tab):
    t, _ = tab
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, F in zip(axes, (1, 16)):
        for m, (name, c, ls) in COST_LINES.items():
            pts = t[F].get(m)
            if not pts:
                continue
            Ts = sorted(pts)
            ax.plot(Ts, [pts[T][0] for T in Ts], ls, color=c, marker="o", ms=4,
                    label=(FIXED if F == 1 else STREAM) if m == "read" else name,
                    lw=2.2 if m == "read" else 1.4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("record length T")
        ax.set_title("one channel" if F == 1 else f"{F} channels")
        ax.grid(color="#eeeeee")
    axes[0].set_ylabel("instructions per call (valgrind; log scale; lower is cheaper)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Cost of a count, frequencies and decay rates at a 5% false-alarm rate")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(HERE / "fig_cost.png", dpi=150)
    plt.close(fig)


# ── 2. counting components: against the standard selectors ─────────────────────────────────────
SELECTORS = (("K_signal", "Entroptics"), ("GD", "Gavish–Donoho"), ("AIC", "AIC"), ("MDL", "MDL"))


def rank_tables():
    """Exact-count accuracy on planted components (experiment 17), and the spurious count on
    serially correlated noise with nothing planted (experiment 18, rho > 1).  Both are averaged
    over the cases every selector covers (AIC and MDL do not apply to square records), so each
    column compares like with like."""
    t17 = md_table("17. K_signal against the standard rank selectors")
    t17 = [r for r in t17 if all(num(r[f"acc {k}"]) is not None for k, _ in SELECTORS)]
    acc = {name: (float(np.mean([num(r[f"acc {k}"]) for r in t17])), len(t17), len(t17))
           for k, name in SELECTORS}
    t18 = md_table("18. The nulls under coloured noise")
    t18 = [r for r in t18 if float(r["rho"]) > 1 and all(num(r[f"mean {k}"]) is not None for k, _ in SELECTORS)]
    spur = {name: float(np.mean([num(r[f"mean {k}"]) for r in t18])) for k, name in SELECTORS}
    return acc, spur


def fig_rank(acc):
    names = [n for _, n in SELECTORS]
    vals = [acc[n][0] for n in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(names, vals, color=[BLUE, GREY, GREY, GREY])
    for b, n in zip(bars, names):
        a, used, tot = acc[n]
        note = f"{a:.3f}" + ("" if used == tot else f"\n({used} of {tot} cases)")
        ax.text(b.get_x() + b.get_width() / 2, a + 0.005, note, ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.8, 1.07)
    ax.set_ylabel("share of cases counted exactly")
    ax.set_title("Counting planted components (higher is better)")
    ax.grid(axis="y", color="#eeeeee")
    fig.tight_layout()
    fig.savefig(HERE / "fig_rank.png", dpi=150)
    plt.close(fig)


# ── 3. the noise test ──────────────────────────────────────────────────────────────────────────
def null_table():
    """Per noise family: the default floor's false-alarm rate pooled over the record shapes, its
    binomial standard error, the closed-form edge's pooled rate, and whether the default holds the
    level -- at most ``FAR`` beyond its noise, at the gate's family-wise z (``entroptics.gate``).
    Pooled, not the worst shape: an exact test runs AT its level in every shape, and the worst of
    six noisy rates sits above it by chance."""
    from statistics import NormalDist
    fam = {}
    for r in rows("screen_null.jsonl"):
        if "null" in r:
            fam.setdefault(r["null"], []).append(r)
    z = NormalDist().inv_cdf(1.0 - FAR / max(len(fam), 1))
    out = []
    for name, rs in fam.items():
        n = sum(r["records"] for r in rs)
        p = sum(r["false_alarm_rate"] * r["records"] for r in rs) / n
        p_mp = sum(r["false_alarm_rate_mp"] * r["records"] for r in rs) / n
        se = (FAR * (1 - FAR) / n) ** 0.5
        out.append((name, p, se, p_mp, p <= FAR + z * se))
    return out


def detect_table():
    out = {}
    for r in rows("screen_null.jsonl"):
        if "detect" in r:
            out.setdefault(r["detect"], []).append((r["found_rate"], r["exact_rate"]))
    return out


def fig_null(tab):
    fams = [t[0] for t in tab][::-1]
    rate = np.array([t[1] for t in tab][::-1])
    err = np.array([t[2] for t in tab][::-1])
    rate_mp = np.array([t[3] for t in tab][::-1])
    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(7.5, 0.4 * len(fams) + 1.4))
    ax.barh(y + 0.18, 100 * rate, 0.34, xerr=100 * err, color=BLUE, label="Entroptics (exact floor)")
    ax.barh(y - 0.18, 100 * rate_mp, 0.34, color=GREY, label="closed-form edge (mp)")
    ax.axvline(100 * FAR, color="#d62728", lw=1.2, ls="--", label="the 5% you allow")
    ax.set_yticks(y, fams)
    ax.set_xlabel("records in which a component is wrongly reported, % (six shapes pooled, 1200 each)")
    ax.set_title("Components reported in pure noise (lower is better)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "fig_null.png", dpi=150)
    plt.close(fig)


def main():
    ft, ct = fft_table(), cost_table()
    dt_, kt = decay_table(), count_table()
    acc, spur = rank_tables()
    nt, dt = null_table(), detect_table()
    fig_fft(ft)
    fig_cost(ct)
    fig_rank(acc)
    fig_null(nt)

    print("| selector | counted exactly (planted components) | spurious components on correlated noise |")
    print("|---|---|---|")
    for _, name in SELECTORS:
        a, used, tot = acc[name]
        sp = spur[name]
        cnt = f"{a:.3f}" + ("" if used == tot else f" ({used} of {tot} cases)")
        print(f"| {name} | {cnt} | {'n/a' if sp is None else f'{sp:.1f}'} |")
    print()
    print("| noise | Entroptics (exact floor) | closed-form edge (mp) | within the 5% you allow? |")
    print("|---|---|---|---|")
    for fam, v, se, v_mp, ok in nt:
        print(f"| {fam} | {100 * v:.1f}% ± {100 * se:.1f} | {100 * v_mp:.1f}% | {'yes' if ok else 'no'} |")
    print()
    print("| planted signal | found | counted exactly |")
    print("|---|---|---|")
    for k, v in dt.items():
        f = [a for a, _ in v]
        e = [b for _, b in v]
        print(f"| {k} | {100 * min(f):.0f}–{100 * max(f):.0f}% | {100 * min(e):.0f}–{100 * max(e):.0f}% |")
    print()
    print("| record | tone | " + " | ".join(n for _, n in METHODS) + " | best |")
    print("|---|---|" + "---|" * (len(METHODS) + 1))
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
    print("| record | tone | true decay | " + " | ".join(n for _, n in DECAY) + " | best |")
    print("|---|---|---|" + "---|" * (len(DECAY) + 1))
    for t in dt_:
        cells = [("lost" if v is None else f"{v:.4f}") for v in (t["got"][n] for _, n in DECAY)]
        cells = [f"**{c}**" if n == t["best"] else c for c, (_, n) in zip(cells, DECAY)]
        print(f"| {t['case']} | {t['f']:g} | {t['true']:g} | " + " | ".join(cells) + f" | {t['best'] or '—'} |")
    print()
    print(f"| record | true | {AUTO} | FFT pipeline |")
    print("|---|---|---|---|")
    for r in kt:
        c = [str(r["auto"]), str(r["pipe"])]
        d = [abs(r["auto"] - r["true"]), abs(r["pipe"] - r["true"])]
        c = [f"**{x}**" if e == min(d) else x for x, e in zip(c, d)]
        print(f"| {r['case']} | {r['true']} | " + " | ".join(c) + " |")
    print()
    t, kk = ct
    for F in (1, 16):
        print(f"F = {F}, instructions per call (M)")
        print("| T | FFT alone (a spectrum only) | FFT pipeline | Entroptics | Entroptics / pipeline |")
        print("|---|---|---|---|---|")
        for T in sorted(t[F]["read"]):
            f, p, r = (t[F][m][T] for m in ("fft", "pipeline", "read"))
            kr, kp = kk.get(F, {}).get(T, (None, None))
            print(f"| {T} | {_mi(f)} | {_mi(p)} (K {kp}) | {_mi(r)} (K {kr}) | {r[0] / p[0]:.3g} |")
        print()


if __name__ == "__main__":
    main()
