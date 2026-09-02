"""
Experiment 18 -- the nulls under coloured noise, which is the case they are not calibrated for.

Every null in the paper is calibrated on i.i.d. draws (exp3, exp4, exp7), but the ordered axis of
a real record is a correlated process -- that is what section 4 measures.  Marchenko-Pastur, the
Johnstone centring and the Tracy-Widom edge all assume the N rows are independent draws, and the
coherence null of Definition 5.3 tests exchangeability, which serial correlation breaks by
construction.  So the calibration and the intended use disagree, and the size of the disagreement
has never been measured.

This plants NO signal.  It draws AR(1) rows at a known correlation length and asks what the
instrument says about pure coloured noise:

  * K_signal  -- should be 0.  Anything above that is the floor counting correlation as signal.
  * coherence -- SHOULD fire: adjacent rows really are more alike than a random re-ordering.
                 That is the read working, not a false alarm, and it is reported to keep the two
                 apart.

The standard selectors of exp17 are run on the SAME records, because the question that matters is
whether this is a property of THIS floor or of the iid null every one of them is calibrated
against.  A serially correlated field genuinely moves the bulk -- the singular values really are
above the iid edge -- so no method reading against that edge can be right here, and what separates
them is how far each one is wrong.

rho = 1 is the i.i.d. case and is the control row.

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Projection

import common as C
from exp17_rank_baselines import k_gavish_donoho, _wax_kailath

SHAPES = [(200, 200), (300, 120), (40, 600)]
RHOS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]       # rho = 1 is iid (the control)
N_SEEDS = 40
SEED0 = 1800
ALPHA = 0.05


def _ar1_rows(T: int, F: int, rho: float, seed: int) -> np.ndarray:
    """Rows correlated along the ORDERED axis with correlation length rho; channels independent.

    x_t = phi x_{t-1} + e_t with phi = exp(-1/rho), scaled to unit marginal variance so the
    record's amplitude does not move with rho and the floor sees the same scale at every row.
    """
    g = C.rng(seed)
    phi = 0.0 if rho <= 1.0 else float(np.exp(-1.0 / rho))
    e = g.standard_normal((T, F))
    if phi == 0.0:
        return e
    x = np.empty((T, F))
    x[0] = e[0]
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]
    return x * np.sqrt(1.0 - phi ** 2)


def run() -> dict:
    rows = []
    spurious = []
    worst_rate, worst_at = 0.0, None
    for (T, F) in SHAPES:
        for rho in RHOS:
            ks, zs, gd, md, ai = [], [], [], [], []
            for s in range(N_SEEDS):
                W = _ar1_rows(T, F, rho, seed=SEED0 + s)
                pr = Projection(W)
                ks.append(int(pr.K_signal))
                zs.append(float(pr.coherence))
                gd.append(k_gavish_donoho(W))
                md.append(_wax_kailath(W, "mdl"))
                ai.append(_wax_kailath(W, "aic"))
            ks = np.array(ks); zs = np.array(zs)
            false_rate = float(np.mean(ks > 0))          # any resolved mode is a false alarm here
            if false_rate > worst_rate:
                worst_rate, worst_at = false_rate, (T, F, rho)
            avg = lambda v: "n/a" if v[0] is None else round(float(np.mean(v)), 1)
            spurious.append((float(ks.mean()), avg(gd), avg(md), avg(ai)))
            rows.append([f"{T}x{F}", rho, round(float(ks.mean()), 2), round(false_rate, 3),
                         avg(gd), avg(md), avg(ai), round(float(zs.mean()), 2)])

    table = C.md_table(["shape", "rho", "mean K_signal", "P(K_signal > 0)",
                        "mean GD", "mean MDL", "mean AIC", "mean coherence z"], rows)
    # mean spurious count at rho > 1, over the cells where each method is defined
    col_cells = [sp for sp, r in zip(spurious, rows) if r[1] > 1.0]
    def _mean(i):
        v = [c[i] for c in col_cells if c[i] != "n/a"]
        return float(np.mean(v)) if v else float("nan")
    spur = {"K_signal": _mean(0), "GD": _mean(1), "MDL": _mean(2), "AIC": _mean(3)}
    gentlest = min((k for k in spur if not np.isnan(spur[k])), key=lambda k: spur[k])
    iid = [r for r in rows if r[1] == 1.0]
    iid_rate = float(np.mean([r[3] for r in iid]))
    col = [r for r in rows if r[1] > 1.0]
    col_rate = float(np.mean([r[3] for r in col]))

    headline = (
        f"On pure AR(1) noise with NO planted signal, the derived floor resolves a mode in "
        f"{col_rate:.1%} of draws at rho > 1 against {iid_rate:.1%} in the i.i.d. control "
        f"(nominal alpha = {ALPHA}); the worst cell is {worst_rate:.1%} at shape "
        f"{worst_at[0]}x{worst_at[1]}, rho = {worst_at[2]:g}.  This is NOT specific to the "
        f"derived floor: on the same records the standard selectors report "
        + ", ".join(f"{k} {spur[k]:.1f}" for k in ("K_signal", "GD", "MDL", "AIC")
                    if not np.isnan(spur[k]))
        + f" spurious modes on average, so every iid-calibrated selector over-reads and "
        f"{gentlest} over-reads the least.  The coherence read fires as it should -- adjacent rows "
        f"genuinely are more alike than a re-ordering -- so the two must not be read as one "
        f"number.")
    concl = (
        "A serially correlated field genuinely moves the bulk -- the singular values really do sit "
        "above the iid edge -- so this is a property of the null every one of these methods is "
        "calibrated against, not of any one estimator. What the comparison measures is how far "
        "each is wrong when the assumption is broken. "
        "The floor's calibration assumes independent rows and the ordered axis of a real record "
        "does not supply them. What this measures is how far the false-alarm rate moves when that "
        "assumption is broken deliberately, at correlation lengths spanning the range section 4 "
        "reads off real data. A rate at or near the i.i.d. control means the floor tolerates "
        "serial correlation at that shape; a rate above it is the floor counting correlation as "
        "signal, and the number is what a reader needs in order to judge whether it matters for "
        "their records.")

    return dict(
        title="18. The nulls under coloured noise",
        setup=(f"AR(1) rows at rho in {RHOS} (rho = 1 is the i.i.d. control), shapes {SHAPES}, "
               f"{N_SEEDS} seeds each, NO planted signal. Projection(W).K_signal and .coherence."),
        table=table,
        metrics=dict(false_rate_iid=iid_rate, false_rate_coloured=col_rate,
                     worst_rate=worst_rate, mean_spurious=spur, gentlest=gentlest),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
