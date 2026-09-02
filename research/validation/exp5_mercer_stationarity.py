"""
Experiment 5 -- Mercer ratio rho as a stationarity diagnostic (Prop 4.7 as a tool).

Ground truth: a stationary record (one AR(1) coefficient throughout) and a
nonstationary record of the same size (a regime switch: the AR(1) coefficient jumps
partway through, changing the correlation length).  A sub-window slides along the
ordered axis, reading Aperture(sub-window).mercer.ratio at each position.  Prop 4.7
says rho is O(1) and constant for a stationary process; a drift/jump flags
nonstationarity.  The coefficient of variation (CV = std/mean) of rho across windows
quantifies the separation -- low for stationary, high for the regime switch.

Deterministic (fixed seeds).  Re-runnable: `python exp5_mercer_stationarity.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

T, F = 1200, 32
WIN = 300
STEP = 30
PHI_STATIONARY = 0.85
PHI_SWITCH = [0.5, 0.97]     # two regimes -> correlation length jumps at the seam
SEED = 505
N_SEEDS = 12                 # average the CV separation over seeds (not a single realization)


def _rho_track(W):
    rhos, centers = [], []
    for start in range(0, T - WIN + 1, STEP):
        sub = W[start:start + WIN]
        r = Aperture(sub, window=None).mercer.ratio
        if np.isfinite(r):
            rhos.append(float(r)); centers.append(start + WIN // 2)
    return np.array(centers), np.array(rhos)


def _cv(x):
    x = np.asarray(x, float)
    return float(np.std(x) / np.mean(x)) if np.mean(x) != 0 else float("nan")


def run() -> dict:
    # Average the CV separation over N_SEEDS realizations (not a single record).
    cvs_s, cvs_n = [], []
    for s in range(N_SEEDS):
        Ws = C.ar1(T, F, PHI_STATIONARY, seed=SEED + 2 * s)
        Wn = C.regime_switch(T, F, PHI_SWITCH, seed=SEED + 2 * s + 1)
        cvs_s.append(_cv(_rho_track(Ws)[1]))
        cvs_n.append(_cv(_rho_track(Wn)[1]))
    cvs_s, cvs_n = np.array(cvs_s), np.array(cvs_n)
    cv_s, cv_n = float(np.mean(cvs_s)), float(np.mean(cvs_n))
    sd_s, sd_n = float(np.std(cvs_s)), float(np.std(cvs_n))

    rows = [["stationary (phi=%.2f)" % PHI_STATIONARY, round(cv_s, 4), round(sd_s, 4)],
            ["regime switch (phi %.2f->%.2f)" % (PHI_SWITCH[0], PHI_SWITCH[1]),
             round(cv_n, 4), round(sd_n, 4)]]
    table = C.md_table(["record", "mean CV(rho)", "std CV over seeds"], rows)

    ratio = cv_n / cv_s if cv_s > 0 else float("inf")
    headline = (
        f"Across sliding sub-windows the Mercer ratio rho is nearly constant for the "
        f"stationary record (mean CV={cv_s:.3f} over {N_SEEDS} seeds) but drifts/jumps "
        f"{ratio:.0f}x more for the regime switch (mean CV={cv_n:.3f}).")
    concl = ("The Mercer ratio is a working stationarity diagnostic: constant rho (low CV) "
             "under stationarity, a marked drift (high CV) at a regime change -- "
             f"substantiating Prop 4.7 recast as a diagnostic ({ratio:.0f}x mean-CV separation "
             f"over {N_SEEDS} seeds, not a single realization).")

    return dict(
        title="5. Mercer ratio rho as a stationarity diagnostic",
        setup=(f"Stationary AR(1) phi={PHI_STATIONARY} vs a regime switch phi {PHI_SWITCH}, "
               f"both (T,F)=({T},{F}); {N_SEEDS} seeds each.  Slide a window of {WIN} (step "
               f"{STEP}); read Aperture(window).mercer.ratio at each position; report the "
               f"seed-averaged CV(rho)."),
        table=table,
        metrics=dict(cv_stationary=cv_s, cv_nonstationary=cv_n, cv_ratio=ratio,
                     cv_stationary_sd=sd_s, cv_nonstationary_sd=sd_n),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
