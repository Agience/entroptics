"""
Experiment 10 -- The observable lift: a nonlinear trajectory becomes a linear operator.

Ground truth: trajectories whose one-step map is not linear in the frame's own coordinates.
Read raw, the operator has nothing to advance.  Read in delay coordinates (Definition 9.6) the
same instrument forecasts the held-out tail, and the forecast is what the lift is for.

The control is the trajectory in a random order: identical shape, identical marginals, no
dynamics.  It is the control that carries the claim, because the resolved-mode COUNT does not --
the count rises with the observable's dimension d*F and with whiteness, so a shuffled trajectory
can resolve more modes than the real one.  Forecast skill separates them; a count does not.

Deterministic (fixed seeds).  Re-runnable: `python exp10_koopman_lift.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import koopman_lift, delay_embed

import common as C

T = 600
DEPTH = 24
HOLD = 100
SEED = 1010


def _nonlinear(kind, seed):
    """Trajectories with no linear one-step map on their own coordinates."""
    g = C.rng(seed)
    t = np.linspace(0, 60, T)
    if kind == "logistic":
        x = np.zeros(T); x[0] = 0.31
        for i in range(T - 1):
            x[i + 1] = 3.72 * x[i] * (1.0 - x[i])
        return np.stack([x, np.roll(x, 1)], 1) + 0.01 * g.standard_normal((T, 2))
    if kind == "oscillator":
        return np.stack([np.sin(t) * np.cos(0.13 * t), np.cos(t) ** 3], 1) \
            + 0.01 * g.standard_normal((T, 2))
    raise ValueError(kind)


def _forecast_error(W, d=DEPTH, hold=HOLD):
    """One-step error on a held-out tail, relative to repeating the last frame.

    Scored on the NEW frame only: Z_{t+1} shares d-1 of its d blocks with Z_t, so scoring the
    whole delay vector measures that shift."""
    F = W.shape[1]
    dyn = koopman_lift(W[:-hold], d=d)                  # fitted on the head alone
    Z = np.asarray(delay_embed(W, d))
    err = base = 0.0
    for i in range(len(Z) - hold - 1, len(Z) - 1):
        pred = np.asarray(dyn.predict(Z[i]))[-F:]
        err += float(np.sum((pred - Z[i + 1][-F:]) ** 2))
        base += float(np.sum((Z[i][-F:] - Z[i + 1][-F:]) ** 2))
    return err / max(base, 1e-30)


def run():
    rows, ratios = [], []
    for i, kind in enumerate(("logistic", "oscillator")):
        W = _nonlinear(kind, SEED)
        g = C.rng(SEED + 100 + i)
        Wsh = W[g.permutation(len(W))]                  # order destroyed, marginals kept
        e_real, e_shuf = _forecast_error(W), _forecast_error(Wsh)
        k_real = int(koopman_lift(W, d=DEPTH).resolved())
        k_shuf = int(koopman_lift(Wsh, d=DEPTH).resolved())
        ratios.append(e_shuf / max(e_real, 1e-30))
        rows.append([kind, round(e_real, 4), round(e_shuf, 4), round(e_shuf / e_real, 1),
                     k_real, k_shuf])
    table = C.md_table(
        ["trajectory", "forecast error", "shuffled control", "ratio", "modes", "modes (shuffled)"],
        rows)
    worst = min(ratios)
    headline = (f"in delay coordinates the operator forecasts the held-out tail at "
                f"{rows[0][1]} and {rows[1][1]} of persistence, against {rows[0][2]} and "
                f"{rows[1][2]} for the same trajectories in a random order -- a separation of "
                f"{worst:.0f}x or better.")
    concl = ("The lift buys a forecast, and the shuffled control is what shows it: the resolved "
             "count does not separate dynamics from disorder, since it rises with the observable's "
             "dimension and with whiteness.")
    return dict(
        title="10. The observable lift: nonlinear trajectory to linear operator",
        setup=(f"logistic map and an amplitude-modulated oscillator, T={T}, delay depth d={DEPTH}, "
               f"one-step forecast on a held-out tail of {HOLD}, against the same trajectory shuffled."),
        table=table,
        metrics=dict(forecast_error=[r[1] for r in rows], shuffled=[r[2] for r in rows],
                     min_ratio=worst, modes=[r[4] for r in rows], modes_shuffled=[r[5] for r in rows]),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
