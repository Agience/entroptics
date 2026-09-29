"""
Experiment 12 -- Scale profile: structure appears at the window that contains it.

Ground truth: a signal whose ordered structure occupies a known extent.  The resolved window of
Definition 3.6 -- the shortest window in which a mode stands above the floor -- is read against the
planted period.  A partial period of a carrier shared across the channels is itself a shared mode,
so a strong enough carrier can resolve in a window shorter than its period; the table reports where
that happens rather than assuming it cannot.

Deterministic (fixed seeds).  Re-runnable: `python exp12_scale_profile.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

T, F = 512, 24
PERIODS = [16, 32, 64, 128]
SEED = 1212


def _oscillation(period, seed):
    """A single ordered mode of known period over a noise floor: its structure needs a
    window of at least one period to be visible at all."""
    g = C.rng(seed)
    t = np.arange(T)
    carrier = np.sin(2 * np.pi * t / period)[:, None]
    return carrier @ (2.5 * g.standard_normal((1, F))) + g.standard_normal((T, F))


def run():
    rows, res_windows = [], []
    for i, p in enumerate(PERIODS):
        W = _oscillation(p, SEED + i)
        prof = Aperture(W, window=None).scale_profile()
        rw = int(prof.resolved_window)
        dw = int(prof.dominant_window)
        res_windows.append(rw)
        ks = np.asarray(prof.K_signal)
        rows.append([p, rw, dw, int(ks.max()), int((ks >= 1).sum()), len(ks)])
    table = C.md_table(
        ["planted period", "resolved window", "dominant window", "max K", "windows resolving", "windows swept"],
        rows)
    rho = C.spearman(PERIODS, res_windows)
    mono = all(a <= b for a, b in zip(res_windows, res_windows[1:]))
    shorter = [p for p, w in zip(PERIODS, res_windows) if w < p]
    headline = (f"the resolved windows for planted periods {PERIODS} are {res_windows} "
                f"(Spearman {rho:+.2f}, monotone: {mono}); {len(shorter)} of {len(PERIODS)} "
                f"resolve in a window shorter than their period, since a partial period shared "
                f"across {F} channels is already a mode at this amplitude.")
    concl = ("The resolved window rises with the planted extent, and the shortest window that "
             "resolves is set by the carrier's strength as well as its period.")
    return dict(
        title="12. Scale profile: structure versus observation window",
        setup=(f"a single ordered mode of known period over a unit noise floor, T={T}, F={F}; "
               f"periods {PERIODS}, trailing windows log-spaced to T."),
        table=table,
        metrics=dict(spearman_resolved_window_vs_period=rho, monotone=mono,
                     resolved_windows=res_windows),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
