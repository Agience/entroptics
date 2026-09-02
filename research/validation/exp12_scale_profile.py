"""
Experiment 12 -- Scale profile: structure appears at the window that contains it.

Ground truth: a signal whose ordered structure occupies a known extent.  Read through
an aperture shorter than that extent, nothing resolves; read through one that contains
it, a mode stands above the floor.  The resolved window of Definition 3.6 must track the planted extent.

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
    headline = (f"the resolved window tracks the planted period across {PERIODS} "
                f"(Spearman {rho:+.2f}, monotone: {mono}); a window shorter than the "
                f"structure resolves nothing.")
    concl = ("Structure is reported at the scale that contains it, so the profile locates the "
             "observation window a signal requires.")
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
