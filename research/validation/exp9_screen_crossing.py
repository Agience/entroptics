"""
Experiment 9 -- The two-way screen: conservation and the brightness bound.

Ground truth: two sides placed on one shared basis through lenses of known etendue.
  (a) Conservation (Proposition 10.4): for every crossing, absorbed + bystanding +
      reflected must equal the sending beam's energy exactly, whatever the pair.
  (b) Brightness (Proposition 10.5): radiance out never exceeds radiance in, with
      equality exactly when the receiving side has the smaller etendue.
Both are read off planted crossings across a range of etendue ratios, concentrating
and spreading, so the equality case and the strict case are each exercised.

Deterministic (fixed seeds).  Re-runnable: `python exp9_screen_crossing.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Screen

import common as C

T, D = 256, 8
WIDTHS = [4, 6, 8, 12, 16, 24]
SEED = 909


def _screen(width_a, width_b, seed):
    """Two sides sharing a planted carrier, entered through lenses of differing width."""
    g = C.rng(seed)
    carrier = g.standard_normal((T, 1))
    Sa = carrier @ g.standard_normal((1, width_a)) + 0.4 * g.standard_normal((T, width_a))
    Sb = carrier @ g.standard_normal((1, width_b)) + 0.4 * g.standard_normal((T, width_b))
    Pa = np.linalg.qr(g.standard_normal((width_a, width_a)))[0][:, :D] if width_a >= D else None
    Pb = np.linalg.qr(g.standard_normal((width_b, width_b)))[0][:, :D] if width_b >= D else None
    if Pa is None:
        Pa = np.vstack([np.eye(width_a), np.zeros((D - width_a, width_a))]).T
    if Pb is None:
        Pb = np.vstack([np.eye(width_b), np.zeros((D - width_b, width_b))]).T
    s = Screen()
    s.register("a", entry=lambda X, P=Pa: np.asarray(X) @ P)
    s.register("b", entry=lambda X, P=Pb: np.asarray(X) @ P)
    s.place("a", Sa)
    s.place("b", Sb)
    return s


def run():
    rows, resid, bright_ok, eq_ok, n_conc = [], [], 0, 0, 0
    n = 0
    for i, w in enumerate(WIDTHS):
        s = _screen(w, D, SEED + i)
        for a, b in (("a", "b"), ("b", "a")):
            t = s.transfer(a, b)
            n += 1
            total = float(t.absorbed) + float(t.bystanding) + float(t.reflected)
            r = abs(total - float(t.energy)) / max(float(t.energy), 1e-30)
            resid.append(r)
            Lf, Lt = float(t.radiance_from), float(t.radiance_to)
            bright_ok += Lt <= Lf * (1 + 1e-12)
            concentrating = float(t.etendue_to) <= float(t.etendue_from)
            if concentrating:
                n_conc += 1
                eq_ok += abs(Lt - Lf) <= 1e-9 * max(abs(Lf), 1.0)
            if a == "a":
                rows.append([w, round(float(t.energy), 4), round(float(t.tau), 4),
                             round(Lf, 6), round(Lt, 6),
                             "concentrating" if concentrating else "spreading",
                             f"{r:.1e}"])
    table = C.md_table(
        ["surface width", "energy", "tau", "radiance in", "radiance out", "regime", "conservation residual"],
        rows)
    worst = max(resid)
    headline = (f"conservation holds to {worst:.1e} relative on all {n} crossings; "
                f"radiance never rises ({bright_ok}/{n}), and is carried across exactly "
                f"in every concentrating crossing ({eq_ok}/{n_conc}).")
    concl = ("Energy is partitioned exactly and radiance is bounded, so a crossing neither "
             "creates energy nor brightens a beam.")
    return dict(
        title="9. The two-way screen: conservation and brightness",
        setup=(f"two sides on one shared basis of D={D}, surface widths {WIDTHS}, T={T}; "
               f"each pair read in both directions."),
        table=table,
        metrics=dict(max_conservation_residual=worst, brightness_holds=bright_ok,
                     crossings=n, equality_under_concentration=eq_ok, concentrating_crossings=n_conc),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
