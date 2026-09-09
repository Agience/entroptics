"""
Experiment 19 -- A state with a hole in it is not a state.

Every other read in the construction treats an unobserved cell as ABSENT and reads what is
there.  An operator cannot: it is fitted to PAIRS (x_t, x_{t+1}), and a state missing a
component is not the state the system was in.

Two ways of handling the hole, on the same records and the same seeds:

  zeroed    substitute 0 for the missing cell.  The transition INTO that component then looks
            like decay toward zero, so every rate reads faster than it truly is -- and it gets
            worse the more is dropped, always in the same direction.
  carried   fill the hole with what the record itself says belongs there: x_t = A x_{t-1}, the
            operator read off the observed cells predicting the missing ones, iterated to a
            fixed point.  This is what `Aperture(W).rates()` does with a NaN.

Ground truth: the planted system's slowest rate, -log(0.985).  The carried read must not depend
on how much was dropped; the zeroed read must, badly.

The control that keeps it straight: a record with NO dynamics must not ACQUIRE any at any
dropout.  Filling holes from a fitted operator could invent a mode; it must not.

Section 9 of the paper quotes these values.  They previously appeared only in a docstring in
`src/tests/test_dynamics.py`, which asserts a bound on them rather than writing them down --
the pattern `research/validation/check_paper.py` exists to catch.

Deterministic (fixed seeds).  Re-runnable: `python exp19_operator_gaps.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

T, F = 2000, 12
NOISE = 0.1
#: the planted slow modes; the slowest rate is -log(0.985)
MAGS, THETAS = (0.985, 0.960), (0.30, 0.11)
DROPS = (0.0, 0.05, 0.20, 0.35, 0.50)
SEED = 1
DROP_SEED = 0
NOISE_T, NOISE_F, NOISE_SEED = 1500, 12, 3


def _planted_system():
    """A linear system with two known slow modes, kept excited by process noise."""
    g = C.rng(SEED)
    A = np.zeros((F, F))
    for i, (m, a) in enumerate(zip(MAGS, THETAS)):
        A[2 * i:2 * i + 2, 2 * i:2 * i + 2] = C.rot_block(m, a)
    A[4:, 4:] = np.diag(g.uniform(0.2, 0.5, F - 4))
    x, tr = g.standard_normal(F), []
    for _ in range(T):
        x = A @ x + NOISE * g.standard_normal(F)
        tr.append(x.copy())
    return np.array(tr), float(np.sort(-np.log(np.array(MAGS)))[0])


def _slowest(W):
    return float(np.sort(np.asarray(Aperture(W, window=None).rates().alpha))[0])


def run() -> dict:
    W0, truth = _planted_system()
    g = C.rng(DROP_SEED)
    masks = {q: (g.random(W0.shape) < q if q else np.zeros(W0.shape, bool)) for q in DROPS}

    rows = []
    carried, zeroed = {}, {}
    for q in DROPS:
        m = masks[q]
        Wc = W0.copy(); Wc[m] = np.nan       # absent -> carried by the record's own operator
        Wz = W0.copy(); Wz[m] = 0.0          # absent -> substituted zero
        carried[q] = _slowest(Wc)
        zeroed[q] = _slowest(Wz)
        rows.append([f"{q:.0%}", round(carried[q], 4), round(zeroed[q], 4),
                     round(carried[q] / truth, 3), round(zeroed[q] / truth, 3)])
    table = C.md_table(["dropped", "slowest rate (carried)", "slowest rate (zeroed)",
                        "carried / truth", "zeroed / truth"], rows)

    # the control: a record with no operator must not acquire one
    n = C.rng(NOISE_SEED).standard_normal((NOISE_T, NOISE_F))
    rows_n = []
    for q in (0.0, 0.20, 0.50):
        X = n.copy()
        if q:
            X[C.rng(NOISE_SEED + 1).random(X.shape) < q] = np.nan
        slow = _slowest(X)
        rows_n.append([f"{q:.0%}", round(slow, 4), round(float(np.exp(-slow)), 4),
                       "no" if np.exp(-slow) < 0.5 else "YES -- invented"])
    table_n = C.md_table(["dropped", "slowest rate", "mode magnitude", "persistent mode?"], rows_n)

    carried_spread = max(carried.values()) - min(carried.values())
    zero_factor = zeroed[0.35] / zeroed[0.0]

    table = ("**(a) the same record, two ways of treating the hole**\n\n" + table
             + "\n\n**(b) the control: white noise must not acquire a mode**\n\n" + table_n)

    headline = (
        f"On a planted operator whose slowest rate is {truth:.4f}, carrying the gaps leaves the "
        f"read independent of how much was dropped ({carried[0.0]:.4f} at none, "
        f"{carried[0.5]:.4f} at half, a spread of {carried_spread:.4f} across 0-50%), while "
        f"substituting zero biases every rate upward and worsens monotonically -- "
        f"{zeroed[0.0]:.4f} at none to {zeroed[0.35]:.4f} at 35%, a factor of {zero_factor:.0f}. "
        f"White noise acquires no persistent mode at any dropout.")
    concl = (
        "Substituting zero for an unobserved cell is not a neutral choice: it asserts that the "
        "system was AT zero, and a transition into zero reads as decay. The bias is one-sided and "
        "grows with the fraction dropped, so it cannot be corrected by reweighting the accumulated "
        "sums -- the pairs are wrong, not the weights. Carrying the gap with the record's own "
        "operator removes the dependence on the dropped fraction, and the noise control shows it "
        "does not manufacture a mode where none exists.")

    return dict(
        title="19. A state with a hole in it is not a state",
        setup=(f"A linear system with planted slow modes at magnitudes {MAGS}, T={T}, F={F}, "
               f"process noise {NOISE}; cells dropped at {[f'{q:.0%}' for q in DROPS]} and read "
               f"two ways -- carried (NaN, the library's own fill) and zeroed. Control: white "
               f"noise at ({NOISE_T}, {NOISE_F})."),
        table=table,
        metrics=dict(truth=truth, carried=carried, zeroed=zeroed,
                     carried_spread=carried_spread, zeroed_factor_at_35pct=zero_factor),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
