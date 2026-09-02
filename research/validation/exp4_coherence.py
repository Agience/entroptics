"""
Experiment 4 -- Coherence detects order and is null-calibrated.

(a) Order detection.  A smoothly ordered signal (a few low-frequency temporal modes,
    so adjacent rows are alike) vs the same rows randomly permuted.  Projection(W).coherence
    is a deterministic z-score against the exact row-permutation null, reading
    large for the ordered signal and ~0 for the permuted one.

(b) Null calibration.  Pure iid Gaussian noise across many shapes/seeds: the z-score
    has mean ~ 0 and, since both its null mean and null variance are exact
    (Def 5.3, the Cliff-Ord/Mantel second moment), an empirical std ~ 1.  The one-sided
    P(z>2) sits near 0.023; the small residual is the permutation distribution's own
    tail skew at small N (the standardisation is exact, the tail shape is not normal).

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Projection

import common as C

# (a) order detection
ORD_SHAPE = (240, 40)
N_MODES = 3
N_PAIRS = 40

# (b) null calibration
NULL_SHAPES = [(120, 30), (200, 50), (80, 200), (300, 60), (150, 150)]
N_NULL = 400


def run() -> dict:
    # ── (a) ordered vs permuted ──
    z_ord, z_perm = [], []
    for s in range(N_PAIRS):
        W = C.ordered_smooth(*ORD_SHAPE, n_modes=N_MODES, seed=404 + s)
        z_ord.append(Projection(W).coherence)
        g = C.rng(50000 + s)
        Wp = W[g.permutation(W.shape[0])]
        z_perm.append(Projection(Wp).coherence)
    z_ord = np.array(z_ord); z_perm = np.array(z_perm)

    # ── (b) iid null calibration ──
    zs = []
    k = 0
    for (T, F) in NULL_SHAPES:
        for _ in range(N_NULL):
            zs.append(Projection(C.rng(70000 + k).standard_normal((T, F))).coherence)
            k += 1
    zs = np.array(zs)
    p_gt2 = float(np.mean(zs > 2.0))

    rows_a = [["ordered (smooth)", round(float(z_ord.mean()), 2), round(float(z_ord.std()), 2),
               round(float(z_ord.min()), 2)],
              ["rows permuted", round(float(z_perm.mean()), 3), round(float(z_perm.std()), 2),
               round(float(z_perm.min()), 2)]]
    table_a = C.md_table(["signal", "mean z", "std z", "min z"], rows_a)

    rows_b = [[f"{len(zs)} iid draws", round(float(zs.mean()), 3), round(float(zs.std()), 3),
               round(p_gt2, 4), 0.0228]]
    table_b = C.md_table(["null sample", "mean z", "std z", "P(z>2)", "N(0,1) target"], rows_b)

    table = "**(a) order detection**\n\n" + table_a + "\n\n**(b) iid null calibration**\n\n" + table_b

    headline = (
        f"Ordered signals read z={z_ord.mean():.1f} (min {z_ord.min():.1f}) while the "
        f"SAME rows permuted read z={z_perm.mean():.2f} (~0); over {len(zs)} iid-noise "
        f"draws the null has mean {zs.mean():.3f}, std {zs.std():.3f}, and P(z>2)={p_gt2:.4f} "
        f"(N(0,1) target 0.023).")
    concl = ("Coherence sharply separates ordered from permuted (an order-of-magnitude z "
             "gap) and is null-centred at ~0 with std ~1 (the exact Cliff-Ord/Mantel null "
             "variance, Def 5.3); P(z>2) sits near the 0.023 N(0,1) target, the small "
             "residual being the permutation distribution's tail skew, not the standardisation.")

    return dict(
        title="4. Coherence detects order and is null-calibrated",
        setup=(f"(a) smooth ordered signal {ORD_SHAPE} vs its own row permutation, {N_PAIRS} pairs. "
               f"(b) iid Gaussian across shapes {NULL_SHAPES}, {len(zs)} draws."),
        table=table,
        metrics=dict(z_ordered_mean=float(z_ord.mean()), z_ordered_min=float(z_ord.min()),
                     z_permuted_mean=float(z_perm.mean()), z_permuted_std=float(z_perm.std()),
                     null_mean=float(zs.mean()), null_std=float(zs.std()), p_z_gt2=p_gt2),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
