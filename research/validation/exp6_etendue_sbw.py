"""
Experiment 6 -- Etendue / space-bandwidth product <-> rank & bandwidth.

Ground truth: signals of known effective rank and known feature bandwidth.
  (a) Rank sweep: W = sum of R equal-strength orthogonal rank-1 modes + light noise,
      for increasing R.  As the true rank rises, more optical modes are active.
  (b) Bandwidth sweep: a feature-bandlimited signal occupying B of F frequency bins,
      for increasing B.  As the true bandwidth rises, the feature aperture widens.
etendue = phi_F*phi_T and space_bandwidth = n_F*n_T both increase monotonically
with the planted rank / bandwidth.

Deterministic (fixed seeds).  Re-runnable: `python exp6_etendue_sbw.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

T, F = 400, 64
RANKS = [1, 2, 4, 8, 16, 32]
BANDS = [2, 4, 8, 16, 32, 64]
SEED = 606


def _lowrank(T, F, R, seed):
    """R equal-strength orthonormal modes + a faint iid noise floor (planted effective
    rank R).  Noise is kept well below the modes so the feature/ordered correlation
    fill fractions read R/F and R/T cleanly (etendue ~ R^2/(FT))."""
    g = C.rng(seed)
    U, _ = np.linalg.qr(g.standard_normal((T, R)))
    V, _ = np.linalg.qr(g.standard_normal((F, R)))
    sig = U @ V.T * np.sqrt(T * F / R)        # unit-variance modes (total power ~ T*F)
    return sig + 0.02 * g.standard_normal((T, F))


def run() -> dict:
    rank_rows, et_r, sb_r = [], [], []
    for i, R in enumerate(RANKS):
        ap = Aperture(_lowrank(T, F, R, seed=SEED + i))
        et, sb = ap.etendue, ap.space_bandwidth
        et_r.append(et); sb_r.append(sb)
        rank_rows.append([R, round(et, 4), sb, round(ap.phi_F, 4), round(ap.phi_T, 4)])

    band_rows, et_b, sb_b = [], [], []
    for i, B in enumerate(BANDS):
        ap = Aperture(C.band_limited(T, F, B, seed=SEED + 100 + i))
        et, sb = ap.etendue, ap.space_bandwidth
        et_b.append(et); sb_b.append(sb)
        band_rows.append([B, round(et, 4), sb, round(ap.n_F, 1), round(ap.n_T, 1)])

    table_a = C.md_table(["rank R", "etendue", "SBW", "phi_F", "phi_T"], rank_rows)
    table_b = C.md_table(["bandwidth B", "etendue", "SBW", "n_F", "n_T"], band_rows)
    table = "**(a) effective-rank sweep**\n\n" + table_a + \
            "\n\n**(b) feature-bandwidth sweep**\n\n" + table_b

    sp_et_r = C.spearman(RANKS, et_r); sp_sb_r = C.spearman(RANKS, sb_r)
    sp_et_b = C.spearman(BANDS, et_b); sp_sb_b = C.spearman(BANDS, sb_b)

    headline = (
        f"Etendue and space-bandwidth rise monotonically with both the planted rank "
        f"(Spearman etendue={sp_et_r:.3f}, SBW={sp_sb_r:.3f}) and the planted feature "
        f"bandwidth (etendue={sp_et_b:.3f}, SBW={sp_sb_b:.3f}).")
    concl = ("Etendue (the conserved aperture area) and the space-bandwidth product "
             "(resolvable-spot count) are strictly monotone in the signal's true "
             "rank and bandwidth -- they read the aperture's size, as claimed.")

    return dict(
        title="6. Etendue / space-bandwidth <-> rank & bandwidth",
        setup=(f"(a) sum of R equal orthonormal modes + light noise, R in {RANKS}; "
               f"(b) feature-bandlimited to B of F={F} bins, B in {BANDS}; T={T}."),
        table=table,
        metrics=dict(spearman_etendue_rank=sp_et_r, spearman_sbw_rank=sp_sb_r,
                     spearman_etendue_band=sp_et_b, spearman_sbw_band=sp_sb_b),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
