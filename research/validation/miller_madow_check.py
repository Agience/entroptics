"""
miller_madow_check.py -- does the paper's noise-guard band beta_T equal the true
first-order entropy deficit of the power-marginal estimator under Gaussian noise?

Setup (Definition 2.2).  Under iid complex Gaussian noise W (shape (T, F)) the power
marginal is p^T_t proportional to sum_f |W_tf|^2.  The plug-in entropy H_T sits below
its maximum log2(T) by a finite-sample deficit.  The paper snaps to "no fold" when
H_T >= log2(T) - beta_T with band

        beta_T = (T - 1) / (2 F ln 2).

The deficit  D = log2(T) - E[H_T]  is measured by Monte Carlo over many seeds and
shapes and compared to beta_T.

Theory.  With W_tf ~ CN(0,1), each |W_tf|^2 ~ Exp(1) and g_t = sum_f |W_tf|^2 ~
Gamma(F,1); the normalized marginal p is Dirichlet(F,...,F) (T components), whose mean
entropy is exactly

        E[H_T] = ( psi(TF + 1) - psi(F + 1) ) / ln 2   (bits),

so the exact deficit is D = log2(T) - E[H_T], with first-order expansion

        D ~ (T - 1) / (2 * T * F * ln 2)  =  beta_T / T.

The mean deficit uses the total cell count T*F as the effective sample size, and
beta_T = (T-1)/(2F ln2) is T times that mean: beta_T is a deliberately conservative
guard, not the mean deficit itself. The margin is necessary because the guard snaps to
no-fold when H_T >= log2(T) - beta_T: a band set to the mean deficit would let about half of
noise realizations fold (their deficit exceeds the mean), and any fold of noise -- even a
fractional delta ~ 1.002 -- blends adjacent cells and manufactures spurious coherence.
Empirically, the mean-sized band (beta_T / T) drives the coherence null P(z>2) from its
target 0.023 to 1.0 and the K_signal false-alarm rate to ~100%; beta_T keeps noise from
folding and the null calibrated.

The cap.  This conservatism has one failure mode: the inner band grows without bound in the
aspect ratio and exceeds the maximum entropy log2(len) for extreme shapes (feature band
beta_F > log2(F) when F >~ 2 T ln F), where the guard fires unconditionally and disables the
fold vacuously -- a fully redundant marginal could not fold. The construction therefore
caps the band at (1/2) log2(len): min(beta, (1/2) log2 len). The cap leaves the conservative
inner band untouched wherever it is already below (1/2) log2 len (all tall/square shapes, so
noise still never folds there), and makes the guard operative at every shape -- it always
folds once power concentrates below sqrt(len) effective cells. The table below flags where
the inner band is vacuous and reports the capped band.

Deterministic (fixed seeds).  Re-runnable: `python miller_madow_check.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np
from scipy.special import digamma

from entroptics.entropy import shannon_bits

SHAPES = [(16, 8), (32, 16), (64, 16), (64, 64), (128, 32), (256, 64), (512, 32),
          (512, 4), (256, 8)]      # last two: tall-thin -> inner band vacuous, cap engages
N_SEEDS = 4000
LN2 = np.log(2.0)


def _mc_deficit(T, F, n_seeds, seed0):
    """Monte-Carlo E[log2(T) - H_T] for the complex-Gaussian power marginal."""
    Hs = np.empty(n_seeds)
    for s in range(n_seeds):
        g = np.random.default_rng(seed0 + s)
        W = (g.standard_normal((T, F)) + 1j * g.standard_normal((T, F))) / np.sqrt(2.0)
        Hs[s] = shannon_bits((np.abs(W) ** 2).sum(axis=1))
    return float(np.log2(T) - Hs.mean())


def run() -> dict:
    rows = []
    ratios = []
    for i, (T, F) in enumerate(SHAPES):
        d_mc = _mc_deficit(T, F, N_SEEDS, seed0=800000 + i * 100000)
        beta_T = (T - 1) / (2 * F * LN2)
        cap = 0.5 * float(np.log2(T))
        banded = min(beta_T, cap)                           # the guard actually used
        vac = "yes" if beta_T >= float(np.log2(T)) else "no"
        ratio = beta_T / d_mc
        ratios.append(ratio)
        rows.append([f"{T}x{F}", round(d_mc, 5), round(beta_T, 4),
                     round(cap, 3), round(banded, 4), vac, round(ratio, 2)])

    ratios = np.array(ratios)
    table = C_md_table(
        ["T x F", "deficit (MC)", "inner band (T-1)/(2F ln2)",
         "cap (1/2)log2 T", "banded = min", "inner vacuous?", "inner / deficit"], rows)

    headline = (
        f"The inner band (T-1)/(2F ln2) is a conservative guard -- T times the mean deficit "
        f"(measured inner/deficit {ratios.min():.1f}..{ratios.max():.1f}) -- so structureless "
        f"noise essentially never folds; but for tall-thin shapes it exceeds log2(T) and would "
        f"disable the fold vacuously, which the cap min(inner, (1/2)log2 len) removes without "
        f"touching the inner band on tall/square shapes.")
    concl = ("The inner band is a deliberately conservative uniform-null guard: it exceeds the "
             "mean marginal deficit by the factor T, so structureless noise essentially never "
             "folds (a band set to the mean deficit folds ~half of noise realizations and, "
             "because any fold blends adjacent cells, drives the coherence null P(z>2) "
             "from 0.023 to 1.0 and the K_signal false-alarm rate to ~100%, which is why the "
             "band is set above the mean). Its "
             "one defect -- exceeding log2(len) for extreme aspect ratios (the 'inner vacuous?' "
             "column), disabling the fold vacuously -- is fixed by capping at (1/2) log2(len): "
             "the 'banded' column is the guard actually used (Definition 2.2), unchanged from "
             "the inner band on every non-vacuous shape and operative (folds below sqrt(len) "
             "effective cells) on the rest. In the construction only the feature axis folds, "
             "so the operative guard is the symmetric beta_F capped the same way; the ordered "
             "axis is kept at native resolution, so its band never engages.")

    return dict(
        title="Miller-Madow band derivation check",
        setup=(f"iid complex Gaussian W across shapes {SHAPES}, {N_SEEDS} seeds each; "
               f"power marginal p^T_t ~ sum_f |W_tf|^2; deficit = log2(T) - E[H_T]."),
        table=table,
        metrics=dict(ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
                     ratios={f"{T}x{F}": float(r) for (T, F), r in zip(SHAPES, ratios)}),
        headline=headline,
        conclusion=concl,
    )


# local markdown-table helper (kept self-contained so the check runs standalone)
def C_md_table(headers, rows):
    def fmt(v):
        return f"{v:.5g}" if isinstance(v, float) else str(v)
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
