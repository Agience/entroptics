"""
Experiment 17 -- K_signal against the standard rank selectors.

Experiment 3 shows K_signal recovers a planted mode count.  The question that answers on its
own is "compared to what".  This runs the established selectors on the SAME planted signals at
the SAME seeds, so the comparison is like for like:

  * K_signal      -- the derived Tracy-Widom floor of section 8, no fitted constant.
  * GD 4/sqrt(3)  -- Gavish & Donoho's optimal hard threshold for singular values, unknown-noise
                     form: tau = omega(beta) * y_median, omega from the published polynomial.
                     At beta = 1 the known-noise constant is 4/sqrt(3); this is its practical
                     sibling, which is what a user without a known sigma would actually run.
  * MDL, AIC      -- Wax & Kailath model-order selection on the sample covariance eigenvalues,
                     the canonical answer to "how many sources".
  * oracle        -- the true K.  Not a method: the ceiling every row is scored against.

Nothing here is tuned.  Each selector is run in its standard form and reported as measured,
including where it beats K_signal.

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Projection

import common as C

SHAPES = [(200, 200), (600, 40), (40, 600), (300, 120)]
KS = [1, 3, 5]
SNRS = [1.0, 2.0, 4.0]          # the calibrated operating range of exp3, plus one above it
N_SEEDS = 30
SEED0 = 303                     # SAME seeds as exp3, so the signals are identical


def _omega(beta: float) -> float:
    """Gavish-Donoho's optimal-hard-threshold coefficient for unknown noise (published cubic)."""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def k_gavish_donoho(W: np.ndarray) -> int:
    """Optimal hard threshold at the median singular value, unknown-noise form."""
    T, F = W.shape
    beta = min(T, F) / max(T, F)
    sv = np.linalg.svd(W, compute_uv=False)
    tau = _omega(beta) * float(np.median(sv))
    return int(np.sum(sv > tau))


def _wax_kailath(W: np.ndarray, criterion: str):
    """Wax-Kailath AIC / MDL on the sample second-moment eigenvalues.

    p variables, n snapshots, eigenvalues l_1 >= ... >= l_p.  For each candidate k the criterion
    contrasts the geometric and arithmetic means of the p-k smallest eigenvalues against a
    free-parameter penalty k(2p-k).  Returns the minimising k, or None where the method does not
    apply.

    Two things this has to get right or it reports nonsense.  The model is ZERO-MEAN, so the
    uncentred second moment is the matrix: centring costs a degree of freedom and leaves a zero
    eigenvalue that sits in every tail, which drives mean(log lambda) down for all k and makes the
    criterion prefer k ~ p-1.  And the derivation needs n > p; at n <= p the covariance is
    singular by construction and no k is meaningful, so this returns None rather than a number.
    """
    T, F = W.shape
    p, n = (F, T) if F <= T else (T, F)           # variables x snapshots, whichever way round
    if n <= p:
        return None                               # not applicable: singular by construction
    S = (W.conj().T @ W) / n if F <= T else (W @ W.conj().T) / n
    lam = np.sort(np.real(np.linalg.eigvalsh(S)))[::-1]
    if lam[-1] <= 0:
        return None
    best, best_k = np.inf, 0
    for k in range(0, p):
        tail = lam[k:]
        m = p - k
        if m <= 1:                                # log(g/a) is identically 0 at m = 1
            break
        log_g = float(np.mean(np.log(tail)))
        a = float(np.mean(tail))
        ll = -n * m * (log_g - np.log(a))         # -log likelihood ratio, >= 0
        free = k * (2 * p - k)
        pen = 0.5 * free * np.log(n) if criterion == "mdl" else float(free)
        val = ll + pen
        if val < best:
            best, best_k = val, k
    return int(best_k)


def run() -> dict:
    methods = ["K_signal", "GD", "MDL", "AIC"]
    hits = {m: [] for m in methods}
    rows = []

    for (T, F) in SHAPES:
        for K in KS:
            for snr in SNRS:
                got = {m: [] for m in methods}
                for s in range(N_SEEDS):
                    W = C.planted_lowrank(T, F, K, snr, seed=SEED0 + s)
                    got["K_signal"].append(Projection(W).K_signal)
                    got["GD"].append(k_gavish_donoho(W))
                    got["MDL"].append(_wax_kailath(W, "mdl"))
                    got["AIC"].append(_wax_kailath(W, "aic"))
                acc, mean = {}, {}
                for m in methods:
                    vals = [v for v in got[m] if v is not None]
                    if not vals:                       # method does not apply at this shape
                        acc[m], mean[m] = None, None
                        continue
                    acc[m] = float(np.mean(np.array(vals) == K))
                    mean[m] = float(np.mean(vals))
                    hits[m].append(acc[m])
                fmt = lambda v, nd: "n/a" if v is None else round(v, nd)
                rows.append([f"{T}x{F}", K, snr]
                            + [fmt(acc[m], 3) for m in methods]
                            + [fmt(mean[m], 2) for m in methods])

    table = C.md_table(["shape", "K_true", "snr"]
                       + [f"acc {m}" for m in methods]
                       + [f"mean {m}" for m in methods], rows)
    overall = {m: float(np.mean(hits[m])) for m in methods if hits[m]}
    covered = {m: len(hits[m]) for m in methods}
    ranked = sorted(overall.items(), key=lambda kv: -kv[1])
    winner, w_acc = ranked[0]
    ours = overall["K_signal"]

    verdict = ("K_signal is the most accurate of the four"
               if winner == "K_signal" else
               f"{winner} is more accurate than K_signal ({w_acc:.3f} vs {ours:.3f})")

    headline = (
        f"On the exp3 planted signals at the same seeds, exact-count accuracy: "
        + ", ".join(f"{m} {overall[m]:.3f} ({covered[m]}/{len(rows)} cells)" for m, _ in ranked)
        + f".  {verdict}.  AIC/MDL are undefined where the snapshot count does not exceed the "
        f"variable count, and are reported n/a there rather than guessed.")
    concl = (
        "The comparison is like for like -- identical signals, identical seeds, each selector in "
        "its standard form with nothing tuned. K_signal carries no fitted constant and no known "
        "sigma; GD's unknown-noise form estimates its scale from the median singular value, and "
        "Wax-Kailath's AIC/MDL assume an iid noise floor over p variables with n snapshots, which "
        "is strained on the wide shapes where p > n.")

    return dict(
        title="17. K_signal against the standard rank selectors",
        setup=(f"The exp3 planted signals: K in {KS} at snr in {SNRS} (units of the iid bulk "
               f"edge), shapes {SHAPES}, {N_SEEDS} seeds each, seeded identically to exp3. "
               f"Selectors: derived floor (K_signal), Gavish-Donoho optimal hard threshold "
               f"(unknown-noise form), Wax-Kailath MDL and AIC."),
        table=table,
        metrics=dict(exact_accuracy=overall, winner=winner),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
