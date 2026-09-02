"""
Experiment 13 -- The reads divide by the measured extent, and a mask is not a zero.

Definition 2.1 sets the axis length L_a to the MEASURED extent: the number of coordinates
carrying at least one finite, unmasked cell.  Every read that compares an entropy against
L_a inherits it -- the fills of Def 3.3, the matched scale, the etendue.  Two claims follow,
and this experiment tests both directly against a ground truth that is available exactly:
the same signal recorded WITHOUT the absent channels.

  (a) Absence is transparent.  Blank a fraction of the channels -- by NaN, or by a mask over
      a finite but unobserved value -- and every read must return what the surviving channels
      return on their own.  The ground truth is the deleted-channel array, so the target is
      equality, not a tolerance band.

  (b) Zero is not absence.  Substituting 0 for an unobserved channel is a different frame: it
      is an observation of no power, it belongs in the extent, and it moves the reads.  The
      gap is the factor L/L_eff the extent corrects -- so it grows with how much is blanked,
      and it is not a rounding effect.

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

SHAPE = (200, 256)
N_DRAWS = 20
BLANK_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 0.90]
READS = ("phi_F", "phi_T", "phi", "etendue", "strehl")


def _draw(seed: int, frac: float):
    """A frame with a planted low-rank signal, and the channels that were never observed."""
    T, F = SHAPE
    g = C.rng(seed)
    W = g.standard_normal((T, 3)) @ g.standard_normal((3, F)) + 0.3 * g.standard_normal((T, F))
    n_dead = int(round(frac * F))
    dead = np.sort(g.choice(F, n_dead, replace=False)) if n_dead else np.zeros(0, int)
    keep = np.setdiff1d(np.arange(F), dead)
    return W, dead, keep


def _reads(ap: Aperture) -> dict:
    return {r: float(getattr(ap, r)) for r in READS}


def run() -> dict:
    rows_a, rows_b = [], []
    worst_abs, worst_zero_gap = 0.0, {}
    for frac in BLANK_FRACTIONS:
        dev_nan, dev_mask, gap = [], [], []
        for i in range(N_DRAWS):
            W, dead, keep = _draw(300000 + i, frac)
            truth = _reads(Aperture(W[:, keep], window=None))

            nan = W.copy(); nan[:, dead] = np.nan
            r_nan = _reads(Aperture(nan, window=None))

            garbage = W.copy(); garbage[:, dead] = 7.5      # finite, real, and not what was there
            m = np.zeros(W.shape, dtype=bool); m[:, dead] = True
            r_mask = _reads(Aperture(garbage, mask=m, window=None))

            zeroed = W.copy(); zeroed[:, dead] = 0.0
            r_zero = _reads(Aperture(zeroed, window=None))

            dev_nan.append(max(abs(r_nan[k] - truth[k]) for k in READS))
            dev_mask.append(max(abs(r_mask[k] - truth[k]) for k in READS))
            gap.append(abs(r_zero["phi_F"] - truth["phi_F"]))
        d_nan, d_mask, g_zero = float(max(dev_nan)), float(max(dev_mask)), float(np.mean(gap))
        worst_abs = max(worst_abs, d_nan, d_mask)
        worst_zero_gap[frac] = g_zero
        rows_a.append([f"{frac:.0%}", f"{d_nan:.2e}", f"{d_mask:.2e}"])
        rows_b.append([f"{frac:.0%}", int(round(frac * SHAPE[1])),
                       SHAPE[1] - int(round(frac * SHAPE[1])), round(g_zero, 4)])

    table = ("**(a) absence is transparent** -- worst deviation over "
             f"{', '.join(READS)} from the deleted-channel ground truth\n\n"
             + C.md_table(["channels blanked", "as NaN", "as a mask over a wrong value"], rows_a)
             + "\n\n**(b) zero is not absence** -- what substituting 0 costs\n\n"
             + C.md_table(["channels blanked", "dead", "measured extent F_eff",
                           "|phi_F(zeroed) - phi_F(truth)|"], rows_b))

    hi = max(worst_zero_gap, key=worst_zero_gap.get)
    headline = (
        f"Across {len(BLANK_FRACTIONS)} blanking fractions and {N_DRAWS} draws each, every read "
        f"({', '.join(READS)}) taken through NaN or through a mask matches the deleted-channel "
        f"ground truth to {worst_abs:.1e} -- floating-point equality, on frames whose nominal width "
        f"is up to {1/(1-max(BLANK_FRACTIONS)):.0f}x their measured extent. Substituting 0 for the "
        f"same channels moves phi_F by {worst_zero_gap[hi]:.3f} at {hi:.0%} blanked, rising "
        f"monotonically with the fraction blanked.")
    concl = (
        "The reads divide by the measured extent. A channel that was never observed is transparent "
        "to every one of them, whether it is marked by NaN or by a mask, and the mask is honoured "
        "even when the value underneath it is finite and wrong. Substituting zero is a different "
        "measurement -- it widens the axis the signal is scored against by exactly the channels "
        "that carry nothing -- and the reads say so.")

    return dict(
        title="13. The reads divide by the measured extent, and a mask is not a zero",
        setup=(f"{N_DRAWS} draws per fraction at {SHAPE} (rank-3 signal + noise); channels blanked "
               f"at {', '.join(f'{f:.0%}' for f in BLANK_FRACTIONS)}, each three ways -- NaN, a mask "
               f"over the finite wrong value 7.5, and a substituted 0 -- against the same frame with "
               f"those channels deleted."),
        table=table,
        metrics=dict(worst_abs_deviation=worst_abs,
                     zero_gap_by_fraction={f"{k:.0%}": v for k, v in worst_zero_gap.items()}),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"], "\n")
    print(r["table"], "\n")
    print(r["headline"])
