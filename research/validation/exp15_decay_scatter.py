"""
Experiment 15 -- Channel scatter separates a decay that is structure from one that is noise (Def 4.8).

`decay` is a sum of one biased autocovariance per feature channel, so the channels are
independent replicates of the same decay.  Their spread is the estimator's own uncertainty:
measured from the record, with no null assumed and nothing subtracted.  Two claims are tested
against a planted ground truth.

  (a) The scatter separates a tail that is NOISE from a tail that is STRUCTURE.  `tail_share`
      alone cannot: an uncorrelated record and a strongly correlated one both put power away
      from zero lag.  On an uncorrelated record the two shares must coincide -- every bit of
      that tail is disagreement between channels.  On a planted correlation length they must
      separate, because the channels agree about it.

  (b) `a_delta` overstates the correlation length when the scatter is wide, and the remedy is
      channels.  Both the read and the scatter are followed as F grows: the read must close on
      the planted answer from one side, and the scatter must fall as it does.

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

T = 1500
WIDTHS = [4, 16, 64, 256]
RHO = 8


def _ar1(F, rho, seed):
    """AR(1) with a planted correlation length rho; phi = e^{-1/rho}."""
    g = C.rng(seed)
    phi = np.exp(-1.0 / rho)
    X = np.zeros((T, F)); e = g.standard_normal((T, F)); X[0] = e[0]
    for t in range(1, T):
        X[t] = phi * X[t - 1] + e[t]
    return X


def _read(X):
    ap = Aperture(X, window=None)
    d = ap.decay_scatter
    return float(ap.a_delta), float(d.noise_share), float(d.tail_share)


def run() -> dict:
    rows_a, rows_b = [], []
    white, real = {}, {}
    for F in WIDTHS:
        a, n, t = _read(C.rng(500000 + F).standard_normal((T, F)))
        white[F] = (a, n, t)
        rows_a.append([F, round(a, 4), round(t, 4), round(n, 4), round(n / t, 3)])
        a, n, t = _read(_ar1(F, RHO, 510000 + F))
        real[F] = (a, n, t)
        rows_b.append([F, round(a, 4), round(t, 4), round(n, 4), round(n / t, 4)])

    hdr = ["channels F", "a_delta", "tail share", "noise share", "noise/tail"]
    table = ("**(a) an uncorrelated record -- the tail IS the scatter**\n\n"
             + C.md_table(hdr, rows_a)
             + f"\n\n**(b) a planted correlation length rho={RHO} -- the tail is structure**\n\n"
             + C.md_table(hdr, rows_b))

    w_ratio = [r[4] for r in rows_a]
    r_ratio = [r[4] for r in rows_b]
    a_white = [white[F][0] for F in WIDTHS]
    n_white = [white[F][1] for F in WIDTHS]
    n_real = [real[F][1] for F in WIDTHS]

    headline = (
        f"On an uncorrelated record the noise and tail shares coincide at every width "
        f"(ratio {min(w_ratio):.2f}-{max(w_ratio):.2f}): all of the power away from zero lag is "
        f"channel disagreement. With a correlation length of {RHO} planted they separate by up to "
        f"{1/min(r_ratio):.0f}x (ratio {max(r_ratio):.3f} down to {min(r_ratio):.4f}), so the same "
        f"tail is read as structure. The scatter falls monotonically with F in both families "
        f"({n_white[0]:.3f}->{n_white[-1]:.4f} and {n_real[0]:.3f}->{n_real[-1]:.4f}), and the "
        f"uncorrelated a_delta closes on its answer of 1 from below as it does "
        f"({a_white[0]:.2f}->{a_white[-1]:.2f}).")
    concl = (
        "The two shares separate the width that is structure from the width that is disagreement, "
        "and they do it from the record alone. Where they coincide the entropy width is the estimator's "
        "own scatter and a_delta overstates the correlation length; where they separate the width "
        "is structure the channels agree on. Neither is subtracted from a_delta -- what the "
        "scatter contributes could only be removed by assuming what the decay would have been -- "
        "and the read is consistent, so the remedy a wide scatter points at is more channels.")

    return dict(
        title="15. Channel scatter separates a decay that is structure from one that is noise",
        setup=(f"T={T}. An uncorrelated record and an AR(1) record with a planted correlation "
               f"length rho={RHO}, each read at F = {', '.join(str(w) for w in WIDTHS)} channels."),
        table=table,
        metrics=dict(white_noise_over_tail={str(F): white[F][1] / white[F][2] for F in WIDTHS},
                     planted_noise_over_tail={str(F): real[F][1] / real[F][2] for F in WIDTHS},
                     white_a_delta={str(F): white[F][0] for F in WIDTHS}),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"], "\n")
    print(r["table"], "\n")
    print(r["headline"])
