"""
Experiment 14 -- A read is a property of the signal, not of the units it was recorded in.

Whitening (Def 8.1) rescales each channel by its own robust MAD, so every read downstream of it
is scale-free by construction: multiplying a record by a constant must not move anything.  That
invariance is only real if no test anywhere in the path compares a DIMENSIONAL quantity against a
fixed number -- a MAD carries the record's units, and a fixed cut on one is a statement about units.

  (a) The same signal, over 36 orders of magnitude of recording scale.  Every read is compared
      against the same signal at unit scale; the target is equality, not a tolerance band.

  (b) The regime a fixed floor destroys.  A read at strain scale (1e-21) is not a corner case:
      it is what an instrument reports before anyone rescales it.

  (c) Uniform quantization, 16 down to 2 bits.  Quantization noise is real noise and the derived
      floor should measure it; what it must not do is diverge, which is what happens when a
      channel whose samples collapse onto one level is divided by a manufactured scale.

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture, Projection

import common as C

SHAPE = (300, 64)
RANK = 3
SCALES = [1e12, 1e6, 1e3, 1.0, 1e-6, 1e-15, 1e-21, 1e-30]
BITS = [16, 10, 8, 6, 4, 3, 2]
READS = ("phi", "phi_T", "phi_F", "etendue", "strehl", "a_delta", "focus")


def _signal(seed: int):
    g = C.rng(seed)
    T, F = SHAPE
    return g.standard_normal((T, RANK)) @ g.standard_normal((RANK, F)) \
        + 0.5 * g.standard_normal((T, F))


def _reads(W) -> dict:
    ap = Aperture(W, window=None)
    p = Projection(W)
    out = {r: float(getattr(ap, r)) for r in READS}
    out["K_signal"] = float(p.K_signal)
    out["coherence"] = float(p.coherence)
    return out


def run() -> dict:
    W = _signal(400000)
    ref = _reads(W)
    keys = list(ref)

    rows_a, worst = [], 0.0
    for s in SCALES:
        r = _reads(W * s)
        dev = max(abs(r[k] - ref[k]) / max(abs(ref[k]), 1e-300) for k in keys)
        worst = max(worst, dev)
        rows_a.append([f"{s:.0e}", int(r["K_signal"]), round(r["coherence"], 4),
                       round(r["phi_F"], 6), f"{dev:.1e}"])
    table_a = C.md_table(["recording scale", "K_signal", "coherence", "phi_F",
                          "worst relative deviation"], rows_a)

    # ── (b) the same reads with a FIXED absolute floor in place of the derived one ──
    # The fixed floor is the derived floor's own value at unit scale, then held constant while
    # the recording scale moves.  Whitening is what makes the derived floor scale-free, so the
    # counterfactual compares the RAW singular spectrum against that constant.
    phi_fixed = float(Projection(W).noise_floor)
    sv_unit = np.linalg.svd(W, compute_uv=False)
    rows_b = []
    for s in SCALES:
        sv = sv_unit * s                      # singular values are homogeneous of degree 1
        k_fixed = int(np.sum(sv > phi_fixed))
        k_derived = int(Projection(W * s).K_signal)
        rows_b.append([f"{s:.0e}", k_derived, k_fixed,
                       "-" if k_fixed == k_derived else ("under" if k_fixed < k_derived else "over")])
    table_b = C.md_table(["recording scale", "K_signal (derived floor)",
                          "K_signal (fixed floor)", "failure"], rows_b)
    n_wrong = sum(1 for r in rows_b if r[3] != "-")
    k_true = int(Projection(W).K_signal)

    clean = _signal(400001)
    lo, hi = float(clean.min()), float(clean.max())
    exact = Projection(clean)
    rows_c = []
    for b in BITS:
        lv = 2 ** b - 1
        q = np.round((clean - lo) / (hi - lo) * lv) / lv * (hi - lo) + lo
        step = (hi - lo) / lv
        p = Projection(q)
        rows_c.append([b, round(step / np.sqrt(12.0), 5), int(p.K_signal),
                       round(float(p.noise_floor), 3),
                       round(float(p.noise_floor / exact.noise_floor), 3)])
    rows_c.append(["exact", 0.0, int(exact.K_signal), round(float(exact.noise_floor), 3), 1.0])
    table_c = C.md_table(["bits", "quantization sigma", "K_signal", "noise floor",
                          "floor / unquantized"], rows_c)

    ratios = [float(r[4]) for r in rows_c[:-1]]
    table = ("**(a) the same signal at every recording scale**\n\n" + table_a
             + "\n\n**(b) the regime a fixed absolute floor destroys**\n\n" + table_b
             + "\n\n**(c) uniform quantization: the floor measures it and stays finite**\n\n" + table_c)

    headline = (
        f"Across {len(SCALES)} recording scales spanning {int(np.log10(max(SCALES)/min(SCALES)))} "
        f"orders of magnitude, all {len(keys)} reads are invariant to {worst:.0e} relative -- "
        f"K_signal holds at {int(ref['K_signal'])} and the coherence at {ref['coherence']:.3f} at "
        f"strain scale (1e-21) exactly as at unit scale. Replacing that derived floor with a fixed "
        f"absolute one -- its own value at unit scale -- breaks the read at {n_wrong} of "
        f"{len(SCALES)} scales, reading K_signal = {rows_b[-1][2]} against a true {k_true} at the "
        f"smallest scale and {rows_b[0][2]} at the largest. Under uniform quantization from 16 bits "
        f"down to 2 the derived floor stays within a factor of "
        f"{max(max(ratios), 1/min(ratios)):.2f} of the unquantized floor at every depth.")
    concl = (
        "Nothing in the read path tests a quantity that carries units against a fixed number. The "
        "one place that must -- deciding whether a channel has any scale to whiten by -- takes the "
        "frame's own pooled MAD times the working dtype's epsilon, so it moves with the record and "
        "with the arithmetic. A channel with no spread is given no scale at all, "
        "which is what keeps coarsely quantized input finite: at 2 bits a channel's samples collapse "
        "onto one level, and dividing by any manufactured stand-in would lift round-off to unit "
        "amplitude and be resolved as signal.")

    return dict(
        title="14. A read is a property of the signal, not of the units it was recorded in",
        setup=(f"A rank-{RANK} signal plus noise at {SHAPE}. (a) multiplied by "
               f"{', '.join(f'{s:.0e}' for s in SCALES)}, every read compared against unit scale; "
               f"(c) uniformly quantized to {', '.join(str(b) for b in BITS)} bits over its own range."),
        table=table,
        metrics=dict(worst_relative_deviation=worst,
                     floor_ratio_by_bits={str(r[0]): float(r[4]) for r in rows_c[:-1]}),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"], "\n")
    print(r["table"], "\n")
    print(r["headline"])
