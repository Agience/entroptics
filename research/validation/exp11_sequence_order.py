"""
Experiment 11 -- Symbol sequences: order is detected, and the null is calibrated.

Ground truth: sequences over a finite alphabet with a known amount of order.
  (a) A Markov chain of known correlation reads a block-entropy departure from its own
      permutation ensemble (Definition 11.3); an i.i.d. sequence over the same alphabet
      does not, because a permutation preserves symbol frequencies exactly.
  (b) The finite-sequence saturation of Proposition 11.2 is exhibited directly: H_n
      turns over once the number of possible n-words exceeds the observed windows.

Deterministic (fixed seeds).  Re-runnable: `python exp11_sequence_order.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import sequence as S

import common as C

N = 4000
ALPHA = 4
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 0.95]
SEED = 1111
DRAWS = 40


def _markov(strength, seed, n=N, k=ALPHA):
    """A chain that repeats its previous symbol with probability `strength`, else draws
    uniformly: strength 0 is i.i.d., strength -> 1 is maximally ordered.  Symbol
    frequencies stay near uniform at every strength, so only the ORDER differs."""
    g = C.rng(seed)
    out = np.empty(n, dtype=int)
    out[0] = g.integers(k)
    for i in range(1, n):
        out[i] = out[i - 1] if g.random() < strength else g.integers(k)
    return out


def run():
    rows, fired = [], []
    for i, s in enumerate(STRENGTHS):
        seq = _markov(s, SEED + i)
        r = S.surrogate_test(seq, draws=DRAWS, n_max=6, seed=0)
        # H_1 is invariant under permutation by construction, so the order signal is n >= 2;
        # ordering LOWERS the block entropies, so the departure is negative.
        zs = np.asarray(r["z"], dtype=float)[1:]
        z = float(np.nanmax(np.abs(zs)))
        h1 = float(S.block_entropies(seq, 1)[0])
        rows.append([s, round(h1, 4), round(z, 1), r["onset"] if r["onset"] else "-",
                     "yes" if z > 3.0 else "no"])
        fired.append(z)
    table = C.md_table(["repeat prob.", "H_1 (bits)", "max |z| (n>=2)", "onset n", "order detected"], rows)

    # saturation: H_n must turn over once k^n exceeds the observed windows
    seq = _markov(0.5, SEED)
    Hs = [float(S.block_entropies(seq, n)[n - 1]) for n in range(1, 9)]
    cap = [float(np.log2(N - n + 1)) for n in range(1, 9)]
    sat = next((n for n in range(1, 9) if Hs[n - 1] > 0.9 * cap[n - 1]), None)
    sat_rows = [[n, round(Hs[n - 1], 3), round(cap[n - 1], 3),
                 round(ALPHA ** n / (N - n + 1), 3)] for n in range(1, 9)]
    table2 = C.md_table(["n", "H_n", "log2(N-n+1)", "words / windows"], sat_rows)

    headline = (f"an i.i.d. sequence reads |z|={fired[0]:.2f} against its own permutation ensemble; "
                f"at repeat probability {STRENGTHS[-1]} the same read gives |z|={fired[-1]:.0f}, "
                f"with H_1 unchanged throughout.")
    concl = ("Order is detected by the permutation null, not by an entropy rate, and the block "
             "entropies saturate at log2(N-n+1) exactly as the finite sequence requires.")
    return dict(
        title="11. Symbol sequences: order detection and the saturation bound",
        setup=(f"repeat-probability chains over an alphabet of {ALPHA}, N={N}, "
               f"{DRAWS} surrogate draws, n_max=6."),
        table=table + "\n\n" + table2,
        metrics=dict(z_iid=fired[0], z_ordered=fired[-1], saturation_onset_n=sat),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
