"""Regenerate every number in PAPER_coupling.md.

    python research/supplemental/coupling/reproduce.py            # everything
    python research/supplemental/coupling/reproduce.py --quick    # 20k re-pairings, for a smoke test

Writes, into ``tables/`` beside this file:

    brute_force.csv     the closed-form permutation variance against brute-force re-pairing,
                        at 20k / 100k / 200k draws, five shapes, real and complex
    sign.csv            planted-sign recovery at five coupling strengths
    null.csv            null calibration over 1600 independent pairs
    embedding.csv       the T=2, D=1 counterexample: what the Hermitian form reports

The brute-force check is the paper's main empirical claim and it is one command.  Nothing
here reimplements the statistic: the coupling is taken through the public ``Screen`` API
(``register`` -> ``place`` -> ``coupling``), and the only thing computed directly is the
brute-force permutation itself, which is the reference the closed form is being checked
against and therefore must not come from the library.

Runtime is dominated by the 200,000-draw brute force at five shapes; expect a few minutes.
``--quick`` cuts it to 20,000 and is not what the paper reports.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# Pinned before numpy is reached: OpenBLAS splits a reduction across its pool, so the summation
# order -- and the last bit of every eigendecomposition -- depends on how many threads it has.
# The committed tables are diffed.  setdefault, so an operator who chose a value keeps it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLES = HERE / "tables"

# Run against the working tree's src/, as the validation suite does, rather than whatever wheel
# happens to be installed: the paper reports what this checkout produces.
sys.path.insert(0, str(REPO / "src"))

from entroptics import Screen  # noqa: E402

# ---------------------------------------------------------------------------------------
# the experiment constants, matching research/validation/exp7_coupling.py
# ---------------------------------------------------------------------------------------

#: Brute-force cases.  The last two are COMPLEX, and they are the point: the closed form is a
#: statement about the real embedding, and a real-only check cannot see the difference.
VAR_CASES = [(40, 5, False), (64, 8, False), (120, 3, False), (64, 6, True), (96, 4, True)]

#: Draw counts the convergence is reported at.  The permutation distribution is not normal
#: (section 4), so the variance estimate converges slowly and the paper reports the whole
#: sequence rather than one number: the residual is sampling error and must be seen to shrink.
DRAWS = (20_000, 100_000, 200_000)

SIGN_SHAPE, N_SIGN = (96, 6), 60
RHOS = [1.0, 0.5, 0.0, -0.5, -1.0]

NULL_SHAPES, N_NULL, FAR = [(64, 4), (128, 6), (96, 12), (200, 5)], 400, 0.05


def rng(seed):
    return np.random.default_rng(seed)


def _couple(a, b, far=FAR):
    """The coupling, through the public screen API.

    ``Screen.coupling`` is the exported path; it registers an identity entry per side, places
    both on one shared basis, and returns the full record.  Nothing here reaches past it."""
    s = Screen(far=far)
    s.register("a", entry=lambda x: x)
    s.register("b", entry=lambda x: x)
    s.place("a", a)
    s.place("b", b)
    return s.coupling("a", "b")


def _pair(rho, seed, T, D, amp=1.5):
    """Two (T, D) sides sharing a planted carrier at signed strength ``rho``."""
    g = rng(seed)
    carrier = g.standard_normal((T, 1))
    a = amp * carrier + g.standard_normal((T, D))
    b = rho * amp * carrier + g.standard_normal((T, D))
    return a, b


# ---------------------------------------------------------------------------------------
# (1) the brute force: the closed form against uniform row re-pairing
# ---------------------------------------------------------------------------------------

def brute_force(n_max):
    """Theorem 2.2's Var_pi[Re S] = tr(C_A C_B)/(T-1) against sampled re-pairings.

    The permutation is done here rather than in the library on purpose: it is the REFERENCE the
    closed form is checked against, so drawing it from the same code would check nothing.  One
    stream of ``n_max`` draws per shape, with the variance read at each cut point, so the
    convergence is measured on nested samples rather than on independent runs -- the sequence
    then shows the estimate settling and not three unrelated numbers."""
    rows = []
    for (T, D, cplx) in VAR_CASES:
        g = rng(120000 + T * D)
        if cplx:
            A = g.standard_normal((T, D)) + 1j * g.standard_normal((T, D))
            B = g.standard_normal((T, D)) + 1j * g.standard_normal((T, D))
        else:
            A, B = g.standard_normal((T, D)), g.standard_normal((T, D))
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        # the Grams are formed on the REAL EMBEDDING, as the library does; Re S is then the
        # real Frobenius inner product of the embedded frames
        Ar = np.hstack([Ac.real, Ac.imag]) if cplx else Ac
        Br = np.hstack([Bc.real, Bc.imag]) if cplx else Bc
        closed = float(np.sum((Ar.T @ Ar) * (Br.T @ Br)) / (T - 1))

        draws = np.empty(n_max)
        for i in range(n_max):
            draws[i] = float(np.sum(Ar[g.permutation(T)] * Br))

        row = {"shape": f"({T}, {D})", "kind": "complex" if cplx else "real",
               "closed_form": f"{closed:.2f}"}
        for n in DRAWS:
            if n > n_max:
                row[f"ratio_{n}"] = ""
                continue
            row[f"ratio_{n}"] = f"{float(draws[:n].var()) / closed:.4f}"
        row["standardised_mean"] = f"{float(draws.mean() / np.sqrt(closed)):.4f}"
        rows.append(row)
        print(f"  {row['shape']:>10} {row['kind']:>8}  closed {closed:10.2f}  "
              + "  ".join(f"{n//1000}k {row[f'ratio_{n}'] or '-':>6}" for n in DRAWS))
    return rows


# ---------------------------------------------------------------------------------------
# (2) planted sign recovery
# ---------------------------------------------------------------------------------------

def sign_recovery():
    T, D = SIGN_SHAPE
    rows = []
    for rho in RHOS:
        cs = [_couple(*_pair(rho, 90000 + 137 * i + int(100 * rho), T, D)) for i in range(N_SIGN)]
        signs = np.array([c.sign for c in cs])
        expect = 0 if rho == 0.0 else int(np.sign(rho))
        rows.append({"planted_rho": f"{rho:+.1f}", "expected_sign": expect,
                     "sign_agreement": f"{float(np.mean(signs == expect)):.3f}",
                     "mean_strength": f"{float(np.mean([c.strength for c in cs])):.3f}",
                     "resolved_rate": f"{float(np.mean([c.resolved for c in cs])):.3f}"})
        print(f"  rho {rho:+.1f}  agreement {rows[-1]['sign_agreement']}  "
              f"strength {rows[-1]['mean_strength']:>7}  resolved {rows[-1]['resolved_rate']}")
    return rows


# ---------------------------------------------------------------------------------------
# (3) null calibration on independent pairs
# ---------------------------------------------------------------------------------------

def null_calibration():
    zs, fired, k = [], 0, 0
    for (T, D) in NULL_SHAPES:
        for _ in range(N_NULL):
            c = _couple(*_pair(0.0, 150000 + k, T, D))
            zs.append(c.z)
            fired += int(c.resolved)
            k += 1
    zs = np.array(zs)
    row = {"n_pairs": len(zs), "mean_z": f"{float(zs.mean()):.3f}",
           "std_z": f"{float(zs.std()):.3f}", "fire_rate": f"{fired / len(zs):.4f}",
           "nominal_far": FAR}
    print(f"  {len(zs)} pairs   mean {row['mean_z']}   std {row['std_z']}   "
          f"fires {row['fire_rate']} against nominal {FAR}")
    return [row]


# ---------------------------------------------------------------------------------------
# (4) the counterexample the real embedding exists for
# ---------------------------------------------------------------------------------------

def embedding_counterexample():
    """T=2, D=1, a~ = (1, -1), b~ = (i, -i).

    Both of the two re-pairings give Re S = 0, so the TRUE permutation variance is exactly 0.
    The Hermitian Gram form reports tr(A~^H A~ B~^H B~)/(T-1) = 4.  The real embedding gives 0
    and is what makes the standardisation exact off the real axis."""
    a = np.array([[1.0 + 0j], [-1.0 + 0j]])
    b = np.array([[0 + 1j], [0 - 1j]])
    T = 2
    ac, bc = a - a.mean(0), b - b.mean(0)

    # every re-pairing, enumerated -- T = 2, so there are exactly two
    res = [float(np.real(np.sum(np.conj(ac[p]) * bc))) for p in ([0, 1], [1, 0])]
    true_var = float(np.var(res))

    hermitian = float(np.real(np.trace((ac.conj().T @ ac) @ (bc.conj().T @ bc))) / (T - 1))

    ar = np.hstack([ac.real, ac.imag])
    br = np.hstack([bc.real, bc.imag])
    embedded = float(np.sum((ar.T @ ar) * (br.T @ br)) / (T - 1))

    print(f"  Re S over both re-pairings: {res}")
    print(f"  true variance {true_var:.1f}   Hermitian form {hermitian:.1f}   "
          f"real embedding {embedded:.1f}")
    return [{"quantity": "true Var[Re S] over all re-pairings", "value": f"{true_var:.1f}"},
            {"quantity": "Hermitian Gram form", "value": f"{hermitian:.1f}"},
            {"quantity": "real embedding (Theorem 2.2)", "value": f"{embedded:.1f}"}]


# ---------------------------------------------------------------------------------------

def _write(name, rows):
    TABLES.mkdir(exist_ok=True)
    out = TABLES / name
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out.relative_to(REPO)}")


def main():
    quick = "--quick" in sys.argv
    n_max = 20_000 if quick else max(DRAWS)
    if quick:
        print("--quick: 20,000 re-pairings.  NOT what the paper reports.\n")

    print("(4) the T=2, D=1 counterexample")
    emb = embedding_counterexample()

    print("\n(2) planted sign recovery")
    sign = sign_recovery()

    print("\n(3) null calibration, 1600 independent pairs")
    null = null_calibration()

    print(f"\n(1) brute force: closed form against up to {n_max:,} uniform row re-pairings")
    bf = brute_force(n_max)

    print("\nwriting tables")
    _write("embedding.csv", emb)
    _write("sign.csv", sign)
    _write("null.csv", null)
    _write("brute_force.csv", bf)

    worst = max(abs(float(r[f"ratio_{n_max}"]) - 1.0) for r in bf)
    print(f"\nworst deviation from the closed form at {n_max:,} draws: {worst:.2%}")


if __name__ == "__main__":
    main()
