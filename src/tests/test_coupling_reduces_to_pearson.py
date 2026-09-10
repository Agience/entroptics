"""What `reads.coupling` IS at one shared coordinate, pinned so the docstring cannot drift.

At `D = 1` the read reduces exactly to Pearson's r: the frames are two columns, `S` is their
centred inner product, and the Pitman-Hoeffding permutation variance collapses to
`|A~|^2 |B~|^2 / (T - 1)`, so

    strength == corrcoef(a, b)          and          z == r * sqrt(T - 1)

This is worth a test rather than a comment because a caller who does not know it will reach for
`coupling` on two scalar columns believing the exact null buys them something over `corrcoef`.
It does not.  What it buys is the DECISION layer, which is the second half of this file.

The failure modes these tests watch for, stated first so they can fail:
  - the reduction quietly stops holding (a change to the centring, the norms, or the variance),
    so the docstring's claim becomes false and callers are misled about what they called;
  - the decision layer disappears -- `strength` stops being zeroed below `far` -- and an
    unresolved correlation starts reporting as a small real one;
  - `z` is silently gated too, leaving a caller at `D = 1` with no route to the raw statistic;
  - the reduction is wrongly assumed to survive above `D = 1`, where `strength` is one signed
    cosine of two FRAMES and is NOT the mean of the per-coordinate correlations.
"""
import numpy as np
import pytest

from entroptics import reads


def _pair(a, b):
    return np.asarray(a, dtype=float)[:, None], np.asarray(b, dtype=float)[:, None]


# ── the reduction itself ─────────────────────────────────────────────────────

@pytest.mark.parametrize("tag,make", [
    ("exact affine, positive", lambda x, g: 3.5 * x - 2.0),
    ("exact affine, negative", lambda x, g: -0.25 * x + 7.0),
    ("noisy",                  lambda x, g: 0.75 * x + g.standard_normal(len(x))),
    ("weakly related",         lambda x, g: 0.10 * x + g.standard_normal(len(x))),
])
def test_strength_is_pearson_r_at_one_coordinate(tag, make):
    g = np.random.default_rng(0)
    x = g.standard_normal(300)
    A, B = _pair(x, make(x, g))
    c = reads.coupling(A, B)
    r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    assert c.resolved, f"{tag}: expected a resolved read to compare strength against"
    assert c.strength == pytest.approx(r, abs=1e-13)


@pytest.mark.parametrize("T", [8, 50, 300, 1001])
def test_z_is_r_times_sqrt_T_minus_one(T):
    g = np.random.default_rng(T)
    x = g.standard_normal(T)
    A, B = _pair(x, 0.6 * x + g.standard_normal(T))
    c = reads.coupling(A, B)
    r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    assert c.z == pytest.approx(r * np.sqrt(T - 1), abs=1e-11)


# ── the half that is NOT corrcoef: the decision layer ────────────────────────

def test_unresolved_strength_is_exactly_zero_while_r_is_not():
    """The one thing the read adds at `D = 1`, and the reason it is not a drop-in for corrcoef.

    Independent columns still have a nonzero sample correlation.  `coupling` reports no coupling
    -- exactly 0.0, not a small number -- because the read's contract is a decision at `far`.
    """
    g = np.random.default_rng(0)
    A, B = _pair(g.standard_normal(300), g.standard_normal(300))
    c = reads.coupling(A, B)
    r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    assert not c.resolved
    assert c.strength == 0.0
    assert abs(r) > 1e-3, "control is vacuous unless the raw r is measurably nonzero"


def test_z_survives_the_gate_so_the_raw_statistic_stays_reachable():
    g = np.random.default_rng(0)
    A, B = _pair(g.standard_normal(300), g.standard_normal(300))
    c = reads.coupling(A, B)
    r = float(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    assert not c.resolved
    assert c.z == pytest.approx(r * np.sqrt(len(A) - 1), abs=1e-12)
    assert c.z != 0.0


# ── and the reduction must NOT be assumed above D = 1 ────────────────────────

def test_above_one_coordinate_strength_is_not_the_mean_of_column_correlations():
    """`strength` is one signed cosine of two frames, not a per-column r averaged.

    Without this the docstring's "the reduction stops" is an assertion nobody checked, and a
    caller could read a multi-column `strength` as an average correlation.
    """
    g = np.random.default_rng(3)
    T, D = 400, 5
    A = g.standard_normal((T, D))
    B = np.empty_like(A)
    scales = np.array([50.0, 1.0, 1.0, 1.0, 1.0])      # one column dominates the Frobenius norm
    for j in range(D):
        rho = 0.95 if j == 0 else 0.05
        B[:, j] = scales[j] * (rho * A[:, j] + np.sqrt(1 - rho ** 2) * g.standard_normal(T))
        A[:, j] = scales[j] * A[:, j]
    c = reads.coupling(A, B)
    per_column = np.array([np.corrcoef(A[:, j], B[:, j])[0, 1] for j in range(D)])
    assert c.resolved
    assert abs(c.strength - per_column.mean()) > 0.2, (
        "the dominant column should pull the frame cosine well away from the column mean; "
        "if these agree the test has stopped discriminating")
