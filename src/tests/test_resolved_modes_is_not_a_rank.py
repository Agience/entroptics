"""`spectral_optics(...).resolved_modes` is a noise-floor count, not a rank and not a model order.

This pins a SCOPE BOUNDARY, in the idiom of `test_absent_is_not_zero.py`: the read is correct at
what it does, and the test exists so nobody rebuilds on it expecting something it never claimed.

The temptation is specific and worth naming.  Identifying a linear operator from a correlation
sequence `C(tau) = sum_i c_i lambda_i^tau` needs a MODEL ORDER -- how many `lambda_i` to keep --
and every standard answer is a chosen number (an information criterion picks a penalty, a
singular-value cut picks a level, a stability window picks a width).  `resolved_modes` looks like
the constant-free way out, because the floor is derived from the data rather than supplied.  It is
not: measured on the Hankel embedding of a sum of decaying exponentials at ZERO noise, it returns
1 for true order 2 and 3, while the same matrix has exact numerical rank 2 and 3.

The reason is structural rather than a floor set wrong.  The floor separates signal from a noise
sea; these modes are all real and wildly unequal in scale, so the smaller ones sit under it with
`top_share` at 0.994 to 0.996.  Asking "which modes clear noise" is a different question from
"how many modes are there", and only the first has a constant-free answer.

The failure modes these tests watch for, stated first so they can fail:
  - the docstring's scope note goes stale because the behaviour changed and nobody noticed;
  - the construction stops being a fair test -- the Hankel matrix must actually HAVE the rank
    claimed, with well-separated singular values, or the read is being blamed for noise;
  - somebody wires `resolved_modes` into a model-order path on the strength of the k = 1 case,
    which is the one order where it happens to be right.
"""
import numpy as np
import pytest

from entroptics import reads


def _hankel(seq, rows):
    seq = np.asarray(seq, dtype=float)
    cols = len(seq) - rows + 1
    return np.stack([seq[i:i + cols] for i in range(rows)], axis=0).T      # (T, N)


def _sequence(lams, coefs, ntau=256):
    tau = np.arange(ntau)
    return sum(ci * (li ** tau) for li, ci in zip(lams, coefs))


ORDERS = {
    1: ([0.80], [1.0]),
    2: ([0.90, 0.40], [1.0, 0.7]),
    3: ([0.92, 0.60, 0.25], [1.0, 0.6, 0.4]),
}


def _numerical_rank(H):
    sv = np.linalg.svd(H, compute_uv=False)
    return int((sv > sv[0] * max(H.shape) * np.finfo(float).eps).sum()), sv


# ── the construction has to be fair before the read can be judged on it ──────

@pytest.mark.parametrize("k", sorted(ORDERS))
def test_the_hankel_embedding_really_has_the_claimed_rank(k):
    lams, coefs = ORDERS[k]
    H = _hankel(_sequence(lams, coefs), rows=12)
    rank, sv = _numerical_rank(H)
    assert rank == k, f"the test's own construction is wrong: rank {rank}, wanted {k}"
    if k > 1:
        assert sv[k - 1] > 1e-3 * sv[0], (
            "the sub-leading mode must be well clear of float noise, or this measures "
            "conditioning rather than the read")


# ── the boundary itself ──────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("noise", [0.0, 1e-6, 1e-3])
def test_resolved_modes_does_not_recover_model_order(k, noise):
    lams, coefs = ORDERS[k]
    seq = _sequence(lams, coefs)
    if noise:
        seq = seq + noise * np.random.default_rng(4).standard_normal(len(seq))
    H = _hankel(seq, rows=12)
    so = reads.spectral_optics(H)
    assert so.resolved_modes == 1, (
        f"resolved_modes returned {so.resolved_modes} for true order {k}. If this now tracks "
        f"the order, the scope note in spectral_optics' docstring is stale -- update it.")
    assert so.top_share > 0.99, (
        "the documented mechanism is that one mode carries nearly all the variance; "
        f"top_share is {so.top_share:.4f}, so the explanation in the docstring no longer fits")


def test_the_singular_values_do_recover_it():
    """The alternative the docstring points callers at, so the note is actionable."""
    for k, (lams, coefs) in ORDERS.items():
        H = _hankel(_sequence(lams, coefs), rows=12)
        rank, _ = _numerical_rank(H)
        assert rank == k


def test_order_one_is_the_case_that_misleads():
    """`resolved_modes` IS right at k = 1 -- which is why the boundary needs stating."""
    lams, coefs = ORDERS[1]
    so = reads.spectral_optics(_hankel(_sequence(lams, coefs), rows=12))
    assert so.resolved_modes == 1
