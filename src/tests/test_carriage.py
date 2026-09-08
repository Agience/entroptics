"""The weight-permutation null: what a weighted aggregation's WEIGHTS carry.

`reads.carriage` is the third of the library's permutation nulls, and each nulls a different
object: `projection.coherence` nulls structure WITHIN one frame's order, `reads.coupling` nulls
a relation BETWEEN two frames on a shared basis, and this one nulls the contribution of the
WEIGHTS to an aggregation.  The statistic is

    A = sum_i (w_i - mean w)(X_i - mean X),      sum_i w_i X_i = n*mean(w)*mean(X) + A

so A is the only part of the weighted sum a re-pairing of weights to samples can move.

The failure modes these tests watch for, stated first so they can fail:
  - the closed-form permutation moments are wrong, so `carried` is not 1 under the null and the
    per-coordinate `z` is mis-scaled (the whole read is those two moments);
  - the false-alarm rate does not land on the reader's `far`, so `resolved` cannot be believed;
  - constant weights or a constant stack read as evidence rather than as exactly zero;
  - `effective_n` drifts from Kish's (sum w)^2/sum w^2, which is what makes a signed aggregation
    report its own cost;
  - numpy and torch disagree, or a repeated read is not bit-identical.
"""
import numpy as np
import pytest

from entroptics import carriage
from entroptics.reads import Carriage


# ── the permutation moments, against brute force ─────────────────────────────

@pytest.mark.parametrize("shape,n,kind", [
    ((4,), 40, "gaussian"),
    ((4,), 40, "signs"),
    ((3, 5), 60, "signs"),
    ((2, 2), 25, "lognormal"),
])
def test_permutation_moments_are_exact(shape, n, kind):
    """E_pi[A] = 0 and Var_pi[A_c] = S_w g_c/(n-1), against brute-force permutation."""
    rng = np.random.default_rng(11)
    X = rng.standard_normal((n,) + shape)
    w = {"gaussian": lambda: rng.standard_normal(n),
         "signs": lambda: rng.choice([-1.0, 1.0], size=n),
         "lognormal": lambda: rng.lognormal(size=n)}[kind]()

    Xc = X - X.mean(axis=0, keepdims=True)
    wc = w - w.mean()
    S_w = float((wc ** 2).sum())
    g = (Xc ** 2).sum(axis=0)
    closed_var = S_w * g / (n - 1)

    draws = 20000
    acc = np.zeros(shape)
    accm = np.zeros(shape)
    for _ in range(draws):
        Ap = np.tensordot(rng.permutation(wc), Xc, axes=(0, 0))
        acc += Ap ** 2
        accm += Ap
    mc_var = acc / draws
    mc_mean = accm / draws

    assert np.abs(mc_mean).max() < 0.15 * np.sqrt(closed_var).max()
    assert np.allclose(mc_var / closed_var, 1.0, atol=0.05)


def test_carried_is_one_under_the_null():
    """Weights independent of the frames carry nothing: E[carried] = 1."""
    rng = np.random.default_rng(5)
    vals = [carriage(rng.standard_normal((150, 30)),
                     rng.choice([-1.0, 1.0], size=150)).carried
            for _ in range(300)]
    assert abs(float(np.mean(vals)) - 1.0) < 0.05


def test_false_alarm_rate_lands_on_far():
    """`resolved` averages far*P under the null -- the level is the reader's, and it holds."""
    rng = np.random.default_rng(6)
    P, trials = 40, 300
    for far, tol in ((0.05, 0.9), (0.20, 2.0)):
        res = [carriage(rng.standard_normal((150, P)),
                        rng.choice([-1.0, 1.0], size=150), far=far).resolved
               for _ in range(trials)]
        assert abs(float(np.mean(res)) - far * P) < tol


def test_planted_truth_is_recovered():
    """Weights tied to a direction in the frames are detected, monotonically in the amplitude.

    Averaged over draws: a single draw is not a statement about the read.  Under the null
    `carried` has a spread of order 0.2 about 1, so two amplitudes whose planted signal is
    below that spread order themselves by noise -- the claim is about the expectation."""
    rng = np.random.default_rng(7)
    n, P, draws = 400, 40, 12
    u = rng.standard_normal(P)
    u /= np.linalg.norm(u)
    got = []
    for amp in (0.0, 0.1, 0.3, 0.6):
        vals = []
        for _ in range(draws):
            Z = rng.standard_normal((n, P))
            w = rng.choice([-1.0, 1.0], size=n)
            vals.append(carriage(Z + amp * w[:, None] * u[None, :], w).carried)
        got.append(float(np.mean(vals)))
    assert got == sorted(got), got
    assert got[0] == pytest.approx(1.0, abs=0.15)      # amp 0 is the null
    assert got[-1] > 3.0                               # and the planted direction is found


def test_effective_n_is_kish():
    """(sum w)^2 / sum w^2, exactly -- and n*mean(w)^2 for unit-magnitude signed weights."""
    rng = np.random.default_rng(8)
    X = rng.standard_normal((200, 6))
    for p in (0.5, 0.6, 0.9):
        w = rng.choice([-1.0, 1.0], size=200, p=[1 - p, p])
        c = carriage(X, w)
        assert c.effective_n == pytest.approx(w.sum() ** 2 / (w ** 2).sum())
        assert c.effective_n == pytest.approx(len(w) * w.mean() ** 2)


# ── degenerate inputs: exactly zero, never evidence ──────────────────────────

def test_constant_weights_carry_nothing():
    """A is identically zero by construction; that is the read being zero, not a null missed."""
    rng = np.random.default_rng(9)
    X = rng.standard_normal((50, 7))
    c = carriage(X, np.full(50, 3.0))
    assert c.carried == 0.0 and c.resolved == 0
    assert c.effective_n == pytest.approx(50.0)      # uniform weights: full effective size


def test_constant_stack_has_nothing_to_carry():
    rng = np.random.default_rng(10)
    c = carriage(np.ones((50, 7)), rng.choice([-1.0, 1.0], size=50))
    assert c.carried == 0.0 and c.resolved == 0


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_samples(n):
    c = carriage(np.ones((n, 4)), np.ones(n))
    assert isinstance(c, Carriage) and c.carried == 0.0 and np.asarray(c.z).size == 0


def test_unobserved_coordinates_are_dropped():
    """A coordinate not measured on every sample is not a coordinate the aggregation ran over."""
    rng = np.random.default_rng(12)
    X = rng.standard_normal((60, 10))
    X[:, 3] = np.nan
    c = carriage(X, rng.choice([-1.0, 1.0], size=60))
    assert np.asarray(c.z).size == 9


def test_refuses_a_shape_that_is_not_a_stack():
    rng = np.random.default_rng(13)
    with pytest.raises(ValueError):
        carriage(rng.standard_normal(20), np.ones(20))
    with pytest.raises(ValueError):
        carriage(rng.standard_normal((20, 4)), np.ones((20, 2)))
    with pytest.raises(ValueError):
        carriage(rng.standard_normal((20, 4)), np.ones(19))


# ── the library contract: determinism and backend parity ─────────────────────

def test_deterministic():
    rng = np.random.default_rng(14)
    X = rng.standard_normal((80, 5, 3))
    w = rng.choice([-1.0, 1.0], size=80)
    a, b = carriage(X, w), carriage(X, w)
    assert a.carried == b.carried and a.resolved == b.resolved
    assert np.array_equal(np.asarray(a.z), np.asarray(b.z))


def test_numpy_torch_parity():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(15)
    X = rng.standard_normal((70, 6, 2))
    w = rng.choice([-1.0, 1.0], size=70)
    a = carriage(X, w)
    b = carriage(torch.as_tensor(X), torch.as_tensor(w))
    assert a.carried == pytest.approx(b.carried, rel=1e-12, abs=1e-12)
    assert a.resolved == b.resolved
    assert a.effective_n == pytest.approx(b.effective_n, rel=1e-12, abs=1e-12)
    assert np.allclose(np.asarray(a.z), b.z.numpy(), rtol=1e-10, atol=1e-10)


def test_scale_behaviour():
    """`carried` is a ratio of two quadratics in the same data, so it is invariant to a
    rescale of either the weights or the frames -- and that is a property, not a choice."""
    rng = np.random.default_rng(16)
    X = rng.standard_normal((90, 8))
    w = rng.choice([-1.0, 1.0], size=90)
    base = carriage(X, w).carried
    assert carriage(3.7 * X, w).carried == pytest.approx(base, rel=1e-10)
    assert carriage(X, -2.5 * w).carried == pytest.approx(base, rel=1e-10)
