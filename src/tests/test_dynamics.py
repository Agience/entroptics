"""The streaming dynamical operator: exact decay-rate recovery, additive splicing,
bit-exact resume, and warm-start accumulator identities."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.dynamics import Dynamics, dynamics
from conftest import build_W


def test_undersampled_truncation_far_is_caller_settable():
    """The under-sampled DMD truncation (T<F) uses the caller's operating point, not a
    hard-wired far: a laxer far keeps more modes, and far=0.05 is the behaviour-preserving
    default.  Well-sampled records never truncate, so this only moves the T<F fit."""
    rng = np.random.default_rng(0)
    U = rng.standard_normal((30, 3)); V, _ = np.linalg.qr(rng.standard_normal((60, 3)))
    W = U @ V.T + 0.3 * rng.standard_normal((30, 60))       # T=30 < F=60 (under-sampled)
    n_lax = dynamics(W, far=0.5).rates().n_modes
    n_default = dynamics(W).rates().n_modes
    n_strict = dynamics(W, far=1e-4).rates().n_modes
    assert n_lax >= n_default >= n_strict                    # laxer far -> more modes kept
    assert n_lax > n_strict                                  # far actually moves the truncation
    assert dynamics(W, far=0.05).rates().n_modes == n_default


def test_rates_recover_known_operator():
    """A pure linear system x_{t+1} = diag(mu) x_t must give alpha_k = -log|mu_k|."""
    mus = np.array([0.90, 0.95, 0.80])
    T = 300
    x = np.zeros((T, 3))
    x[0] = [1.0, 1.0, 1.0]
    for t in range(1, T):
        x[t] = mus * x[t - 1]
    r = dynamics(x).rates()
    got = sorted(float(a) for a in np.asarray(r.alpha))
    exp = sorted(-np.log(mus))
    assert got == pytest.approx(exp, abs=1e-6)


def test_rates_recover_rotation_frequency():
    """A rotating mode's beta = arg(mu) must recover the rotation angle."""
    theta = 0.3
    R = 0.98 * np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    T = 400
    x = np.zeros((T, 2))
    x[0] = [1.0, 0.0]
    for t in range(1, T):
        x[t] = R @ x[t - 1]
    r = dynamics(x).rates()
    assert np.max(np.abs(np.asarray(r.alpha) - (-np.log(0.98)))) < 1e-6
    assert np.min(np.abs(np.abs(np.asarray(r.beta)) - theta)) < 1e-6


def test_splice_is_additive():
    """forgetting=1 accumulators are additive: adjacent merge == whole-stream fit."""
    W = build_W(7)
    k = 37
    whole = dynamics(W[:k]).merge(dynamics(W[k:]), adjacent=True)
    full = dynamics(W)
    assert np.max(np.abs(np.asarray(whole.Pxx) - np.asarray(full.Pxx))) < 1e-10
    assert np.max(np.abs(np.asarray(whole.Pyx) - np.asarray(full.Pyx))) < 1e-10
    assert whole.n_pairs == full.n_pairs


def test_resume_is_bit_exact():
    """state() -> from_state() -> continue must be identical to an uninterrupted run."""
    W = build_W(3)
    a = dynamics(W[:40])
    resumed = Dynamics.from_state(a.state())
    for t in range(40, 64):
        a.update(W[t])
        resumed.update(W[t])
    assert np.array_equal(np.asarray(a.Pxx), np.asarray(resumed.Pxx))
    assert np.array_equal(np.asarray(a.Pyx), np.asarray(resumed.Pyx))


def test_merge_with_empty_operator():
    W = build_W(7)
    full = dynamics(W)
    assert Dynamics(W.shape[1]).merge(full).n_frames == full.n_frames
    assert full.merge(Dynamics(W.shape[1])).n_frames == full.n_frames


def test_seed_warmstart_accumulators():
    """seed(A_prior, weight=w): Pxx = w*I, Pyx = w*A_prior (initial propagator == A_prior)."""
    F = 4
    A_prior = np.random.default_rng(0).standard_normal((F, F))
    d = Dynamics(F).seed(A_prior, weight=3.0)
    assert np.allclose(np.asarray(d.Pxx), 3.0 * np.eye(F))
    assert np.allclose(np.asarray(d.Pyx), 3.0 * A_prior)


def test_aperture_splice_optics_reproduces_full():
    """Splicing two BATCH halves rebuilds the window -> optics == full-signal optics."""
    W = build_W(7)
    whole = A.Aperture(W[:40]).splice(A.Aperture(W[40:]))
    o_splice, o_full = whole.optics(), A.Aperture(W).optics()
    for k in o_full:
        if isinstance(o_full[k], float):
            assert o_splice[k] == pytest.approx(o_full[k], rel=1e-9, abs=1e-12), k


def test_dmd_wellposed_when_undersampled():
    """Under-sampling (n_pairs < 2F, e.g. a short T<F cutout): the DMD truncates to the
    resolved signal rank, so it does not overfit into spurious |mu| > 1.  Noise reads a
    margin < 1 (it forgets); a coherent mode is still resolved."""
    import math
    r = np.random.default_rng(0)
    T, F = 40, 60                                        # T < F -> rank-deficient
    noise = dynamics(r.standard_normal((T, F)))
    fo = noise.forgetting()
    assert fo["margin"] < 1.0 and fo["forgets"]          # no overfit into a spurious persistent mode
    v = r.standard_normal(F); v /= np.linalg.norm(v)
    X = np.array([math.cos(0.1 * t) * 3.0 * v + 0.1 * r.standard_normal(F) for t in range(T)])
    assert dynamics(X).resolved() >= 1                    # a real coherent mode still resolves


def test_reconstruct_decay_shape_and_normalisation():
    W = build_W(7)
    c = np.asarray(dynamics(W).reconstruct_decay(20))
    assert c.shape == (20,)
    assert c[0] == pytest.approx(1.0, abs=1e-9)     # normalised so C(0) = 1
