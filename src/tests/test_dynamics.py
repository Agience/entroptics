"""The streaming dynamical operator: exact decay-rate recovery, additive splicing,
bit-exact resume, and warm-start accumulator identities."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.dynamics import Dynamics, dynamics, carry_over_gaps
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
    """Splicing two batch halves rebuilds the window -> optics == full-signal optics."""
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


# ── the scalar-sequence moment pencil (hankel_spectrum) + generic jackknife ───────────────
def test_hankel_spectrum_recovers_known_transfer_modes():
    """The moment pencil on a finite exponential sum C(tau)=sum_k w_k lam_k^tau recovers the lam_k
    exactly (Prony is exact with enough moments) -- the scalar-series analogue of
    test_rates_recover_known_operator for the covariance DMD."""
    from entroptics import hankel_spectrum
    lam = np.array([0.5, 0.2]); w = np.array([0.7, 0.3])
    c = np.array([float(np.sum(w * lam ** t)) for t in range(9)])
    hs = hankel_spectrum(c, 2)
    got = np.sort(np.asarray(hs.evals))[::-1]
    assert np.allclose(got[:2], [0.5, 0.2], atol=1e-8)
    assert abs(hs.leading - 0.5) < 1e-8
    assert abs(hs.rate + np.log(0.5)) < 1e-8            # rate = -log(lambda_1)
    assert hs.psd > -1e-9                               # PSD (eigenvalues >= 0); H0 here is exactly rank-2,
                                                        # so its min eigenvalue is a roundoff-zero (sign is
                                                        # platform-dependent), not a negative indefinite mode


def test_hankel_spectrum_matches_handrolled_pencil():
    """Bit-for-bit agreement with the hand-rolled reflection-positive pencil used by the mass-gap
    scripts, so results here reproduce the same published numbers."""
    rng = np.random.default_rng(3)
    lam = np.array([0.6, 0.33, 0.15]); w = np.array([0.5, 0.3, 0.2])
    c = np.array([float(np.sum(w * lam ** t)) for t in range(12)]) + 1e-4 * rng.standard_normal(12)
    n = 3
    cN = c / c[0]                                        # the exact 8_7 `pencil` body:
    idx = np.add.outer(np.arange(n + 1), np.arange(n + 1))
    H0, H1 = cN[idx], cN[idx + 1]
    ww, V = np.linalg.eigh(H0); keep = ww > 1e-6 * ww.max()
    Vr = V[:, keep] / np.sqrt(ww[keep]); M = Vr.T @ H1 @ Vr
    ev_ref = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
    from entroptics import hankel_spectrum
    ev_got = np.asarray(hankel_spectrum(c, n).evals)
    assert ev_got.shape == ev_ref.shape and np.allclose(ev_got, ev_ref, rtol=0, atol=1e-12)


def test_jackknife_mean_matches_closed_form():
    """Delete-one jackknife SE of the sample mean equals the textbook s/sqrt(N)."""
    from entroptics import jackknife
    x = np.random.default_rng(0).standard_normal(50)
    est, se = jackknife(x, lambda s: float(np.mean(s)))
    assert abs(est - x.mean()) < 1e-12
    assert abs(se - x.std(ddof=1) / np.sqrt(len(x))) < 1e-12


def test_jackknife_binned_matches_massgap_convention():
    """Binned (delete-one-bin) jackknife reproduces sqrt((G-1)/G sum (theta_g-mean)^2), the
    mass-gap scripts' error convention."""
    from entroptics import jackknife
    X = np.random.default_rng(1).standard_normal((64, 4))
    read = lambda s: float(np.mean(s))
    _, se = jackknife(X, read, n_bins=8)
    N, G = 64, 8; groups = np.array_split(np.arange(N), G)
    th = np.array([read(X[np.setdiff1d(np.arange(N), g)]) for g in groups])
    ref = np.sqrt((G - 1) / G * np.sum((th - th.mean()) ** 2))
    assert abs(se - ref) < 1e-12


def test_multistep_predict_exact_and_backward_compatible():
    """predict(steps=h) / rollout are the spectral forecast x_h = sum_k phi_k mu_k^h b_k -- exact
    vs A^h on a linear system, stable at large h, and the default steps=1 gives the full A @ x
    update."""
    import numpy as np
    from entroptics.dynamics import dynamics
    rng = np.random.default_rng(0); F = 6
    Q, _ = np.linalg.qr(rng.standard_normal((F, F)))
    A = Q @ np.diag([0.95, 0.9, 0.8, 0.7, 0.5, 0.3]) @ Q.T
    x0 = rng.standard_normal(F); traj = [x0]
    for _ in range(60):
        traj.append(A @ traj[-1])
    dyn = dynamics(np.array(traj)); xs = np.array(traj)[20]
    for h in (1, 5, 25, 50):
        pred = np.asarray(dyn.predict(xs, steps=h))
        assert np.max(np.abs(pred - np.linalg.matrix_power(A, h) @ xs)) < 1e-9   # exact, no A^h blowup
    roll = np.asarray(dyn.rollout(xs, 8))
    assert roll.shape == (8, F)
    assert np.allclose(np.asarray(dyn.predict(xs)), np.asarray(dyn.predict(xs, steps=1)))   # steps defaults to 1


def test_koopman_lift_resolves_nonlinear():
    """A nonlinear oscillator resolves no operator raw but is linear in delay coordinates."""
    import numpy as np
    from entroptics import Aperture, koopman_lift, delay_embed
    t = np.linspace(0, 40, 400)
    traj = np.stack([np.sin(t) + 0.3 * np.sin(3 * t), np.cos(t)], 1)
    assert delay_embed(traj, 5).shape == (396, 10)
    raw = Aperture(traj, window=None).dynamics()          # one path to a fitted operator
    assert raw.resolved() < koopman_lift(traj, d=12).resolved()   # lift resolves modes


# ── missing cells: an operator is read off PAIRS, so a state with a hole is not a state ──

def _planted_system(T=2000, F=12, seed=1, noise=0.1):
    """A linear system with two known slow modes, kept excited by process noise."""
    g = np.random.default_rng(seed)
    mag, th = np.array([0.985, 0.960]), np.array([0.30, 0.11])
    A = np.zeros((F, F))
    for i, (m, a) in enumerate(zip(mag, th)):
        A[2*i:2*i+2, 2*i:2*i+2] = m * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    A[4:, 4:] = np.diag(g.uniform(0.2, 0.5, F - 4))
    x, tr = g.standard_normal(F), []
    for _ in range(T):
        x = A @ x + noise * g.standard_normal(F)
        tr.append(x.copy())
    return np.array(tr), float(np.sort(-np.log(mag))[0])


def _slowest(W):
    return float(np.sort(np.asarray(A.Aperture(W, window=None).rates().alpha))[0])


def test_dropout_does_not_bias_the_decay_rates():
    """Zeroing a missing cell makes the transition INTO it look like decay toward zero, so every
    rate reads faster than it is -- the read rose from 0.0195 to 0.4597 (23x) as dropout went from
    0 to 35%, always in the same direction.  The gaps are carried by the record's own operator
    instead, so the read stops depending on how much was dropped."""
    W0, _ = _planted_system()
    rng = np.random.default_rng(0)
    ref = _slowest(W0)
    for q in (0.05, 0.20, 0.35, 0.50):
        W = W0.copy()
        W[rng.random(W.shape) < q] = np.nan
        assert _slowest(W) == pytest.approx(ref, rel=0.10), f"dropout {q:.0%} moved the rate"


def test_carrying_the_gaps_invents_no_operator():
    """The control that keeps it honest: a record with no dynamics must not acquire any.  White
    noise has no persistent mode, and must still have none after half of it is dropped."""
    rng = np.random.default_rng(3)
    N = rng.standard_normal((1500, 12))
    clean = np.sort(np.asarray(A.Aperture(N, window=None).rates().alpha))[0]
    for q in (0.20, 0.50):
        X = N.copy()
        X[rng.random(X.shape) < q] = np.nan
        slow = np.sort(np.asarray(A.Aperture(X, window=None).rates().alpha))[0]
        assert np.exp(-slow) < 0.5, "a persistent mode appeared in noise"
        assert slow > 0.5 * clean


def test_a_record_without_gaps_is_returned_unchanged():
    """The fill is reached only by records that need it: no gaps, no work and no copy."""
    W = np.random.default_rng(4).standard_normal((50, 8))
    assert carry_over_gaps(W) is W


def test_single_frame_streaming_is_not_biased_by_dropout():
    """A block can be completed from its own record; one frame cannot, so the operator STANDING
    NOW predicts the hole.  Zeroing it read the slowest rate 0.0150 at no dropout and 0.68 at
    half -- a factor of 45 -- because a zero next-state is decay toward zero."""
    W0, _ = _planted_system(T=4000)
    ref = None
    for q in (0.0, 0.05, 0.20, 0.35):
        W = W0.copy()
        if q:
            W[np.random.default_rng(7).random(W.shape) < q] = np.nan
        d = Dynamics(W.shape[1])
        for row in W:
            d.update(row)
        slow = float(np.sort(np.asarray(d.rates().alpha))[0])
        if ref is None:
            ref = slow
        else:
            assert slow == pytest.approx(ref, rel=0.15), f"dropout {q:.0%} moved the rate"


def test_the_carried_inverse_matches_a_fresh_solve():
    """`Pxx^+` is carried through each frame's rank-1 update instead of resolved, so the predictor
    costs what the accumulators cost and is never stale.  It has to stay the same inverse."""
    rng = np.random.default_rng(5)
    F = 10
    W = rng.standard_normal((600, F))
    d = Dynamics(F)
    for row in W:
        d.update(row)
    assert d.Pinv is not None
    fresh = np.linalg.pinv(np.asarray(d.Pxx), hermitian=True)
    assert np.allclose(np.asarray(d.Pinv), fresh, rtol=1e-6, atol=1e-10)


def test_block_ingest_drops_the_carried_inverse():
    """A block moves the accumulators wholesale, so an inverse carried through single frames no
    longer belongs to them and must be reseeded."""
    rng = np.random.default_rng(6)
    F = 8
    d = Dynamics(F)
    for row in rng.standard_normal((3 * F, F)):
        d.update(row)
    assert d.Pinv is not None
    d.update_block(rng.standard_normal((20, F)))
    assert d.Pinv is None
