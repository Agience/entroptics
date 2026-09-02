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


# ── the scalar-sequence MOMENT PENCIL (hankel_spectrum) + generic jackknife ───────────────
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
    """Bit-for-bit agreement with the hand-rolled reflection-positive pencil the mass-gap scripts
    used, so relocating the read into the viewer changes no published number."""
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
