"""The operator/prediction surface and the multi-scale reads:
Dynamics.propagator_full / predict, reads.scale_profile, spectral_optics.resolved_power."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.dynamics import Dynamics, dynamics
from entroptics.reads import scale_profile, spectral_optics
from conftest import build_W


# ── the full-space propagator and one-step prediction ─────────────────────────

def test_propagator_and_predict_recover_known_operator():
    """x_{t+1} = M x_t  =>  Pyx Pxx^+ = M on the reachable subspace, so predict = M x."""
    M = np.array([[0.9, 0.1], [0.0, 0.8]])
    T = 100
    x = np.zeros((T, 2))
    x[0] = [1.0, 0.5]
    for t in range(1, T):
        x[t] = M @ x[t - 1]
    dyn = dynamics(x)
    Ahat = np.asarray(dyn.propagator_full())
    # the trajectory spans R^2, so the operator equals M there
    assert np.max(np.abs(Ahat @ x[50] - x[51])) < 1e-8
    assert np.max(np.abs(np.asarray(dyn.predict(x[-1])) - M @ x[-1])) < 1e-8


def test_propagator_full_shape():
    dyn = dynamics(build_W(7))
    assert np.asarray(dyn.propagator_full()).shape == (48, 48)


def test_empty_operator_predict_is_zero():
    d = Dynamics(4)
    assert np.asarray(d.propagator_full()).shape == (4, 4)
    assert np.allclose(np.asarray(d.predict(np.ones(4))), 0.0)


def test_predict_via_aperture(W):
    ap = A.Aperture(W)
    p = np.asarray(ap.predict(W[-1]))
    assert p.shape == (W.shape[1],) and np.all(np.isfinite(p))


def test_rcond_truncation_runs():
    dyn = dynamics(build_W(3))
    assert np.asarray(dyn.propagator_full(rcond=1e-6)).shape == (48, 48)


# ── resolved_power (the "power above the noise sea") ───────────────────────────

def test_resolved_power_matches_definition(W):
    sp = spectral_optics(W)
    ev = np.asarray(sp.eigenvalues, float)
    above = ev[ev > sp.noise_floor]
    assert sp.resolved_power == pytest.approx(float(np.sum(above - sp.noise_floor)),
                                              rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_resolved_power_nonnegative(seed):
    assert spectral_optics(build_W(seed)).resolved_power >= 0.0


def test_resolved_power_in_optics_dict(W):
    o = A.Aperture(W).optics()
    assert "resolved_power" in o and o["resolved_power"] >= 0.0
    assert A.Aperture(W).resolved_power == o["resolved_power"]


# ── scale_profile (structure vs observation window, in CELLS) ─────────────────

def test_scale_profile_shapes(W):
    sp = scale_profile(W)
    n = len(sp.windows)
    assert n >= 1
    assert len(sp.K_signal) == n and len(sp.coherence) == n
    assert len(sp.a_delta) == n and len(sp.phi_T) == n
    assert np.all(np.diff(sp.windows) > 0)             # strictly increasing window sizes
    assert sp.windows[-1] <= W.shape[0]                # windows are in cells, bounded by T
    assert sp.resolved_window == 0 or sp.resolved_window in sp.windows
    assert sp.dominant_window in sp.windows


def test_scale_profile_custom_windows(W):
    sp = scale_profile(W, windows=[16, 32, 64, 80])
    assert list(sp.windows) == [16, 32, 64, 80]
    assert set(sp.transitions).issubset(set(sp.windows))


def test_scale_profile_windows_in_cells_not_seconds(W):
    """The read must be domain-neutral: windows are ordered-axis cells, never scaled
    by any time unit."""
    sp = scale_profile(W)
    assert sp.windows.dtype.kind in "iu"               # integer cell counts
    assert sp.resolved_window == int(sp.resolved_window)


def test_scale_profile_via_aperture(W):
    sp = A.Aperture(W).scale_profile(windows=[20, 40, 80])
    assert list(sp.windows) == [20, 40, 80]


# ── dominance, robust noise, >2D raise ───────────────────────────────────────

def test_dominance_matches_definition_and_bounds(W):
    sp = spectral_optics(W)
    ev = np.asarray(sp.eigenvalues, float)
    exp = min(1.0, max(0.0, (ev.max() - 1.0) / (ev.size - 1)))
    assert 0.0 <= sp.dominance <= 1.0
    assert sp.dominance == pytest.approx(exp, rel=1e-9, abs=1e-12)


def test_dominance_in_optics_dict(W):
    o = A.Aperture(W).optics()
    assert 0.0 <= o["dominance"] <= 1.0
    assert A.Aperture(W).dominance == o["dominance"]


def test_robust_null_provider_is_deterministic(W):
    from entroptics import null_providers as nulls
    a = spectral_optics(W, null=nulls.robust)
    b = spectral_optics(W, null=nulls.robust)
    assert a.noise_floor == b.noise_floor and a.resolved_modes == b.resolved_modes
    assert np.isfinite(a.noise_floor)


def test_non_callable_null_raises(W):
    with pytest.raises(TypeError):
        spectral_optics(W, null="bogus")        # the floor is a provider callback, not a name


def test_operator_forgetting_and_resolved_reads():
    # stage-(a) foundation: the streaming operator produces the forgetting margin (the
    # aperture-forgetting axiom read) and a resolved K_signal from Pxx -- both O(F^3) once,
    # no O(T^2) batch op.  Noise forgets and resolves ~nothing; a coherent mode has a
    # near-unit margin and resolves.
    import math
    from entroptics.dynamics import Dynamics
    r = np.random.default_rng(0); F = 30
    noise = Dynamics(F)
    for _ in range(200):
        noise.update(r.standard_normal(F))
    fo = noise.forgetting()
    assert fo["forgets"] and fo["margin"] < 0.7 and noise.resolved() <= 3
    v = r.standard_normal(F); v /= np.linalg.norm(v)
    osc = Dynamics(F)
    for t in range(200):
        osc.update(math.cos(0.12 * t) * v + 0.02 * r.standard_normal(F))
    assert osc.forgetting()["margin"] > 0.9          # near-persistent, well above the noise floor
    assert osc.resolved() >= 1                        # resolves the coherent mode from Pxx


def test_update_block_matches_per_frame_loop():
    # the vectorised block ingest equals the per-frame recurrence exactly at forgetting=1,
    # including the boundary when streamed in chunks (so streaming and batch agree).
    from entroptics.dynamics import Dynamics
    rng = np.random.default_rng(0); T, F = 800, 20
    X = rng.standard_normal((T, F))
    loop = Dynamics(F)
    for t in range(T):
        loop.update(X[t])
    blk = Dynamics(F); blk.update_block(X)
    assert np.allclose(loop.Pxx, blk.Pxx) and np.allclose(loop.Pyx, blk.Pyx)
    assert loop.n_pairs == blk.n_pairs
    chunks = Dynamics(F)
    for c in np.array_split(X, 5):
        chunks.update_block(c)                        # boundary transitions across chunks
    assert np.allclose(chunks.Pxx, blk.Pxx) and np.allclose(chunks.Pyx, blk.Pyx)


def test_aperture_window_is_adaptive_minimum_not_a_clock():
    # streaming-only + ADAPTIVE forgetting: `window` is a MINIMUM.  Pure noise (no active
    # signal) forgets to it; a persistent coherent signal is NEVER truncated; the operator
    # accumulates all frames regardless.
    import math
    from entroptics import Aperture
    # noise, T=400 -> forget to the minimum
    noise = np.random.default_rng(0).standard_normal((400, 24))
    apn = Aperture(noise, window=128)
    assert apn.W.shape[0] == 128
    assert apn.dynamics().n_frames == 400                     # operator sees all frames (global)
    assert apn.margin == pytest.approx(apn.dynamics().forgetting()["margin"])
    # a persistent coherent oscillation, T=600 -> kept beyond the minimum (still active)
    r = np.random.default_rng(1); F = 24; v = r.standard_normal(F); v /= np.linalg.norm(v)
    X = np.array([math.cos(0.05 * t) * 3.0 * v + 0.05 * r.standard_normal(F) for t in range(600)])
    apc = Aperture(X, window=128)
    assert apc.W.shape[0] > 128                               # never truncate an active signal
    # a short signal (<= minimum) is kept whole
    assert Aperture(np.random.default_rng(2).standard_normal((80, 24)), window=128).W.shape[0] == 80


def test_operator_resolves_coherent_signal_under_heavy_rfi():
    # sparse is NOT incoherent: a coherent mode buried under heavy RFI (masked NaN cells +
    # dead channels) is still resolved -- missing data contributes nothing to the operator,
    # the valid cells carry the signal.  Noise+RFI stays incoherent (no false signal).
    import math
    from entroptics.dynamics import Dynamics
    r = np.random.default_rng(1); T, F = 400, 30
    v = r.standard_normal(F); v /= np.linalg.norm(v)
    X = np.array([math.cos(0.05 * t) * 3.0 * v + 0.1 * r.standard_normal(F) for t in range(T)])
    Xm = X.copy()
    Xm[r.random((T, F)) < 0.5] = np.nan              # 50% scattered RFI
    Xm[:, [5, 12, 19]] = np.nan                       # fully-dead channels
    d = Dynamics(F).update_block(Xm)
    assert d.resolved() >= 1 and d.forgetting()["margin"] > 0.7     # coherence survives RFI
    N = r.standard_normal((T, F)); N[r.random((T, F)) < 0.5] = np.nan
    dn = Dynamics(F).update_block(N)
    assert dn.resolved() <= 2 and dn.forgetting()["margin"] < 0.7   # missing data -> no false signal


def test_aperture_prefers_reference_null_when_given():
    # the library PREFERS the reference-calibrated null when a signal-free reference is given
    # (else the derived mp default); an explicit null= overrides both.
    import math
    from entroptics import Aperture, null_providers as nulls
    r = np.random.default_rng(0); T, F = 120, 30
    ref = [r.standard_normal((T, F)) for _ in range(50)]           # signal-free reference
    u = r.standard_normal(T); u /= np.linalg.norm(u)
    v = r.standard_normal(F); v /= np.linalg.norm(v)
    W = r.standard_normal((T, F)) + 4.0 * (math.sqrt(T) + math.sqrt(F)) * np.outer(u, v)
    assert Aperture(W).resolved() >= 1                             # no reference -> mp
    ap = Aperture(W, reference=ref)
    assert ap.resolved() >= 1                                      # reference -> reference_null
    assert ap._effective_null("bulk").__name__ == "reference_null"
    assert Aperture(W, reference=ref, null=nulls.robust)._effective_null("bulk") is nulls.robust


def test_operator_significance_consistent_with_resolved():
    # the operator's per-mode p-values and its resolved count are the same object at the
    # same far: resolved() == #(p_k < far), the streaming form of the screen identity.
    from entroptics.dynamics import Dynamics
    r = np.random.default_rng(3); T, F = 400, 80
    X = r.standard_normal((T, F)) @ r.standard_normal((F, F))     # correlated feature structure
    d = Dynamics(F); d.update_block(X)
    sig = d.significance()
    assert int((sig.pvalue < 0.05).sum()) == d.resolved()
    assert (sig.pvalue >= 0.0).all() and (sig.pvalue <= 1.0).all()


def test_randomized_eig_matches_exact_on_lowrank():
    # the O(F^2 k) randomized path resolves the same count as the exact O(F^3) eig on a
    # low-rank signal (a few strong modes over a noise bulk) -- the feature-side lever.
    import math
    from entroptics.dynamics import Dynamics
    for s in range(6):
        r = np.random.default_rng(10 + s); T, F, K = 300, 300, 4
        W = r.standard_normal((T, F))
        for _ in range(K):
            u = r.standard_normal(T); u /= np.linalg.norm(u)
            v = r.standard_normal(F); v /= np.linalg.norm(v)
            W = W + 4.0 * (math.sqrt(T) + math.sqrt(F)) * np.outer(u, v)
        d = Dynamics(F); d.update_block(W)
        assert d.resolved(k=12) == d.resolved()                  # sketch == exact, low-rank


def test_operator_reads_run_on_torch_and_match_numpy():
    # every operator read is backend-agnostic (GPU proxy): torch stays on-device and equals numpy.
    torch = pytest.importorskip("torch")
    from entroptics.dynamics import Dynamics
    X = np.random.default_rng(1).standard_normal((500, 24))
    dn = Dynamics(24); dn.update_block(X)
    dt = Dynamics(24); dt.update_block(torch.as_tensor(X))
    assert isinstance(dt.Pxx, torch.Tensor)                          # never left the backend
    assert dn.resolved() == dt.resolved()
    assert abs(dn.phi_F() - dt.phi_F()) < 1e-9
    assert abs(dn.feature_entropy() - dt.feature_entropy()) < 1e-9
    assert abs(dn.forgetting()["margin"] - dt.forgetting()["margin"]) < 1e-9


def test_spectral_optics_raises_on_high_d():
    with pytest.raises(ValueError):
        spectral_optics(np.random.default_rng(0).standard_normal((4, 4, 4)))
    with pytest.raises(ValueError):
        spectral_optics(np.random.default_rng(0).standard_normal(8))   # 1-D


# ── dominant_decay_rate: the dominant (slowest) operator mode's rate ───────────

def test_dominant_decay_rate_recovers_known_rate_and_is_deterministic():
    """The dominant (slowest) mode's rate -log|mu_1| from the operator.  A rank-1 signal
    decaying at rate alpha has dominant eigenvalue e^{-alpha}, so the read returns alpha.
    It is deterministic (operator eigenvalues)."""
    alpha, T = 0.25, 60
    v = np.array([1.0, 0.5, -0.3, 0.7])
    W = np.exp(-alpha * np.arange(T))[:, None] * v[None, :]        # (T, F) rank-1 decay
    ap = A.Aperture(W)
    assert ap.dominant_decay_rate == pytest.approx(alpha, abs=1e-6)
    assert A.Aperture(W).dominant_decay_rate == ap.dominant_decay_rate   # deterministic


def test_connected_decay_rate_reads_the_fluctuation_and_is_deterministic():
    """The dominant (slowest) mode's rate -log|mu_1| on the CONNECTED (mean-subtracted) spectrum.
    For a large PERSISTENT (constant) mode plus a decaying fluctuation, the raw dominant mode is
    the persistent one (rate ~0) and the connected read is the fluctuation's decay rate.
    Deterministic (operator eigenvalues)."""
    alpha, T = 0.30, 60
    u = np.array([1.0, -1.0, 0.5, -0.5])
    W = 10.0 * np.ones((T, 4)) + np.exp(-alpha * np.arange(T))[:, None] * u[None, :]
    ap = A.Aperture(W)
    assert abs(ap.dominant_decay_rate) < 1e-3         # raw: the persistent (constant) mode, rate ~0
    assert ap.connected_decay_rate > 0.1              # connected: the decaying fluctuation
    assert A.Aperture(W).connected_decay_rate == ap.connected_decay_rate   # deterministic
