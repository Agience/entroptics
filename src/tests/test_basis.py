"""The write path: Aperture.basis() -- encode, decode, split, the round-trip certificate, drift."""
import numpy as np
import pytest

from entroptics import Aperture, Basis, Drift, RoundTrip


def _process(seed, F=40, K=3, hetero=True):
    rng = np.random.default_rng(seed)
    Bt = np.linalg.qr(rng.standard_normal((F, K)))[0].T
    nz = np.exp(rng.uniform(-0.5, 0.5, F)) if hetero else np.ones(F)
    amp = np.array([6.0, 4.0, 3.0])[:K]

    def rec(T=600):
        return (rng.standard_normal((T, K)) * amp) @ Bt + rng.standard_normal((T, F)) * nz + 5.0
    return rng, rec, nz


def test_the_basis_is_the_resolved_rank_and_orthonormal():
    _, rec, _ = _process(0)
    b = Aperture(rec()).basis()
    assert isinstance(b, Basis) and b.K == 3 and b.F == 40
    assert np.allclose(b.B @ b.B.conj().T, np.eye(3), atol=1e-12)
    assert np.all(np.diff(b.power) <= 0)                     # strongest first


def test_the_round_trip_is_certified_at_round_off():
    _, rec, _ = _process(1)
    W = rec()
    b = Aperture(W).basis()
    c = b.certify(W)
    assert isinstance(c, RoundTrip)
    for v in (c.defect, c.inverse, c.idempotent, c.lossless):
        assert v < 1e-12
    assert 0.0 < c.captured < 1.0


def test_split_is_lossless_and_decode_inverts_encode():
    _, rec, _ = _process(2)
    b = Aperture(rec()).basis()
    W2 = rec()                                               # a later frame, coded by the shared basis
    resolved, residual = b.split(W2)
    assert np.max(np.abs(resolved + residual - W2)) < 1e-12 * np.max(np.abs(W2))
    A = b.encode(W2)
    assert A.shape == (600, 3)
    assert np.allclose(b.encode(b.decode(A)), A, atol=1e-10)
    assert np.allclose(b.decode(A), resolved, atol=1e-10)


def test_a_frame_in_the_span_comes_back_exactly():
    _, rec, _ = _process(3)
    b = Aperture(rec()).basis()
    A = np.random.default_rng(3).standard_normal((50, 3))
    frame = b.decode(A)
    resolved, residual = b.split(frame)
    assert np.allclose(resolved, frame, atol=1e-10) and np.max(np.abs(residual)) < 1e-10


def test_gaps_and_unmeasured_channels():
    _, rec, _ = _process(4)
    W = rec()
    W[:, 7] = np.nan                                         # a channel the source never measured
    b = Aperture(W).basis()
    assert np.all(b.B[:, 7] == 0) and np.isnan(b.centre[7])
    W2 = rec()
    W2[5, 3] = np.nan
    resolved, residual = b.split(W2)
    assert np.isnan(resolved[5, 3]) and np.isnan(residual[5, 3])
    assert np.isnan(resolved[:, 7]).all()
    ok = np.isfinite(resolved)
    assert np.max(np.abs((resolved + residual - W2)[ok])) < 1e-12 * np.nanmax(np.abs(W2))


def test_a_noiseless_channel_is_its_centre():
    _, rec, _ = _process(5)
    W = rec()
    W[:, 2] = 1.25
    b = Aperture(W).basis()
    resolved, _ = b.split(rec())
    assert np.all(resolved[:, 2] == 1.25)


def test_drift_reads_nothing_on_the_same_process_and_finds_new_structure():
    """Both directions: a later record from the same process gives K == 0 (heteroscedastic
    channel noise included), and the same record with one planted extra mode gives K >= 1."""
    rng, rec, nz = _process(6)
    b = Aperture(rec()).basis()
    same = b.drift(rec())
    assert isinstance(same, Drift) and same.K == 0
    u = np.linalg.qr(rng.standard_normal((40, 1)))[0][:, 0]
    assert b.drift(rec() + np.outer(rng.standard_normal(600) * 1.5, u)).K >= 1
    W3 = rec()
    W3[200:260, 10:16] += 2.0 * np.hanning(60)[:, None] * nz[10:16]   # a local transient
    assert b.drift(W3).K >= 1


def test_select_and_refusals():
    _, rec, _ = _process(7)
    ap = Aperture(rec())
    b = ap.basis()
    s = b.select([2, 0, 2])
    assert s.K == 2 and np.array_equal(s.B, b.B[[2, 0]])
    with pytest.raises(ValueError):
        b.select([3])
    with pytest.raises(ValueError):
        ap.basis(modes=[-1])
    with pytest.raises(ValueError):
        ap.basis(modes=[0.5])
    with pytest.raises(ValueError):
        b.encode(np.zeros((10, 39)))
    assert ap.basis(modes=[1]).K == 1


def test_complex_records():
    rng = np.random.default_rng(8)
    T, F = 400, 24
    Bt = np.linalg.qr(rng.standard_normal((F, 2)) + 1j * rng.standard_normal((F, 2)))[0].T
    W = (rng.standard_normal((T, 2)) * [5, 3]) @ Bt + rng.standard_normal((T, F)) + 1j * rng.standard_normal((T, F))
    b = Aperture(W).basis()
    c = b.certify(W)
    assert b.K == 2 and max(c.defect, c.inverse, c.idempotent, c.lossless) < 1e-12


def test_a_noise_record_has_an_empty_basis():
    rng = np.random.default_rng(9)
    b = Aperture(rng.standard_normal((600, 40))).basis()
    assert b.K == 0
    W2 = rng.standard_normal((600, 40))
    resolved, residual = b.split(W2)
    assert np.allclose(resolved, b.centre[None, :]) and b.encode(W2).shape == (600, 0)


def _complex_process(seed, F=12, K=2):
    rng = np.random.default_rng(seed)
    Bt = np.linalg.qr(rng.standard_normal((F, K)) + 1j * rng.standard_normal((F, K)))[0].T

    def draw(T=500):
        return ((rng.standard_normal((T, K)) + 1j * rng.standard_normal((T, K))) * [5, 3][:K]) @ Bt             + rng.standard_normal((T, F)) + 1j * rng.standard_normal((T, F))
    return draw


def test_drift_complement_matches_a_dense_complement_on_complex_records():
    """The Householder complement must be the complement of the span the residual is orthogonal
    to: its coordinates carry the whole residual, with the same singular values as a dense one."""
    draw = _complex_process(10)
    b = Aperture(draw()).basis()
    W2 = draw()
    d = b.drift(W2)
    R = np.asarray(d.projection.W)
    act = np.isfinite(b.centre)
    X = W2[:, act] - b.centre[act]
    from entroptics.basis import noise_scale
    s = noise_scale(X, b.B[:, act] * b.scale[act], b.scale[act])
    Xs = X / s
    _, sv, Vh = np.linalg.svd(b.B[:, act] * b.scale[act] / s, full_matrices=False)
    Ba = Vh[:b.K]
    Q = np.linalg.qr(Ba.T, mode="complete")[0]
    dense = (Xs - (Xs @ Ba.conj().T) @ Ba) @ Q.conj()[:, b.K:]
    assert np.allclose(np.linalg.svd(R, compute_uv=False), np.linalg.svd(dense, compute_uv=False),
                       rtol=1e-10)
    assert np.linalg.norm(R) == pytest.approx(np.linalg.norm(Xs - (Xs @ Ba.conj().T) @ Ba), rel=1e-12)


def test_complex_drift_reads_nothing_on_the_same_process():
    draw = _complex_process(11)
    b = Aperture(draw()).basis()
    assert b.K == 2
    assert [b.drift(draw()).K for _ in range(4)] == [0, 0, 0, 0]


def test_certify_on_a_noiseless_channel_is_clean():
    """The suite turns a RuntimeWarning into an error, so a 0/0 on the noiseless channel fails."""
    _, rec, _ = _process(12)
    W = rec()
    W[:, 2] = 7.0
    c = Aperture(W).basis().certify(W)
    assert max(c.defect, c.inverse, c.idempotent, c.lossless) < 1e-12


def test_drift_holds_its_level_when_the_span_weighs_on_few_channels():
    """Removing a span takes its share of each channel's noise (sigma^2 (1 - P_ff)).  On 12
    channels with a rank-2 span that share is large and uneven; unless the noise scale is
    corrected for it, white noise reads as structure (20-45% of records before the correction)."""
    rng = np.random.default_rng(13)
    Br = np.linalg.qr(rng.standard_normal((12, 2)))[0].T

    def rec():
        return (rng.standard_normal((500, 2)) * [5, 3]) @ Br + rng.standard_normal((500, 12))
    b = Aperture(rec()).basis()
    assert [b.drift(rec()).K for _ in range(6)] == [0] * 6


def test_noise_scale_is_the_exact_moment_solution():
    """The Woodbury solve equals the dense one, m = |I - P|^2 x, for real and complex spans; a
    variance solved at or below zero is one the data does not resolve, and the metric stands."""
    from entroptics.basis import noise_scale
    rng = np.random.default_rng(20)
    for cx in (False, True):
        F, K, T = 9, 3, 50
        D = rng.standard_normal((T, F)) + (1j * rng.standard_normal((T, F)) if cx else 0)
        Bs = rng.standard_normal((K, F)) + (1j * rng.standard_normal((K, F)) if cx else 0)
        s0 = np.exp(rng.uniform(-1, 1, F))
        Ba = np.linalg.svd(Bs / s0, full_matrices=False)[2]
        X = D / s0
        R = X - (X @ Ba.conj().T) @ Ba
        A = np.abs(np.eye(F) - Ba.conj().T @ Ba) ** 2
        x = np.linalg.solve(A, np.mean(np.abs(R) ** 2, axis=0))
        assert np.allclose(noise_scale(D, Bs, s0), np.sqrt(np.where(x > 0, x, 1.0)) * s0, rtol=1e-10)


def test_noise_scale_is_unbiased_in_a_wrong_metric():
    """Negative control on the metric: with s0 off by up to 2x, the mean solved variance still
    lands on the true one (the plain RMS off the span misses by the span's share)."""
    from entroptics.basis import noise_scale
    rng = np.random.default_rng(21)
    F, K, T = 12, 2, 400
    sig = np.exp(rng.uniform(-0.5, 0.5, F))
    Bs = rng.standard_normal((K, F))
    s0 = sig * np.exp(rng.uniform(-0.7, 0.7, F))
    est = np.mean([noise_scale(rng.standard_normal((T, F)) * sig, Bs, s0) ** 2 for _ in range(200)], axis=0)
    assert np.allclose(est / sig ** 2, 1.0, atol=0.03)


def test_the_basis_scale_is_the_noise_not_the_signal():
    rng, rec, nz = _process(22)
    b = Aperture(rec()).basis()
    assert np.allclose(b.scale / nz, 1.0, atol=0.12)


def test_noise_scale_with_a_channel_at_half_weight():
    """d_f = 1 - 2 p_f is 0 when a channel puts half its weight in the span; Woodbury would divide by
    it, the system itself is sound, and the answer must still be the dense solution."""
    from entroptics.basis import noise_scale
    rng = np.random.default_rng(23)
    F = 5
    b = np.zeros(F)
    b[0] = np.sqrt(0.5)
    b[1:] = rng.standard_normal(F - 1)
    b[1:] *= np.sqrt(0.5) / np.linalg.norm(b[1:])
    D = rng.standard_normal((400, F))
    A = np.abs(np.eye(F) - np.outer(b, b)) ** 2
    R = D - np.outer(D @ b, b)
    dense = np.sqrt(np.clip(np.linalg.lstsq(A, np.mean(R ** 2, axis=0), rcond=None)[0], 0, None))
    got = noise_scale(D, b[None, :], np.ones(F))
    assert np.all(np.isfinite(got)) and np.allclose(got, dense, rtol=1e-10)


def test_a_channel_inside_the_span_keeps_its_metric():
    """A channel lying in the span leaves no residual, so its noise cannot be seen: its metric
    stands, and nothing divides by the singular system (this crashed before)."""
    from entroptics import Basis
    from entroptics.basis import noise_scale
    rng = np.random.default_rng(24)
    b = Basis(B=np.eye(10)[3:4], centre=np.zeros(10), scale=np.ones(10), power=np.ones(1))
    assert b.drift(rng.standard_normal((300, 10))).K == 0
    s = noise_scale(rng.standard_normal((300, 10)), np.r_[1.0, 1e-9 * np.ones(9)][None, :], np.ones(10))
    assert s[0] == 1.0 and np.all(np.isfinite(s))


@pytest.mark.parametrize("F,K", [(2, 1), (4, 2), (5, 3), (7, 4)])
def test_unidentifiable_noise_moves_the_metric_only_as_far_as_the_data_can(F, K):
    """With more channels than the residual's covariance has numbers, per-channel noise is not
    identifiable.  The estimate must stay finite and bounded by what the metric already knew (it
    was off by 100-1400% before), not return the pseudo-solution of a singular system."""
    from entroptics.basis import noise_scale
    rng = np.random.default_rng(25 + F)
    sig = np.exp(rng.uniform(-0.5, 0.5, F))
    Bs = rng.standard_normal((K, F))
    s0 = sig * np.exp(rng.uniform(-0.3, 0.3, F))
    est = noise_scale(rng.standard_normal((20000, F)) * sig, Bs, s0)
    assert np.all(np.isfinite(est)) and np.all(est > 0)
    assert np.max(np.abs(np.log(est / sig))) <= np.max(np.abs(np.log(s0 / sig))) + 0.1


def test_drift_with_no_room_outside_the_span():
    """When the span fills every channel that is left, nothing lies outside it to read: K = 0 and
    no projection, rather than a Projection of a record with no columns (a crash before)."""
    from entroptics import Basis
    b = Basis(B=np.eye(3), centre=np.zeros(3), scale=np.ones(3), power=np.ones(3))
    d = b.drift(np.random.default_rng(26).standard_normal((100, 3)))
    assert d.K == 0 and d.projection is None


def test_a_real_span_in_a_complex_dtype_is_counted_as_real():
    """Whether |I - P|^2 can have full rank depends on the span, not its dtype: a real span stored
    as complex must give what it gives as real (Woodbury ran rank-deficient on it before)."""
    from entroptics.basis import noise_scale
    rng = np.random.default_rng(27)
    sig = np.array([1.0, 2.0, 0.5, 1.5])
    Bs = rng.standard_normal((2, 4))
    D = rng.standard_normal((200000, 4)) * sig
    a = noise_scale(D.astype(complex), Bs.astype(complex), np.ones(4))
    b = noise_scale(D, Bs, np.ones(4))
    assert np.allclose(a, b, rtol=1e-8)
    # 4 channels, 3 identifiable numbers: the estimate improves on the metric, it cannot reach truth
    assert np.max(np.abs(np.log(b / sig))) < np.max(np.abs(np.log(1.0 / sig)))


def test_round_off_rank_is_never_read_as_information():
    """Complex K = 3 of F = 4: |I - P|^2 has rank 1, and round-off singular values just above the
    default cutoff amplified sampling scatter (5 of these 40 ended further from the truth than the
    metric before the rank was capped at its structural count)."""
    from entroptics.basis import noise_scale
    worse = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        F, K = 4, 3
        sg = np.exp(rng.uniform(-0.5, 0.5, F))
        Bs = rng.standard_normal((K, F)) + 1j * rng.standard_normal((K, F))
        s0 = sg * np.exp(rng.uniform(-0.3, 0.3, F))
        D = (rng.standard_normal((20000, F)) + 1j * rng.standard_normal((20000, F))) * sg / np.sqrt(2)
        est = noise_scale(D, Bs, s0)
        worse += np.max(np.abs(np.log(est / sg))) > np.max(np.abs(np.log(s0 / sg))) + 0.1
    assert worse == 0


def test_a_phase_rotated_real_span_is_never_worse_than_its_metric():
    """A real span with a phase per channel has a real span's |P|^2, so its system can be short of
    rank though the span reads as complex.  With exact moments the estimate must never be further
    from the truth than the metric (in x = sigma^2 / s0^2); it was up to 1e12 times further."""
    from entroptics.basis import _noise_scale
    for t in range(60):
        rng = np.random.default_rng(t)
        F, K, T = 4, 2, 4000
        sig = np.exp(rng.uniform(-0.8, 0.8, F))
        Bs = rng.standard_normal((K, F)) * np.exp(1j * rng.uniform(0, 2 * np.pi, F))
        Z = rng.standard_normal((T, F))
        Z = Z @ np.linalg.inv(np.linalg.cholesky(np.cov(Z.T, bias=True)).T)   # exactly white
        s, identified = _noise_scale(Z * sig, Bs, np.ones(F))
        assert not identified
        assert np.linalg.norm(s ** 2 - sig ** 2) <= np.linalg.norm(1 - sig ** 2) * (1 + 1e-9)


def test_drift_says_when_the_noise_is_not_identified():
    """Few channels outside the span cannot tell every channel's noise apart; drift says so
    rather than holding a level it cannot hold.  The 4-channel span is spread over every channel
    at equal strength, so both of its modes clear the read's edge (a mode confined to one channel
    of four, or a second mode small beside the first, does not)."""
    for F, K, amp, expect in ((4, 2, [6, 6], False), (40, 3, [6, 4, 3], True)):
        rng = np.random.default_rng(F)
        Bt = (np.array([[1.0, 1, 1, 1], [1, -1, 1, -1]]) / 2.0 if F == 4
              else np.linalg.qr(rng.standard_normal((F, K)))[0].T)
        nz = np.exp(rng.uniform(-0.5, 0.5, F))

        def rec():
            return (rng.standard_normal((600, K)) * amp) @ Bt + rng.standard_normal((600, F)) * nz
        assert Aperture(rec()).basis().drift(rec()).identified is expect


def test_a_channel_nearly_inside_a_rank_one_span_is_not_called_identified():
    """K = 1, one channel within 1e-7 of the span: Woodbury's 1 x 1 inner matrix is always its own
    largest value, so its round-off must be judged against the sum it cancels (scales of 1e14 came
    back as identified before).  A channel exactly inside the span is not identified either."""
    from entroptics.basis import _noise_scale
    for seed in (0, 1739, 4103, 5945):
        rng = np.random.default_rng(seed)
        F = 4
        b = 3e-7 * (rng.standard_normal(F) + 1j * rng.standard_normal(F))
        b[0] = 1
        Z = rng.standard_normal((200, F)) + 1j * rng.standard_normal((200, F))
        Z -= Z.mean(0)
        Z = Z @ np.linalg.inv(np.linalg.cholesky(Z.conj().T @ Z / 200).conj().T)
        s, identified = _noise_scale(Z, b[None, :], np.ones(F))
        assert np.allclose(s, 1.0, atol=1e-6) and not identified
    s, identified = _noise_scale(np.random.default_rng(1).standard_normal((300, 5)),
                                 np.eye(5)[:1], np.ones(5))
    assert s[0] == 1.0 and not identified
