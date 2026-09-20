"""Mathematical invariants of the optics reads -- the identities a paper/Lean would
state (and that must hold for any input, not just the golden signal)."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.reads import (decay, concentration, etendue, phi_T, phi_F, strehl,
                              coupling, decay_scatter)
from entroptics.entropy import geometry
from conftest import build_W, build_Wc


# ── algebraic identities between reads ────────────────────────────────────────

def test_etendue_is_product_of_axis_fills(W):
    ap = A.Aperture(W)
    assert ap.etendue == pytest.approx(ap.phi_F * ap.phi_T, rel=1e-12)


def test_magnification_is_reciprocal_of_phi(W):
    ap = A.Aperture(W)
    assert ap.magnification == pytest.approx(1.0 / ap.phi, rel=1e-12)


def test_space_bandwidth_is_product_of_mode_counts(W):
    ap = A.Aperture(W)
    assert ap.space_bandwidth == ap.n_F * ap.n_T


def test_at_diffraction_limit_matches_magnification(W):
    ap = A.Aperture(W)
    assert ap.at_diffraction_limit == (abs(ap.magnification - 1.0) < 1e-9)


# ── bounds every read must respect ────────────────────────────────────────────

@pytest.mark.parametrize("seed", [1, 2, 3, 7])
def test_phi_in_unit_interval(seed):
    ap = A.Aperture(build_W(seed))
    for p in (ap.phi, ap.phi_T, ap.phi_F, ap.etendue):
        assert 0.0 < p <= 1.0 + 1e-12


@pytest.mark.parametrize("seed", [1, 2, 3, 7])
def test_strehl_in_unit_interval(seed):
    assert 0.0 <= A.Aperture(build_W(seed)).strehl <= 1.0 + 1e-12


# ── the entropy noise-guard: pure noise must not fold ─────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pure_noise_does_not_fold(seed):
    """iid noise sits inside the fold band of max entropy -> geometry snaps
    delta to 1.0 on both axes (no spurious adjacent-row correlation)."""
    rng = np.random.default_rng(1000 + seed)
    g = A.Aperture(rng.standard_normal((64, 32))).geometry
    assert g["delta_T"] == 1.0
    assert g["delta_F"] == 1.0


# ── decay / autocorrelation properties ────────────────────────────────────────

def test_decay_peak_at_zero_lag(W):
    """C(0) = variance >= |C(tau)| for all tau (Cauchy-Schwarz on the connected ACF)."""
    c = np.asarray(decay(W))
    assert c[0] >= np.abs(c).max() - 1e-9


def test_decay_of_constant_is_zero(W):
    """A constant signal has no fluctuation -> connected autocovariance is 0."""
    c = np.asarray(decay(np.full((32, 8), 3.7)))
    assert np.allclose(c, 0.0)


def test_toeplitz_autocovariance_is_psd(W):
    """Lemma behind mercer_certificate: the biased ACF's Toeplitz matrix is PSD."""
    c = np.asarray(decay(W))
    T = c.size
    idx = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    lam = np.linalg.eigvalsh(c[idx])
    assert lam.min() >= -1e-8 * max(1.0, abs(lam.max()))


# ── concentration: axial (focus) vs directional (resultant) are distinct ───────

def test_concentration_antipodal_is_axial_not_directional():
    u = np.array([1.0, 0.0, 0.0])
    c = concentration(np.vstack([u] * 10 + [-u] * 10))
    assert c.focus == pytest.approx(1.0, abs=1e-9)      # axially aligned
    assert c.resultant == pytest.approx(0.0, abs=1e-9)  # directionally cancelled


def test_concentration_aligned_is_fully_concentrated():
    u = np.array([1.0, 0.0, 0.0])
    c = concentration(np.vstack([u] * 20))
    assert c.focus == pytest.approx(1.0, abs=1e-9)
    assert c.resultant == pytest.approx(1.0, abs=1e-9)


# ── etendue / phi invariance: relabeling + per-variable phase (not arbitrary mix) ─
#
# The per-axis fills phi_T, phi_F (hence etendue = phi_F * phi_T) are functions of the
# unit-diagonal correlation eigenvalues of an axis' variables.  Relabelling those
# variables (a permutation) permutes the correlation matrix -> same eigenvalues; a
# per-variable phase D (a diagonal unitary) maps C -> D C D^H, which preserves both
# the unit diagonal and the eigenvalues.  So phi is invariant under permutation +
# per-variable phase of its own variable axis.  It is not invariant under an arbitrary
# orthonormal/unitary mixing of the variables (that changes the correlation itself) --
# the negative control below pins exactly that distinction.

def _phase(rng, n):
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, n))

def _rand_orthonormal(rng, n):
    """A random real orthonormal (n, n) matrix via QR (numpy-only, no scipy)."""
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))          # fix signs for a proper orthonormal frame


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_phi_F_invariant_under_feature_relabel_and_phase(seed):
    """phi_F's variables are the feature columns: permute + per-feature phase them
    (samples/rows untouched) -> phi_F is invariant to float tolerance."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 100)
    F = W.shape[1]
    perm = rng.permutation(F)
    Wt = W.astype(complex)[:, perm] * _phase(rng, F)[None, :]
    assert phi_F(W) == pytest.approx(phi_F(Wt), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_phi_T_invariant_under_ordered_relabel(seed):
    """phi_T's variables are the ordered rows, so RELABELING them cannot change how power is
    distributed across them (PAPER Prop 3.5).

    Relabeling only.  A per-time phase is not a gauge of an ordered record -- there is no
    convention under which one time sample's sign is arbitrary and the next one's is not -- and
    holding phi_T invariant under it would mean not removing what is constant in time, which is
    how a static bandpass came to read as a coherent temporal mode.  Per-CHANNEL phase, which IS
    an instrument convention, is a gauge and is tested on phi_F below."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 200)
    perm = rng.permutation(W.shape[0])
    assert phi_T(W) == pytest.approx(phi_T(W[perm, :]), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_a_static_bandpass_is_not_a_temporal_mode(seed):
    """A per-channel level constant in time carries no order at all: shuffling the rows leaves it
    untouched.  It reached the ordered correlation as one vector added to every time point, and
    the read scored it as coherence -- 0.99 on white noise, above what a real periodic signal
    scores.  The ordered reads are formed on the connected screen, so it cannot."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 400)
    ref_T, ref_s = phi_T(W), strehl(W)
    for amp in (1.0, 10.0, 100.0):
        X = W + rng.standard_normal(W.shape[1]) * amp
        assert phi_T(X) == pytest.approx(ref_T, rel=1e-9)
        assert strehl(X) == pytest.approx(ref_s, rel=1e-9)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_etendue_invariant_under_axis_relabeling(seed):
    """etendue = phi_F * phi_T is invariant under relabelling (permutation) of the
    variables on both axes, plus a global phase (the complex path)."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 300)
    T, F = W.shape
    Wt = (W.astype(complex) * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    Wt = Wt[rng.permutation(T), :][:, rng.permutation(F)]
    assert etendue(W) == pytest.approx(etendue(Wt), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_phi_F_changes_under_arbitrary_feature_mixing(seed):
    """Negative control: an arbitrary orthonormal mixing of the feature variables
    changes the unit-diagonal correlation, so phi_F is not invariant -- the claim is
    specific to relabelling + per-variable phase, not general basis change."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 400)
    Q = _rand_orthonormal(rng, W.shape[1])
    assert abs(phi_F(W) - phi_F(W @ Q)) > 1e-3


# ── complex inputs flow end to end ────────────────────────────────────────────

def test_complex_optics_all_finite(Wc):
    o = A.Aperture(Wc).optics()
    for k, v in o.items():
        if isinstance(v, float):
            assert np.isfinite(v) or k == "shape_factor", k


# ── the fold needs continuity, not just concentration ─────────────────────────

def _line_frame(seed=0, T=96, F=64):
    """Sparse and continuous: a narrow line on an axis where adjacency means something."""
    rng = np.random.default_rng(seed)
    line = np.exp(-((np.arange(F) - 20.0) ** 2) / 8.0)
    return rng.standard_normal((T, 1)) @ line[None, :] + 0.05 * rng.standard_normal((T, F))


def _nominal_frame(seed=0, T=96, F=64):
    """Sparse and nominal: a few active but unrelated channels -- concentrated exactly like the
    line frame, and folding it would average signals that have nothing to do with each other."""
    rng = np.random.default_rng(seed)
    X = 0.05 * rng.standard_normal((T, F))
    X[:, [3, 17, 41]] += 4.0 * rng.standard_normal((T, 3))
    return X


def test_feature_adjacency_separates_continuous_from_nominal():
    from entroptics.entropy import feature_adjacency
    assert feature_adjacency(_line_frame()) > 3.0
    assert abs(feature_adjacency(_nominal_frame())) < 2.0


def test_a_continuous_axis_folds():
    X = _line_frame()
    assert geometry(X)["n_F"] < X.shape[1]


def test_a_nominal_axis_does_not_fold_however_concentrated():
    """Concentration alone must not license a fold: these two frames have nearly
    the same feature-marginal entropy, yet only the continuous one may be folded."""
    from entroptics.entropy import feature_adjacency
    line, nominal = _line_frame(), _nominal_frame()
    assert geometry(nominal)["H_F"] < np.log2(nominal.shape[1])      # genuinely concentrated
    assert geometry(nominal)["n_F"] == nominal.shape[1]              # and yet: no fold
    assert geometry(line)["n_F"] < line.shape[1]                     # while this one folds


def test_batched_fold_matches_the_per_frame_fold_on_a_nominal_axis():
    """The batched monitor calls the same fold decision, so bit-identity survives the gate."""
    from entroptics.projection import read_batch, Projection as _S
    frames = [_nominal_frame(0), _line_frame(1), _nominal_frame(2)]
    for f, br in zip(frames, read_batch(frames)):
        sc = _S(f)
        assert br.K_signal == sc.K_signal
        assert br.noise_floor == pytest.approx(float(sc.noise_floor), rel=1e-12)


def test_space_bandwidth_is_a_capacity_and_etendue_times_it_is_the_content():
    """SBW counts the spots the screen COULD carry; etendue x SBW counts the ones it does.

    Stated as the identity: on a screen the fold leaves at native
    resolution, `n_a` is the axis length, so `etendue * SBW` reduces exactly to the product of
    the two axis participation numbers `2^H` -- 1 for a single mode, larger as modes are added.
    The capacity alone cannot tell those frames apart, which is the reason to say so."""
    rng = np.random.default_rng(11)
    T, F = 40, 8
    one_mode = np.outer(rng.standard_normal(T), rng.standard_normal(F))
    many = rng.standard_normal((T, F))

    caps, contents = [], []
    for X in (one_mode, many):
        ap = A.Aperture(X, window=None)
        content = ap.etendue * ap.space_bandwidth
        # the identity: etendue * SBW == 2^H_T * 2^H_F of the axis CORRELATION spectra
        direct = (ap.phi_T * ap.W.shape[0]) * (ap.phi_F * ap.W.shape[1])
        assert content == pytest.approx(direct, rel=1e-12)
        caps.append(ap.space_bandwidth); contents.append(content)

    assert caps[0] == caps[1]                      # capacity cannot separate them
    assert contents[0] == pytest.approx(1.0)       # one mode fills exactly one spot
    assert contents[1] > contents[0]               # more modes fill more of the same capacity
    assert all(c <= caps[0] * (1 + 1e-9) for c in contents)   # content never exceeds capacity


def test_a_pedestal_cannot_change_the_transfer_function():
    """`decay` reads the record it is given.  It used to choose between a coherent and an
    incoherent read from whether any sample was negative -- so subtracting a baseline from an
    intensity, which changes no physics, crossed that branch and moved `a_delta` by 3.6x."""
    rng = np.random.default_rng(0)
    T = 256
    kern = np.exp(-np.arange(T) / 12.0)
    field = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0,
                                rng.standard_normal((T, 32)))
    intensity = field ** 2

    for record in (field, intensity):
        ref = decay(record)
        for pedestal in (-1.0, -1e-3, 0.0, 3.0, 50.0):          # straddles the old branch
            assert np.allclose(np.asarray(decay(record + pedestal)), np.asarray(ref), rtol=1e-9)


def test_the_incoherent_read_is_the_caller_squaring_the_record():
    """The incoherent read did not disappear with the branch -- it is stated instead of guessed,
    and it is exactly the coherent read of the intensity."""
    rng = np.random.default_rng(1)
    field = rng.standard_normal((128, 16))
    assert np.allclose(np.asarray(decay(field ** 2)), np.asarray(decay(np.abs(field) ** 2)))
    assert not np.allclose(np.asarray(decay(field)), np.asarray(decay(field ** 2)))


def test_a_record_that_never_varied_has_no_decay():
    """Subtracting a constant record's own mean leaves the arithmetic's residue, which squared
    into a Gram reads as a decay.  The floor is the T*eps that the mean itself carries."""
    for value in (0.0, 3.7, -12.5, 1e6):
        c = np.asarray(decay(np.full((20, 10), value)))
        assert np.all(c == 0.0), f"constant {value} left a decay"
    varied = np.asarray(decay(np.random.default_rng(0).standard_normal((20, 10))))
    assert varied[0] > 0.0                                       # a record that did vary still reads


def test_a_coordinate_one_side_never_observed_is_not_a_shared_one():
    """`coupling` refuses two different bases outright.  The same argument applies inside one
    basis: a coordinate only one side measured was never compared, so it must not sit in the
    denominator.  Left in, it contributed nothing to the cross term but still carried the other
    side's norm, and the reported strength read about a tenth low."""
    rng = np.random.default_rng(0)
    T, F = 200, 24
    carrier = rng.standard_normal((T, 2))
    a = carrier @ rng.standard_normal((2, F)) + 0.4 * rng.standard_normal((T, F))
    b = carrier @ rng.standard_normal((2, F)) + 0.4 * rng.standard_normal((T, F))

    for n in (4, 8, 12):
        dead = np.sort(rng.choice(F, n, replace=False))
        keep = np.setdiff1d(np.arange(F), dead)
        one_sided = a.copy(); one_sided[:, dead] = np.nan

        got = coupling(one_sided, b)
        truth = coupling(a[:, keep], b[:, keep])          # the coordinates actually compared
        assert got.strength == pytest.approx(truth.strength, rel=1e-12)
        assert got.z == pytest.approx(truth.z, rel=1e-12)
        assert got.sign == truth.sign


def test_coupling_reports_nothing_when_no_coordinate_was_measured_on_both():
    """Two sides that never overlap have nothing to be coupled about."""
    rng = np.random.default_rng(1)
    T, F = 100, 8
    a = rng.standard_normal((T, F)); b = rng.standard_normal((T, F))
    a[:, F // 2:] = np.nan
    b[:, :F // 2] = np.nan
    c = coupling(a, b)
    assert c.strength == 0.0 and c.sign == 0 and not c.resolved


@pytest.mark.parametrize("rho", [None, 2, 8])
def test_the_diffraction_limit_converges_as_channels_are_added(rho):
    """`C` is averaged over the feature axis, so the sampling noise in its tail falls as 1/F.
    That noise widens the entropy, and a wider entropy reads as a longer correlation -- a narrow
    record overstates it.  The read has to close on one answer as channels are added, and to
    approach it from the same side every time; on an uncorrelated record that answer is 1."""
    T = 4000

    def frame(F):
        g = np.random.default_rng(1)
        if rho is None:
            return g.standard_normal((T, F))
        phi = np.exp(-1.0 / rho)
        X = np.zeros((T, F)); e = g.standard_normal((T, F)); X[0] = e[0]
        for t in range(1, T):
            X[t] = phi * X[t - 1] + e[t]
        return X

    seen = [float(A.Aperture(frame(F), window=None).a_delta) for F in (4, 16, 64, 256)]
    assert seen == sorted(seen), f"not monotone in F: {seen}"          # approached from one side
    assert seen[-1] - seen[-2] < seen[1] - seen[0]                     # and closing, not drifting
    if rho is None:
        assert 0.9 < seen[-1] <= 1.0                                   # an uncorrelated record -> 1


# ── how much of a decay the record's own channels disagree about ─────────────

def _ar1_frame(T, F, rho, seed=1):
    phi = np.exp(-1.0 / rho)
    g = np.random.default_rng(seed)
    X = np.zeros((T, F)); e = g.standard_normal((T, F)); X[0] = e[0]
    for t in range(1, T):
        X[t] = phi * X[t - 1] + e[t]
    return X


def test_the_channel_terms_sum_to_the_decay_itself():
    """`decay_scatter` reads the scatter of the per-channel autocovariances that `decay` sums.
    If the two ever computed different objects the scatter would describe something else, so the
    identity is what keeps them one read."""
    rng = np.random.default_rng(0)
    for T, F in ((300, 8), (800, 32)):
        X = rng.standard_normal((T, F))
        Xc = X - X.mean(0)
        n = 1 << int(np.ceil(np.log2(2 * T)))
        spec = np.fft.fft(Xc, n=n, axis=0)
        per_channel = np.real(np.fft.ifft(spec * np.conj(spec), axis=0))[:T] / T
        assert np.allclose(per_channel.sum(1), np.asarray(decay(X)), atol=1e-10)


def test_scatter_separates_a_tail_that_is_noise_from_a_tail_that_is_signal():
    """The point of the read: `tail_share` alone cannot say whether power away from zero lag is
    structure or sampling noise.  The channels can -- they agree about structure and disagree
    about noise."""
    rng = np.random.default_rng(0)
    white = decay_scatter(rng.standard_normal((1500, 64)))
    real_ = decay_scatter(_ar1_frame(1500, 64, 8))

    # an uncorrelated record: every bit of the tail is scatter, so the two shares coincide
    assert white.noise_share == pytest.approx(white.tail_share, rel=0.25)
    # a real correlation length: the tail is large and the channels agree about it
    assert real_.tail_share > 0.5
    assert real_.noise_share < 0.1 * real_.tail_share


@pytest.mark.parametrize("rho", [None, 8])
def test_scatter_falls_as_channels_are_added(rho):
    """More replicates, less disagreement -- the number has to move the way the cure does."""
    def frame(F):
        if rho is None:
            return np.random.default_rng(0).standard_normal((1500, F))
        return _ar1_frame(1500, F, rho)
    shares = [decay_scatter(frame(F)).noise_share for F in (4, 16, 64, 256)]
    assert shares == sorted(shares, reverse=True), f"not falling in F: {shares}"


def test_scatter_counts_only_the_channels_that_were_observed():
    """A channel nothing was observed in is not a replicate."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((600, 32))
    X[:, :8] = np.nan
    assert decay_scatter(X).channels == 24


def test_one_channel_has_nothing_to_disagree_with():
    """Scatter across replicates needs replicates; with a single channel there is no measurement
    to make and none is invented."""
    d = decay_scatter(np.random.default_rng(3).standard_normal((400, 1)))
    assert np.isnan(d.noise_share) and np.isnan(d.tail_share)


# ── the PERIODIC decay: a lag read on an axis that closes ─────────────────────
#
# `decay(W, periodic=True)` reads the ordered axis as a ring.  Its defining property is an
# EQUALITY, not an approximation -- C(tau) == C(T - tau) for every tau on every record -- so the
# tests below assert it with `==` and not with `approx`.  The consumer is a correlator whose
# target satisfies rho(d) = rho(n - d) as a theorem; a read that satisfied it only to within
# round-off would be a different object from the one the theorem is about.


def _circular_decay(X, level):
    """C(tau) = (1/T) sum_i Re<Xc_i, Xc_{i+tau mod T}>, written the slow, obvious way -- an
    independent statement of the definition, so the read is checked against the definition and
    not against a rearrangement of itself."""
    Xc = np.asarray(X) - level
    T = Xc.shape[0]
    return np.array([float((np.conj(Xc) * np.roll(Xc, -t, axis=0)).sum().real) / T
                     for t in range(T)])


@pytest.mark.parametrize("T", [2, 3, 4, 5, 16, 17])
def test_periodic_decay_is_exactly_symmetric_in_the_lag(T):
    """C(tau) == C(T - tau) EXACTLY: the two are the same sum walked the other way round the
    ring, so the read computes one and mirrors it.  Bit equality, at every length, odd and even."""
    X = np.random.default_rng(T).standard_normal((T, 5))
    c = np.asarray(decay(X, periodic=True))
    assert c.size == T
    for t in range(1, T):
        assert c[t] == c[T - t], f"T={T} lag {t}: {c[t]!r} != {c[T - t]!r}"


def test_periodic_decay_is_exactly_symmetric_on_a_complex_record(Wc):
    """Symmetry is a property of how the lag is counted, so it does not depend on the dtype."""
    c = np.asarray(decay(Wc, periodic=True))
    T = c.size
    for t in range(1, T):
        assert c[t] == c[T - t]


def test_periodic_and_linear_disagree_where_the_axis_does_not_close():
    """The two reads are different objects, and the negative control says so.  On this record the
    linear read is [12.5, 0.5, -2.25, -4.5] and the periodic one [12.5, -4, -4.5, -4]: the linear
    read averages lag 3 over the single pair that has a partner, the periodic one wraps and
    averages it over all four."""
    X = np.array([[1.0], [2.0], [3.0], [10.0]])
    lin = np.asarray(decay(X))
    per = np.asarray(decay(X, periodic=True))
    assert lin == pytest.approx([12.5, 0.5, -2.25, -4.5], abs=1e-12)
    assert per == pytest.approx([12.5, -4.0, -4.5, -4.0], abs=1e-12)
    assert not np.allclose(lin, per)
    assert lin[1] != lin[3]                      # the default read carries no such symmetry
    assert per[1] == per[3]


@pytest.mark.parametrize("seed", [0, 3, 9])
def test_periodic_decay_is_the_circular_definition(seed):
    """Against the definition itself, spelled out with np.roll -- a direct lag sum checked against
    a direct lag sum, and no transform on either side.  Wiener-Khinchin licenses this read; it is
    not a step the read or its test takes (see the module header and ``decay``'s docstring)."""
    X = np.random.default_rng(seed).standard_normal((32, 6))
    c = np.asarray(decay(X, periodic=True))
    assert c == pytest.approx(_circular_decay(X, X.mean(0)), abs=1e-12)
    # and the ring is genuinely closed: rolling the record does not move the correlator
    rolled = np.asarray(decay(np.roll(X, 7, axis=0), periodic=True))
    assert rolled == pytest.approx(c, abs=1e-12)


def test_the_periodic_lag_sum_is_a_squared_magnitude_at_every_level():
    """The lag sum telescopes: sum_tau C(tau) = (1/T)|sum_t (X_t - L)|^2 for EVERY level L.

    So it is non-negative whatever L is, and the record's own mean is simply the L that attains
    the floor of zero -- not a degeneracy that supplying a level repairs.  Pinned because the
    opposite reading is easy to reach and would make ``sum_tau C(tau) > 0`` look like a fact
    about correlation, when it only asks whether sum_t X_t differs from T*L."""
    rng = np.random.default_rng(11)
    for _ in range(50):
        T, F = int(rng.integers(2, 12)), int(rng.integers(1, 5))
        X = rng.standard_normal((T, F)) * float(10.0 ** rng.integers(-3, 4))
        level = float(rng.standard_normal() * 10)
        s = float(np.sum(np.asarray(decay(X, periodic=True, disconnected=level))))
        assert s == pytest.approx(float(np.sum((X - level).sum(axis=0) ** 2) / T), rel=1e-9)
        assert s >= 0.0
    # and the own-mean case sits exactly on that floor
    for seed in (0, 1, 2):
        X = np.random.default_rng(seed).standard_normal((24, 4))
        assert float(np.sum(np.asarray(decay(X, periodic=True)))) == pytest.approx(0.0, abs=1e-9)


def test_a_supplied_level_is_the_one_removed():
    """The caller's level lands: the read matches the definition taken at THAT level, and at a
    level that is not the record's own it is a different correlator at every lag -- which is the
    reason to pass one, the ensemble's level being what the target is defined with."""
    rng = np.random.default_rng(5)
    X = rng.standard_normal((24, 4)) + 3.0                 # sits well away from the given level
    level = 2.5                                            # an "ensemble" level, not this record's
    c = np.asarray(decay(X, periodic=True, disconnected=level))
    assert c == pytest.approx(_circular_decay(X, level), abs=1e-12)
    assert not np.allclose(c, np.asarray(decay(X, periodic=True)))   # a different object
    for t in range(1, c.size):
        assert c[t] == c[c.size - t]                       # still exactly symmetric


def test_a_supplied_level_equal_to_the_mean_reproduces_the_default(W):
    """The argument is not decorative: handing over exactly what the default computes returns
    exactly what the default returns, on both the linear and the periodic path."""
    mean = W.mean(0)
    assert np.asarray(decay(W, disconnected=mean)) == pytest.approx(np.asarray(decay(W)), abs=1e-12)
    assert (np.asarray(decay(W, periodic=True, disconnected=mean))
            == pytest.approx(np.asarray(decay(W, periodic=True)), abs=1e-12))


def test_a_per_channel_level_is_accepted_and_a_wrong_shape_is_refused(W):
    """``(F,)`` and ``(1, F)`` are both accepted -- the second is what
    ``ensemble.mean(0, keepdims=True)`` hands you, which is the natural way to produce an
    ensemble level.  A term of any other shape or kind is refused rather than broadcast, so a
    level that was meant to land cannot quietly fail to."""
    F = W.shape[1]
    T = W.shape[0]
    assert np.asarray(decay(W, periodic=True, disconnected=np.zeros(F))).size == T
    assert np.asarray(decay(W, periodic=True, disconnected=W.mean(0, keepdims=True))).size == T
    assert (np.asarray(decay(W, periodic=True, disconnected=W.mean(0, keepdims=True)))
            == pytest.approx(np.asarray(decay(W, periodic=True, disconnected=W.mean(0))),
                             abs=1e-12))
    for bad in (np.zeros(F + 1), np.zeros((2, F)), np.zeros((F, 1))):
        with pytest.raises(ValueError, match="per LIVE feature channel"):
            decay(W, periodic=True, disconnected=bad)
    with pytest.raises(ValueError, match="complex"):
        decay(W, periodic=True, disconnected=1.0 + 2.0j)


def test_a_non_finite_level_is_refused(W):
    """A NaN or inf level makes every lag NaN, and NaN != NaN -- so the exact C(tau) == C(T-tau)
    symmetry this read exists for would silently stop holding.  The guarantee is unqualified, so
    the input that would break it is refused."""
    for bad in (np.nan, np.inf, -np.inf, np.full(W.shape[1], np.nan)):
        with pytest.raises(ValueError, match="finite"):
            decay(W, periodic=True, disconnected=bad)


def test_a_boolean_level_is_refused(W):
    """``disconnected=True`` would silently subtract 1.0.  A level is a number, not a flag."""
    for bad in (True, False, np.bool_(True), np.array([True] * W.shape[1])):
        with pytest.raises(ValueError, match="not a flag"):
            decay(W, periodic=True, disconnected=bad)


def test_the_periodic_read_refuses_a_record_with_a_dead_row():
    """The ring's premise is that the last row neighbours the first.  ``live_view`` drops a
    fully-dead row, which on the default read is one fewer sample but here splices two rows that
    were never adjacent and returns a profile exactly symmetric about the WRONG period -- a wrong
    answer nothing downstream can detect.  So it raises instead of repairing."""
    X = np.random.default_rng(0).standard_normal((16, 3))
    X[5, :] = np.nan
    with pytest.raises(ValueError, match="periodic"):
        decay(X, periodic=True)
    assert np.asarray(decay(X)).size == 15            # the linear read has no such premise
    # a dead COLUMN is fine -- the ring runs along the ordered axis, not the feature axis
    Xc = np.random.default_rng(1).standard_normal((16, 3)); Xc[:, 1] = np.nan
    assert np.asarray(decay(Xc, periodic=True)).size == 16
    # so is a scattered gap: the row survives, imputed
    Xg = np.random.default_rng(2).standard_normal((16, 3)); Xg[5, 1] = np.nan
    assert np.asarray(decay(Xg, periodic=True)).size == 16
    # and a MASK that kills a row is refused the same way -- the route in does not matter
    Xm = np.random.default_rng(3).standard_normal((16, 3))
    m = np.zeros((16, 3), bool)
    m[7, :] = True
    with pytest.raises(ValueError, match="periodic"):
        decay(Xm, m, periodic=True)


def test_a_record_with_nothing_observed_cannot_produce_a_profile():
    """``live_view`` fills an unseen column with 0.0, which centres to 0 under the record's own
    mean but against a SUPPLIED level reads as a constant deviation -- returning level^2 * F at
    every lag from a record containing no data.  Emptiness is detected, not inferred."""
    empty = np.full((8, 3), np.nan)
    for kwargs in ({}, {"disconnected": 2.0}, {"disconnected": None},
                   {"disconnected": np.zeros(3)}):
        assert np.allclose(np.asarray(decay(empty, periodic=True, **kwargs)), 0.0), kwargs
        assert np.allclose(np.asarray(decay(empty, **kwargs)), 0.0), kwargs
    # fully masked is the same statement made a different way
    live = np.random.default_rng(0).standard_normal((8, 3))
    assert np.allclose(
        np.asarray(decay(live, np.ones((8, 3), bool), periodic=True, disconnected=2.0)), 0.0)


def test_an_unresolvable_record_is_zeroed_whoever_supplied_the_level():
    """The round-off guard tests the RESIDUAL, so it covers a supplied level too.  A record with
    no resolvable variation, held against a level that coincides with its mean, leaves only the
    backward error of the subtraction -- which must not be returned as a decay."""
    X = np.full((8, 3), 1e8)
    X[0, 0] = np.nextafter(1e8, 2e8)                  # one ulp: below the arithmetic's resolution
    for level in (X.mean(0), float(X.mean())):
        c = np.asarray(decay(X, periodic=True, disconnected=level))
        assert np.all(c == 0.0), c
        assert float(np.sum(c)) == 0.0
    # but the guard must NOT fire where the residual is real: a high-variance record whose mean
    # happens to equal the supplied level is a perfectly good correlator
    W2 = np.random.default_rng(4).standard_normal((32, 4)) * 5
    assert not np.allclose(np.asarray(decay(W2, periodic=True, disconnected=W2.mean(0))), 0.0)


def test_disconnected_none_subtracts_nothing(W):
    """``None`` is a caller saying the record is already connected -- distinct from omitting the
    argument, which is the record's own mean."""
    c = np.asarray(decay(W, periodic=True, disconnected=None))
    assert c == pytest.approx(_circular_decay(W, 0.0), abs=1e-12)
    assert not np.allclose(c, np.asarray(decay(W, periodic=True)))


def test_a_flat_record_against_a_supplied_level_is_not_zeroed():
    """The round-off guard belongs to the record's OWN mean: subtracting it from a constant leaves
    nothing but backward error.  Against a supplied level a constant record is a real, constant
    deviation, and its correlator is that constant's power at every lag -- a measurement, which
    the guard must not delete."""
    X = np.full((8, 3), 4.0)
    assert np.allclose(np.asarray(decay(X, periodic=True)), 0.0)          # own mean -> round-off
    c = np.asarray(decay(X, periodic=True, disconnected=1.0))             # deviation of 3.0
    assert c == pytest.approx(np.full(8, 3.0 * 3.0 * 3), abs=1e-12)       # 3 channels x 9


def test_the_default_decay_is_untouched_by_the_new_options(W, Wc):
    """One path, not two: the periodic read is an option on the existing lag read, and the read
    every existing caller already makes must be the same arithmetic it was."""
    for X in (W, Wc, np.array([[1.0], [2.0], [3.0], [10.0]])):
        Xc = np.asarray(X) - np.asarray(X).mean(0)
        T = Xc.shape[0]
        lin = np.array([float((np.conj(Xc[:T - t]) * Xc[t:]).sum().real) / T for t in range(T)])
        assert np.asarray(decay(X)) == pytest.approx(lin, abs=1e-10)
