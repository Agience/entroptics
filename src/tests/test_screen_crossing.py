"""The two-way screen: the signed coupling read and the Screen that carries it.

The load-bearing claim is that the coupling is a measurement with an exact null -- so the
first test here validates the closed-form permutation variance against brute force, and the
rest check the sign is measured (planted aligned / anti-aligned / independent), that the
level is calibrated, and that the screen reads on the un-folded basis.
"""
import numpy as np
import pytest

from entroptics import Screen, Beam
from entroptics.entropy import geometry
from entroptics.reads import Coupling, coupling, etendue as E_etendue
from entroptics.projection import Projection


def _pair(seed=0, T=64, D=6, rho=0.0, amp=1.0):
    """Two (T, D) sides sharing a planted carrier at strength ``rho`` (signed)."""
    rng = np.random.default_rng(seed)
    carrier = rng.standard_normal((T, 1))
    a = amp * carrier + rng.standard_normal((T, D))
    b = rho * amp * carrier + rng.standard_normal((T, D))
    return a, b


# ── the exact permutation null (the claim everything else rests on) ───────────

@pytest.mark.parametrize("seed", [0, 3])
def test_permutation_variance_matches_brute_force(seed):
    """Var_pi[Re S] = tr(C_a C_b)/(T-1) -- the closed form against 20k actual permutations."""
    rng = np.random.default_rng(seed)
    T, D = 40, 5
    A = rng.standard_normal((T, D)); B = rng.standard_normal((T, D))
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    closed = float(np.sum((Ac.T @ Ac) * (Bc.T @ Bc)) / (T - 1))
    draws = np.empty(20000)
    for i in range(draws.size):
        draws[i] = float(np.sum(Ac[rng.permutation(T)] * Bc))
    assert draws.mean() == pytest.approx(0.0, abs=4.0 * draws.std() / np.sqrt(draws.size))
    assert draws.var() == pytest.approx(closed, rel=0.05)


def test_z_is_the_standardised_alignment(W):
    """The reported z is exactly Re S / sqrt(closed-form variance) -- no hidden rescaling."""
    A, B = _pair(1, rho=0.8)
    Ac = A - A.mean(0); Bc = B - B.mean(0)
    T = A.shape[0]
    expect = float(np.sum(Ac * Bc)) / np.sqrt(np.sum((Ac.T @ Ac) * (Bc.T @ Bc)) / (T - 1))
    assert coupling(A, B).z == pytest.approx(expect, rel=1e-12)


# ── the sign is measured (acceptance criterion 2) ─────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 5, 9])
def test_aligned_sides_couple_positive(seed):
    c = coupling(*_pair(seed, rho=1.0, amp=1.5))
    assert c.resolved and c.sign == +1 and c.strength > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 9])
def test_anti_aligned_sides_couple_negative(seed):
    c = coupling(*_pair(seed, rho=-1.0, amp=1.5))
    assert c.resolved and c.sign == -1 and c.strength < 0.0


def test_independent_sides_read_zero():
    """Unrelated sides: nothing resolves, so the coupling is exactly 0.0 -- not a small number."""
    c = coupling(*_pair(4, rho=0.0))
    assert not c.resolved and c.sign == 0 and c.strength == 0.0


def test_false_alarm_rate_matches_the_level():
    """The level is calibrated: independent sides fire at about `far`, not more."""
    fired = sum(coupling(*_pair(1000 + s, T=64, D=4, rho=0.0), far=0.05).resolved
                for s in range(200))
    assert fired / 200.0 < 0.12          # 0.05 + ~3 binomial standard errors


def test_sign_flips_with_the_side_but_magnitude_does_not():
    A, B = _pair(2, rho=0.9, amp=1.5)
    pos, neg = coupling(A, B), coupling(A, -B)
    assert pos.sign == -neg.sign
    assert pos.strength == pytest.approx(-neg.strength, rel=1e-12)


# ── bounds and identities ─────────────────────────────────────────────────────

def test_self_coupling_is_unit_strength():
    A, _ = _pair(6)
    c = coupling(A, A)
    assert c.sign == +1 and c.strength == pytest.approx(1.0, rel=1e-12)
    assert coupling(A, -A).strength == pytest.approx(-1.0, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_strength_and_tightness_are_bounded(seed):
    c = coupling(*_pair(seed, rho=0.5))
    assert -1.0 - 1e-12 <= c.strength <= 1.0 + 1e-12
    assert 0.0 <= c.tightness <= 1.0 + 1e-12


def test_coupling_is_symmetric_in_its_two_sides():
    A, B = _pair(8, rho=0.7)
    assert coupling(A, B).z == pytest.approx(coupling(B, A).z, rel=1e-12)


def test_scale_invariance_of_sign_and_strength():
    A, B = _pair(3, rho=0.8, amp=1.5)
    base, scaled = coupling(A, B), coupling(10.0 * A, 0.1 * B)
    assert base.sign == scaled.sign
    assert base.strength == pytest.approx(scaled.strength, rel=1e-10)


def test_tightness_separates_one_mode_from_many():
    """One shared mode reads tight; independently shared modes read loose."""
    rng = np.random.default_rng(21)
    T, D = 256, 8
    one = rng.standard_normal((T, 1))
    tight = coupling(one @ rng.standard_normal((1, D)) + 0.1 * rng.standard_normal((T, D)),
                     one @ rng.standard_normal((1, D)) + 0.1 * rng.standard_normal((T, D)))
    many = rng.standard_normal((T, D))
    loose = coupling(many + 0.1 * rng.standard_normal((T, D)),
                     many + 0.1 * rng.standard_normal((T, D)))
    assert tight.tightness > 0.9 > loose.tightness


# ── the sign exists only where the lens carries one ──────────────────────────

def test_no_shared_basis_refuses_rather_than_inventing_a_sign():
    """Different widths are two bases.  The only basis-free statistic is non-negative and has
    no exact null, so there is no sign to report, and this raises."""
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="ONE shared basis"):
        coupling(rng.standard_normal((64, 5)), rng.standard_normal((64, 8)))


def test_a_phase_carrying_side_has_no_sign():
    """A U(1)-like pair -- one side is the other turned by a quarter turn -- couples with
    full phase and no sign: the sign is only the real shadow of the phase."""
    rng = np.random.default_rng(15)
    A = rng.standard_normal((80, 4)) + 1j * rng.standard_normal((80, 4))
    c = coupling(A, 1j * A)
    assert c.sign == 0 and c.strength == 0.0
    assert c.phase == pytest.approx(np.pi / 2, abs=1e-9)
    assert coupling(A, A).phase == pytest.approx(0.0, abs=1e-9)
    assert abs(coupling(A, -A).phase) == pytest.approx(np.pi, abs=1e-9)


def test_complex_sides_couple_on_the_real_alignment():
    """The real embedding is exact: Re<a,b>_C standardises like any real pair."""
    rng = np.random.default_rng(17)
    A = rng.standard_normal((96, 4)) + 1j * rng.standard_normal((96, 4))
    c = coupling(A, A + 0.2 * (rng.standard_normal((96, 4)) + 1j * rng.standard_normal((96, 4))))
    assert c.resolved and c.sign == +1


# ── degenerate inputs ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(2, 4), (0, 4), (5, 0)])
def test_degenerate_shapes_read_zero(shape):
    z = np.zeros(shape)
    c = coupling(z, z)
    assert isinstance(c, Coupling) and c.strength == 0.0 and not c.resolved


def test_constant_side_reads_zero():
    A, _ = _pair(0)
    c = coupling(A, np.ones_like(A))
    assert c.strength == 0.0 and not c.resolved


def test_mismatched_ordered_axis_raises():
    with pytest.raises(ValueError, match="ORDERED axis"):
        coupling(np.zeros((10, 3)), np.zeros((11, 3)))


def test_non_2d_raises():
    with pytest.raises(ValueError, match="2-D"):
        coupling(np.zeros(10), np.zeros((10, 3)))


def test_deterministic():
    A, B = _pair(0, rho=0.6)
    assert coupling(A, B) == coupling(A, B)


def test_numpy_torch_parity():
    torch = pytest.importorskip("torch")
    A, B = _pair(5, rho=0.7)
    n, t = coupling(A, B), coupling(torch.from_numpy(A), torch.from_numpy(B))
    assert t.z == pytest.approx(n.z, rel=1e-10)
    assert t.strength == pytest.approx(n.strength, rel=1e-10)
    assert t.sign == n.sign and t.resolved == n.resolved


# ── the Screen ──────────────────────────────────────────────────────────────

def _rotation(D, seed=0):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((D, D)))
    return q


def _linear_lens_of(Q):
    return dict(entry=lambda X: np.asarray(X) @ Q, inverse=lambda C: np.asarray(C) @ Q.T)


def _crossing(seed=0, D=6, rho=1.0):
    Qa, Qb = _rotation(D, 1), _rotation(D, 2)
    m = Screen()
    m.register("a", **_linear_lens_of(Qa))
    m.register("b", **_linear_lens_of(Qb))
    sa, sb = _pair(seed, D=D, rho=rho, amp=1.5)
    m.place("a", sa @ Qa.T)          # surfaces chosen so entry lands the planted pair
    m.place("b", sb @ Qb.T)
    return m, sa, sb


def test_registered_lens_round_trips_losslessly():
    """Acceptance 1: render(entry(x)) is x again, and the certificate says so."""
    m, _, _ = _crossing()
    x = np.random.default_rng(3).standard_normal((64, 6))
    assert m.lossless("a", x) == pytest.approx(0.0, abs=1e-12)
    cert = m.certify("a", x)
    assert cert.lossless and cert.sigma_top < cert.noise_floor


def _structured_surface(seed=4, T=64, D=6):
    rng = np.random.default_rng(seed)
    return 3.0 * rng.standard_normal((T, 1)) + rng.standard_normal((T, D))


def test_a_lens_that_drops_a_resolved_direction_fails_its_certificate():
    """The leak the certificate exists to catch: structure the aperture CAN resolve, gone."""
    m, Q = Screen(), _rotation(6, 1)

    def drop_first(C):
        C = np.array(C, copy=True)
        C[:, 0] = 0.0                                    # lose one whole direction of the basis
        return C @ Q.T

    m.register("lossy", entry=lambda X: np.asarray(X) @ Q, inverse=drop_first)
    x = _structured_surface()
    cert = m.certify("lossy", x)
    assert cert.residual > 0.1 and cert.K_signal >= 1 and not cert.lossless


def test_a_lens_worse_than_returning_nothing_fails_its_certificate():
    """The other leak: a residual that resolves nothing but exceeds the null conversion."""
    m, Q = Screen(), _rotation(6, 1)
    noise = np.random.default_rng(9).standard_normal((64, 6)) * 20.0
    m.register("junk", entry=lambda X: np.asarray(X) @ Q, inverse=lambda C: noise)
    cert = m.certify("junk", _structured_surface())
    assert cert.residual > 1.0 and not cert.lossless


def test_cross_lens_render_goes_through_the_shared_basis():
    """Acceptance 1: enter from one side, exit to the other -- one composition, no table."""
    m, _, _ = _crossing()
    out = m.render("b")
    assert out is not None and out.shape == (64, 6)
    assert np.all(np.isfinite(out))


def test_screen_couple_measures_the_planted_sign():
    """Acceptance 2, through the front door."""
    assert _crossing(0, rho=1.0)[0].couple("a", "b") > 0.0
    assert _crossing(0, rho=-1.0)[0].couple("a", "b") < 0.0
    assert _crossing(0, rho=0.0)[0].couple("a", "b") == 0.0


def test_couple_matches_the_underlying_read():
    m, sa, sb = _crossing(7)
    assert m.couple("a", "b") == pytest.approx(coupling(sa, sb).strength, rel=1e-12)


def test_balance_sums_to_zero():
    """Acceptance 3: the balance point is the origin."""
    m, _, _ = _crossing()
    bal = m.balance()
    assert bal.total == pytest.approx(0.0, abs=1e-9)
    assert set(bal.offsets) == {"a", "b"} and set(bal.closed) == {"a", "b"}
    assert all(bal.closed.values())                    # the derived zero closes by construction
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in bal.residual.values())


def test_balance_reports_the_dc_a_side_gave_up():
    """`offsets` is how much DC the side carried to reach its zero."""
    m, _, _ = _crossing()
    before = m.balance().offsets["a"]
    m.place("a", m._placed["a"] + 50.0)       # push the side off zero
    assert m.balance().offsets["a"] > 10.0 * max(before, 1.0)


def test_a_declared_zero_that_closes_reads_closed():
    X = _structured_surface(seed=5, T=150, D=5)
    m = Screen()
    m.register("own", entry=lambda x: np.asarray(x), zero=lambda F: np.asarray(F).mean(0))
    m.place("own", X)
    b = m.balance()
    assert b.residual["own"] == pytest.approx(0.0, abs=1e-9)
    assert b.pvalue["own"] == pytest.approx(1.0) and b.closed["own"]


def test_a_declared_zero_placed_elsewhere_reads_open():
    """The calibrated test: a zero displaced from where the system balances is detected."""
    X = _structured_surface(seed=5, T=150, D=5)
    m = Screen()
    m.register("off", entry=lambda x: np.asarray(x), zero=lambda F: np.asarray(F).mean(0) + 0.5)
    m.place("off", X)
    b = m.balance()
    assert b.residual["off"] > 0.5 and b.pvalue["off"] < 0.05 and not b.closed["off"]


def test_the_closure_test_reads_the_residual_norm():
    """It uses ||r|| alone, so its power is the same in every direction -- worth knowing, since
    a direction-aware test would be sharper where the covariance is small."""
    rng = np.random.default_rng(5)
    X = 3.0 * rng.standard_normal((150, 1)) + rng.standard_normal((150, 5))
    ps = []
    for off in (np.full(5, 0.5), 0.5 * np.r_[1, -1, 1, -1, 1]):     # along and across the carrier
        m = Screen()
        m.register("g", entry=lambda x: np.asarray(x),
                   zero=lambda F, o=off: np.asarray(F).mean(0) + o)
        m.place("g", X)
        ps.append(m.balance().pvalue["g"])
    assert ps[0] == pytest.approx(ps[1], rel=1e-12)


def test_the_closure_test_fires_at_its_level():
    """Independent draws whose declared zero IS the truth: the test fires at about `far`."""
    rng = np.random.default_rng(11)
    fired = 0
    trials = 300
    for _ in range(trials):
        X = rng.standard_normal((90, 5))                    # true mean is exactly 0
        m = Screen(far=0.05)
        m.register("g", entry=lambda x: np.asarray(x), zero=lambda F: 0.0)
        m.place("g", X)
        fired += int(not m.balance().closed["g"])
    assert fired / trials < 0.12                            # 0.05 + ~3 binomial se


def test_read_uses_the_unfolded_basis():
    """Acceptance 4: a sparse-but-continuous frame legitimately folds under Projection; the
    screen read keeps the native basis and still resolves the structure in it."""
    rng = np.random.default_rng(19)
    T, D = 96, 64
    line = np.exp(-((np.arange(D) - 20.0) ** 2) / 8.0)     # a narrow line on a continuous axis
    X = rng.standard_normal((T, 1)) @ line[None, :] + 0.05 * rng.standard_normal((T, D))
    assert geometry(X)["n_F"] < D                          # the fold IS licensed here ...
    assert Projection(X).F_eff < D                             # ... and Projection takes it
    m = Screen()
    m.register("id", entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
    m.place("id", X)
    r = m.read()
    assert r.D == D and r.K_signal >= 1                    # the screen keeps all 64 columns
    assert m.basis().shape[0] == D and r.basis_dim >= 1


def test_read_reports_the_joint_frame():
    m, _, _ = _crossing()
    r = m.read()
    assert r.n_lenses == 2 and r.T == 64 and r.D == 6
    assert r.noise_floor > 0.0 and np.isfinite(r.coherence) and r.a_delta > 0.0


def test_resolution_is_none_when_nothing_resolves():
    """The null is no outgoing signal -- never a synthesised one."""
    m = Screen()
    m.register("id", entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
    m.place("id", np.zeros((64, 6)))
    assert m.resolution() is None and m.render("id") is None


def test_entry_only_lens_cannot_render_or_certify():
    m = Screen()
    m.register("in", entry=lambda x: np.asarray(x))
    m.place("in", np.random.default_rng(0).standard_normal((32, 4)))
    with pytest.raises(ValueError, match="entry-only"):
        m.render("in")
    with pytest.raises(ValueError, match="entry-only"):
        m.certify("in", np.zeros((32, 4)))


def test_a_side_must_share_the_basis():
    """The shared basis is the meeting place, and it is the one thing required."""
    m = Screen()
    m.register("a", entry=lambda x: np.asarray(x))
    m.register("b", entry=lambda x: np.asarray(x))
    m.place("a", np.zeros((32, 4)))
    with pytest.raises(ValueError, match="shared basis"):
        m.place("b", np.zeros((32, 5)))


# ── containment: a beam carried on its OWN ordered axis ───────────────────────

def _own_orders(seed=3, D=6):
    """Two beams sharing a concept direction but ordered differently -- bank transactions run
    on real time, purchases on their own event order."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((1, D))
    m = Screen()
    for g in ("accounting", "purchases"):
        m.register(g, entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
    m.place("accounting", 4.0 * rng.standard_normal((200, 1)) @ w + rng.standard_normal((200, D)))
    m.place("purchases",  4.0 * rng.standard_normal((37, 1)) @ w + rng.standard_normal((37, D)))
    return m


def test_beams_may_carry_their_own_ordered_axes():
    """Containment: the ordered axis is not shared and does not need to be."""
    m = _own_orders()
    assert m.orders == {"accounting": 200, "purchases": 37} and not m.shares_order


def test_the_basis_reads_work_across_different_orders():
    """Everything that meets on the shared basis is order-free."""
    m = _own_orders()
    assert m.basis().shape[0] == 6
    for g in ("accounting", "purchases"):
        assert m.beam(g).energy > 0.0 and m.directions(g).shape[0] == 6
        assert 0.0 < m.beam(g).etendue <= 1.0
    bal = m.balance()                                  # per side, so it closes per side
    assert set(bal.offsets) == {"accounting", "purchases"} and bal.frame is None


def test_two_beams_on_their_own_orders_still_meet():
    """The meeting point: a purchase is a concept both beams resolve, so energy crosses even
    though no row of a bank statement lines up with a row of a purchase log."""
    m = _own_orders()
    t = m.transfer("accounting", "purchases")
    assert t.modes_from >= 1 and t.modes_to >= 1
    assert t.participation > 0.0 and t.delivered > 0.0
    assert t.energy == pytest.approx(t.bystanding + t.reflected + t.delivered, rel=1e-12)
    assert m.transfer("purchases", "accounting").participation > 0.0      # and the other way


def test_render_across_orders_needs_only_a_concept():
    m = _own_orders()
    concept = m._placed["purchases"]
    assert m.render("accounting", concept).shape == (37, 6)


@pytest.mark.parametrize("call", [
    lambda m: m.couple("accounting", "purchases"),
    lambda m: m.joint(),
    lambda m: m.read(),
    lambda m: m.resolution(),
])
def test_row_paired_reads_refuse_across_orders(call):
    """Never fit: with no row pairing there is no quantity to measure, so the read raises --
    with a message that says which reads do work."""
    with pytest.raises(ValueError, match="ROW BY ROW"):
        call(_own_orders())


def test_unknown_lens_and_unplaced_lens_raise():
    m = Screen()
    with pytest.raises(KeyError):
        m.place("nope", np.zeros((8, 2)))
    m.register("a", entry=lambda x: np.asarray(x))
    with pytest.raises(KeyError, match="nothing placed"):
        m.coupling("a", "a")


def test_entry_must_return_a_2d_frame():
    m = Screen()
    m.register("bad", entry=lambda x: np.zeros(5))
    with pytest.raises(ValueError, match="2-D"):
        m.place("bad", np.zeros((8, 2)))


# ── each system carries its OWN laws (domain-agnosticism, concretely) ─────────

def test_default_energy_is_the_frames_own_power():
    m = Screen()
    m.register("id", entry=lambda x: np.asarray(x))
    X = _structured_surface()
    m.place("id", X)
    Xc = X - X.mean(0)
    assert np.allclose(m.energy("id"), (Xc ** 2).sum(axis=1))


def test_a_system_supplies_its_own_energy_law():
    """A system whose energy is not its variance says so, and the screen uses ITS law."""
    m = Screen()
    m.register("mine", entry=lambda x: np.asarray(x),
               energy=lambda F: np.abs(np.asarray(F)).sum(axis=1))     # L1, not power
    X = _structured_surface()
    m.place("mine", X)
    Xc = X - X.mean(0)
    assert np.allclose(m.energy("mine"), np.abs(Xc).sum(axis=1))
    assert not np.allclose(m.energy("mine"), (Xc ** 2).sum(axis=1))


def test_energy_is_measured_about_the_sides_own_zero():
    """The law receives the balanced frame -- a side that wants the raw origin says zero=0."""
    X = _structured_surface() + 7.0
    m = Screen()
    m.register("balanced", entry=lambda x: np.asarray(x))
    m.register("raw", entry=lambda x: np.asarray(x), zero=lambda F: 0.0)
    m.place("balanced", X)
    assert np.allclose(m.energy("balanced"), ((X - X.mean(0)) ** 2).sum(axis=1))
    m.clear()
    m.place("raw", X)
    assert np.allclose(m.energy("raw"), (X ** 2).sum(axis=1))


def test_a_system_supplies_its_own_zero():
    """Balance is read where the system balances, not where the arithmetic mean is."""
    X = _structured_surface() + 7.0
    m = Screen()
    m.register("mean", entry=lambda x: np.asarray(x))                        # derived default
    m.register("mine", entry=lambda x: np.asarray(x), zero=lambda F: 7.0)    # its own law
    m.place("mean", X)
    assert np.allclose(m.balanced("mean"), X - X.mean(0))
    m.clear()
    m.place("mine", X)
    assert np.allclose(m.balanced("mine"), X - 7.0)


def test_a_declared_zero_that_does_not_close_is_visible():
    """total is a measurement, not a tautology: a wrong zero leaves the frame open."""
    X = _structured_surface() + 7.0
    m = Screen()
    m.register("off", entry=lambda x: np.asarray(x), zero=lambda F: 0.0)   # declares no zero at all
    m.place("off", X)
    assert m.balance().total > 1.0


def test_a_zero_that_does_not_broadcast_raises():
    m = Screen()
    m.register("bad", entry=lambda x: np.asarray(x), zero=lambda F: np.zeros(3))
    m.place("bad", np.zeros((32, 6)))
    with pytest.raises(ValueError, match="broadcast"):
        m.balanced("bad")


def test_non_callable_law_raises():
    m = Screen()
    with pytest.raises(ValueError, match="energy must be callable"):
        m.register("x", entry=lambda v: v, energy=3.0)


def test_one_system_carries_several_forces():
    """A system reading/writing multiple lens forces registers one lens per force; they
    couple and transfer like any other pair, with no grouping machinery needed."""
    rng = np.random.default_rng(31)
    shared = rng.standard_normal((96, 1))
    m = Screen()
    for name, sign in (("sys:em", 1.0), ("sys:weak", -1.0), ("sys:strong", 1.0)):
        m.register(name, entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
        m.place(name, sign * 2.0 * shared + rng.standard_normal((96, 5)))
    assert m.read().n_lenses == 3
    assert m.couple("sys:em", "sys:strong") > 0.0
    assert m.couple("sys:em", "sys:weak") < 0.0
    assert m.transfer("sys:em", "sys:weak").energy > 0.0


def test_a_system_supplies_its_own_noise_floor():
    """What counts as signal in a detector is not what counts as signal in a market: the
    side that owns the physics owns the floor that scores it."""
    X = _structured_surface(seed=8, T=96, D=6)
    m = Screen()
    m.register("open", entry=lambda x: np.asarray(x), null=lambda ctx: 0.0)          # all modes
    m.register("shut", entry=lambda x: np.asarray(x), null=lambda ctx: float("inf"))  # none
    m.register("default", entry=lambda x: np.asarray(x))
    for g in ("open", "shut", "default"):
        m.clear()
        m.place(g, X)
        assert m.directions(g).shape[1] == {"open": 6, "shut": 0}.get(g, m.directions(g).shape[1])
    m.clear()
    m.place("shut", X)
    assert len(m.beam("shut").modes) == 0                    # its own floor, not the screen's


def test_a_side_that_resolves_nothing_by_its_own_floor_receives_nothing():
    """The per-side floor is load-bearing, not decorative: it gates what can be received."""
    X = _structured_surface(seed=9, T=96, D=6)
    m = Screen()
    m.register("send", entry=lambda x: np.asarray(x))
    m.register("deaf", entry=lambda x: np.asarray(x), null=lambda ctx: float("inf"))
    m.place("send", X)
    m.place("deaf", X)
    t = m.transfer("send", "deaf")
    assert t.modes_to == 0 and t.participation == 0.0 and t.bystanding == pytest.approx(t.energy)


def test_non_callable_null_raises():
    m = Screen()
    with pytest.raises(ValueError, match="null must be"):
        m.register("x", entry=lambda v: v, null=0.5)


def test_participation_is_measured_in_the_senders_own_energy_law():
    """An L1 total divided by an L2 share would not describe the same quantity: the fraction
    goes through the same law as the total."""
    X = _structured_surface(seed=11, T=96, D=6)
    l1 = lambda F: np.abs(np.asarray(F)).sum(axis=1)
    m = Screen()
    m.register("send", entry=lambda x: np.asarray(x), energy=l1)
    m.register("recv", entry=lambda x: np.asarray(x))
    m.place("send", X)
    m.place("recv", X)
    t = m.transfer("send", "recv")
    Xc, V = m.balanced("send"), m.directions("recv")
    matched = (Xc @ V) @ V.T
    assert t.energy == pytest.approx(l1(Xc).sum(), rel=1e-12)
    assert t.pertinent == pytest.approx(l1(matched).sum(), rel=1e-12)     # L1 throughout
    l2_share = (np.linalg.norm(matched) ** 2) / (np.linalg.norm(Xc) ** 2)
    assert t.participation != pytest.approx(l2_share, rel=1e-6)           # and NOT the L2 share


def test_a_beam_is_the_three_quantities_a_crossing_needs():
    m, _, _ = _crossing()
    s = m.beam("a")
    assert s.lens == "a"
    assert s.energy == pytest.approx(float(np.sum(m.energy("a"))), rel=1e-12)
    assert s.etendue == pytest.approx(float(E_etendue(m.balanced("a"))), rel=1e-12)
    assert len(s.modes) == m.directions("a").shape[1] == s.basis.shape[1]
    assert s.basis.shape[0] == 6 and s.flow.shape == (64,)
    assert set(m.beam()) == {"a", "b"}


# ── transfer: what matches, what fits, what crosses ───────────────────────────

def _wide_narrow(seed=0, T=96, D=8):
    rng = np.random.default_rng(seed)
    m = Screen()
    m.register("wide", entry=lambda x: np.asarray(x))
    m.register("narrow", entry=lambda x: np.asarray(x))
    m.place("wide", rng.standard_normal((T, D)))
    m.place("narrow", rng.standard_normal((T, 1)) @ rng.standard_normal((1, D)))
    return m


@pytest.mark.parametrize("a,b", [("wide", "narrow"), ("narrow", "wide")])
def test_energy_accounting_closes_exactly(a, b):
    """energy == bystanding + reflected + delivered, in both directions."""
    t = _wide_narrow().transfer(a, b)
    assert t.energy == pytest.approx(t.bystanding + t.reflected + t.delivered, rel=1e-12)


@pytest.mark.parametrize("a,b", [("wide", "narrow"), ("narrow", "wide")])
def test_the_two_root_behaviours_account_for_everything(a, b):
    """Wave energy at a boundary is absorbed or transmitted -- there is no third root.
    Reflection is transmission with a direction flip, not a category of its own."""
    t = _wide_narrow().transfer(a, b)
    assert t.energy == pytest.approx(t.absorbed + t.transmitted, rel=1e-12)
    assert t.absorbed == pytest.approx(t.delivered, rel=1e-12)
    assert t.transmitted == pytest.approx(t.bystanding + t.reflected, rel=1e-12)


def test_absorption_is_condensation():
    """The absorbed energy is exactly what condensed into the receiver's concepts."""
    m, _ = _shared_concept()
    t = m.transfer("cat", "observer")
    assert t.absorbed == pytest.approx(sum(c.energy for c in t.condensation), rel=1e-10)


def test_a_message_that_finds_no_home_is_wholly_transmitted():
    """Nothing absorbed, everything still propagating -- and retrievable, going forward."""
    t = _wide_narrow().transfer("narrow", "wide")
    assert t.absorbed == 0.0 and t.transmitted == pytest.approx(t.energy, rel=1e-12)


@pytest.mark.parametrize("a,b", [("wide", "narrow"), ("narrow", "wide")])
def test_transfer_fractions_are_bounded(a, b):
    t = _wide_narrow().transfer(a, b)
    assert 0.0 <= t.participation <= 1.0 and 0.0 < t.tau <= 1.0
    assert 0.0 < t.match <= 1.0 and t.reflected >= 0.0 and t.bystanding >= 0.0


def test_energy_can_exist_without_being_pertinent():
    """The receiving side resolves nothing, so none of the sender's energy is in this
    interaction: it bystands entirely -- not reflected, not lost, just not pertinent."""
    t = _wide_narrow().transfer("narrow", "wide")
    assert t.energy > 0.0 and t.modes_to == 0
    assert t.participation == 0.0 and t.delivered == 0.0 and t.reflected == 0.0
    assert t.bystanding == pytest.approx(t.energy, rel=1e-12)


def test_a_narrow_side_cannot_hold_a_wide_signal():
    """The brightness theorem: pertinent energy that does not fit is reflected."""
    t = _wide_narrow().transfer("wide", "narrow")
    assert t.etendue_to < t.etendue_from and t.tau < 1.0
    assert t.participation > 0.0 and t.reflected > 0.0
    assert t.delivered == pytest.approx(t.pertinent * t.tau, rel=1e-12)


def test_matched_sides_transfer_in_full_both_ways():
    """Full two-way transfer is etendue match."""
    X = _structured_surface(seed=5, T=96, D=8)
    m = Screen()
    for name in ("a", "b"):
        m.register(name, entry=lambda x: np.asarray(x))
        m.place(name, X)
    for a, b in (("a", "b"), ("b", "a")):
        t = m.transfer(a, b)
        assert t.match == pytest.approx(1.0, rel=1e-12) and t.tau == pytest.approx(1.0, rel=1e-12)
        assert t.reflected == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a,b", [("wide", "narrow"), ("narrow", "wide")])
def test_brightness_theorem_holds(a, b):
    """radiance_to <= radiance_from: a passive screen cannot make a side brighter than the
    side that fed it.  The second-law ceiling nonimaging optics derives its limit from."""
    t = _wide_narrow().transfer(a, b)
    assert t.radiance_to <= t.radiance_from + 1e-12


def test_concentrating_at_the_limit_conserves_radiance():
    """The ideal-concentrator equality case: squeezing into less phase space costs energy
    (tau < 1) but not brightness -- radiance is conserved exactly."""
    t = _wide_narrow().transfer("wide", "narrow")
    assert t.concentration > 1.0 and t.tau < 1.0
    assert t.radiance_to == pytest.approx(t.radiance_from, rel=1e-12)


def test_dilution_transfers_everything_and_costs_brightness():
    """The other direction: all the pertinent energy crosses, spread over more phase space,
    so radiance falls by exactly the etendue ratio."""
    rng = np.random.default_rng(41)
    T, D = 96, 8
    m = Screen()
    m.register("narrow", entry=lambda x: np.asarray(x))
    m.register("wide", entry=lambda x: np.asarray(x))
    tight = rng.standard_normal((T, 1)) @ rng.standard_normal((1, D))
    m.place("narrow", tight + 0.02 * rng.standard_normal((T, D)))
    m.place("wide", tight + 2.0 * rng.standard_normal((T, D)))
    t = m.transfer("narrow", "wide")
    assert t.concentration < 1.0 and t.tau == pytest.approx(1.0, rel=1e-12)
    assert t.reflected == pytest.approx(0.0, abs=1e-12)
    assert t.radiance_to == pytest.approx(t.radiance_from * t.concentration, rel=1e-10)
    assert t.radiance_to < t.radiance_from


def _shared_concept(seed=3, D=6, Ta=200, Tb=60):
    """Two beams, their own orders, one planted concept in common."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((1, D)); w /= np.linalg.norm(w)
    m = Screen()
    for g in ("cat", "observer"):
        m.register(g, entry=lambda x: np.asarray(x))
    m.place("cat", 4 * rng.standard_normal((Ta, 1)) @ w + rng.standard_normal((Ta, D)))
    m.place("observer", 4 * rng.standard_normal((Tb, 1)) @ w + rng.standard_normal((Tb, D)))
    return m, w.ravel()


def test_energy_condenses_into_the_receivers_concepts():
    """A crossing is not a response -- it is energy condensing as a concept in the receiving
    lens.  The condensation says which concept, not merely how much."""
    m, w = _shared_concept()
    t = m.transfer("cat", "observer")
    assert len(t.condensation) == t.modes_to >= 1
    top = max(t.condensation, key=lambda c: c.energy)
    assert abs(float(top.direction @ w)) > 0.7          # it condensed into the shared concept
    assert top.energy > 0.0


def test_condensation_is_measured_in_the_receivers_basis():
    """The concepts are the receiver's -- the sender's energy arrives in the other lens's
    directions, which is what makes it a concept over there."""
    m, _ = _shared_concept()
    t = m.transfer("cat", "observer")
    V = m.directions("observer")
    for c in t.condensation:
        assert np.allclose(c.direction, V[:, c.index])


def test_condensation_accounts_for_everything_that_matched():
    """Each condensation beam reports what matched into its concept, and agrees with its own
    frame.  The capacity limit scales the total that crosses; which concepts it favours is left
    open, so no per-concept delivery is claimed."""
    m, _ = _shared_concept()
    t = m.transfer("cat", "observer")
    assert sum(c.energy for c in t.condensation) == pytest.approx(t.pertinent, rel=1e-10)
    for c in t.condensation:
        assert c.energy == pytest.approx(float((c.frame ** 2).sum()), rel=1e-9)


def test_a_stateful_floor_tracks_what_it_scores():
    """A provider that sharpens on its own local sample is told what it is scoring."""
    seen = []
    class Tracking:
        def update(self, frame): seen.append(np.asarray(frame).shape)
        def __call__(self, ctx): return float(np.median(ctx.spectrum)) if ctx.spectrum is not None else 0.0
    m = Screen()
    m.register("s", entry=lambda x: np.asarray(x), null=Tracking())
    m.place("s", np.random.default_rng(0).standard_normal((40, 4)))
    assert seen == [(40, 4)]


def test_nothing_condenses_where_the_receiver_resolves_nothing():
    t = _wide_narrow().transfer("narrow", "wide")
    assert t.modes_to == 0 and t.condensation == [] and t.delivered == 0.0


def test_what_does_not_condense_carries_on():
    """A beam is a carrier of information whether or not it condenses.  Meeting a side that
    resolves nothing matching does not consume it -- it passes through, still carrying, and
    condenses later at a screen that can resolve it."""
    rng = np.random.default_rng(7)
    D = 6
    w = rng.standard_normal((1, D)); w /= np.linalg.norm(w)      # what the message is about
    msg = 4 * rng.standard_normal((120, 1)) @ w + 0.1 * rng.standard_normal((120, D))
    g = dict(entry=lambda x: np.asarray(x))

    wrong = Screen()                       # a side that resolves nothing at all
    wrong.register("msg", **g); wrong.register("deaf", entry=lambda x: np.asarray(x),
                                               null=lambda ctx: float("inf"))
    wrong.place("msg", msg); wrong.place("deaf", msg)
    t1 = wrong.transfer("msg", "deaf")
    assert t1.delivered == 0.0 and t1.condensation == []          # nothing collapses here
    carried = wrong.uncondensed("msg", "deaf")
    assert np.allclose(carried, wrong.balanced("msg"))            # all of it goes on, intact

    right = Screen()                       # a side that does resolve the concept
    right.register("msg", **g); right.register("reader", **g)
    right.place("msg", carried)
    right.place("reader", 4 * rng.standard_normal((80, 1)) @ w + rng.standard_normal((80, D)))
    t2 = right.transfer("msg", "reader")
    assert t2.delivered > 0.0 and len(t2.condensation) >= 1       # and condenses there
    assert abs(float(max(t2.condensation, key=lambda c: c.energy).direction @ w.ravel())) > 0.7


def test_uncondensed_energy_is_exactly_what_bystands():
    """The complementary projection: what carries on is what did not condense."""
    m, _ = _shared_concept()
    t = m.transfer("cat", "observer")
    left = m.uncondensed("cat", "observer")
    assert float((left ** 2).sum()) == pytest.approx(t.bystanding, rel=1e-10)


# ── realise: what the lens's OWN conversion actually delivers ────────────────

def _realise_case(inverse, seed=3, D=6):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((1, D)); w /= np.linalg.norm(w)
    Q = np.linalg.qr(rng.standard_normal((D, D)))[0]
    m = Screen()
    m.register("src", entry=lambda x: np.asarray(x))
    m.register("dst", entry=lambda x: np.asarray(x) @ Q, inverse=lambda c: inverse(c, Q))
    m.place("src", 4 * rng.standard_normal((200, 1)) @ w + rng.standard_normal((200, D)))
    m.place("dst", 4 * rng.standard_normal((60, 1)) @ w + rng.standard_normal((60, D)))
    return m


def test_a_lossless_conversion_reaches_the_etendue_bound():
    r = _realise_case(lambda c, Q: np.asarray(c) @ Q.T).realise("src", "dst")
    assert r.efficiency == pytest.approx(1.0, rel=1e-10)
    assert r.realised == pytest.approx(r.ideal, rel=1e-10) and r.shortfall == pytest.approx(0.0, abs=1e-9)
    assert r.passive


def test_a_lossy_conversion_falls_short_of_the_bound():
    """Halving the amplitude quarters the energy under the derived quadratic law."""
    r = _realise_case(lambda c, Q: 0.5 * np.asarray(c) @ Q.T).realise("src", "dst")
    assert r.efficiency == pytest.approx(0.25, rel=1e-9)
    assert r.shortfall > 0.0 and r.passive


def test_dropping_a_direction_costs_exactly_that_direction():
    r = _realise_case(lambda c, Q: (np.asarray(c) * np.r_[0, 1, 1, 1, 1, 1]) @ Q.T).realise("src", "dst")
    assert 0.0 < r.efficiency < 1.0 and r.shortfall > 0.0


def test_realised_is_the_bound_times_the_efficiency():
    """The two loss mechanisms stay separate: phase space (tau) and conversion (efficiency)."""
    m = _realise_case(lambda c, Q: 0.7 * np.asarray(c) @ Q.T)
    r, t = m.realise("src", "dst"), m.transfer("src", "dst")
    assert r.ideal == pytest.approx(t.delivered, rel=1e-12)
    assert r.realised == pytest.approx(r.ideal * r.efficiency, rel=1e-12)


def test_a_non_passive_conversion_is_reported_not_clipped():
    """A conversion returning more than it was given is a real finding about that lens."""
    r = _realise_case(lambda c, Q: 3.0 * np.asarray(c) @ Q.T).realise("src", "dst")
    assert r.efficiency > 1.0 and not r.passive


def test_realise_needs_an_invertible_receiver():
    m = Screen()
    for g in ("a", "b"):
        m.register(g, entry=lambda x: np.asarray(x))
        m.place(g, np.random.default_rng(0).standard_normal((40, 4)))
    with pytest.raises(ValueError, match="entry-only"):
        m.realise("a", "b")


def test_nothing_realises_where_nothing_condenses():
    m = _wide_narrow()
    frame = m._placed["wide"]                     # re-registering clears the placement
    m.register("wide", entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
    m.place("wide", frame)
    r = m.realise("narrow", "wide")
    assert r.realised == 0.0 and r.efficiency == 0.0


def test_transfer_needs_both_sides_placed():
    m = Screen()
    m.register("a", entry=lambda x: np.asarray(x))
    m.place("a", np.zeros((32, 4)))
    with pytest.raises(KeyError, match="nothing placed"):
        m.transfer("a", "b")


def test_one_lens_meets_many_others_on_many_screens():
    """A screen is ONE meeting.  The same lens -- same conversions, same laws -- meets
    other sides on other screens, carrying a beam per meeting.  No network object needed."""
    g = dict(entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
    rng = np.random.default_rng(0)
    carrier = rng.standard_normal((96, 1))
    m1, m2 = Screen(), Screen()
    for m, other, sign in ((m1, "vision", 1.0), (m2, "market", -1.0)):
        m.register("language", **g)
        m.register(other, **g)
        m.place("language", 2 * carrier + rng.standard_normal((96, 5)))
        m.place(other, sign * 2 * carrier + rng.standard_normal((96, 5)))
    assert m1.couple("language", "vision") > 0.0        # each screen measures its OWN crossing
    assert m2.couple("language", "market") < 0.0
    assert len(m1.beam("language").modes) == len(m2.beam("language").modes)


def test_a_beam_is_a_bundle_of_beams():
    """A string-theory string, not a wire: the carrier is extended, and each constituent mode
    is a beam of one -- same three quantities (how much, how much room, where) at every depth."""
    rng = np.random.default_rng(23)
    T, D = 128, 6
    m = Screen()
    m.register("s", entry=lambda x: np.asarray(x))
    m.place("s", np.hstack([3.0 * rng.standard_normal((T, 1))] * 3 +
                           [2.0 * rng.standard_normal((T, 1))] * 3) + rng.standard_normal((T, D)))
    b = m.beam("s")
    assert len(b.modes) >= 2 and len(b.modes) == b.basis.shape[1]
    for k, mode in enumerate(b.modes):
        assert mode.index == k
        assert 0.0 < mode.etendue <= 1.0 + 1e-12         # a mode carries a real patch of phase space
        assert mode.etendue == pytest.approx(mode.phi_T * mode.phi_F, rel=1e-12)
        assert mode.energy > 0.0 and mode.direction.shape == (D,)
    assert [mo.energy for mo in b.modes] == sorted((mo.energy for mo in b.modes), reverse=True)
    assert np.allclose(b.basis[:, 0], b.modes[0].direction)


def test_a_mode_IS_a_beam():
    """Self-similar in the type, not only in the wording: each mode is a Beam, so it carries
    the same record and the recursion bottoms out at a leaf spanning one direction."""
    m, _, _ = _crossing()
    b = m.beam("a")
    assert b.modes and all(isinstance(mo, Beam) for mo in b.modes)
    for mo in b.modes:
        assert mo.is_leaf and mo.modes == []          # a leaf decomposes no further
        assert mo.basis.shape[1] == 1                 # and spans exactly one direction
        assert np.allclose(mo.direction, mo.basis[:, 0])
    assert b.index == -1 and all(mo.index == k for k, mo in enumerate(b.modes))


def test_etendue_is_the_fill_product_at_every_depth():
    """The same identity holds for a side and for each mode it carries."""
    rng = np.random.default_rng(5)
    T, D = 160, 6
    a, b = rng.standard_normal((T, 1)), rng.standard_normal((T, 1))
    X = np.hstack([3*a, 3*a, 3*a, 2*b, 2*b, 2*b]) + rng.standard_normal((T, D))
    m = Screen(); m.register("s", entry=lambda x: np.asarray(x)); m.place("s", X)
    whole = m.beam("s")
    for lvl in [whole] + list(whole.modes):
        assert lvl.etendue == pytest.approx(lvl.phi_T * lvl.phi_F, rel=1e-12)
        assert 0.0 < lvl.etendue <= 1.0 + 1e-12


def _lens_case(entry, seed=4, T=120, F=10, D=6, rank1=False):
    """A lens whose surface is DELIBERATELY wider than the shared basis (F=10 -> D=6).

    A lens exists to convert between differently-shaped spaces, so a square conversion is the
    one shape at which a space error cannot surface -- and it also lets a screen-space frame be
    pushed back through ``entry``, measuring ``entry . entry`` while looking correct.  Returns
    the screen AND the surface, because linearity is measured on what the conversion takes in.

    ``rank1`` plants a single carrier, so the side resolves one mode and there is no pair to
    convert apart."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal((T, 1)), rng.standard_normal((T, 1))
    half = F // 2
    X = (np.hstack([3*a] * F) if rank1 else
         np.hstack([3*a] * (F - half) + [2*b] * half)) + 0.3 * rng.standard_normal((T, F))
    Q = np.linalg.qr(rng.standard_normal((F, F)))[0][:, :D]        # (F, D): F in, D out
    m = Screen(); m.register("s", entry=lambda x: entry(np.asarray(x), Q)); m.place("s", X)
    return m, X


def test_a_linear_lens_passes_the_modes_independently():
    """One lens serves a beam of ANY number of modes because it is linear -- each mode
    transforms on its own, which is what lets a mode be split off and recombined."""
    m, X = _lens_case(lambda x, Q: x @ Q)
    L = m.linear("s", X)
    assert L.modes >= 2 and L.linear
    assert L.additivity == pytest.approx(0.0, abs=1e-12)
    assert L.homogeneity == pytest.approx(0.0, abs=1e-12)


def test_an_affine_lens_is_linear_as_the_screen_sees_it():
    """Its departure is a constant, and a side's zero absorbs constants -- so the modes do
    pass independently in the balanced frame the screen reads."""
    m, X = _lens_case(lambda x, Q: x @ Q + 1.0)
    L = m.linear("s", X)
    assert L.linear
    assert L.additivity == pytest.approx(0.0, abs=1e-12)


def test_a_nonlinear_lens_mixes_the_modes():
    """A departure that survives balancing means the modes interact inside the conversion, so
    converting a mode alone stops agreeing with converting the beam whole."""
    m, X = _lens_case(lambda x, Q: np.tanh(x @ Q))
    L = m.linear("s", X)
    assert not L.linear and L.additivity > 1e-3 and L.homogeneity > 1e-3


def test_additivity_needs_two_modes_to_compare():
    """With one resolved mode there is no pair to convert apart; homogeneity still applies."""
    m, X = _lens_case(lambda x, Q: (x @ Q) ** 2, rank1=True)
    L = m.linear("s", X)
    assert L.modes < 2 and np.isnan(L.additivity)
    assert L.homogeneity > 1e-3 and not L.linear


def test_linearity_is_measured_on_a_non_square_lens():
    """A lens converts BETWEEN shapes, so linearity must be read on the surface it takes in.

    The surface is the argument because ``entry`` converts surface to screen: the placed frame is
    already through it, and is in the screen's coordinates."""
    m, X = _lens_case(lambda x, Q: x @ Q)
    assert X.shape[1] != np.asarray(m.basis()).shape[0]      # surface width != basis width
    L = m.linear("s", X)                                     # must not raise
    assert L.linear and L.additivity == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="2-D surface"):
        m.linear("s", np.zeros(7))


def test_a_beam_splits_into_modes_and_recombines():
    """The algebra: a beam is the sum of its modes, exactly.  Splitting is reading a mode's
    frame, merging is summing frames, and the two are inverses on the resolved sector."""
    rng = np.random.default_rng(5)
    T, D = 160, 6
    a, b = rng.standard_normal((T, 1)), rng.standard_normal((T, 1))
    X = np.hstack([3*a, 3*a, 3*a, 2*b, 2*b, 2*b]) + rng.standard_normal((T, D))
    m = Screen(); m.register("s", entry=lambda x: np.asarray(x)); m.place("s", X)
    beam = m.beam("s")
    assert beam.frame.shape == (T, D)
    assert np.allclose(sum(mo.frame for mo in beam.modes), beam.frame)
    for mo in beam.modes:
        assert np.linalg.matrix_rank(mo.frame) == 1        # amplitude x one direction
        assert float(np.linalg.norm(mo.direction)) == pytest.approx(1.0, rel=1e-10)


def test_a_split_mode_places_on_another_screen_unchanged():
    """A mode taken off one screen is a signal like any other: it carries its energy with it."""
    rng = np.random.default_rng(5)
    T, D = 160, 6
    X = np.hstack([3*rng.standard_normal((T, 1))]*3 + [2*rng.standard_normal((T, 1))]*3)         + rng.standard_normal((T, D))
    m = Screen(); m.register("s", entry=lambda x: np.asarray(x)); m.place("s", X)
    mode = m.beam("s").modes[0]
    other = Screen(); other.register("m0", entry=lambda x: np.asarray(x))
    other.place("m0", mode.frame)
    assert other.beam("m0").energy == pytest.approx(mode.energy, rel=1e-9)


def test_a_footprint_is_an_extracted_signal():
    """A footprint carries its T-signal and its F-signal, not only their fills, so a single
    mode can be filtered off and passed along as a beam."""
    from entroptics import Aperture
    rng = np.random.default_rng(3)
    Nt, Nf = 64, 200
    v = np.zeros(Nf); v[30:34] = 1.0; v /= np.linalg.norm(v)
    u = rng.standard_normal(Nt); u -= u.mean(); u /= np.linalg.norm(u)
    W = 6.0 * (np.sqrt(Nt) + np.sqrt(Nf)) * np.outer(u, v) + rng.standard_normal((Nt, Nf))
    fps = Aperture(W).footprints
    assert fps and all(isinstance(f, Beam) and f.is_leaf for f in fps)
    top = fps[0]
    assert top.profile.shape[1] == 1 and top.basis.shape[1] == 1     # one T-signal, one F-signal
    assert np.linalg.matrix_rank(top.frame) == 1
    assert top.etendue == pytest.approx(top.phi_T * top.phi_F, rel=1e-12)
    # the F-signal peaks where the mode was planted (in the projection's folded coordinates)
    F_eff = top.basis.shape[0]
    assert 30 <= int(np.argmax(np.abs(top.basis[:, 0]))) * Nf / F_eff <= 34


def test_flux_is_signed_against_propagation():
    """A beam splits by direction at a boundary: forward counts positive, backward negative,
    and absorption leaves the field carrying no direction at all."""
    t = _wide_narrow().transfer("wide", "narrow")
    assert t.flux == pytest.approx(t.bystanding - t.reflected, rel=1e-12)
    assert t.energy == pytest.approx(t.absorbed + t.transmitted, rel=1e-12)
    assert t.transmitted == pytest.approx(t.bystanding + t.reflected, rel=1e-12)


def test_flux_is_forward_when_nothing_reflects():
    """With nothing turned back, every travelling joule is still going the way it arrived."""
    t = _wide_narrow().transfer("narrow", "wide")
    assert t.reflected == pytest.approx(0.0, abs=1e-12)
    assert t.flux == pytest.approx(t.transmitted, rel=1e-12) and t.flux > 0.0


def test_condensation_entries_are_beams():
    """What lands in a concept is itself a beam, so it can be read like any other."""
    m, _ = _shared_concept()
    for c in m.transfer("cat", "observer").condensation:
        assert isinstance(c, Beam) and c.is_leaf and c.energy >= 0.0


def test_a_mode_reads_like_the_beam_it_belongs_to():
    """Self-similarity is structural: a mode answers every question the whole beam does, so a
    bundle can be read at any depth without knowing which depth it is."""
    m, _, _ = _crossing()
    b = m.beam("a")
    for attr in ("energy", "etendue", "phi_T", "phi_F", "flow", "basis", "profile",
                 "frame", "modes", "is_leaf", "index", "lens"):
        assert hasattr(b, attr) and hasattr(b.modes[0], attr)
    for lvl in (b, b.modes[0]):
        assert np.isfinite(lvl.energy) and np.isfinite(lvl.etendue)
        assert lvl.etendue == pytest.approx(lvl.phi_T * lvl.phi_F, rel=1e-12)


def test_a_beam_is_cheap_and_its_costly_fields_resolve_on_access():
    """One read per side: the beam arrives without paying for the T x T eigendecomposition its
    fills need, or for the bundle."""
    m, _, _ = _crossing()
    b = m.beam("a")
    assert b._fills is None or callable(b._fills)      # unresolved until asked
    assert callable(b._modes)
    _ = b.etendue, b.modes                              # asking resolves them
    assert not callable(b._fills) and not callable(b._modes)
    assert b.etendue == m.beam("a").etendue             # and the value is stable


def test_couple_is_the_only_symmetric_read():
    """Direction is the geometry: whether two sides co-resolve is a fact about the pair, but
    a crossing is not -- wide into narrow reflects, narrow into wide dilutes."""
    m = _wide_narrow()
    assert m.couple("wide", "narrow") == pytest.approx(m.couple("narrow", "wide"), rel=1e-12)
    fwd, rev = m.transfer("wide", "narrow"), m.transfer("narrow", "wide")
    assert fwd.tau != pytest.approx(rev.tau, rel=1e-6)
    assert fwd.concentration == pytest.approx(1.0 / rev.concentration, rel=1e-10)


def test_n_lenses_carry_n_conversions():
    """Registration is additive and per-side: no pairwise table exists to grow."""
    m = Screen()
    for name in ("a", "b", "c", "d"):
        m.register(name, entry=lambda x: np.asarray(x), inverse=lambda c: np.asarray(c))
    assert len(m.lenses) == 4 and m.placed == []
