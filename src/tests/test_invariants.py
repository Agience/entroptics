"""Mathematical invariants of the optics reads -- the identities a paper/Lean would
state (and that must hold for ANY input, not just the golden signal)."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.reads import decay, concentration, etendue, phi_T, phi_F
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


# ── the entropy noise-guard: pure noise must NOT fold ─────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_pure_noise_does_not_fold(seed):
    """iid noise sits inside the Miller-Madow band of max entropy -> geometry snaps
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
    """LEMMA behind mercer_certificate: the biased ACF's Toeplitz matrix is PSD."""
    c = np.asarray(decay(W))
    T = c.size
    idx = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    lam = np.linalg.eigvalsh(c[idx])
    assert lam.min() >= -1e-8 * max(1.0, abs(lam.max()))


# ── concentration: axial (focus) vs directional (resultant) are DISTINCT ───────

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


# ── etendue / phi invariance: relabeling + per-variable phase (NOT arbitrary mix) ─
#
# The per-axis fills phi_T, phi_F (hence etendue = phi_F * phi_T) are functions of the
# UNIT-DIAGONAL correlation eigenvalues of an axis' variables.  Relabelling those
# variables (a permutation) permutes the correlation matrix -> same eigenvalues; a
# per-variable phase D (a diagonal unitary) maps C -> D C D^H, which preserves both
# the unit diagonal and the eigenvalues.  So phi is invariant under permutation +
# per-variable phase of ITS OWN variable axis.  It is NOT invariant under an arbitrary
# orthonormal/unitary MIXING of the variables (that changes the correlation itself) --
# the negative control below pins exactly that distinction.

def _phase(rng, n):
    return np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, n))

def _rand_orthonormal(rng, n):
    """A random real orthonormal (n, n) matrix via QR (numpy-only, no scipy)."""
    Q, R = np.linalg.qr(rng.standard_normal((n, n)))
    return Q * np.sign(np.diag(R))          # fix signs for a proper orthonormal frame


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_phi_F_invariant_under_feature_relabel_and_phase(seed):
    """phi_F's variables are the FEATURE columns: permute + per-feature phase them
    (samples/rows untouched) -> phi_F is invariant to float tolerance."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 100)
    F = W.shape[1]
    perm = rng.permutation(F)
    Wt = W.astype(complex)[:, perm] * _phase(rng, F)[None, :]
    assert phi_F(W) == pytest.approx(phi_F(Wt), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_phi_T_invariant_under_ordered_relabel_and_phase(seed):
    """phi_T's variables are the ORDERED rows: permute + per-row phase them
    (samples/columns untouched) -> phi_T is invariant to float tolerance."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 200)
    T = W.shape[0]
    perm = rng.permutation(T)
    Wt = W.astype(complex)[perm, :] * _phase(rng, T)[:, None]
    assert phi_T(W) == pytest.approx(phi_T(Wt), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_etendue_invariant_under_axis_relabeling(seed):
    """etendue = phi_F * phi_T is invariant under RELABELLING (permutation) of the
    variables on both axes, plus a global phase (the complex path)."""
    W = build_W(seed)
    rng = np.random.default_rng(seed + 300)
    T, F = W.shape
    Wt = (W.astype(complex) * np.exp(1j * rng.uniform(0, 2 * np.pi)))
    Wt = Wt[rng.permutation(T), :][:, rng.permutation(F)]
    assert etendue(W) == pytest.approx(etendue(Wt), rel=0, abs=1e-10)


@pytest.mark.parametrize("seed", [1, 2, 7])
def test_phi_F_changes_under_arbitrary_feature_mixing(seed):
    """Negative control: an ARBITRARY orthonormal mixing of the feature variables
    changes the unit-diagonal correlation, so phi_F is NOT invariant -- the claim is
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
