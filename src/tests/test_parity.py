"""Backend parity: the one code path must give the same numbers on numpy and torch.
Skipped entirely when torch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from entroptics import aperture as A
from entroptics.projection import Projection, coherence
from entroptics.reads import decay
from entroptics.entropy import normalize
from entroptics import tensor as TZ
from conftest import build_W, build_Wc


def _max_scalar_diff(a: dict, b: dict) -> float:
    return max(abs(float(a[k]) - float(b[k]))
               for k in a if isinstance(a[k], (int, float)) and np.isfinite(a[k]))


def test_optics_parity(W):
    o_np = A.Aperture(W).optics()
    o_t = A.Aperture(torch.as_tensor(W)).optics()
    assert _max_scalar_diff(o_np, o_t) < 1e-10


def test_optics_parity_complex(Wc):
    o_np = A.Aperture(Wc).optics()
    o_t = A.Aperture(torch.as_tensor(Wc)).optics()
    assert _max_scalar_diff(o_np, o_t) < 1e-9


def test_decay_parity(W):
    c_np = np.asarray(decay(W))
    c_t = decay(torch.as_tensor(W)).cpu().numpy()
    assert np.max(np.abs(c_np - c_t)) < 1e-10


def test_coherence_is_backend_identical(W):
    """Deterministic coherence -> bit-identical (no RNG divergence)."""
    sc_np = Projection(W)
    sc_t = Projection(torch.as_tensor(W))
    assert abs(sc_np.coherence - float(sc_t.coherence)) < 1e-10
    assert sc_np.K_signal == sc_t.K_signal


def test_screen_singular_values_parity(W):
    sc_np = Projection(W)
    sc_t = Projection(torch.as_tensor(W))
    s_np = np.sort(np.asarray(sc_np.S))
    s_t = np.sort(sc_t.S.cpu().numpy())
    assert np.max(np.abs(s_np - s_t)) < 1e-9


def test_tensor_fidelity_parity(W):
    data_np = np.nan_to_num(normalize(W))
    te_np = A.Aperture(W).tensor(d=8)
    fid_np = TZ.tensor_fidelity(data_np, te_np)
    Wt = torch.as_tensor(W)
    te_t = A.Aperture(Wt).tensor(d=8)
    fid_t = TZ.tensor_fidelity(torch.nan_to_num(normalize(Wt)), te_t)
    assert abs(fid_np - fid_t) < 1e-10


def test_torch_stays_on_backend(W):
    """A torch input keeps its optics arrays in torch (on-device)."""
    ap = A.Aperture(torch.as_tensor(W))
    assert type(ap.W).__module__.split(".")[0] == "torch"
    assert type(ap.spectral.eigenvalues).__module__.split(".")[0] == "torch"


def test_predict_parity(W):
    """The full propagator and one-step forecast must agree on numpy and torch."""
    p_np = np.asarray(A.Aperture(W).predict(W[-1]))
    p_t = A.Aperture(torch.as_tensor(W)).predict(torch.as_tensor(W[-1])).cpu().numpy()
    assert np.max(np.abs(p_np - p_t)) < 1e-9


def test_resolved_batch_parity():
    """The batched resolved read must give the same answer on numpy and torch-CPU -- K_signal
    byte-identical (exact integer count), energy / floor to fp precision.  (torch-CPU uses the same
    exact SVD realisation as numpy; the Fermi matrix-sign path is CUDA-only.)"""
    from entroptics import resolved_batch
    rng = np.random.default_rng(5)
    X = np.stack([([0.0, 3.0, 8.0][i % 3]) * np.outer(rng.standard_normal(80), rng.standard_normal(32))
                  + rng.standard_normal((80, 32)) for i in range(10)])
    for fold in (True, False):
        rn = resolved_batch(X, fold=fold, energy=True)
        rt = resolved_batch(torch.as_tensor(X), fold=fold, energy=True)
        assert np.array_equal(np.asarray(rn.K_signal), np.asarray(rt.K_signal))   # byte-identical count
        assert np.max(np.abs(np.asarray(rn.energy) - rt.energy.cpu().numpy())) < 1e-9
        assert np.max(np.abs(np.asarray(rn.noise_floor) - rt.noise_floor.cpu().numpy())) < 1e-9


def test_resolved_batch_stays_on_backend():
    """A torch stack keeps its outputs in torch (on-device)."""
    from entroptics import resolved_batch
    X = torch.as_tensor(np.random.default_rng(6).standard_normal((6, 64, 16)))
    r = resolved_batch(X, fold=False, energy=True, basis=True)
    assert type(r.energy).__module__.split(".")[0] == "torch"
    assert type(r.projector).__module__.split(".")[0] == "torch"


def test_fields_over_planes_parity():
    """The geometry-preserving N-D reduction must agree on numpy and torch."""
    from entroptics import fields
    field = np.random.default_rng(4).standard_normal((5, 8, 8))
    op_np = fields.over_planes(field, (1, 2))
    op_t = fields.over_planes(torch.as_tensor(field), (1, 2))
    assert abs(op_np - op_t) < 1e-9


# ── the paths where a backend could disagree about "nothing is here" ─────────

_DEGENERATE_READS = ("phi", "phi_T", "phi_F", "etendue", "strehl", "a_delta", "focus")


def _degenerate_frames():
    """Frames built to land on each decision that asks whether something is there at all: a
    channel with no spread, channels never measured, scattered gaps, and a record small enough
    that the resolution floor is what separates signal from round-off."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((200, 3)) @ rng.standard_normal((3, 32))         + 0.3 * rng.standard_normal((200, 32))
    flat = base.copy(); flat[:, 5] = 2.5                       # zero variance -> no scale to whiten by
    dead = base.copy(); dead[:, [2, 9, 17]] = np.nan           # never measured -> outside the extent
    gaps = base.copy(); gaps[3, 4] = gaps[80, 20] = np.nan     # scattered -> filled from a counted mean
    return {"plain": base, "a flat channel": flat, "dead channels": dead,
            "scattered gaps": gaps, "tiny scale": base * 1e-25}


@pytest.mark.parametrize("name", list(_degenerate_frames()))
def test_backends_agree_on_the_degenerate_decisions(name):
    """These reads each cross a threshold derived from the working dtype's epsilon, so numpy and
    torch could in principle land on opposite sides of one.  Compared against each other rather
    than against stored numbers: whatever the right answer is, one backend must not invent it."""
    X = _degenerate_frames()[name]
    an = A.Aperture(X, window=None)
    at = A.Aperture(torch.as_tensor(X, dtype=torch.float64), window=None)

    for k in _DEGENERATE_READS:
        vn, vt = float(getattr(an, k)), float(getattr(at, k))
        assert np.isnan(vn) == np.isnan(vt), f"{k}: one backend read a value, the other did not"
        if not np.isnan(vn):
            assert vt == pytest.approx(vn, rel=1e-10), k

    assert Projection(X).K_signal == Projection(torch.as_tensor(X, dtype=torch.float64)).K_signal


def test_backends_agree_that_a_powerless_screen_is_not_a_mode():
    """The fill is NaN when no mode carries power (see test_edges).  Both backends must refuse,
    or the same frame reads as a perfect single mode on one of them."""
    for build in (lambda: np.zeros((40, 8)), lambda: np.full((40, 8), np.nan)):
        Z = build()
        an, at = A.Aperture(Z, window=None), A.Aperture(torch.as_tensor(Z, dtype=torch.float64), window=None)
        assert np.isnan(an.phi) and np.isnan(at.phi)
        assert np.isnan(an.phi_F) and np.isnan(at.phi_F)
