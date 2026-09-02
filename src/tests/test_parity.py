"""Backend parity: the ONE code path must give the same numbers on numpy and torch.
Skipped entirely when torch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from entroptics import aperture as A
from entroptics.screen import Screen, coherence
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
    sc_np = Screen(W)
    sc_t = Screen(torch.as_tensor(W))
    assert abs(sc_np.coherence - float(sc_t.coherence)) < 1e-10
    assert sc_np.K_signal == sc_t.K_signal


def test_screen_singular_values_parity(W):
    sc_np = Screen(W)
    sc_t = Screen(torch.as_tensor(W))
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
    """A torch input must keep its optics arrays in torch (on-device), not silently
    fall back to numpy."""
    ap = A.Aperture(torch.as_tensor(W))
    assert type(ap.W).__module__.split(".")[0] == "torch"
    assert type(ap.spectral.eigenvalues).__module__.split(".")[0] == "torch"


def test_predict_parity(W):
    """The full propagator and one-step forecast must agree on numpy and torch."""
    p_np = np.asarray(A.Aperture(W).predict(W[-1]))
    p_t = A.Aperture(torch.as_tensor(W)).predict(torch.as_tensor(W[-1])).cpu().numpy()
    assert np.max(np.abs(p_np - p_t)) < 1e-9


def test_fields_over_planes_parity():
    """The geometry-preserving N-D reduction must agree on numpy and torch."""
    from entroptics import fields
    field = np.random.default_rng(4).standard_normal((5, 8, 8))
    op_np = fields.over_planes(field, (1, 2))
    op_t = fields.over_planes(torch.as_tensor(field), (1, 2))
    assert abs(op_np - op_t) < 1e-9
