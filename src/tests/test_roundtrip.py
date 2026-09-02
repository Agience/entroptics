"""Round-trip / reconstruction properties: tensor HOSVD, pack/unpack."""
import numpy as np
import pytest

from entroptics import aperture as A
from entroptics.projection import Projection
from entroptics.entropy import normalize
from entroptics import tensor as TZ
from conftest import build_W, build_Wc


def test_tensor_full_rank_roundtrip_is_exact(W):
    """With no rank truncation the HOSVD is exact -> overlap-add reconstructs the
    (whitened) data, so fidelity ~ 1."""
    data = np.nan_to_num(normalize(W))
    d = 4
    T_prime = W.shape[0] - d + 1
    te = TZ.tensor_embed(data, d=d, rank=(T_prime, d, W.shape[1]))
    assert TZ.tensor_fidelity(data, te) > 0.999


def test_tensor_fidelity_in_unit_interval(W):
    te = A.Aperture(W).tensor(d=8)
    fid = TZ.tensor_fidelity(np.nan_to_num(normalize(W)), te)
    assert 0.0 <= fid <= 1.0


def test_tensor_reconstruct_shape(W):
    te = A.Aperture(W).tensor(d=8)
    rec = np.asarray(TZ.tensor_reconstruct(te))
    assert rec.shape[1] == W.shape[1]


def test_pack_unpack_is_lossless(W):
    te = A.Aperture(W).tensor(d=8)
    packed = TZ.pack_factors(te)
    r_t = te["U_time"].shape[1]
    r_d = te["U_lag"].shape[1]
    r_f = te["U_freq"].shape[1]
    te2 = TZ.unpack_factors(packed, r_t, r_d, r_f, te["d"],
                            te["U_freq"].shape[0], te["T_prime"])
    rec1 = np.asarray(TZ.tensor_reconstruct(te))
    rec2 = np.asarray(TZ.tensor_reconstruct(te2))
    assert np.max(np.abs(rec1 - rec2)) == pytest.approx(0.0, abs=1e-12)


def test_complex_tensor_roundtrip(Wc):
    te = TZ.tensor_read(Wc, d=6)
    data = np.nan_to_num(normalize(Wc))
    assert 0.0 <= TZ.tensor_fidelity(data, te) <= 1.0


def test_normalize_masks_are_nan_not_zero():
    """Masked cells must be missing (NaN), never zero-filled (a zero drags the fold)."""
    W = build_W(7)
    mask = np.zeros(W.shape, bool)
    mask[10:14, 5:9] = True
    out = normalize(W, mask)
    assert np.all(np.isnan(out[mask]))
    assert np.all(np.isfinite(out[~mask]))
