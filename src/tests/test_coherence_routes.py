"""Coherence is the same number whichever side its moments are taken on.

The z-score's moments are sums over the N x N row Gram.  On a long, narrow screen they are taken
on the feature side instead (never forming N x N); both are exact, so they must agree to round-off,
real and complex, at any lag."""
import numpy as np
import pytest

from entroptics.projection import _coherence_feature_side, _coherence_z, coherence


def _gram_side(X, lag):
    Y = np.concatenate([X.real, X.imag], axis=1) if np.iscomplexobj(X) else X
    G = Y @ Y.T
    G = G / np.max(np.abs(G))
    R = G ** 2
    d = np.diagonal(R)
    S1 = R.sum() - d.sum()
    S2 = (R * R).sum() - (d * d).sum()
    rowsum = R.sum(axis=1) - d
    return _coherence_z(S1, S2, float(np.sum(rowsum ** 2)), float(np.diagonal(R, lag).sum()),
                        X.shape[0], lag, R)


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("lag", [1, 4])
def test_both_sides_agree(complex_, lag):
    rng = np.random.default_rng(lag + 10 * complex_)
    X = rng.standard_normal((1200, 6))
    X = np.cumsum(X, axis=0) * 0.03 + X                    # some ordered structure
    if complex_:
        X = X + 1j * rng.standard_normal(X.shape)
    Y = np.concatenate([X.real, X.imag], axis=1) if complex_ else X
    a, b = _gram_side(X, lag), _coherence_feature_side(Y, lag)
    assert b == pytest.approx(a, rel=1e-10, abs=1e-12)
    assert coherence(X, lag) == pytest.approx(a, rel=1e-10, abs=1e-12)


def test_a_long_screen_never_forms_the_row_gram():
    X = np.random.default_rng(0).standard_normal((100_000, 4))   # N x N would be 80 GB
    z = coherence(X, 1)
    assert np.isfinite(z) and abs(z) < 6
