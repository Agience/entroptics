"""The ordered-axis correlation spectrum is read on the smaller Gram.

The ordered-axis read correlates the T time points across the F channels: the T x T matrix
R = Y^H Y with Y the centred frame, each column scaled to unit norm. Its nonzero eigenvalues are
exactly those of the F x F matrix Y Y^H, and the rest are zeros. So the spectrum costs
O(T F^2 + min(T, F)^3) instead of O(T^3), and the reads built on it (the fill fraction, the
Strehl ratio) are unchanged, because zeros add nothing to an entropy or a trace.
"""
import numpy as np
import pytest

from entroptics import reads


def _direct(W):
    """The ordered-axis spectrum computed the long way: the full T x T correlation."""
    X = W - W.mean(axis=0)
    M = X.T                                   # samples = channels, variables = time points
    Mc = M - M.mean(axis=0)
    C = Mc.conj().T @ Mc
    dg = np.real(np.diag(C))
    d = np.where(dg > 0, np.sqrt(dg), 1.0)
    C = C / np.outer(d, d)
    return np.clip(np.linalg.eigvalsh(C), 0.0, None)[::-1]


@pytest.mark.parametrize("T,F", [(300, 8), (64, 64), (40, 90), (257, 3)])
def test_the_ordered_axis_spectrum_equals_the_full_correlation(T, F):
    W = np.random.default_rng(T * 1000 + F).standard_normal((T, F))
    got = np.asarray(reads.axis_spectrum(W, 0))
    want = _direct(W)
    assert got.shape == want.shape
    # DERIVED: a symmetric eigensolve of an n x n matrix is backward stable to n * eps * ||R||
    # (Higham 2002, Thm 8.x); the two routes each carry that much, and ||R|| = the top eigenvalue.
    n = max(T, F)
    bound = 2 * n * np.finfo(float).eps * want[0]
    assert np.max(np.abs(got - want)) <= bound


def test_the_ordered_axis_read_never_decomposes_the_long_side(monkeypatch):
    """On a tall frame the read must not form or decompose a T x T matrix."""
    T, F = 2048, 8
    W = np.random.default_rng(7).standard_normal((T, F))
    sizes = []
    real = np.linalg.eigvalsh

    def spy(a, *args, **kwargs):
        sizes.append(np.shape(a)[-1])
        return real(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvalsh", spy)
    reads.axis_spectrum(W, 0)
    assert sizes and max(sizes) <= min(T, F)


def test_the_ordered_fill_is_unchanged_on_a_planted_mode():
    """phi_T, the read the spectrum feeds, matches the full-correlation value."""
    rng = np.random.default_rng(3)
    T, F = 400, 12
    W = rng.standard_normal((T, F)) + 3.0 * np.outer(np.sin(np.arange(T) / 7.0), rng.standard_normal(F))
    ev = _direct(W)
    ev = ev[ev > 0]
    p = ev / ev.sum()
    want = 2.0 ** (-np.sum(p * np.log2(p))) / T
    got = reads.phi_T(W)
    # DERIVED: the fill is a smooth function of the spectrum, so it inherits the eigensolve's
    # relative round-off, n * eps, on each of the two routes.
    assert abs(got - want) <= 2 * T * np.finfo(float).eps * want
