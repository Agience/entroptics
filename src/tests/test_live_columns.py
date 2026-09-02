"""The floor's null is sized by the channels that exist.

`noise_floor` is the edge of an N x F iid ensemble, so F decides where it sits.  A column of
exact zeros carries no observation from that ensemble, and this module produces such columns
by two routes: `project` mean-imputes an all-missing screen cell to 0, and `normalize` returns a
channel it could not scale as zeros.  `live_columns` excludes both from F.

The failure modes these tests watch for:
  - appending columns that carry no observation moves the floor and changes K_signal;
  - appending columns that carry a real observation does not (the control, without which the
    rule degenerates to "ignore whatever is inconvenient");
  - and the floor and `mode_significance` disagree on F, breaking `K_signal == #(p < far)`.
"""
import numpy as np
import pytest

from entroptics.projection import live_columns, mode_significance, noise_floor


def _rank2(seed=4, N=90, F=20, amp=1.0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((2, F))
    C = rng.standard_normal((N, 2)) * amp
    return C @ B + rng.standard_normal((N, F))


def _k(S, far=0.05):
    sv = np.linalg.svd(S, compute_uv=False)
    return int(np.count_nonzero(sv > noise_floor(S, far=far)))


def test_dead_columns_do_not_move_the_floor():
    """600 columns of nothing must not change where the edge of the noise sits."""
    S = _rank2()
    bare = noise_floor(S)
    for npad in (50, 200, 600):
        padded = np.concatenate([S, np.zeros((S.shape[0], npad))], axis=1)
        assert live_columns(padded) == S.shape[1]
        assert noise_floor(padded) == pytest.approx(bare, rel=1e-12)


def test_dead_columns_used_to_admit_noise_as_signal():
    """`noise_sigma2` divides median row energy by a denominator sized from `shape`.  Dead
    columns add no energy but do add width, so sizing that denominator by the raw array width, in place of `live_columns`, undercounts sigma^2, sinks the floor, and admits noise as
    signal."""
    from entroptics.null_providers import apply_floor
    S = _rank2()
    padded = np.concatenate([S, np.zeros((S.shape[0], 600))], axis=1)
    sv = np.linalg.svd(padded, compute_uv=False)

    def k_at(F):
        f = apply_floor(None, spectrum=None, data=padded, shape=(padded.shape[0], F),
                        far=0.05, kind="projection", seed=0)
        return int(np.count_nonzero(sv > f))

    assert k_at(padded.shape[1]) > 2, "the old array-width sizing over-counted; if this stops " \
                                      "being true the demonstration is stale, not the fix"
    assert k_at(live_columns(padded)) == 2 == _k(padded)


def test_real_columns_DO_move_the_floor():
    """The control.  More genuine channels is a genuinely wider ensemble."""
    rng = np.random.default_rng(9)
    S = _rank2()
    widened = np.concatenate([S, rng.standard_normal((S.shape[0], 200))], axis=1)
    assert live_columns(widened) > live_columns(S)
    assert noise_floor(widened) != pytest.approx(noise_floor(S), rel=1e-9)


def test_the_floor_and_the_evidence_agree_on_the_width():
    """`K_signal == #(p_k < far)` is an identity between the two reads, so they must size the
    null the same way. Sizing only one of them by the live width would break it silently."""
    S = np.concatenate([_rank2(), np.zeros((90, 300))], axis=1)
    far = 0.05
    assert _k(S, far) == int(np.count_nonzero(mode_significance(S).pvalue < far))


def test_a_fully_dead_screen_does_not_invent_a_width():
    """Floored at 1 so the Johnstone shape stays non-degenerate — never widened to the array."""
    assert live_columns(np.zeros((8, 64))) == 1
