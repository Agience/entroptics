"""`reads.level_shift_scan` -- the contiguous-run scan statistic for a level shift of unknown size
and location.  Pins it against a brute-force reference, over batches, with unobserved (NaN)
positions, and the below-baseline case; and checks the top-level re-export."""
import numpy as np
import pytest

from entroptics.reads import level_shift_scan
import entroptics


def _brute(profile, reach=None):
    """O(n^2) reference: the run [a, b) with the largest positive S^2/n, lowest (a, b) on a tie."""
    p = np.asarray(profile, dtype=float)
    n = p.size
    e = np.where(np.isfinite(p), p, 0.0)
    na = n if reach is None else max(1, min(int(reach), n))
    best_stat, best_a, best_b = 0.0, 0, 1
    for a in range(na):
        for b in range(a + 1, n + 1):
            s = float(e[a:b].sum())
            if s > 0.0:
                v = s * s / (b - a)
                if v > best_stat:                      # strict: keep the first (lowest a, then b)
                    best_stat, best_a, best_b = v, a, b
    if not np.isfinite(p).any():
        return np.nan, 0, 1
    return best_stat, best_a, best_b


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("reach", [None, 1, 5])
def test_matches_bruteforce_1d(seed, reach):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 40))
    # deviations: mostly negative/noise with a planted positive run, so a real run exists
    p = rng.normal(-0.2, 1.0, n)
    a0 = int(rng.integers(0, n - 1))
    b0 = int(rng.integers(a0 + 1, n + 1))
    p[a0:b0] += rng.uniform(0.5, 3.0)
    stat, a, b = level_shift_scan(p, reach=reach)
    rstat, ra, rb = _brute(p, reach=reach)
    assert np.isclose(float(stat), rstat, rtol=0, atol=1e-9)
    assert (int(a), int(b)) == (ra, rb)


def test_batched_matches_per_slice():
    rng = np.random.default_rng(0)
    P = rng.normal(0.0, 1.0, (3, 4, 17))
    P[..., 5:9] += 2.0
    stat, a, b = level_shift_scan(P)
    assert stat.shape == a.shape == b.shape == (3, 4)
    for idx in np.ndindex(3, 4):
        s1, a1, b1 = level_shift_scan(P[idx])
        assert np.isclose(float(stat[idx]), float(s1), atol=1e-9)
        assert (int(a[idx]), int(b[idx])) == (int(a1), int(b1))


def test_nan_positions_contribute_nothing():
    # a clean run of +1 in [2,5); NaN elsewhere must not change the located run
    p = np.array([np.nan, np.nan, 1.0, 1.0, 1.0, np.nan, np.nan])
    stat, a, b = level_shift_scan(p)
    assert (int(a), int(b)) == (2, 5)
    assert np.isclose(float(stat), 3.0)                # S=3, n=3 -> 9/3 = 3
    # a planted NaN inside the run is skipped (counts as 0), run still spans it
    q = np.array([-1.0, 2.0, np.nan, 2.0, -1.0])
    s2, a2, b2 = level_shift_scan(q)
    assert (int(a2), int(b2)) == (1, 4)


def test_all_unobserved_is_nan():
    stat, a, b = level_shift_scan(np.full(6, np.nan))
    assert np.isnan(float(stat))


def test_below_baseline_via_negation():
    p = np.array([0.0, 0.0, -2.0, -2.0, 0.0])
    stat, a, b = level_shift_scan(-p)                  # find the run that dips below
    assert (int(a), int(b)) == (2, 4)
    assert np.isclose(float(stat), 8.0)                # S=4 over n=2 -> 16/2 = 8


def test_top_level_reexport_is_same_object():
    assert entroptics.level_shift_scan is level_shift_scan


def test_no_run_above_the_baseline_reads_zero():
    stat, a, b = level_shift_scan(np.array([-1.0, -2.0, 0.0, -0.5]))
    assert float(stat) == 0.0 and (int(a), int(b)) == (0, 1)
