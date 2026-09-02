"""Spectral proximity: the Marchenko-Pastur deviation read and its probe.

Ported alongside `entroptics/proximity.py` from `agience-mantle`'s independent
reimplementation. Focuses on the properties that make the module's guarantees real: the bulk-edge/prediction identity, determinism, scale-covariance, the common-prefix
comparison rule, and the probe's exactness against a brute-force scan.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from entroptics.null_providers import johnstone
from entroptics.proximity import (
    SpectrumProbe, bulk_edge, centre, common_prefix, effective_width, mp_deviation,
    mp_spectrum, spectral_distance,
)


def _frame(seed=0, n=60, d=32, k=0, snr=0.0):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, d))
    for _ in range(k):
        u = rng.normal(size=n)
        w = rng.normal(size=d)
        M = M + snr * np.outer(u / np.linalg.norm(u), w / np.linalg.norm(w))
    return M


# ── the bulk edge is the k -> 0 limit of the per-mode prediction ──────────────────────────────

def test_the_bulk_edge_is_the_top_of_the_predicted_support():
    for n, d in [(60, 32), (32, 60), (8, 128), (120, 16)]:
        M = _frame(seed=n * 7 + d, n=n, d=d)
        edge = bulk_edge(M)
        top = float(mp_spectrum(M)[0])
        # s_MP(1) is the first order statistic, not the support point itself, so it sits just
        # under the edge -- not equal to it, but bounded by it and close.
        assert top <= edge * (1.0 + 1e-9)
        assert top / edge > 0.9, f"s_MP(1) should sit close under the edge: {top} vs {edge}"


def test_effective_width_is_exactly_a_live_count_on_the_two_valued_case():
    """`effective_width` generalises a live-channel count; on the two-valued case (every live
    channel carrying the SAME per-channel second moment v, the rest exactly zero) it must
    recover the count exactly.

    Twelve independently-drawn Gaussian columns fall short -- each has its own sample
    variance, so the "count" is only approximately 12. Reusing one base column (sign-flipped,
    which leaves the second moment unchanged) forces every live channel's v_j to be identical,
    which is what the two-valued case in the derivation actually assumes."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=40)
    A = np.zeros((40, 20))
    for j in range(12):
        A[:, j] = base if j % 2 == 0 else -base
    assert effective_width(centre(A)) == pytest.approx(12.0, rel=1e-9)


def test_effective_width_is_continuous_not_discrete():
    """A live-channel count flips on a tiny perturbation; the effective width must not."""
    rng = np.random.default_rng(4)
    A = rng.normal(size=(40, 20))
    A[:, 12:] = 0.0
    nudged = A.copy()
    nudged[:, 12] = 1e-9 * rng.normal(size=40)      # a channel that is almost dead
    w0 = effective_width(centre(A))
    w1 = effective_width(centre(nudged))
    assert abs(w1 - w0) < 1e-3, "an eight-order-of-magnitude nudge moved the width too much"


# ── determinism, scale-covariance, and what the normalisation is stated over ──────────────────

def test_the_read_is_bit_identical_on_repeat():
    M = _frame(seed=11, n=40, d=24)
    assert np.array_equal(mp_deviation(M), mp_deviation(M))


@pytest.mark.parametrize("c", [1e-4, 1.0, 1e4])
def test_scaling_the_frame_scales_the_read(c):
    M = _frame(seed=12, n=40, d=24, k=2, snr=3.0)
    dev = mp_deviation(M)
    scaled = mp_deviation(c * M)
    assert np.allclose(scaled, c * dev, rtol=1e-6, atol=1e-9 * max(1.0, abs(c)))


def test_the_read_is_not_scale_invariant():
    """The opposite of the check above: a read that is scale-invariant has thrown away the
    magnitude this module exists to keep."""
    M = _frame(seed=13, n=40, d=24)
    assert not np.allclose(mp_deviation(M), mp_deviation(2.0 * M))


def test_a_mode_carrying_no_energy_does_not_read_zero_deviation():
    """A structurally-zero singular value is not read as "at its prediction" -- it is read as
    below it, by the prediction's own value.

    The frame must carry noise on every channel (not exact zeros): a handful of genuinely dead
    channels collapses the effective width down with the live signal, which pushes the
    Marchenko-Pastur aspect ratio `b = rows/F_eff` past 1 and puts even the "empty" tail modes
    in the law's structurally-zero atom -- prediction and actual both 0, deviation 0, which
    would prove nothing about the claim under test. Full-width noise plus a planted low-rank
    signal keeps `F_eff` large and `b` well under 1, so the tail modes carry a genuine
    non-trivial prediction to be measured against."""
    rng = np.random.default_rng(21)
    M = rng.normal(size=(64, 384))
    for _ in range(5):
        u = rng.normal(size=64)
        w = rng.normal(size=384)
        M = M + 60.0 * np.outer(u / np.linalg.norm(u), w / np.linalg.norm(w))
    dev = mp_deviation(M)
    tail = dev[10:]              # well past the planted rank-5 signal
    assert np.all(tail < 0.0), "a noise-only tail mode read as non-negative deviation"


# ── common_prefix: dropped, never invented ─────────────────────────────────────────────────────

def test_records_of_different_length_compare_on_the_common_prefix():
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([5.1, 3.9, 3.2])
    x, y = common_prefix(a, b)
    assert x.tolist() == [5.0, 4.0, 3.0]
    assert y.tolist() == [5.1, 3.9, 3.2]


def test_spectral_distance_is_symmetric_and_zero_on_equal_records():
    a = np.array([5.0, 4.0, 3.0])
    b = np.array([5.1, 3.7, 2.9])
    assert spectral_distance(a, a) == 0.0
    assert spectral_distance(a, b) == pytest.approx(spectral_distance(b, a))


# ── the probe: exact, not approximate ──────────────────────────────────────────────────────────

def _brute_force_within(gallery, query, radius):
    return sorted(
        (spectral_distance(query, g), i)
        for i, g in enumerate(gallery) if spectral_distance(query, g) <= radius
    )


def _brute_force_nearest(gallery, query, k):
    ranked = sorted((spectral_distance(query, g), i) for i, g in enumerate(gallery))
    return ranked[:k]


def test_within_returns_exactly_what_a_full_scan_returns():
    rng = np.random.default_rng(42)
    gallery = [mp_deviation(_frame(seed=s, n=30, d=20, k=2, snr=4.0)) for s in range(40)]
    probe = SpectrumProbe(gallery)
    for trial in range(8):
        q = mp_deviation(_frame(seed=1000 + trial, n=30, d=20, k=2, snr=4.0))
        radius = float(rng.uniform(0.5, 5.0))
        got = sorted((h.distance, h.index) for h in probe.within(q, radius))
        want = _brute_force_within(gallery, q, radius)
        assert got == want, f"probe disagreed with a full scan at radius {radius}"


@pytest.mark.parametrize("k", [1, 3])
def test_nearest_returns_exactly_what_a_full_scan_returns(k):
    gallery = [mp_deviation(_frame(seed=s, n=30, d=20, k=2, snr=4.0)) for s in range(40)]
    probe = SpectrumProbe(gallery)
    for trial in range(8):
        q = mp_deviation(_frame(seed=2000 + trial, n=30, d=20, k=2, snr=4.0))
        got = sorted((h.distance, h.index) for h in probe.nearest(q, k=k))
        want = _brute_force_nearest(gallery, q, k)
        assert got == want, f"probe disagreed with a full scan at k={k}"


def test_the_probe_bound_holds_at_every_component_not_just_the_first():
    """The universal bound `|x_j - y_j| <= ||x - y||` that makes the probe exact -- checked
    directly, not just through the probe's own behaviour."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        x = mp_deviation(_frame(seed=int(rng.integers(1, 1_000_000)), n=30, d=20))
        y = mp_deviation(_frame(seed=int(rng.integers(1, 1_000_000)), n=30, d=20))
        a, b = common_prefix(x, y)
        dist = float(np.linalg.norm(a - b))
        assert np.all(np.abs(a - b) <= dist + 1e-9)


def test_an_empty_probe_and_a_degenerate_query_do_not_raise():
    probe = SpectrumProbe([])
    assert probe.within(np.array([1.0, 2.0]), 1.0) == []
    assert probe.nearest(np.array([1.0, 2.0]), k=3) == []
    nonempty = SpectrumProbe([np.array([1.0, 2.0, 3.0])])
    hits = nonempty.within(np.array([]), 10.0)   # an empty query defaults its key to 0.0
    assert isinstance(hits, list)


def test_within_rejects_a_negative_radius():
    probe = SpectrumProbe([np.array([1.0, 2.0])])
    with pytest.raises(ValueError):
        probe.within(np.array([1.0, 2.0]), -1.0)


# ── johnstone is reused, not re-derived ──────────────────────────────────────────────────────

def test_bulk_edge_uses_null_providers_johnstone_directly():
    """`bulk_edge` must be expressible purely in terms of `null_providers.johnstone` and the
    frame's own de-biased variance -- if it silently grew a second centring/scaling derivation
    the two would be free to drift apart."""
    M = _frame(seed=99, n=50, d=30, k=1, snr=2.0)
    A = centre(M)
    N = int(A.shape[0])
    from entroptics.proximity import _noise_sigma2_at, effective_width as _ew
    F_eff = _ew(A)
    mu, _sigma_J = johnstone(N, F_eff)
    expected = math.sqrt(_noise_sigma2_at(A, N, F_eff) * mu)
    assert bulk_edge(M) == expected


# ── design guards, carried over from the module's original home ──────────────────────────────

def test_no_constant_can_be_reintroduced():
    """The module's whole claim is that every number in it is derived from the frame, not
    typed in. Measured directly from the source: `{0, 1e-30, 1.0, 2, 3,
    9.0}` are the admissible literals (loop/index bookkeeping, the `+1e-30` divide-by-zero
    guards, and the fixed low-order TW1/MP algebra), and nothing else. A new tunable —
    a size floor, a length normalisation, a significance level — would show up here as a new
    numeric literal before it showed up in any behavioural test."""
    import ast
    import inspect

    from entroptics import proximity as _P

    tree = ast.parse(inspect.getsource(_P))
    found = {n.value for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
             and not isinstance(n.value, bool)}
    allowed = {0, 1e-30, 1.0, 2, 3, 9.0}
    assert found <= allowed, sorted(found - allowed)


def test_the_distance_itself_was_not_touched():
    """`spectral_distance` is used exactly as specified: no floor, no normalisation, no
    weight — the three compensations that would each be a chosen constant.

    Checked on the AST of the function BODY, docstring excluded, so the statement is about
    what runs: cut to the common prefix, then take one norm of one difference. Any size-aware
    repair would have to add a call, a comparison or a division here."""
    import ast
    import inspect

    fn = ast.parse(inspect.getsource(spectral_distance)).body[0]
    body = [n for n in fn.body if not (isinstance(n, ast.Expr)
                                       and isinstance(n.value, ast.Constant))]
    calls = {n.func.attr for n in ast.walk(ast.Module(body=body, type_ignores=[]))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    names = {n.id for n in ast.walk(ast.Module(body=body, type_ignores=[]))
             if isinstance(n, ast.Name)}
    assert calls == {"norm"}, calls
    assert names <= {"common_prefix", "np", "float", "x", "y", "a", "b"}, names
    assert not any(isinstance(n, (ast.Compare, ast.IfExp, ast.If))
                   for n in ast.walk(ast.Module(body=body, type_ignores=[]))), (
        "a comparison in the distance is how a floor or a length rule gets in")
