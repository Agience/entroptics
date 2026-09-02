"""What the library refuses, and why each refusal exists.

A read that quietly reshapes, truncates or substitutes its input returns a number about a signal
nobody handed in.  Every refusal below is therefore part of the contract, not defensive noise, and
each test names the wrong answer that would be returned if the check were removed.

An audit found 35 of the library's 65 ``raise`` statements never executed under test -- the
messages were unverified, so a check could be deleted, or its message could name the wrong
argument, without anything failing.
"""
import numpy as np
import pytest

import entroptics as E
from entroptics import Aperture, Screen
from entroptics import reads as R


# ── the front door needs a 2-D signal, and needs one at all ──────────────────

def test_aperture_refuses_a_non_2d_signal():
    """(T, F) is the whole model: one ordered axis, one feature axis.  A 1-D array has no feature
    axis to resolve and a 3-D one has two candidates; either would have to be guessed."""
    with pytest.raises(ValueError, match="2-D"):
        Aperture(np.zeros(64))
    with pytest.raises(ValueError, match="2-D"):
        Aperture(np.zeros((4, 8, 8)))


def test_a_streaming_aperture_refuses_to_read_before_it_has_data():
    """A streaming aperture with no frames yet has nothing to read.  Returning zeros would be a
    measurement of a signal that was never observed."""
    ap = Aperture(window=64)
    with pytest.raises(ValueError, match="no data yet"):
        _ = ap.W


# ── the batched read needs a stack, and a real resource envelope ─────────────

def test_resolved_batch_refuses_anything_that_is_not_a_stack():
    """``(B, T, F)`` -- a single frame is not a stack of one, because the caller who passed a
    frame meant a frame, and silently promoting it hides the mistake."""
    with pytest.raises(ValueError, match=r"\(B, T, F\)"):
        E.resolved_batch(np.zeros((64, 16)))
    with pytest.raises(ValueError, match=r"\(B, T, F\)"):
        E.resolved_batch(np.zeros((2, 3, 4, 5)))


def test_resource_limits_refuses_what_it_cannot_read_as_an_envelope():
    """``limits`` is a resource envelope; anything else is a caller error, not a default."""
    with pytest.raises(TypeError, match="ResourceLimits"):
        E.resolved_batch(np.zeros((2, 32, 8)), limits="4GB")
    assert E.ResourceLimits.coerce(None) == E.ResourceLimits()
    assert E.ResourceLimits.coerce({"threads": 2}).threads == 2


# ── the level is a probability ───────────────────────────────────────────────

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_far_must_be_a_probability(bad):
    """``far`` is a false-alarm RATE.  0 and 1 are not levels a tail can be taken at, and a value
    outside (0, 1) has no quantile at all."""
    from entroptics.null_providers import tw1_quantile
    with pytest.raises(ValueError, match="far must be in"):
        tw1_quantile(bad)


# ── the delay embedding and the moment pencil ────────────────────────────────

def test_tensor_refuses_a_delay_deeper_than_the_trajectory():
    """A delay embedding of depth d needs at least d+2 ordered samples to leave a pair to fit."""
    with pytest.raises(ValueError, match="too large for"):
        E.tensor_read(np.zeros((6, 4)), d=8)


def test_dynamics_free_reads_refuse_the_wrong_shape():
    """``hankel_spectrum`` reads a SCALAR sequence; the moment order is a count."""
    with pytest.raises(ValueError, match="n must be >= 1"):
        E.hankel_spectrum(np.zeros(64), n=0)


def test_proximity_refuses_a_non_2d_frame():
    """The spectral digest is of a frame, so it needs one."""
    from entroptics.proximity import mp_spectrum
    with pytest.raises(ValueError, match="2-D"):
        mp_spectrum(np.zeros(32))
    with pytest.raises(ValueError, match="non-empty"):
        mp_spectrum(np.zeros((0, 4)))


# ── the screen ───────────────────────────────────────────────────────────────

def test_screen_refuses_to_read_before_anything_is_placed():
    """A screen with no sides on it has no shared basis to read, and inventing one would report a
    measurement of nothing."""
    s = Screen()
    s.register("a", entry=lambda x: np.asarray(x))
    with pytest.raises(ValueError, match="nothing placed"):
        s.read()


def test_screen_refuses_a_lens_that_does_not_land_on_the_basis():
    """``entry`` must return a 2-D ``(T, D)`` frame on the shared basis: that landing IS the
    contract that makes one screen one meeting place."""
    s = Screen()
    s.register("bad", entry=lambda x: np.asarray(x).ravel())
    with pytest.raises(ValueError, match="2-D"):
        s.place("bad", np.zeros((30, 5)))


def test_screen_refuses_two_sides_on_different_bases():
    """Differing widths are two bases, and a coupling across them has no exact null -- so the
    screen refuses the placement."""
    rng = np.random.default_rng(0)
    s = Screen()
    s.register("a", entry=lambda x: np.asarray(x)[:, :6])
    s.register("b", entry=lambda x: np.asarray(x)[:, :4])
    s.place("a", rng.standard_normal((40, 8)))
    with pytest.raises(ValueError, match="ONE shared basis"):
        s.place("b", rng.standard_normal((40, 8)))


# ── the streaming operator ───────────────────────────────────────────────────

def test_the_operator_refuses_a_state_of_the_wrong_width():
    """The operator is fitted at a fixed feature width; a state of another width is a different
    signal, and advancing it would return coordinates that mean nothing."""
    from entroptics.dynamics import Dynamics
    rng = np.random.default_rng(0)
    d = Dynamics(8)
    d.update_block(rng.standard_normal((60, 8)))
    with pytest.raises(ValueError, match="8-vector"):
        d.update(rng.standard_normal(5))
    with pytest.raises(ValueError, match="blocks of 8-vectors"):
        d.update_block(rng.standard_normal((10, 5)))


def test_an_exact_merge_needs_perfect_memory_on_both_sides():
    """Splicing two streams adds their accumulators. A faded accumulator has already discounted
    its own history, so the sum is not the operator of the concatenated stream."""
    from entroptics.dynamics import Dynamics
    rng = np.random.default_rng(1)
    a, b = Dynamics(6, forgetting=1.0), Dynamics(6, forgetting=0.9)
    a.update_block(rng.standard_normal((40, 6)))
    b.update_block(rng.standard_normal((40, 6)))
    with pytest.raises(ValueError, match="forgetting=1"):
        a.merge(b)


def test_the_scalar_pencil_refuses_what_it_cannot_resolve():
    """A moment pencil of order n needs 2n+2 lags to be determined, and a jackknife needs at
    least two samples to have a spread at all."""
    rng = np.random.default_rng(2)
    from entroptics.dynamics import dynamics
    with pytest.raises(ValueError, match="2-D"):
        dynamics(rng.standard_normal(64))
    with pytest.raises(ValueError, match="correlation lags"):
        E.hankel_spectrum(rng.standard_normal(6), n=3)
    with pytest.raises(ValueError, match="2 samples"):
        E.jackknife(np.array([1.0]), lambda s: float(np.mean(s)))


# ── the screen's placement contract ──────────────────────────────────────────

def _lens(width, D, seed=0):
    P = np.linalg.qr(np.random.default_rng(seed).standard_normal((width, width)))[0][:, :D]
    return lambda X, P=P: np.asarray(X) @ P


def test_a_side_must_be_placed_before_it_can_be_read():
    """Every crossing read is about placed frames. A lens with nothing on it has no beam, no
    aperture and no energy, and inventing them would report a measurement of nothing."""
    s = Screen()
    s.register("a", entry=_lens(8, 6))
    for call in (lambda: s.beam("a"), lambda: s.aperture("a"), lambda: s.energy("a"),
                 lambda: s.directions("a"), lambda: s.balanced("a")):
        with pytest.raises(KeyError, match="nothing placed"):
            call()


def test_update_must_extend_a_side_on_the_basis_it_placed():
    """``update`` appends ordered steps to a side. A block that lands on a different width is a
    different basis, so it extends nothing."""
    rng = np.random.default_rng(3)
    s = Screen()
    s.register("a", entry=_lens(8, 6))
    s.place("a", rng.standard_normal((30, 8)))
    s.register("wide", entry=lambda X: np.asarray(X))
    with pytest.raises(ValueError, match="2-D"):
        s.place("wide", rng.standard_normal(30))


def test_losslessness_needs_a_surface_and_a_round_trip_that_returns_it():
    """The residual is measured between a surface and its round trip, so it needs a 2-D surface,
    and a round trip that returns a different shape has not returned the surface."""
    rng = np.random.default_rng(4)
    s = Screen()
    P = np.linalg.qr(rng.standard_normal((8, 8)))[0][:, :6]
    s.register("a", entry=lambda X: np.asarray(X) @ P, inverse=lambda C: np.asarray(C) @ P.T)
    s.place("a", rng.standard_normal((30, 8)))
    with pytest.raises(ValueError, match="2-D surface"):
        s.certify("a", np.zeros(8))
    s.register("bad", entry=lambda X: np.asarray(X) @ P,
               inverse=lambda C: np.asarray(C)[:, :3])
    s.place("bad", rng.standard_normal((30, 8)))
    with pytest.raises(ValueError, match="round trip returned shape"):
        s.certify("bad", rng.standard_normal((30, 8)))


# ── the remaining refusals ───────────────────────────────────────────────────

def test_a_streaming_aperture_has_no_operator_before_it_has_frames():
    """The operator is accumulated from frames. Handing back an empty one would report a
    dynamics nobody observed."""
    ap = Aperture(window=64)
    with pytest.raises(ValueError, match="no data"):
        ap.dynamics()


def test_a_sampled_null_needs_the_samples_it_resamples():
    """A resampling provider draws from the frame. Given only a pooled covariance there is
    nothing to draw from, so it refuses."""
    from entroptics.null_providers import FloorContext, permutation
    ctx = FloorContext(spectrum=np.array([2.0, 1.0]), data=None, shape=(50, 4),
                       far=0.05, kind="projection", rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="raw samples"):
        permutation(draws=4)(ctx)


def test_principal_directions_needs_a_2d_frame():
    """The directions are of a (T, N) frame; a higher-D field is reduced first (entroptics.fields)
    because which reduction is correct depends on the read."""
    with pytest.raises(ValueError, match="2-D"):
        R.principal_directions(np.zeros((4, 8, 8)))


def test_an_accumulator_pools_only_commensurate_frames():
    """The pooled spectrum is over one feature axis. Frames of differing width are two axes, and
    summing them would report a spectrum of neither."""
    rng = np.random.default_rng(0)
    a, b = R.SpectralAccumulator(6), R.SpectralAccumulator(9)
    a.add(rng.standard_normal((40, 6)))
    b.add(rng.standard_normal((40, 9)))
    with pytest.raises(ValueError, match="feature mismatch"):
        a.merge(b)


def test_two_sides_must_land_on_one_shared_basis():
    """Differing widths are two bases. The screen refuses the placement, so no later crossing can report a magnitude with no exact null."""
    rng = np.random.default_rng(5)
    s = Screen()
    s.register("a", entry=lambda X: np.asarray(X)[:, :6])
    s.register("b", entry=lambda X: np.asarray(X)[:, :4])
    s.place("a", rng.standard_normal((40, 8)))
    with pytest.raises(ValueError, match="ONE shared basis"):
        s.place("b", rng.standard_normal((40, 8)))


def test_a_crossing_needs_both_sides_placed():
    """A crossing is between two beams. A side carrying nothing has no beam to send or receive,
    so the read refuses."""
    rng = np.random.default_rng(6)
    s = Screen()
    s.register("a", entry=_lens(8, 6))
    s.register("b", entry=_lens(8, 6, seed=1))
    s.place("a", rng.standard_normal((30, 8)))
    with pytest.raises(KeyError, match="nothing placed"):
        s.uncondensed("a", "b")


def test_update_holds_the_same_contract_as_place():
    """``update`` extends a side, so it enforces what ``place`` enforced: the entry must land on
    the shared basis as a 2-D frame, a side keeps one basis as it grows, and a first placement by
    update meets the width the other sides already agreed."""
    rng = np.random.default_rng(7)
    s = Screen()

    # entry that returns a 3-D result lands on no basis at all
    s.register("cube", entry=lambda X: np.asarray(X)[:, :, None, None])
    with pytest.raises(ValueError, match="2-D"):
        s.update("cube", rng.standard_normal((20, 6)))

    # a side keeps one basis as it grows
    s.register("a", entry=lambda X: np.asarray(X)[:, :6])
    s.update("a", rng.standard_normal((20, 8)))
    s.register("shrink", entry=lambda X: np.asarray(X)[:, :4])
    s._lenses["a"] = s._lenses["shrink"]                  # same side, narrower entry
    with pytest.raises(ValueError, match="keeps one basis"):
        s.update("a", rng.standard_normal((10, 8)))

    # a first placement made by update still meets the basis the others agreed
    s2 = Screen()
    s2.register("x", entry=lambda X: np.asarray(X)[:, :6])
    s2.register("y", entry=lambda X: np.asarray(X)[:, :4])
    s2.update("x", rng.standard_normal((20, 8)))
    with pytest.raises(ValueError, match="ONE shared basis"):
        s2.update("y", rng.standard_normal((20, 8)))


def test_realise_needs_a_round_trip_that_returns_the_basis():
    """``realise`` measures a crossing through the receiving lens's own conversion, so that
    conversion must hand back the shared basis it was given. A lens whose round trip completes but
    lands on a different width has not returned the basis, and the shortfall it would report would be a shape error."""
    rng = np.random.default_rng(8)
    P = np.linalg.qr(rng.standard_normal((8, 8)))[0][:, :6]

    def wide_entry(X):
        X = np.asarray(X)
        return X @ P if X.shape[1] == 8 else X[:, :3]      # off-width input lands elsewhere

    s = Screen()
    s.register("a", entry=lambda X: np.asarray(X) @ P, inverse=lambda C: np.asarray(C) @ P.T)
    s.register("b", entry=wide_entry,
               inverse=lambda C: np.hstack([np.asarray(C) @ P.T, np.zeros((len(C), 1))]))
    carrier = rng.standard_normal((30, 1))                 # shared, so the crossing is pertinent
    s.place("a", carrier @ rng.standard_normal((1, 8)) + 0.1 * rng.standard_normal((30, 8)))
    s.place("b", carrier @ rng.standard_normal((1, 8)) + 0.1 * rng.standard_normal((30, 8)))
    with pytest.raises(ValueError, match="expected the shared basis"):
        s.realise("a", "b")


# ── the two raises that need a failure injected, not a bad argument ──────────

def test_a_chunked_batch_reraises_a_failure_that_is_not_out_of_memory():
    """`resolved_batch` chunks a large stack and retries a chunk that runs out of memory, halving
    it each time.  Anything else must come straight back out: swallowing it would turn a real
    fault into a silent halving loop that ends with a wrong answer at chunk size 1."""
    from entroptics import batch as B

    X = np.random.default_rng(0).standard_normal((8, 40, 12))
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise RuntimeError("the fold failed for a reason of its own")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(B, "normalize_batch", _boom)
        with pytest.raises(RuntimeError, match="reason of its own"):
            B.resolved_batch(X, limits=E.ResourceLimits(memory_gb=1e-6))
    assert len(calls) == 1, "a non-OOM failure must not be retried"


def test_a_chunked_batch_gives_up_when_one_row_still_will_not_fit():
    """The backoff halves the chunk on an out-of-memory error, so it has a floor: at one row there
    is nothing left to halve and the error is the caller's.  Without the re-raise this loops for
    ever on a box that genuinely cannot fit the read."""
    from entroptics import batch as B

    X = np.random.default_rng(0).standard_normal((8, 40, 12))
    sizes = []

    def _oom(xp, stack, *a, **k):
        sizes.append(int(np.asarray(stack).shape[0]))
        raise RuntimeError("CUDA out of memory")        # what _is_oom matches on

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(B, "normalize_batch", _oom)
        with pytest.raises(RuntimeError, match="out of memory"):
            B.resolved_batch(X, limits=E.ResourceLimits(memory_gb=1e-6))
    assert sizes and sizes[-1] == 1, f"backoff stopped at {sizes[-1]} rows, not 1"
    assert sizes == sorted(sizes, reverse=True), "the chunk must shrink, not grow"


def test_an_unreadable_frame_says_which_read_failed_and_on_what():
    """A non-convergent SVD arrives as a bare LinAlgError naming neither the read nor the frame.
    The proximity read wraps it so the caller learns both, and keeps the original as the cause."""
    from entroptics import proximity as P

    def _no_converge(*a, **k):
        raise np.linalg.LinAlgError("SVD did not converge")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(np.linalg, "svd", _no_converge)
        with pytest.raises(ValueError, match=r"proximity read failed on a \(30, 6\) frame") as ei:
            P.mp_deviation(np.random.default_rng(0).standard_normal((30, 6)))
    assert isinstance(ei.value.__cause__, np.linalg.LinAlgError)
