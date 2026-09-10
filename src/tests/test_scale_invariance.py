"""The central claim, as a property test: the same signal reads the same at any scale.

Step 1 of the pipeline is SCALE.  A read of a signal's STRUCTURE -- a z-score, a share, a mode
count, a correlation, a focus -- is dimensionless, so multiplying the input by a constant must
not move it.  Volts or millivolts, a correlator normalised or raw: same signal, same read.

WHY THIS FILE EXISTS.  `projection.coherence` carried an ABSOLUTE guard, `var_A < 1e-24`.  The
z-score it guards is exactly scale-invariant, but `R` is a squared inner product -- it scales as
the FOURTH power of the screen, so `var_A` scales as the EIGHTH.  One factor of ten in the
caller's units therefore took the read from `z = 18.82` to exactly `0.0` on the identical signal:
strong ordered structure silently reported as none, with no error and no warning.  The guard is
on CANCELLATION, so it belongs at the arithmetic's own resolution -- `scale * shape * macheps`,
the idiom the rest of the library already uses.

An absolute tolerance anywhere in a read of structure is this bug waiting to happen, and the
sweep below is wide on purpose: a floor placed for "reasonable" data is exactly what a caller
working in other units walks into.

The failure modes these tests watch for, stated first so they can fail:
  - a dimensionless read stops being invariant because someone adds an absolute epsilon;
  - the degenerate guards stop working, so a constant or empty screen starts reporting
    structure instead of zero -- the reason a floor is there at all;
  - the invariance is achieved by returning a constant (a read that never moves is invariant
    and useless), so each read must also be shown to RESPOND to real structure.
"""
from __future__ import annotations

import numpy as np
import pytest

import entroptics as E
from entroptics.projection import coherence, noise_floor, mode_significance
from entroptics.null_providers import noise_sigma2
from entroptics import Projection, read_batch

# far inside float64's range at both ends; none of these is pathological
SCALES = [1e-30, 1e-12, 1e-3, 1.0, 1e3, 1e12, 1e30]


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    T, F = 200, 12
    t = np.linspace(0, 8 * np.pi, T)
    return np.stack([np.sin(t + k * 0.4) + 0.15 * rng.standard_normal(T) for k in range(F)],
                    axis=1)


READS = {
    "coherence.z": lambda X: float(coherence(X)),
    "spectral_optics.resolved_modes": lambda X: float(E.reads.spectral_optics(X).resolved_modes),
    "spectral_optics.top_share": lambda X: float(E.reads.spectral_optics(X).top_share),
    "concentration.focus": lambda X: float(E.reads.concentration(X).focus),
    "concentration.resultant": lambda X: float(E.reads.concentration(X).resultant),
    # a genuinely related pair: adjacent rows of a smooth signal are coupled, adjacent rows
    # of white noise are not -- so these reads both stay scale-free AND discriminate.
    "coupling.strength": lambda X: float(E.reads.coupling(X, np.roll(X, 1, axis=0)).strength),
    "coupling.z": lambda X: float(E.reads.coupling(X, np.roll(X, 1, axis=0)).z),
    "coupling.tightness": lambda X: float(E.reads.coupling(X, np.roll(X, 1, axis=0)).tightness),
}


@pytest.mark.parametrize("name", sorted(READS))
def test_dimensionless_reads_are_scale_free(name, signal):
    fn = READS[name]
    ref = fn(signal)
    for c in SCALES:
        got = fn(signal * c)
        assert got == pytest.approx(ref, rel=1e-6, abs=1e-9), (
            f"{name} moved from {ref!r} to {got!r} when the input was scaled by {c:.0e}. "
            f"A dimensionless read reports a property of the signal, not of its units -- "
            f"look for an ABSOLUTE tolerance on a quantity that scales with the data.")


@pytest.mark.parametrize("name", sorted(READS))
def test_each_read_actually_responds_to_structure(name, signal):
    """The control that stops a constant from passing the test above.

    A read that always returns the same number is trivially scale-free.  Each read must
    separate the structured signal from white noise of the same shape, or its invariance
    above proves nothing.
    """
    fn = READS[name]
    noise = np.random.default_rng(7).standard_normal(signal.shape)
    a, b = fn(signal), fn(noise)
    assert not np.isclose(a, b, rtol=1e-3, atol=1e-6), (
        f"{name} gives {a!r} on a structured signal and {b!r} on white noise -- it does not "
        f"discriminate, so its scale-invariance is vacuous")


def test_coherence_degenerate_guards_still_fire():
    """The floor exists for a reason and the reason must survive the fix."""
    assert coherence(np.ones((50, 6))) == 0.0
    assert coherence(np.zeros((50, 6))) == 0.0
    assert coherence(np.full((50, 6), 3.7)) == 0.0


def test_coherence_survives_the_scale_that_used_to_break_it(signal):
    """The regression, named: 1e-3 is where the absolute 1e-24 guard used to trip."""
    ref = float(coherence(signal))
    assert abs(ref) > 1.0, "the fixture must carry real ordered structure for this to mean anything"
    for c in (1e-2, 1e-3, 1e-4, 1e-6):
        assert float(coherence(signal * c)) == pytest.approx(ref, rel=1e-6)

# ── the other half: DIMENSIONED reads must carry their dimension exactly ──────

DIMENSIONED = {
    # name                                fn                                              dim
    "projection.noise_floor":            (lambda X: float(noise_floor(X)),                 1),
    "null_providers.noise_sigma2":       (lambda X: float(noise_sigma2(np, X, X.shape[0], X.shape[1])), 2),
}

WIDE = [1e12, 1e6, 1.0, 1e-6, 1e-15, 1e-40, 1e-100, 1e-140]


@pytest.mark.parametrize("name", sorted(DIMENSIONED))
def test_dimensioned_reads_scale_exactly(name, signal):
    """`R(cX) == c**dim * R(X)`.

    A dimensionless read is invariant; a dimensioned one is not, and testing it as if it were
    hides the bug.  Any departure locates an ABSOLUTE constant added to a quantity that carries
    units -- which is what `noise_sigma2` had: a `+1e-30` on a VARIANCE (dimension 2), inflating
    the reported value 2.79x at a screen scale of 1e-15 and 1.8e+06x at 1e-18.  That same
    computation was copy-pasted at five sites, so the guard has to stay identical at all of them
    or the batch and per-frame reads stop agreeing (`test_resolved_batch`).
    """
    fn, dim = DIMENSIONED[name]
    ref = fn(signal)
    assert ref > 0.0, "the fixture must produce a positive value for a ratio test to mean anything"
    for c in WIDE:
        got = fn(signal * c)
        expect = (c ** dim) * ref
        assert got == pytest.approx(expect, rel=1e-9), (
            f"{name} returned {got!r} at scale {c:.0e}, expected {expect!r} "
            f"(dimension {dim}). Look for an absolute constant added to or clamped onto a "
            f"quantity that carries units.")


def test_a_screen_with_no_energy_has_no_noise():
    """The degenerate case the old absolute constant was really paying for.

    It now lives in `mode_significance`, the one consumer that divides by sigma^2, which is what
    lets the variance itself stay exact rather than floored.
    """
    z = np.zeros((50, 6))
    assert float(noise_sigma2(np, z, 50, 6)) == 0.0
    ms = mode_significance(z)
    assert np.all(np.asarray(ms.pvalue) == 1.0)
    assert np.all(np.asarray(ms.deviate) == 0.0)


# ── the batch path folds on its own marginal, so it needs its own scale gate ──

@pytest.mark.parametrize("c", [1.0, 1e-6, 1e-15, 1e-20, 1e-40])
def test_batch_and_per_frame_agree_at_every_scale(c):
    """`read_batch` must equal `Projection` frame by frame, at ANY scale.

    Bit-identity is the contract the batch path already claims in its docstring; this asserts
    it survives a change of units, which nothing else tests.

    It does NOT currently discriminate the batch fold marginal's former absolute constants
    (`tot > 1e-30` and a `1e-12` probability clip): those were replaced with the exact `> 0`
    tests the per-frame path uses, but no input has been found where either changed a returned
    `F_eff`, so no test here can claim to catch them.
    """
    rng = np.random.default_rng(0)
    N, F = 128, 8
    frames = [(([0.0, 2.0, 5.0, 12.0][i % 4]) * np.outer(rng.standard_normal(N),
                                                         rng.standard_normal(F))
               + rng.standard_normal((N, F))) * c for i in range(8)]
    for r, s in zip(read_batch(frames), (Projection(f) for f in frames)):
        assert r.K_signal == s.K_signal
        assert float(r.noise_floor) == float(s.noise_floor)
        assert np.array_equal(r.S, s.S)


# ── the far ends: silent NaN is worse than a refusal ─────────────────────────

@pytest.mark.parametrize("exponent", [-150, -100, -40, 0, 40, 100, 150])
def test_coherence_returns_a_number_at_extreme_scales(signal, exponent):
    """`coherence` reaches the EIGHTH power of the screen: `R` squares the Gram and the
    permutation moments square its row sums.  Formed on the raw Gram that overflowed to `nan`
    above a screen magnitude of ~1e38 -- returned, not raised, with only a RuntimeWarning that a
    caller may never see.

    The z-score is scale-invariant by construction, so normalising the Gram before squaring is
    exact in value (measured shift 1.2e-14 to 1.2e-13 relative, against a golden tolerance of
    1e-8) and removes the cliff entirely.
    """
    z = float(coherence(signal * (10.0 ** exponent)))
    assert np.isfinite(z), (
        f"coherence returned {z} at scale 1e{exponent}. A read that cannot answer must refuse, "
        f"not hand back NaN -- and this one can answer, since its z-score is scale-free.")
    assert z == pytest.approx(float(coherence(signal)), rel=1e-6)

