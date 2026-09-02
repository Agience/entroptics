"""Batched ensemble reads: ``read_batch`` (screen monitor) and ``spectral_batch`` (correlation
eigvalsh) must be BIT-IDENTICAL to reading each frame with ``Screen`` / ``spectral_optics`` --
batching only changes the loop nesting, never a float -- and must fall back cleanly for a frame
they cannot batch (masked / non-finite / complex / odd shape / too small).  Deterministic seeds."""
import numpy as np
import pytest

from entroptics import Screen, read_batch, BatchRead
from entroptics.reads import spectral_batch, spectral_optics
from entroptics.null_providers import reference_null, top_spectrum_value


def _frames(shape=(128, 8), n=24, seed=0):
    """A mix of pure-noise (no fold) and increasingly structured (folding) frames of ONE shape,
    so both the fold-guard branches and the F_eff grouping are exercised."""
    rng = np.random.default_rng(seed)
    N, F = shape
    return [([0.0, 2.0, 5.0, 12.0][i % 4]) * np.outer(rng.standard_normal(N), rng.standard_normal(F))
            + rng.standard_normal((N, F)) for i in range(n)]


# ── read_batch (screen monitor) ───────────────────────────────────────────────

@pytest.mark.parametrize("shape", [(128, 8), (8, 8), (64, 16), (200, 32)])
def test_read_batch_bit_identical(shape):
    frames = _frames(shape)
    for r, s in zip(read_batch(frames), (Screen(f) for f in frames)):
        assert r.K_signal == s.K_signal
        assert r.sigma_top == s.sigma_top
        assert float(r.noise_floor) == float(s.noise_floor)
        assert np.array_equal(r.S, s.S)


def test_read_batch_with_null_provider():
    """A non-default (reference-null) provider is applied per frame -- still bit-identical."""
    frames = _frames((8, 8), n=20)
    prov = reference_null([Screen(np.random.default_rng(k).standard_normal((8, 8))).sigma_top
                           for k in range(12)])
    for r, s in zip(read_batch(frames, null=prov), (Screen(f, null=prov) for f in frames)):
        assert r.K_signal == s.K_signal
        assert float(r.noise_floor) == float(s.noise_floor)
        assert np.array_equal(r.S, s.S)


def test_read_batch_mixed_fold_widths():
    """Frames whose entropy fold lands on DIFFERENT widths must each be grouped and stay exact."""
    frames = _frames((100, 32), n=30)
    got = read_batch(frames)
    assert len({r.S.shape[0] for r in got}) > 1        # more than one F_eff group is actually present
    for r, s in zip(got, (Screen(f) for f in frames)):
        assert r.K_signal == s.K_signal and np.array_equal(r.S, s.S)


def test_read_batch_fallbacks_match_per_frame():
    """Complex / NaN / odd-shape frames fall back to a per-frame Screen -- still the right answer."""
    rng = np.random.default_rng(1)
    good = rng.standard_normal((32, 8))
    cplx = rng.standard_normal((32, 8)) + 1j * rng.standard_normal((32, 8))
    nan = rng.standard_normal((32, 8)); nan[0, 0] = np.nan
    odd = rng.standard_normal((16, 8))                 # a different shape than the others
    frames = [good, cplx, nan, odd]
    for r, f in zip(read_batch(frames), frames):
        s = Screen(f)
        assert r.K_signal == s.K_signal and np.array_equal(r.S, s.S)


def test_read_batch_types_and_empty():
    assert read_batch([]) == []
    (r,) = read_batch([np.random.default_rng(0).standard_normal((32, 8))])
    assert isinstance(r, BatchRead) and isinstance(r.K_signal, int)


# ── spectral_batch (correlation eigvalsh) ─────────────────────────────────────

@pytest.mark.parametrize("shape", [(8, 8), (49, 8), (64, 16), (128, 32)])
def test_spectral_batch_bit_identical(shape):
    frames = _frames(shape, n=20)
    prov = reference_null([top_spectrum_value(np.random.default_rng(k).standard_normal(shape),
                                              "spectral") for k in range(12)])
    for r, s in zip(spectral_batch(frames, null=prov), (spectral_optics(f, null=prov) for f in frames)):
        assert r.contrast == s.contrast              # every SpectralOptics field, not just contrast:
        assert r.resolved_modes == s.resolved_modes
        assert r.attenuation == s.attenuation
        assert r.top_share == s.top_share
        assert r.dispersion == s.dispersion
        assert np.array_equal(r.eigenvalues, s.eigenvalues)


def test_spectral_batch_default_mp():
    frames = _frames((64, 16), n=15)
    for r, s in zip(spectral_batch(frames), (spectral_optics(f) for f in frames)):
        assert r.contrast == s.contrast and np.array_equal(r.eigenvalues, s.eigenvalues)


def test_spectral_batch_fallbacks_match_per_frame():
    rng = np.random.default_rng(2)
    good = rng.standard_normal((32, 8))
    cplx = rng.standard_normal((32, 8)) + 1j * rng.standard_normal((32, 8))
    nan = rng.standard_normal((32, 8)); nan[1, 1] = np.nan
    small = rng.standard_normal((2, 8))                # T < 3 -> spectral_optics returns the empty read
    for r, f in zip(spectral_batch([good, cplx, nan, small]), [good, cplx, nan, small]):
        s = spectral_optics(f)
        assert r.contrast == s.contrast and np.array_equal(r.eigenvalues, s.eigenvalues)
