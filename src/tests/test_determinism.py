"""The reads are deterministic: no RNG anywhere in the optics/screen path, so
repeated calls are bit-identical (this is what makes the golden a valid contract
and the coherence a reproducible z-score)."""
import numpy as np

from entroptics import aperture as A
from entroptics.projection import Projection, coherence
from conftest import build_W


def test_optics_is_repeatable(W):
    a = A.Aperture(W).optics()
    b = A.Aperture(W).optics()
    for k in a:
        assert a[k] == b[k], k


def test_coherence_is_repeatable(W):
    sc = Projection(W)
    assert coherence(sc.screen) == coherence(sc.screen)


def test_coherence_takes_no_rng():
    """coherence's signature must not carry an rng/n_shuffles knob (it is closed-form)."""
    import inspect
    params = set(inspect.signature(coherence).parameters)
    assert "rng" not in params and "n_shuffles" not in params
    assert params == {"screen", "lag"}


def test_screen_read_takes_no_rng():
    import inspect
    from entroptics.projection import read
    params = set(inspect.signature(read).parameters)
    assert "rng" not in params and "n_shuffles" not in params


def test_repeated_screen_reads_identical(W):
    r1 = Projection(W).read()
    r2 = Projection(W).read()
    assert r1 == r2
