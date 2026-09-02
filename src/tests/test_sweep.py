"""The aperture SWEEP: a fixed-capacity aperture swept across the feature axis, gated by coherence.

The guarantees are structural, not physical: the sweep must (a) return bands only where there IS
coherent structure -- a pure-noise field yields nothing; (b) locate the coherent band by its column
``span``; and (c) read a finite on-pulse width and tail decay there.  Deterministic seeds."""
import numpy as np

from entroptics import sweep


def _field(T=140, F=2048, lo=800, hi=1200, tau=3.0, amp=8.0, seed=0):
    """iid-noise field with a scattered, transient burst confined to columns [lo, hi)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((T, F))
    t = np.arange(T)
    prof = np.exp(-0.5 * ((t - 45) / 1.8) ** 2)
    k = np.exp(-np.arange(T) / tau); k /= k.sum()
    prof = np.convolve(prof, k)[:T]
    W[:, lo:hi] += amp * prof[:, None]
    return W


def test_sweep_gates_to_the_coherent_band():
    bands = sweep(_field(), patch=256, coherence=3.0)
    assert bands, "must find the coherent band"
    for b in bands:
        s0, s1 = b["span"]
        assert s1 > 800 and s0 < 1200, "every returned band must overlap the signal columns"
        assert b["coherence"] >= 3.0, "the coherence gate must hold"
        assert np.isfinite(b["width"]), "on-pulse width must be read"


def test_sweep_skips_pure_noise():
    W = np.random.default_rng(1).standard_normal((140, 2048))
    assert sweep(W, patch=256, coherence=3.0) == [], "no coherent structure -> no bands"


def test_sweep_tail_decay_grows_with_scattering():
    """A longer scattering tail reads a larger tau_decay (the sweep tracks the decay rate)."""
    def med_tau(tau):
        b = sweep(_field(tau=tau, amp=10.0), patch=256, coherence=3.0)
        taus = [x["tau_decay"] for x in b if np.isfinite(x["tau_decay"])]
        return float(np.median(taus)) if taus else np.nan
    assert med_tau(6.0) > med_tau(2.0), "more scattering -> longer tail decay"
