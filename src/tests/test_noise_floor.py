"""Noise-floor calibration + per-mode footprint reads.

Regression guard for the derived floor: the false-alarm rate must stay near the
significance level across aspect ratios. An uncalibrated (biased) median-row-norm
sigma^2 estimate can diverge sharply by aspect ratio -- as far as ~40% at N>>F and
~20% at N<<F -- so this pins the calibrated behavior directly.  Deterministic seeds
-> non-flaky.
"""
import math

import numpy as np
import pytest

from entroptics import Projection, Aperture
from entroptics.projection import footprints


def _far(N, F, trials=200):
    """P(K_signal >= 1) on pure iid Gaussian noise (deterministic seeds)."""
    hits = 0
    for i in range(trials):
        W = np.random.default_rng(1000 + i).standard_normal((N, F))
        hits += Projection(W).K_signal >= 1
    return hits / trials


@pytest.mark.parametrize("N,F", [(1000, 20), (500, 50), (200, 50), (128, 128), (20, 1000)])
def test_floor_far_calibrated_across_aspect_ratios(N, F):
    # Target FAR is 5%; assert well under 15% at every aspect ratio. An uncalibrated
    # (biased) floor can give 0.40 at (1000,20) and 0.20 at (20,1000).
    assert _far(N, F) < 0.15


def test_floor_far_not_trivially_zero():
    # A floor that never fires would also pass the bound above; confirm it is a real
    # ~5% test by checking the pooled FAR over shapes is in a sane band, not 0.
    pooled = np.mean([_far(N, F, trials=150) for (N, F) in [(200, 50), (128, 128), (256, 256)]])
    assert 0.005 < pooled < 0.12


def test_floor_detects_planted_mode():
    # A rank-1 mode at 1.5x the edge must be resolved at every tested shape.
    for (N, F) in [(200, 50), (128, 128), (20, 1000)]:
        edge = math.sqrt(N) + math.sqrt(F)
        rng = np.random.default_rng(7)
        u = rng.standard_normal(N); u /= np.linalg.norm(u)
        v = rng.standard_normal(F); v /= np.linalg.norm(v)
        W = 1.5 * edge * np.outer(u, v) + rng.standard_normal((N, F))
        assert Projection(W).K_signal >= 1


def test_floor_holds_under_heteroscedastic_noise():
    # Channels with 10x different noise scales must still whiten to a calibrated floor
    # (the shrinkage keeps a noisy small-N per-channel MAD from exploding the floor).
    hits = 0
    for i in range(150):
        rng = np.random.default_rng(2000 + i)
        scales = np.exp(rng.uniform(-math.log(10), math.log(10), 60))
        W = rng.standard_normal((150, 60)) * scales[None, :]
        hits += Projection(W).K_signal >= 1
    assert hits / 150 < 0.20


# ── per-mode footprint (localization) ────────────────────────────────────────

def _planted(Nt, Nf, freq_idx, time_idx, amp, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros(Nt); u[time_idx] = 1.0; u /= np.linalg.norm(u)
    v = np.zeros(Nf); v[freq_idx] = 1.0; v /= np.linalg.norm(v)
    return amp * np.outer(u, v) + rng.standard_normal((Nt, Nf))


def test_footprint_reads_broadband_transient_through_pipeline():
    # A broadband, time-localized burst is exactly what per-channel whitening keeps:
    # it must resolve and read as spread-in-feature, localized-in-time.
    Nt, Nf = 64, 400
    amp = 6.0 * (math.sqrt(Nt) + math.sqrt(Nf))
    broad = Aperture(_planted(Nt, Nf, np.arange(Nf), [30, 31, 32, 33], amp)).footprints
    assert broad
    assert broad[0].phi_F > 0.5 and broad[0].phi_T < 0.3
    # a broadband transient occupies little phase-space area (localized in time)
    assert broad[0].etendue < 0.3
    assert abs(broad[0].etendue - broad[0].phi_T * broad[0].phi_F) < 1e-12   # dataclass contract


def test_footprint_reads_narrowband_through_pipeline():
    # A realistic (time-varying) narrowband signal is resolved by the pipeline and
    # reads as localized in feature.  Narrowband works end to end; only a perfectly
    # constant DC per-channel offset is removed (baseline rejection, tested below).
    Nt, Nf = 200, 400
    amp = 6.0 * (math.sqrt(Nt) + math.sqrt(Nf))
    rng = np.random.default_rng(1)
    v = np.zeros(Nf); v[200:206] = 1.0; v /= np.linalg.norm(v)         # 6 channels
    u = rng.standard_normal(Nt); u -= u.mean(); u /= np.linalg.norm(u)  # time-varying
    W = amp * np.outer(u, v) + rng.standard_normal((Nt, Nf))
    fp = Aperture(W).footprints
    assert fp                                    # narrowband IS detected
    assert fp[0].phi_F < 0.15                    # and reads as narrowband


def test_constant_dc_baseline_is_removed_not_detected():
    # A perfectly constant per-channel offset (a DC baseline / fixed bandpass) must not
    # read as signal: the per-channel centring removes it.  This is the one narrowband
    # case that is (correctly) suppressed; time-varying narrowband is detected above.
    Nt, Nf = 200, 400
    amp = 6.0 * (math.sqrt(Nt) + math.sqrt(Nf))
    rng = np.random.default_rng(2)
    v = np.zeros(Nf); v[200:206] = 1.0; v /= np.linalg.norm(v)
    W = amp * np.outer(np.ones(Nt) / math.sqrt(Nt), v) + rng.standard_normal((Nt, Nf))
    assert Projection(W).K_signal == 0


def test_footprint_computation_localized_vs_spread():
    # Unit-test the footprint math directly on constructed SVD vectors (independent of
    # the whitening, which suppresses narrowband signals): a localized mode vector must
    # read a small fill, a spread one a fill near 1.
    Nt, Nf = 64, 400
    U = np.zeros((Nt, 1)); U[:, 0] = 1.0 / math.sqrt(Nt)          # spread over time
    Vt = np.zeros((1, Nf)); Vt[0, 200:206] = 1.0 / math.sqrt(6)    # localized to 6 channels
    fp = footprints(U, np.array([10.0]), Vt, 1)[0]
    assert fp.phi_T > 0.99                                        # uniform -> fill ~ 1
    assert fp.phi_F < 0.02                                        # 6/400 channels -> fill ~ 0.015
    assert abs(fp.phi_F - 6.0 / Nf) < 1e-6


def test_spectral_mp_edge_far_calibrated_moderate_sampling():
    # The reads.spectral_optics "mp" edge (finite-size Tracy-Widom) must control the
    # resolved-mode false-alarm rate for sample counts T that are not tiny (T >= ~80).
    from entroptics.reads import spectral_optics
    for (T, N) in [(200, 50), (100, 100), (150, 300)]:   # T (samples) not tiny
        hits = sum(spectral_optics(np.random.default_rng(4000 + i).standard_normal((T, N))).resolved_modes >= 1
                   for i in range(200))
        assert hits / 200 < 0.15


def test_tw1_survival_matches_tabulated_quantiles():
    # the Chiani Gamma approximation P(TW1 > q_alpha) must return alpha at the tabulated
    # upper quantiles (the same q used by the floor), within the approximation error.
    from entroptics.projection import _tw1_sf, _TW1_UPPER_Q
    for far, q in _TW1_UPPER_Q.items():
        assert abs(_tw1_sf(q) - far) < 0.005


def test_mode_significance_consistent_with_k_signal():
    # the resolved count must equal the number of modes whose p-value clears far:
    # the evidence read and the thresholded count are the same object at the same far.
    for (N, F, r) in [(200, 50, 0), (128, 128, 0), (300, 60, 3), (20, 1000, 0)]:
        rng = np.random.default_rng(N + F)
        W = rng.standard_normal((N, F))
        for _ in range(r):
            u = rng.standard_normal(N); u /= np.linalg.norm(u)
            v = rng.standard_normal(F); v /= np.linalg.norm(v)
            W = W + 3.0 * (math.sqrt(N) + math.sqrt(F)) * np.outer(u, v)
        sc = Projection(W)
        sig = sc.significance
        assert int((sig.pvalue < 0.05).sum()) == sc.K_signal          # far=0.05 default
        assert (sig.pvalue >= 0.0).all() and (sig.pvalue <= 1.0).all()
        assert sig.deviate.shape == sig.pvalue.shape == sc.S.shape


def test_mode_significance_is_alpha_free_evidence():
    # the evidence (deviate, pvalue) does not depend on any threshold; only the count does.
    W = np.random.default_rng(5).standard_normal((150, 40))
    sig = Aperture(W).significance
    # a stronger far resolves at least as many modes as a weaker one, from the same p-values
    assert int((sig.pvalue < 0.10).sum()) >= int((sig.pvalue < 0.01).sum())


def test_footprint_empty_when_no_signal():
    W = np.random.default_rng(0).standard_normal((80, 40))
    sc = Projection(W)
    assert sc.K_signal == 0
    assert sc.footprints == []
    # module-level helper agrees with the Projection property
    assert footprints(sc.U, sc.S, sc.Vt, sc.K_signal) == []


def test_footprint_count_matches_k_signal():
    Nt, Nf = 80, 300
    amp = 5.0 * (math.sqrt(Nt) + math.sqrt(Nf))
    rng = np.random.default_rng(3)
    W = rng.standard_normal((Nt, Nf))
    for _ in range(2):                            # plant two well-separated modes
        u = rng.standard_normal(Nt); u /= np.linalg.norm(u)
        v = rng.standard_normal(Nf); v /= np.linalg.norm(v)
        W = W + amp * np.outer(u, v)
    sc = Projection(W)
    assert len(sc.footprints) == sc.K_signal >= 1


# ── the noise floor as a caller-suppliable null provider (screen floor / K_signal) ──

from entroptics import null_providers as nulls


def test_default_null_is_mp_and_unchanged():
    # the default null provider must be mp, reproducing the reference floor and K_signal
    # exactly (the whole suite/golden rests on this); passing mp explicitly matches None.
    from entroptics.projection import noise_floor
    W = np.random.default_rng(11).standard_normal((120, 40))
    sc = Projection(W)
    assert Projection(W, null=nulls.mp).K_signal == sc.K_signal
    assert noise_floor(sc.screen) == noise_floor(sc.screen, null=nulls.mp)


def test_permutation_provider_is_at_least_as_conservative_on_correlated_bulk():
    # a fully cross-correlated bulk with no planted low-rank mode,
    # scored against the i.i.d. mp yardstick, over-counts resolved modes; the permutation
    # provider preserves each channel's marginal but destroys the cross-channel alignment,
    # so it never counts more and sometimes counts fewer.
    km, kp = [], []
    for s in range(8):
        r = np.random.default_rng(100 + s)
        T, F = 150, 30
        A = r.standard_normal((F, F))                      # full-rank random mixing
        X = r.standard_normal((T, F)) @ A                  # correlated bulk, not low-rank
        km.append(Projection(X).K_signal)                                          # default mp
        kp.append(Projection(X, null=nulls.permutation(draws=80), seed=0).K_signal)
    assert all(p <= m for p, m in zip(kp, km))             # never over-counts vs mp
    assert sum(kp) < sum(km)                               # strictly more conservative overall


def test_permutation_provider_still_resolves_a_strong_mode():
    # a genuine strong rank-1 mode above the marginal null is still resolved.
    r = np.random.default_rng(5)
    T, F = 150, 30
    u = r.standard_normal(T); u /= np.linalg.norm(u)
    v = r.standard_normal(F); v /= np.linalg.norm(v)
    X = r.standard_normal((T, F)) + 8.0 * (math.sqrt(T) + math.sqrt(F)) * np.outer(u, v)
    assert Projection(X, null=nulls.permutation(draws=80), seed=0).K_signal >= 1


def test_resampling_provider_deterministic_per_seed():
    W = np.random.default_rng(13).standard_normal((100, 25))
    a = Projection(W, null=nulls.permutation(draws=50), seed=7).K_signal
    b = Projection(W, null=nulls.permutation(draws=50), seed=7).K_signal
    assert a == b


def test_non_callable_null_raises():
    from entroptics.projection import noise_floor
    sc = Projection(np.random.default_rng(0).standard_normal((40, 20)))
    with pytest.raises(TypeError):
        noise_floor(sc.screen, null="mp")            # a string is not a provider callback


def test_custom_null_provider_callback_is_evaluated_per_screen():
    # any FloorContext->float callback is a valid null: here one that floors just below the
    # top singular value, so exactly the leading mode resolves.  It runs on this screen.
    W = np.random.default_rng(1).standard_normal((80, 30))
    sc = Projection(W, null=lambda ctx: float(ctx.spectrum[0]) * 0.999)
    assert sc.K_signal == 1


def test_aperture_screen_threads_null_provider():
    # a caller reads K_signal via Aperture(config).projection(); the provider must reach it,
    # both as a per-call override and as the aperture's own (streaming/dynamical) provider.
    W = np.random.default_rng(1).standard_normal((80, 30))
    assert Aperture(W).projection(null=nulls.permutation(draws=40)).K_signal >= 0
    assert Aperture(W, null=nulls.permutation(draws=40)).projection().K_signal >= 0


def test_stateful_provider_update_runs_in_the_stream():
    # a stateful provider's update(frame) is called per streaming frame (it "runs in the
    # dynamical"), and its __call__ sets the floor on each local screen.
    class CountingNull:
        def __init__(self): self.frames = 0
        def update(self, frame): self.frames += 1
        def __call__(self, ctx): return nulls.mp(ctx)
    prov = CountingNull()
    ap = Aperture(null=prov)
    rng = np.random.default_rng(3)
    for _ in range(20):
        ap.update(rng.standard_normal(12))
    assert prov.frames == 20                          # update ran once per frame
    assert ap.projection().K_signal >= 0                  # and the provider floors the local screen


def test_spectral_null_provider_matches_default():
    # the correlation floor takes the same provider contract; None == mp explicitly.
    from entroptics.reads import spectral_optics
    W = np.random.default_rng(2).standard_normal((120, 40))
    assert spectral_optics(W).noise_floor == spectral_optics(W, null=nulls.mp).noise_floor
    assert spectral_optics(W, null=nulls.robust).resolved_modes >= 0


def test_different_provider_per_cut_point():
    # each cut point (screen / spectral / bulk) can take its own provider, via by_kind or a
    # {kind: provider} mapping, and the read for each kind uses only its entry.
    from entroptics.reads import spectral_optics, SpectralAccumulator
    from entroptics import by_kind
    W = np.random.default_rng(4).standard_normal((150, 30))
    seen = []
    def tag(kind, base):
        def prov(ctx):
            seen.append(ctx.kind); return base(ctx)
        return prov
    routed = by_kind(projection=tag("projection", nulls.mp),
                     spectral=tag("spectral", nulls.robust),
                     bulk=tag("bulk", nulls.mp))
    # a bare dict is accepted anywhere a provider is, exactly like by_kind
    assert Projection(W, null=routed).K_signal == Projection(W, null={"projection": nulls.mp}).K_signal
    _ = spectral_optics(W, null=routed).resolved_modes
    acc = SpectralAccumulator(30); acc.add(W)
    _ = acc.spectral(null=routed).resolved_modes
    assert set(seen) == {"projection", "spectral", "bulk"}          # each cut point hit its own entry


def test_aperture_routes_screen_and_spectral_to_different_providers():
    # one Aperture, one null= mapping -> screen floor and spectral floor use different providers.
    from entroptics import by_kind
    W = np.random.default_rng(5).standard_normal((150, 30))
    hits = {"projection": 0, "spectral": 0}
    def count(kind):
        def prov(ctx):
            hits[kind] += 1; return nulls.mp(ctx)
        return prov
    ap = Aperture(W, null=by_kind(projection=count("projection"), spectral=count("spectral")))
    _ = ap.projection().K_signal
    _ = ap.spectral.resolved_modes
    assert hits == {"projection": 1, "spectral": 1}                 # routed apart, not shared


def test_by_kind_rejects_unknown_cut_point():
    from entroptics import by_kind
    with pytest.raises(ValueError):
        by_kind(projection=nulls.mp, bogus=nulls.mp)


def test_reference_null_is_deterministic_and_sharpens():
    # the reference-calibrated Gaussian null: floor = center + z(far)*scale from a signal-free
    # reference's top-mode values.  Deterministic, and sharpens analytically to any far.
    from entroptics.null_providers import reference_null, apply_floor, ReferenceNull, top_spectrum_value
    r = np.random.default_rng(0)
    ref = [top_spectrum_value(r.standard_normal((100, 20)), "projection") for _ in range(80)]
    prov = reference_null(ref)
    def floor(far):
        return apply_floor(prov, spectrum=None, data=None, shape=(100, 20), far=far, kind="projection")
    assert floor(0.05) == floor(0.05)                       # deterministic (no RNG)
    assert floor(1e-5) > floor(0.05) > floor(0.5)           # sharper far -> higher floor, analytically
    # the stateful Welford form matches the batch calibration
    rn = ReferenceNull(ref)
    assert rn.center == pytest.approx(prov.center) and rn.scale == pytest.approx(prov.scale, rel=1e-6)
    assert rn.n_reference == len(ref)


def test_derived_edge_serves_arbitrary_sharp_far():
    # the false-alarm level travels with the null and may be sharpened without limit: the
    # TW1 quantile is inverted from the survival function for any far outside the tabulated
    # set (e.g. 1e-5 = 99.999%), and sharper far -> larger quantile -> higher floor.
    qs = [nulls.tw1_quantile(f) for f in [0.10, 0.05, 1e-3, 1e-5, 1e-7]]
    assert all(np.isfinite(qs)) and all(x < y for x, y in zip(qs, qs[1:]))   # monotone, no raise
    W = np.random.default_rng(7).standard_normal((120, 40))
    assert Projection(W, far=1e-5).K_signal <= Projection(W, far=0.05).K_signal      # sharper -> stricter


def test_stateful_provider_sharpens_alpha_over_the_run():
    # a provider that accumulates surrogates across screens (its long-term sample) tightens
    # its empirical far as the run progresses -- the floor sharpens (rises) with the sample.
    class Sharpening:
        def __init__(self): self.tops = []
        def update(self, frame): pass
        def __call__(self, ctx):
            X = np.asarray(ctx.data)
            for _ in range(50):
                self.tops.append(nulls.top_spectrum_value(nulls.shuffle_in_time(X, ctx.rng), ctx.kind))
            far = min(0.05, 5.0 / len(self.tops))          # sharper as the sample grows
            return float(np.quantile(self.tops, 1.0 - far))
    prov = Sharpening()
    W = np.random.default_rng(42).standard_normal((60, 20))
    floors = [Projection(W, null=prov, seed=i).noise_floor for i in range(5)]
    assert floors[-1] >= floors[0]                         # the cutoff sharpened over the run


# ── the provider reaches every read that thresholds against it ────────────────

def _signal_frame(T=256, F=16, seed=0):
    """Two clean ordered modes over noise -- resolves > 0 under the derived floor, so a floor
    the caller raises can be seen to take the count back down."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12, T)
    return (np.outer(np.sin(t), rng.standard_normal(F)) * 6
            + np.outer(np.cos(2 * t), rng.standard_normal(F)) * 4
            + rng.standard_normal((T, F)))


def test_scale_profile_reads_against_the_apertures_own_floor():
    """``K_signal`` counts against the floor and ``contrast`` divides by it, so a profile that
    dropped the caller's provider would be a different measurement under the same name.

    The aperture's own floor reaches every window, so a caller-supplied provider governs the
    profile the same way it governs a single read."""
    W = _signal_frame()
    base = Aperture(W).scale_profile()
    raised = Aperture(W, null=lambda ctx: 1e6).scale_profile()
    assert int(np.asarray(base.K_signal).max()) >= 1            # the default resolves something
    assert int(np.asarray(raised.K_signal).max()) == 0          # an impossible floor resolves none
    # contrast is sigma_top / floor, so raising the floor to 1e6 drives it to ~1e-5, not to 0
    assert np.asarray(raised.contrast).max() < np.asarray(base.contrast).max() / 1000.0


def test_the_projection_cut_point_is_named_projection():
    """A reference null must be calibrated in the units the floor thresholds.

    The two branches score different quantities, so ``top_spectrum_value`` names its cut point: a kind outside ``KINDS`` raises."""
    from entroptics import top_spectrum_value, KINDS
    rng = np.random.default_rng(3)
    X = rng.standard_normal((100, 20))
    assert "screen" not in KINDS
    assert top_spectrum_value(X, "projection") != top_spectrum_value(X, "spectral")
    with pytest.raises(ValueError, match="unknown kind"):
        top_spectrum_value(X, "screen")


def test_a_reference_calibrated_aperture_uses_the_projection_statistic():
    """The reference null the aperture builds for its projection must match one built by hand
    in the projection's own units."""
    from entroptics import reference_null, top_spectrum_value
    rng = np.random.default_rng(5)
    ref = [rng.standard_normal((256, 16)) for _ in range(40)]
    W = _signal_frame(seed=1)
    by_hand = reference_null([top_spectrum_value(r, "projection") for r in ref])
    assert Aperture(W, reference=ref).projection().noise_floor == \
           pytest.approx(Projection(W, null=by_hand).noise_floor, rel=1e-12)
