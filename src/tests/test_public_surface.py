"""The documented public surface, exercised against its own identities.

An audit of executed lines found 51 public callables that no test ever entered -- among them
``Projection.beam`` (the 0.2.1 replacement for the removed ``embeddings``/``vocabulary``/``K``),
``Screen.update``, the ``entroptics.extract(W)`` shortcut the README points at, and the
``self_calibrating_null`` provider.  Untested surface is where the drift was: a default of
``kind="screen"`` survived the 0.2.1 rename in ``self_calibrating_null`` precisely because nothing
called it.

These are identity checks, not smoke tests: each read is pinned to something it must equal.  Note
that the aperture's ``contrast`` / ``top_share`` / ``noise_floor`` / ``resolved_modes`` are the
SPECTRAL cut point (the correlation edge), not the projection's singular-value floor -- two cut
points, deliberately different numbers.
"""
import types

import numpy as np
import pytest

import entroptics as E
from entroptics import Aperture, Projection, Screen
from entroptics import reads as R
from entroptics import entropy as EN


@pytest.fixture
def W():
    rng = np.random.default_rng(0)
    T, F = 192, 24
    t = np.linspace(0, 12, T)
    return (np.outer(np.sin(t), rng.standard_normal(F)) * 6
            + np.outer(np.cos(2 * t), rng.standard_normal(F)) * 3
            + rng.standard_normal((T, F)))


# ── reads: the free functions agree with the front door ──────────────────────

def test_magnification_and_duality_are_one_arithmetic(W):
    """Three call sites read the same reciprocal; they must not drift."""
    ap = Aperture(W, window=None)
    assert R.magnification(W) == pytest.approx(1.0 / R.phi(W), rel=1e-12)
    assert R.magnification(W) == pytest.approx(ap.magnification, rel=1e-12)
    d, ad = R.scale_duality(W), ap.duality
    assert d["magnification"] == pytest.approx(1.0 / d["phi"], rel=1e-12)
    assert d == ad
    assert ad["at_diffraction_limit"] == ap.at_diffraction_limit


def test_duality_of_handles_a_degenerate_phi():
    """phi = 0 carries no scale, so the reciprocal is infinite."""
    assert R.duality_of(0.0)["magnification"] == float("inf")
    assert R.duality_of(1.0)["at_diffraction_limit"] is True
    assert R.duality_of(0.5)["magnification"] == pytest.approx(2.0, rel=1e-12)


def test_axis_sigma_reads_match_the_axis_spectrum(W):
    """``sigma_T``/``sigma_F`` are the leading correlation singular values of their own axis."""
    assert R.sigma_T(W) == pytest.approx(R.axis_read(W, 0).sigma, rel=1e-12)
    assert R.sigma_F(W) == pytest.approx(R.axis_read(W, 1).sigma, rel=1e-12)
    ap = Aperture(W, window=None)
    assert ap.sigma_T == pytest.approx(R.sigma_T(W), rel=1e-12)
    assert ap.sigma_F == pytest.approx(R.sigma_F(W), rel=1e-12)


def test_space_bandwidth_is_the_matched_cell_count(W):
    """The space-bandwidth product is the two axes' matched cell counts multiplied."""
    g = EN.geometry(W)
    assert R.space_bandwidth(W) == int(g["n_F"]) * int(g["n_T"])
    assert Aperture(W, window=None).space_bandwidth == R.space_bandwidth(W)


def test_shape_factor_reads_agree_with_the_front_door(W):
    ap = Aperture(W, window=None)
    assert ap.shape_factor == pytest.approx(ap.a_delta / ap.phi_F, rel=1e-12)
    assert ap.rayleigh_shape_factor == pytest.approx(R.rayleigh_shape_factor(R.decay(W)), rel=1e-12)
    assert ap.fresnel_number(64) == pytest.approx(R.fresnel_number(W, 64), rel=1e-12)


def test_mercer_certificate_agrees_with_the_front_door(W):
    """The certificate is a property of the frame, so both routes to it return the same one."""
    a, b = R.mercer_certificate(W), Aperture(W, window=None).mercer
    assert a.ratio == pytest.approx(b.ratio, rel=1e-12)
    assert a.n_dof == b.n_dof
    assert a.a_delta_temporal == pytest.approx(b.a_delta_temporal, rel=1e-12)
    assert a.a_delta_spectral == pytest.approx(b.a_delta_spectral, rel=1e-12)


# ── Aperture: the geometry and optics properties ──────────────────────────────

def test_geometry_properties_match_the_entropy_read(W):
    """``H_T``/``H_F``/``delta_T``/``delta_F`` are ``entropy.geometry``, surfaced."""
    ap = Aperture(W, window=None)
    g = EN.geometry(ap.W)
    assert ap.H_T == pytest.approx(g["H_T"], rel=1e-12)
    assert ap.H_F == pytest.approx(g["H_F"], rel=1e-12)
    assert ap.delta_T == pytest.approx(g["delta_T"], rel=1e-12)
    assert ap.delta_F == pytest.approx(g["delta_F"], rel=1e-12)


def test_the_spectral_cut_point_properties_are_the_spectral_read(W):
    """These four are the SPECTRAL (correlation-edge) cut point, surfaced -- not the
    projection's singular-value floor, which is a different number by construction."""
    ap = Aperture(W, window=None)
    so = ap.spectral
    assert ap.contrast == pytest.approx(so.contrast, rel=1e-12)
    assert ap.top_share == pytest.approx(so.top_share, rel=1e-12)
    assert ap.noise_floor == pytest.approx(so.noise_floor, rel=1e-12)
    assert ap.resolved_modes == so.resolved_modes
    assert 0.0 <= ap.top_share <= 1.0
    assert ap.noise_floor != pytest.approx(float(ap.projection().noise_floor), rel=1e-6)


def test_spectral_optics_properties_agree_with_the_read(W):
    ap = Aperture(W, window=None)
    so = ap.spectral
    assert ap.attenuation == pytest.approx(so.attenuation, rel=1e-12)
    assert ap.dispersion == pytest.approx(so.dispersion, rel=1e-12)
    assert ap.phase == pytest.approx(so.phase, rel=1e-12)


def test_decay_properties_agree_with_the_diffraction_limit(W):
    ap = Aperture(W, window=None)
    dl = R.diffraction_limit(R.decay(W))
    assert ap.a_delta == pytest.approx(dl.a_delta, rel=1e-12)
    assert ap.correlation_length == pytest.approx(dl.xi, rel=1e-12)


def test_state_round_trips_the_streaming_operator(W):
    """``state``/``from_state`` is the resume contract: the restored aperture reads the same."""
    ap = Aperture(window=64)
    for row in W:
        ap.update(row)
    back = Aperture.from_state(ap.state())
    assert back.rates().dominant == pytest.approx(ap.rates().dominant, rel=1e-12)
    assert back.rates().n_frames == ap.rates().n_frames
    assert repr(ap).startswith("Aperture(")


# ── Projection: the beam and the crossings ────────────────────────────────────

def test_projection_beam_is_the_documented_replacement(W):
    """``beam`` replaced ``embeddings``/``vocabulary``/``K`` in 0.2.1: ``profile`` and ``basis``
    are that same pair, and the beam's modes are the projection's footprints."""
    sc = Projection(W)
    beam = sc.beam
    assert not hasattr(sc, "embeddings") and not hasattr(sc, "vocabulary")
    assert len(beam.modes) == len(sc.footprints) == sc.K_signal
    profile, basis = np.asarray(beam.profile), np.asarray(beam.basis)
    assert profile.shape == (sc.N, sc.K_signal)          # (N, K) rows on the resolved modes
    assert basis.shape == (sc.F_eff, sc.K_signal)        # (F_eff, K) the modes themselves
    # the pair reconstructs the resolved part of the screen it was read from
    approx = profile @ basis.T
    assert approx.shape == (sc.N, sc.F_eff)
    assert np.linalg.norm(approx) > 0.0


def test_projection_and_aperture_cross_to_each_other(W):
    """``ap.projection()`` and ``Projection(W).aperture()`` are the two views of one signal."""
    sc = Projection(W)
    ap = sc.aperture()
    assert Aperture(W, window=None).projection().K_signal == sc.K_signal
    assert ap.W.shape == sc.screen.shape
    assert sc.N == np.asarray(sc.screen).shape[0]
    assert repr(sc).startswith("Projection(")


def test_projection_read_is_the_same_numbers_as_the_object(W):
    """``read(W)`` is the dataclass form of ``Projection(W)`` -- one measurement, two shapes."""
    from entroptics.projection import read as projection_read
    r, sc = projection_read(W), Projection(W)
    assert r.K_signal == sc.K_signal
    assert float(r.noise_floor) == pytest.approx(float(sc.noise_floor), rel=1e-12)


def test_projection_tensor_is_the_delay_embedded_view(W):
    t = Projection(W).tensor()
    assert isinstance(t, dict)
    assert np.asarray(t["core"]).ndim == 3            # (time, lag, freq) Tucker core
    assert np.asarray(t["U_time"]).shape[0] == int(t["T_prime"])
    assert int(t["d"]) >= 1


# ── entropy / extract / providers ─────────────────────────────────────────────

def test_surprisal_bits_is_the_per_event_half_of_the_entropy():
    """I(x) = -log2(observed / total): a certain outcome is 0 bits, a 1-in-8 outcome is 3."""
    assert EN.surprisal_bits(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert EN.surprisal_bits(1.0, 8.0) == pytest.approx(3.0, rel=1e-12)
    assert EN.surprisal_bits(1.0, 2.0) == pytest.approx(1.0, rel=1e-12)


def test_upsample_inverts_downsample_on_a_matched_grid():
    """``upsample`` puts a folded axis back on the array's coordinates, conserving the total."""
    x = np.arange(24, dtype=float)[None, :]
    down = EN.downsample(x, 4, 1)
    up = EN.upsample(down, 24, 1)
    assert down.shape == (1, 4) and up.shape == (1, 24)
    assert up.sum() == pytest.approx(x.sum(), rel=1e-9)


def test_the_filter_has_one_path(W):
    """The filter is ``Aperture.extract`` and nothing else.

    The filter is ``extract.filter_projection``, which takes the projection the caller already
    holds; ``Aperture.extract`` builds that projection and delegates.  The namespace exports
    neither, so ``entroptics.extract`` is the module."""
    assert not hasattr(E, "extract") or isinstance(getattr(E, "extract"), types.ModuleType)
    from entroptics.extract import filter_projection
    ap = Aperture(W, window=None)
    a_clean, a_info = ap.extract()
    b_clean, b_info = filter_projection(ap.projection())
    assert np.array_equal(a_clean, b_clean)          # the door delegates, it does not reimplement
    assert a_info["K_signal"] == b_info["K_signal"]


def _ctx(data):
    from entroptics.null_providers import FloorContext
    s = np.linalg.svd(np.asarray(data), compute_uv=False)
    return FloorContext(spectrum=s, data=np.asarray(data), shape=np.asarray(data).shape,
                        far=0.05, kind="projection", rng=np.random.default_rng(0))


def test_self_calibrating_null_reads_in_the_projection_units():
    """A region-local reference built from signal-free blocks, scored in the units the floor
    thresholds.  Its ``kind`` default named the pre-0.2.1 cut point until nothing called it."""
    from entroptics.null_providers import self_calibrating_null, reference_null, top_spectrum_value
    rng = np.random.default_rng(4)
    noise = rng.standard_normal((600, 16))
    prov = self_calibrating_null(noise, block_rows=100)
    tops = [top_spectrum_value(noise[i:i + 100], "projection") for i in range(0, 501, 100)]
    ctx = _ctx(noise[:100])
    assert prov(ctx) == pytest.approx(reference_null(tops)(ctx), rel=1e-12)


def test_self_calibrating_null_needs_two_blocks():
    from entroptics.null_providers import self_calibrating_null
    with pytest.raises(ValueError, match="2 signal-free blocks"):
        self_calibrating_null(np.zeros((80, 8)), block_rows=100)


def test_reference_null_is_callable_on_a_context():
    from entroptics.null_providers import reference_null
    prov = reference_null([1.0, 2.0, 3.0, 4.0, 5.0])
    assert prov(_ctx(np.zeros((10, 4)))) > 0.0


# ── Screen.update: the incremental placement ──────────────────────────────────

def test_screen_update_extends_a_side_rather_than_replacing_it():
    """``update`` appends ordered steps to one side; ``place`` replaces it.  The extended side
    must carry the rows of both, and the other side must keep its placement."""
    rng = np.random.default_rng(2)
    D = 6
    Pa, Pb = rng.standard_normal((9, D)), rng.standard_normal((7, D))
    s = Screen()
    s.register("a", entry=lambda X: X @ Pa)
    s.register("b", entry=lambda X: X @ Pb)
    first, more = rng.standard_normal((40, 9)), rng.standard_normal((15, 9))
    s.place("a", first)
    s.place("b", rng.standard_normal((40, 7)))
    before = float(np.asarray(s.beam("b").energy))
    s.update("a", more)
    assert np.asarray(s.balanced("a")).shape[0] == 55          # 40 appended with 15
    assert float(np.asarray(s.beam("b").energy)) == pytest.approx(before, rel=1e-12)


# ── namespace conventions ─────────────────────────────────────────────────────

def test_no_module_name_resolves_to_anything_but_its_module():
    """``entroptics.<name>`` is always the module.  No exceptions, nothing to remember.

    Three names once resolved to a function (``dynamics``, ``extract``, ``sweep``).  Each was a
    second path to a front-door reading -- ``Aperture.dynamics()``, ``Aperture.extract()``,
    ``Aperture.sweep()`` -- so none of them is exported and the ambiguity is gone."""
    import glob
    import importlib
    import os

    for path in sorted(glob.glob("src/entroptics/*.py")):
        name = os.path.basename(path)[:-3]
        if name == "__init__":
            continue
        attr = getattr(E, name, None)
        assert attr is None or isinstance(attr, types.ModuleType), f"entroptics.{name} is not a module"
        assert isinstance(importlib.import_module(f"entroptics.{name}"), types.ModuleType)


def test_the_new_read_modules_declare_the_same_kind_of_surface():
    """``proximity`` and ``sequence`` are siblings -- module-scoped read sets, not front-door
    exports -- so they declare their surface the same way."""
    from entroptics import proximity, sequence
    for mod in (proximity, sequence):
        assert getattr(mod, "__all__", None), f"{mod.__name__} declares no __all__"
        for n in mod.__all__:
            assert hasattr(mod, n), f"{mod.__name__}.__all__ names missing {n}"
        # module-scoped on purpose: neither is re-exported at the top level
        assert not (set(mod.__all__) & set(E.__all__))


def _door_readings(seed=0, shape=(96, 16, 8)):
    """Every reading reachable from a front door, as (label, value) pairs."""
    import inspect
    rng = np.random.default_rng(seed)
    T, F, D = shape
    t = np.linspace(0, 9, T)
    frame = np.outer(np.sin(t), rng.standard_normal(F)) * 5 + rng.standard_normal((T, F))
    car = rng.standard_normal((T, 1))
    A = car @ rng.standard_normal((1, D)) + 0.4 * rng.standard_normal((T, D))
    B = car @ rng.standard_normal((1, D)) + 0.4 * rng.standard_normal((T, D))

    scr = Screen()
    scr.register("a", entry=lambda x: np.asarray(x))
    scr.register("b", entry=lambda x: np.asarray(x))
    scr.place("a", A)
    scr.place("b", B)

    out = []
    for host, obj in (("Aperture", Aperture(frame, window=None)),
                      ("Projection", Projection(frame)), ("Screen", scr)):
        # probing must not MUTATE the door: dir() reaches `clear` before `coupling`, and a
        # cleared screen makes every later read raise, so the probe would compare nothing.
        MUTATORS = {"clear", "place", "update", "register", "render", "from_state"}
        for n in dir(obj):
            if n.startswith("_") or n in MUTATORS:
                continue
            try:
                v = getattr(obj, n)
                if callable(v) and not inspect.isclass(v):
                    ps = list(inspect.signature(v).parameters)
                    if not ps:
                        v = v()
                    elif ps[:2] in (["lens_a", "lens_b"], ["lens_from", "lens_to"]):
                        v = v("a", "b")
                    elif ps[:1] == ["lens"] and len(ps) == 1:
                        v = v("a")
                    else:
                        continue
            except Exception:
                continue
            out.append((f"{host}.{n}", v))
    return out, frame, A, B


def _same(a, b):
    import dataclasses as dc
    if dc.is_dataclass(a) and dc.is_dataclass(b):
        return type(a) is type(b) and all(_same(getattr(a, f.name), getattr(b, f.name))
                                          for f in dc.fields(a))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    try:
        A_, B_ = np.asarray(a), np.asarray(b)
        return A_.shape == B_.shape and bool(np.array_equal(A_, B_))
    except Exception:
        return bool(a == b)


def test_no_top_level_name_duplicates_a_front_door_reading():
    """ONE PATH: no exported FREE FUNCTION may return what a front door already returns.

    Checked by VALUE across all three doors and both input arities -- a name check would have
    passed ``mercer_certificate``/``Aperture.mercer``, ``scale_duality``/``Aperture.duality``,
    ``spectral_optics``/``Aperture.spectral`` and ``coupling``/``Screen.coupling``, which wore
    different names on the two paths.

    Scope, deliberately narrow: this compares the top-level namespace against the doors.  It says
    nothing about a door's own members equalling a component's -- ``ap.rates()`` IS
    ``ap.dynamics().rates()`` and ``ap.W`` IS ``ap.projection().W``, because an Aperture OWNS a
    projection and an operator.  That is composition, not a second path, and a rule that called it
    one would have to delete ``Aperture.W``."""
    import inspect

    door, frame, A, B = _door_readings(0)
    door2, frame2, A2, B2 = _door_readings(7, shape=(112, 20, 6))

    # The probe must not be vacuous: walking dir() reaches mutators as well as reads, so these
    # pin that it saw the doors it claims to compare against.
    names = {n for n, _ in door}
    for expect in ("Aperture.etendue", "Aperture.phi", "Projection.K_signal",
                   "Projection.footprints", "Screen.coupling", "Screen.basis"):
        assert expect in names, f"the probe never reached {expect}"
    assert len(door) > 40, f"the probe reached only {len(door)} readings"

    trials = {"frame": (frame,), "pair": (A, B)}
    trials2 = {"frame": (frame2,), "pair": (A2, B2)}
    second = dict(door2)
    dups = []
    for n in sorted(E.__all__):
        f = getattr(E, n, None)
        if inspect.isclass(f) or not callable(f):
            continue
        for label, args in trials.items():
            try:
                r = f(*args)
            except Exception:
                continue                  # wrong arity -> a primitive, not a path
            for door_name, v in door:
                if not _same(r, v):
                    continue
                # A scalar can equal another read by coincidence, so a duplicate has to hold on a
                # second frame of a DIFFERENT SHAPE: the same path agrees on both, and a collision
                # between two shape-derived integers does not survive the change of shape.
                try:
                    r2 = f(*trials2[label])
                except Exception:
                    continue
                if door_name in second and _same(r2, second[door_name]):
                    dups.append(f"entroptics.{n}({label}) == {door_name}")
            break
    assert not dups, "duplicate public paths: " + "; ".join(dups)


# ── the operator's own reads ──────────────────────────────────────────────────

def test_dynamics_feature_reads_are_the_operators_own(W):
    """``phi_F``/``feature_entropy`` on the OPERATOR are the feature spectrum of the accumulated
    dynamics, not the frame's -- the same names on ``Aperture`` read the frame, which is why the
    two live on different objects."""
    dy = Aperture(W, window=None).dynamics()
    F = W.shape[1]
    h, fill = dy.feature_entropy(), dy.phi_F()
    # two distinct reads off the operator: H of the feature POWER MARGINAL, and the fill of its
    # EIGENVALUE spectrum -- both bounded, and neither is the frame-side read of the same name
    assert 0.0 <= h <= np.log2(F) + 1e-9
    assert 0.0 < fill <= 1.0 + 1e-12
    assert fill != pytest.approx(Aperture(W, window=None).phi_F, rel=1e-6)


def test_dynamics_propagator_is_the_reduced_operator(W):
    """``propagator`` is the reduced r x r operator in the POD basis; ``propagator_full`` is the
    same operator lifted back to the F x F feature space."""
    dy = Aperture(W, window=None).dynamics()
    A_red = np.asarray(dy.propagator())
    assert A_red.ndim == 2 and A_red.shape[0] == A_red.shape[1]
    A_full = np.asarray(dy.propagator_full())
    assert A_full.shape == (W.shape[1], W.shape[1])
    assert A_red.shape[0] <= A_full.shape[0]
    # the reduced operator's spectrum is the one `rates` reports
    mu = np.sort_complex(np.linalg.eigvals(A_red))
    assert np.allclose(np.sort_complex(np.asarray(dy.rates().mu)), mu, atol=1e-9)


def test_dynamics_tensors_expose_the_accumulators(W):
    """``tensors`` is the operator opened up: the accumulators it streams into and the reduced
    operator read off them -- what ``state`` persists, in array form."""
    dy = Aperture(W, window=None).dynamics()
    t = dy.tensors()
    assert isinstance(t, dict) and t
    for k, v in t.items():
        if isinstance(v, np.ndarray):
            assert np.all(np.isfinite(v)), k


def test_attenuation_interval_certifies_the_constant(W):
    """A certified interval for the attenuation constant at a given concentration band: the point
    read must sit inside its own interval, and a wider band cannot narrow it."""
    ap = Aperture(W, window=None)
    band = R.concentration_band(W.shape[0], W.shape[1],
                                spec_norm=float(ap.spectral.eigenvalues[0]))
    tight = ap.attenuation_interval(band)
    assert tight.attenuation_lo <= tight.attenuation <= tight.attenuation_hi
    assert tight.attenuation == pytest.approx(ap.attenuation, rel=1e-12)
    assert tight.band == pytest.approx(band, rel=1e-12)
    wide = ap.attenuation_interval(band * 4.0)          # a looser band cannot certify tighter
    assert (wide.attenuation_hi - wide.attenuation_lo) >= (tight.attenuation_hi - tight.attenuation_lo) - 1e-12


def test_reference_null_scores_a_context_at_its_own_level():
    """``ReferenceNull`` is a provider: called on a context it returns the floor its calibration
    sample implies, and a higher sample lifts the floor."""
    from entroptics.null_providers import reference_null
    low = reference_null([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    high = reference_null([10.0, 11.0, 9.0, 10.5, 9.5, 10.0])
    ctx = _ctx(np.zeros((12, 4)))
    assert 0.0 < low(ctx) < high(ctx)


def test_spectrum_probe_length_is_its_channel_count():
    """``len(probe)`` is how many channels the digest was taken over."""
    from entroptics.proximity import mp_spectrum, SpectrumProbe
    rng = np.random.default_rng(0)
    spectra = [mp_spectrum(rng.standard_normal((128, 20))) for _ in range(4)]
    probe = SpectrumProbe(spectra)
    assert isinstance(probe, SpectrumProbe)
    assert len(probe) == 4                       # the probe holds the spectra it was built from


def test_screen_repr_names_its_sides():
    """A screen's repr says which lenses are on it -- the placement is the state worth showing."""
    rng = np.random.default_rng(0)
    P = rng.standard_normal((9, 6))
    s = Screen()
    s.register("left", entry=lambda x: np.asarray(x) @ P)
    assert repr(s).startswith("Screen(")
    s.place("left", rng.standard_normal((40, 9)))
    assert "left" in repr(s)


def test_reference_null_class_is_a_streaming_provider_with_fading_memory():
    """``ReferenceNull`` is the STATEFUL sibling of the ``reference_null`` closure: reference
    values arrive by ``push`` and ``forgetting < 1`` fades the old ones, so the floor tracks a
    drifting noise level as an aperture sweeps.  It has no ``update`` hook on purpose -- the
    streaming aperture must not calibrate it on the data it is thresholding."""
    from entroptics.null_providers import ReferenceNull
    ctx = _ctx(np.zeros((12, 4)))

    quiet, loud = ReferenceNull([1.0] * 8), ReferenceNull([10.0] * 8)
    assert 0.0 < quiet(ctx) < loud(ctx)                    # the provider interface

    assert not hasattr(ReferenceNull([1.0] * 4), "update")  # cannot self-calibrate on the signal

    # the noise level moves from 1 to 10 under both, and only the fading null follows it:
    # perfect memory still carries the old values, so its floor is inflated by the SPREAD of a
    # sample that is really two populations, while the fading one settles on the level now.
    remembering = ReferenceNull([1.0] * 40, forgetting=1.0)
    fading = ReferenceNull([1.0] * 40, forgetting=0.5)
    for _ in range(20):
        remembering.push(10.0)
        fading.push(10.0)
    assert abs(fading(ctx) - 10.0) < abs(remembering(ctx) - 10.0)
    assert fading(ctx) == pytest.approx(10.0, abs=0.5)      # tracks the drift
    assert remembering(ctx) > fading(ctx)                   # and is not just a slower version

    with pytest.raises(ValueError, match="forgetting"):
        ReferenceNull([1.0, 2.0], forgetting=0.0)


# ── the lift's preconditions ──────────────────────────────────────────────────

def test_delay_embed_refuses_what_it_cannot_embed():
    """Takens coordinates need a 2-D trajectory and a depth the trajectory can carry.  Each
    refusal is explicit: a silently reshaped or truncated trajectory is a different signal."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="2-D"):
        E.delay_embed(rng.standard_normal(50), 4)
    with pytest.raises(ValueError, match="delay depth"):
        E.delay_embed(rng.standard_normal((50, 3)), 0)
    with pytest.raises(ValueError, match="trajectory length"):
        E.delay_embed(rng.standard_normal((4, 3)), 9)
    A = rng.standard_normal((10, 2))
    assert E.delay_embed(A, 1) is A                      # d=1 is the trajectory itself


# ── chunking is output-transparent ────────────────────────────────────────────

def test_chunking_cannot_change_the_answer():
    """``ResourceLimits(memory_gb=...)`` splits the read into chunks to bound the working set.
    That is a resource decision, so it must be invisible in the result -- byte-identical counts and
    floors however many chunks it takes, or the caller's memory budget would change their science."""
    rng = np.random.default_rng(3)
    X = np.stack([rng.standard_normal((64, 16)) * s for s in (0.0, 2.0, 5.0, 12.0, 1.0, 8.0)])
    whole = E.resolved_batch(X, energy=True)
    tiny = E.resolved_batch(X, energy=True,
                            limits=E.ResourceLimits(memory_gb=1e-7), _allow_chunk=True)
    assert np.array_equal(np.asarray(whole.K_signal), np.asarray(tiny.K_signal))
    assert np.array_equal(np.asarray(whole.noise_floor), np.asarray(tiny.noise_floor))
    assert np.array_equal(np.asarray(whole.sigma_top), np.asarray(tiny.sigma_top))
    assert np.array_equal(np.asarray(whole.energy), np.asarray(tiny.energy))   # byte-identical
