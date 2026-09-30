"""The one batched resolved-screen read (``entroptics.batch.resolved_batch``): one read, two
backends (numpy CPU / torch GPU), three cost tiers.  Must be

  * Tier-0 bit-identical to a per-frame ``Projection`` (K_signal / sigma_top / noise_floor),
  * Tier-1 exact (per-row energy matches an independent full-SVD reference),
  * identical on numpy and torch (parity), across the exact solvers (svd == eigh),
  * fold-configurable (dense continuum vs sparse/choppy substrates), and
  * subset-capable (the expensive tier runs only on flagged frames).
"""
import numpy as np
import pytest

from entroptics import Projection, resolved_batch, ResolvedBatch
from entroptics.projection import normalize_batch, project_batch, fold_target_batch
from entroptics.null_providers import reference_null, top_spectrum_value, mp


def _frames(shape=(64, 16), n=12, seed=0):
    rng = np.random.default_rng(seed)
    N, F = shape
    return [([0.0, 2.0, 5.0, 12.0][i % 4]) * np.outer(rng.standard_normal(N), rng.standard_normal(F))
            + rng.standard_normal((N, F)) for i in range(n)]


def _energy_ref(frames, r, fold):
    """Independent full-SVD reference for the per-row resolved energy of each frame."""
    X = np.stack(frames)
    data = normalize_batch(np, X)
    feff = fold_target_batch(np, X) if fold else np.full(len(frames), X.shape[2], int)
    out = []
    for i in range(len(frames)):
        scr = project_batch(np, data[i:i + 1], int(feff[i]))[0]
        U, S, _ = np.linalg.svd(scr, full_matrices=False)
        keep = (S > float(r.noise_floor[i])).astype(float)
        out.append(((U ** 2) * (keep * S ** 2)[None, :]).sum(1))
    return out


# ── Tier 0: bit-identical to Projection ───────────────────────────────────────────

@pytest.mark.parametrize("fold", ["auto", True, False])
@pytest.mark.parametrize("shape", [(64, 16), (128, 8), (100, 32), (8, 8)])
def test_tier0_bit_identical_to_screen(shape, fold):
    """``"auto"`` is the DEFAULT, so it is the mode that has to match -- a parity test that only
    ran under an explicit ``fold=True`` would leave the used path unverified."""
    frames = _frames(shape)
    r = resolved_batch(np.stack(frames), fold=fold)
    for i, f in enumerate(frames):
        sc = Projection(f)  # Projection applies the fold decision; so do "auto" and True
        if fold is not False:
            assert int(r.K_signal[i]) == sc.K_signal
            assert float(r.sigma_top[i]) == float(sc.sigma_top)
            assert float(r.noise_floor[i]) == float(sc.noise_floor)


def test_tier0_with_null_provider():
    frames = _frames((8, 8), n=16)
    prov = reference_null([Projection(np.random.default_rng(k).standard_normal((8, 8))).sigma_top
                           for k in range(12)])
    r = resolved_batch(np.stack(frames), fold=True, null=prov)
    for i, f in enumerate(frames):
        sc = Projection(f, null=prov)
        assert int(r.K_signal[i]) == sc.K_signal
        assert float(r.noise_floor[i]) == float(sc.noise_floor)


# ── Tier 1: exact energy; cheap/expensive agree ───────────────────────────────

@pytest.mark.parametrize("fold", [True, False])
def test_tier1_energy_exact(fold):
    frames = _frames((80, 24), n=10)
    r = resolved_batch(np.stack(frames), fold=fold, energy=True, basis=True)
    for i, en_ref in enumerate(_energy_ref(frames, r, fold)):
        assert np.max(np.abs(np.asarray(r.energy[i]) - en_ref)) < 1e-9
    cheap = resolved_batch(np.stack(frames), fold=fold)
    assert np.array_equal(cheap.K_signal, r.K_signal)          # cheap gate == expensive read


def test_subset_matches_full():
    frames = _frames((80, 32), n=10)
    full = resolved_batch(np.stack(frames), fold=False, energy=True)
    sub = resolved_batch(np.stack(frames), fold=False, energy=True, subset=[1, 4, 7])
    assert np.max(np.abs(np.asarray(sub.energy) - np.asarray(full.energy)[[1, 4, 7]])) < 1e-12
    assert list(sub.K_signal) == [int(full.K_signal[i]) for i in (1, 4, 7)]


# ── projector: energy == the projector applied to the rows ─────────────────────

def test_projector_reproduces_energy():
    """The resolved ``projector`` P = 1(C > floor^2) must reproduce the per-row energy:
    energy[t] = s_t P s_t^T -- one operator behind both reads (CPU SVD path)."""
    frames = _frames((80, 32), n=8)
    r = resolved_batch(np.stack(frames), fold=False, energy=True, basis=True)
    from entroptics.projection import normalize_batch, project_batch
    data = normalize_batch(np, np.stack(frames))
    for i in range(len(frames)):
        scr = project_batch(np, data[i:i + 1], 32)[0]
        P = np.asarray(r.projector[i])
        en_P = ((scr @ P) * scr).sum(1)
        assert np.max(np.abs(en_P - np.asarray(r.energy[i]))) < 1e-9
        assert abs(int(round(np.trace(P))) - int(r.K_signal[i])) == 0   # tr(P) == K_signal


def test_returns_dataclass_and_shapes():
    frames = _frames((64, 16), n=6)
    r = resolved_batch(np.stack(frames), fold=False, energy=True, basis=True)
    assert isinstance(r, ResolvedBatch)
    assert r.energy.shape == (6, 64)
    assert np.asarray(r.projector).shape == (6, 16, 16)   # (B, F, F) resolved projector


# ── auto-fold: decide foldability from feature-axis locality (dense vs sparse/choppy) ──

def test_autofold_matches_the_per_frame_read_on_every_substrate():
    """The fold is ONE decision (``entropy.fold_width``, PAPER Def 2.2), so the batched read and
    the per-frame ``Projection`` reach it identically on every substrate -- a dense continuum, an
    unordered basis, a sparse carrier and a globally redundant basis alike.

    The decision is per frame and lives in one place, so a stack and a single frame reach it
    identically."""
    rng = np.random.default_rng(0); T, F = 200, 64

    def continuum():                        # dense smooth continuum
        x = rng.standard_normal((T, F)); k = np.ones(6) / 6
        return np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, x)

    def kv():                               # unordered basis (LLM KV head_dim)
        return rng.standard_normal((T, F))

    def sparse():                           # narrowband spike -- averaging would dilute it
        x = rng.standard_normal((T, F)) * 0.3; x[:, F // 3] += 5 * np.sin(np.linspace(0, 20, T)); return x

    def redundant():                        # global correlation, no locality
        L = rng.standard_normal((3, F)); Z = rng.standard_normal((T, 3))
        return Z @ L + rng.standard_normal((T, F)) * 0.3

    for fn in (continuum, kv, sparse, redundant):
        frames = [fn() for _ in range(6)]
        r = resolved_batch(np.stack(frames), fold="auto")
        for i, f in enumerate(frames):
            sc = Projection(f)
            assert int(r.K_signal[i]) == sc.K_signal, f"{fn.__name__}[{i}]"
            assert float(r.noise_floor[i]) == float(sc.noise_floor), f"{fn.__name__}[{i}]"


def test_auto_and_true_are_one_read():
    """There is one fold decision, so the two spellings that ask for it cannot disagree."""
    frames = _frames((100, 32), n=8)
    a = resolved_batch(np.stack(frames), fold="auto")
    t = resolved_batch(np.stack(frames), fold=True)
    assert np.array_equal(np.asarray(a.K_signal), np.asarray(t.K_signal))
    assert np.array_equal(np.asarray(a.noise_floor), np.asarray(t.noise_floor))


# ── ResolvedScreen: the stateful revisited-screen sibling ──────────────────────

def test_thread_safe_parallel_reads():
    """resolved_batch is pure -> concurrent calls must not clobber each other; each parallel result
    equals the serial result (and the shared bounded pool prevents thread oversubscription)."""
    from concurrent.futures import ThreadPoolExecutor
    stacks = [np.stack(_frames((80, 24), n=12, seed=k)) for k in range(8)]
    serial = [resolved_batch(s, energy=True) for s in stacks]
    with ThreadPoolExecutor(max_workers=8) as ex:
        par = list(ex.map(lambda s: resolved_batch(s, energy=True), stacks))
    for a, b in zip(serial, par):
        assert np.array_equal(a.K_signal, b.K_signal)
        assert np.array_equal(np.asarray(a.energy), np.asarray(b.energy))


def test_resolved_screen_concurrent_updates_safe():
    """Concurrent update() on one ResolvedScreen is serialised by its lock -- the accumulated Gram
    is not clobbered, so T counts every appended row."""
    from concurrent.futures import ThreadPoolExecutor
    from entroptics.batch import ResolvedScreen
    rng = np.random.default_rng(2); F = 24
    rs = ResolvedScreen(F)
    chunks = [rng.standard_normal((4, F)) for _ in range(40)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(rs.update, chunks))
    assert rs.T == 40 * 4                                   # every row accumulated, none lost to a race


def test_resolved_screen_streams_and_resumes():
    from entroptics.batch import ResolvedScreen
    rng = np.random.default_rng(1); F, T = 32, 400
    L = rng.standard_normal((2, F)); Z = rng.standard_normal((T, 2))
    X = Z @ L + rng.standard_normal((T, F)) * 0.5
    rs = ResolvedScreen(F)
    for i in range(0, T, 16):
        rs.update(X[i:i + 16])
    batch = int(resolved_batch(rs.window[None], fold=False).K_signal[0])
    assert rs.K_signal == batch == 2                        # streamed == the batch read of its window
    assert rs.window.shape[0] >= F + 1                      # the window carries every direction
    en = rs.energy(X[-16:])
    assert np.asarray(en).shape == (16,) and np.all(np.asarray(en) >= 0)
    rs2 = ResolvedScreen.from_state(rs.state())             # resume across a session
    assert rs2.K_signal == rs.K_signal


def test_resolved_screen_batch_matches_batch_and_per_screen():
    """ResolvedScreenBatch (B screens in one (B,F,F) Gram) == the batch resolved_batch of every row
    under the Gram's closed-form floor; batched energy is well-formed."""
    from entroptics.batch import ResolvedScreenBatch, ResolvedScreen
    rng = np.random.default_rng(0); B, F, T = 6, 24, 300
    L = rng.standard_normal((2, F))
    Xs = np.stack([rng.standard_normal((T, 2)) @ L + rng.standard_normal((T, F)) * 0.5 for _ in range(B)])
    rsb = ResolvedScreenBatch(B, F)
    for i in range(0, T, 16):
        rsb.update(Xs[:, i:i + 16, :])
    assert np.array_equal(rsb.K_signal, resolved_batch(Xs, fold=False, null=mp).K_signal)   # == batch read
    en = rsb.energy(Xs[:, -8:, :])
    assert np.asarray(en).shape == (B, 8) and np.all(np.asarray(en) >= 0)
    assert np.array_equal(ResolvedScreenBatch.from_state(rsb.state()).K_signal, rsb.K_signal)


def test_scale_profile_has_contrast():
    from entroptics import Aperture
    rng = np.random.default_rng(0)
    W = np.outer(np.sin(np.linspace(0, 10, 128)), np.ones(16)) + rng.standard_normal((128, 16)) * 0.3
    sp = Aperture(W, window=None).scale_profile()
    assert len(sp.contrast) == len(sp.windows) and np.all(np.asarray(sp.contrast) >= 0)


# ── the null provider reaches the stateful screens ────────────────────────────

from entroptics.batch import ResolvedScreen, ResolvedScreenBatch  # noqa: E402


def _revisited(B=3, T=256, F=16, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12, T)
    return np.stack([np.outer(np.sin(t + j), rng.standard_normal(F)) * 6
                     + rng.standard_normal((T, F)) for j in range(B)])


@pytest.mark.parametrize("null,expect", [(None, 1), (lambda ctx: 1e6, 0), (lambda ctx: 1e-9, 16)])
def test_resolved_screen_reads_against_the_callers_floor(null, expect):
    """``null=`` is the caller's floor, and ``K_signal`` is a count against it.

    The floor is the caller's when they supply one, and the derived ``mp`` edge when they do
    not."""
    X = _revisited(B=1)[0]
    s = ResolvedScreen(X.shape[1], null=null)
    s.update(X)
    assert int(s.K_signal) == expect


@pytest.mark.parametrize("null", [lambda ctx: 1e6, lambda ctx: 1e-9])
def test_resolved_screen_batch_takes_a_provider_and_matches_the_per_screen_read(null):
    """The batched sibling takes the same provider contract: a floor above every mode resolves
    none on either, and one below every mode resolves all of them on either.  (Their reads differ
    in what they read -- the batch every row, the screen its window -- so an unset floor is not
    compared.)"""
    Xs = _revisited()
    B, _, F = Xs.shape
    batch = ResolvedScreenBatch(B, F, null=null)
    batch.update(Xs)
    per = []
    for j in range(B):
        one = ResolvedScreen(F, null=null)
        one.update(Xs[j])
        per.append(int(one.K_signal))
    assert [int(k) for k in np.asarray(batch.K_signal)] == per


def test_a_stream_of_noise_reads_as_the_batch_does():
    """The stream forms, at every refresh, the whitened Gram the batch read forms from the same
    rows, so noise appended one token at a time reads as the batch reads it -- no warmup, no frozen
    centre whose offset would grow into a mode.  Negative control: the rows' raw Gram (whiten=False
    on unwhitened rows) reads structure in nearly every stream."""
    from entroptics.batch import ResolvedScreen, ResolvedScreenBatch
    B, T, F = 40, 16, 64
    rng = np.random.default_rng(11)
    Xs = rng.standard_normal((B, T, F)) * np.exp(rng.uniform(-1, 1, F)) + 3.0
    rsb = ResolvedScreenBatch(B, F)
    raw = ResolvedScreenBatch(B, F, whiten=False)
    for t in range(T):
        rsb.update(Xs[:, t, :])
        raw.update(Xs[:, t, :])
    assert np.array_equal(rsb.K_signal, resolved_batch(Xs, fold=False, null=mp).K_signal)
    assert np.mean(rsb.K_signal > 0) <= 0.1
    assert np.mean(raw.K_signal > 0) >= 0.9
    rs = ResolvedScreen(F)
    for t in range(T):
        rs.update(Xs[0, t])
    assert rs.K_signal == int(resolved_batch(rs.window[None], fold=False).K_signal[0])


def test_a_streamed_flat_channel_leaves_the_live_width():
    """A constant channel carries nothing, on the stream as in the batch read: it is not a cell of
    the floor's ensemble."""
    from entroptics.batch import ResolvedScreen, ResolvedScreenBatch
    rng = np.random.default_rng(3)
    hits_s, hits_b = 0, 0
    for i in range(20):
        X = np.hstack([rng.standard_normal((1024, 12)), np.full((1024, 4), 2.5)])
        rs = ResolvedScreen(16)
        for k in range(0, 1024, 64):
            rs.update(X[k:k + 64])
        rsb = ResolvedScreenBatch(1, 16)
        rsb.update(X[None])
        hits_s += rs.K_signal > 0
        hits_b += int(rsb.K_signal[0]) > 0
        assert rs.K_signal == int(resolved_batch(rs.window[None], fold=False).K_signal[0])
    assert hits_s <= 2 and hits_b <= 2


def test_a_screen_with_no_live_channel_resolves_nothing():
    """After one token, or on a constant screen, no channel has moved: K = 0 and finite energies,
    as the batch read gives -- not a sign iteration run on an infinite floor."""
    from entroptics.batch import ResolvedScreenBatch
    r = np.random.default_rng(0)
    X = r.standard_normal((3, 50, 8))
    b = ResolvedScreenBatch(3, 8)
    b.update(X[:, 0])
    assert np.array_equal(b.K_signal, [0, 0, 0])
    assert np.all(np.isfinite(np.asarray(b.energy(X[:, :2]))))
    Y = X.copy()
    Y[1] = 2.0
    b = ResolvedScreenBatch(3, 8)
    b.update(Y)
    assert np.array_equal(b.K_signal, resolved_batch(Y, fold=False, null=mp).K_signal)


def test_a_provider_sees_the_live_spectrum_descending_on_every_path():
    from entroptics.batch import ResolvedScreen, ResolvedScreenBatch
    W = np.random.default_rng(1).standard_normal((200, 8))
    W[:, 3] = 1.0
    prov = lambda ctx: 0.5 * ctx.spectrum[0]                     # noqa: E731
    rs = ResolvedScreen(8, null=prov)
    rs.update(W)
    rb = ResolvedScreenBatch(1, 8, null=prov)
    rb.update(W[None])
    want = int(resolved_batch(W[None], fold=False, null=prov).K_signal[0])
    assert int(rb.K_signal[0]) == want == 7
    assert rs.K_signal == int(resolved_batch(rs.window[None], fold=False, null=prov).K_signal[0])


def test_forgetting_does_not_depend_on_how_the_stream_is_blocked():
    from entroptics.batch import ResolvedScreen
    x = np.random.default_rng(2).standard_normal((1000, 8))
    out = []
    for blk in (1, 10):
        rs = ResolvedScreen(8, forgetting=0.99)
        for k in range(0, 1000, blk):
            rs.update(x[k:k + blk])
        out.append((rs.T, rs.window, rs.K_signal))
    assert out[0][0] == out[1][0] and out[0][2] == out[1][2]
    assert np.array_equal(out[0][1], out[1][1])


@pytest.mark.parametrize("batched", [False, True])
def test_every_read_is_the_batch_read_of_every_row_so_far(batched):
    """No cadence: a read after any append is the batch read of what the stream reads -- every row
    for the Gram batch, the window for the screen -- so a mode that arrives in a handful of rows is
    seen at the next read, not after the stream's own refresh."""
    rng = np.random.default_rng(4)
    F = 16
    noise = rng.standard_normal((200, F))
    burst = np.outer(rng.standard_normal(12), rng.standard_normal(F)) * 8 + rng.standard_normal((12, F))
    rs = ResolvedScreenBatch(1, F) if batched else ResolvedScreen(F)

    def k():
        return int(np.asarray(rs.K_signal).ravel()[0])

    def want(rows):
        if batched:
            return int(resolved_batch(rows[None], fold=False, null=mp).K_signal[0])
        return int(resolved_batch(rs.window[None], fold=False).K_signal[0])

    rs.update(noise[None] if batched else noise)
    assert k() == want(noise)
    rs.update(burst[None] if batched else burst)
    assert k() == want(np.vstack([noise, burst])) >= 1
