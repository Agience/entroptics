"""readme_examples.py -- the README's code examples, run exactly as printed there.

Every output the README shows beside an example is what this script prints.  Seeded and
single-threaded, so it reproduces.

    python readme_examples.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

np.set_printoptions(precision=4, suppress=False)


def _stamp():
    import hashlib
    from pathlib import Path
    import entroptics
    h = hashlib.sha256()
    for p in sorted(Path(entroptics.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))   # git's form: a checkout reproduces it
    print(f"source: {entroptics.__file__} sha256 {h.hexdigest()}", flush=True)


_stamp()


def show(label, value):
    print(f"{label}: {value}", flush=True)


# -- how many real components, and the signal apart from the noise ---------------------------------
from entroptics import Aperture  # noqa: E402

rng = np.random.default_rng(1)
signal = rng.standard_normal((600, 3)) @ rng.standard_normal((3, 40))   # 3 sources over 40 sensors
W = signal + rng.standard_normal((600, 40))                             # plus noise
ap = Aperture(W)
show("components found", ap.projection().K_signal)
show("components in pure noise", Aperture(rng.standard_normal((600, 40))).projection().K_signal)
clean, info = ap.extract()
show("max |clean + residual - W|", float(np.max(np.abs(clean + info["residual"] - W))))
err = lambda a: float(np.linalg.norm(a - signal) / np.linalg.norm(signal))   # noqa: E731
show("error vs the true signal: raw, clean", (round(err(W), 3), round(err(clean), 3)))

# -- the operator: a record is its modes plus an exact residual ----------------------------------------
from entroptics.experimental import operator_read  # noqa: E402

rng = np.random.default_rng(0)
t = np.arange(2048)
x = (np.cos(2 * np.pi * 0.0402 * t)
     + 0.7 * np.cos(2 * np.pi * 0.0926 * t + 1)
     + 0.5 * np.cos(2 * np.pi * 0.15 * t + 2)
     + 0.05 * rng.standard_normal(t.size))

op = operator_read(x)
show("op.K, op.depth", (op.K, op.depth))
f = op.frequency[op.frequency > 0]
show("frequencies", np.sort(f))
show("decay rates", np.asarray(op.modes.alpha)[op.frequency > 0][np.argsort(f)])
S = op.spectrum(np.linspace(0.0, 0.5, 100_001))
show("view points", S.size)
resolved, residual = op.split(x)
show("max |resolved + residual - x|", float(np.max(np.abs(resolved + residual - x))))
show("residual.var() / 0.05**2", float(residual.var() / 0.05 ** 2))

# -- one 2-D record ---------------------------------------------------------------------------------
from entroptics import Aperture  # noqa: E402

W = np.random.default_rng(0).standard_normal((256, 64))
ap = Aperture(W)
show("len(ap.optics())", len(ap.optics()))
show("ap.projection().K_signal on noise", ap.projection().K_signal)

# -- the write path: a basis read once, shared, applied to later frames ----------------------------------
rng = np.random.default_rng(1)
mix = np.linalg.qr(rng.standard_normal((40, 3)))[0].T          # 3 hidden sources over 40 channels
gain = np.exp(rng.uniform(-0.5, 0.5, 40))                        # unequal channel noise


def frames(T):
    return (rng.standard_normal((T, 3)) * [6, 4, 3]) @ mix + gain * rng.standard_normal((T, 40))


b = Aperture(frames(600)).basis()        # read once
show("b.K, b.F", (b.K, b.F))
later = frames(600)                       # frames that arrive afterwards
A = b.encode(later)
show("A.shape", A.shape)
resolved, residual = b.split(later)
show("max |resolved + residual - later|", float(np.max(np.abs(resolved + residual - later))))
c = b.certify(later)
show("certify", c)
show("b.drift(later).K", b.drift(later).K)
new = np.linalg.qr(rng.standard_normal((40, 1)))[0][:, 0]
changed = frames(600) + np.outer(1.5 * rng.standard_normal(600), new)
show("b.drift(changed).K", b.drift(changed).K)

# -- a 1-D record lifted to a 2-D one ---------------------------------------------------------------
from entroptics import delay_embed, koopman_lift  # noqa: E402

rng = np.random.default_rng(2)
y = np.exp(-0.01 * t[:1024]) * np.cos(2 * np.pi * 0.0402 * t[:1024]) + 0.02 * rng.standard_normal(1024)
Z = delay_embed(y[:, None], 16)
show("Z.shape", Z.shape)
m = koopman_lift(y[:, None], 16).modes()
top = np.argmax(np.asarray(m.power) * (np.angle(np.asarray(m.mu)) > 0))
show("lift: frequency, decay", (float(np.angle(m.mu[top]) / (2 * np.pi)), float(m.alpha[top])))

# -- what a single mode fills: etendue x space_bandwidth on a rank-1 record ------------------------------
for T in (256, 512, 1024):
    rng = np.random.default_rng(3)
    W1 = np.outer(rng.standard_normal(T), rng.standard_normal(64))
    a1 = Aperture(W1)
    show(f"rank-1 etendue * space_bandwidth, T = {T}", float(a1.etendue * a1.space_bandwidth))
