"""baseline.py -- the release baseline: sensitivity and accuracy, measured, with their uncertainty.

    python baseline.py [path to a src/] > baseline_metrics.json

Every metric is emitted as ``{"value", "se", "better"}``: ``better`` is ``"higher"``, ``"lower"``, or
``"bound"`` for a level that must hold rather than improve (a false-alarm rate at or under ``far``).
``se`` is the metric's own sampling error across its seeded records, so a candidate is compared
against the baseline beyond the noise the measurement carries (``gate.py``), never against a
chosen tolerance.  Everything is seeded and single-threaded: the same library reproduces every
number exactly.  Cost -- instructions and memory per read -- is measured by ``cost.py``.

  sensitivity  the share of records in which a planted signal is found, on a ladder of strengths
               relative to the noise edge ``sqrt(T) + sqrt(F)`` (a factor of sqrt 2 per rung, from a
               third of the edge to twice it), for four signal shapes, four noise laws and three
               shapes of record, at the library's default floor and ``far = 0.05``;
  accuracy     the share of records whose planted rank is counted exactly (at and above the edge);
               the false-alarm rate on noise alone (a bound, at ``far``); the decay-rate and
               frequency error of the dynamical operator against a known linear system across
               signal-to-noise; and the extract filter's error against the truth of a known burst.

The ladder, the shapes and the record counts are the measurement protocol -- fixed here, never
tuned -- and not constants of the library: nothing the library computes reads them.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "validation"))

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import entroptics as E  # noqa: E402
from entroptics import Aperture, Projection  # noqa: E402

FAR = 0.05
N_REC = 100                                   # records per sensitivity / level cell
RUNGS = [2.0 ** (j / 2) for j in range(-3, 3)]   # 0.35 .. 2 of the edge
SHAPES = ((256, 32), (1024, 64), (32, 512))
SIGNALS = ("mode", "line", "burst", "three")
RANK = {"mode": 1, "line": 1, "burst": 1, "three": 3}


# ── noise laws, each at unit standard deviation so a strength means the same in every one ────
def _gauss(r, T, F):
    return r.standard_normal((T, F))


def _lognormal(r, T, F):
    s = 1.5
    x = np.exp(s * r.standard_normal((T, F)))
    return (x - math.exp(s * s / 2)) / math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))


def _poisson(r, T, F):
    return r.poisson(1.0, (T, F)) - 1.0


def _complex(r, T, F):
    return (r.standard_normal((T, F)) + 1j * r.standard_normal((T, F))) / math.sqrt(2.0)


NOISE = {"gaussian": _gauss, "lognormal_sigma1_5": _lognormal, "poisson1": _poisson, "complex": _complex}


def _unit(v):
    return v / np.linalg.norm(v)


def _signal(kind, r, T, F, strength):
    """A planted structure whose singular values are ``strength`` times the noise edge."""
    edge = math.sqrt(T) + math.sqrt(F)
    if kind == "mode":                          # one direction across every channel
        return strength * edge * np.outer(_unit(r.standard_normal(T)), _unit(r.standard_normal(F)))
    if kind == "line":                          # a narrowband tone on two neighbouring channels
        c = F // 3
        v = np.zeros(F); v[c:c + 2] = 1.0
        return strength * edge * np.outer(_unit(np.sin(2 * np.pi * 0.05 * np.arange(T))), _unit(v))
    if kind == "burst":                         # a short broadband transient
        w = max(4, T // 16)
        u = np.zeros(T); u[T // 2 - w // 2: T // 2 - w // 2 + w] = np.hanning(w)
        return strength * edge * np.outer(_unit(u), _unit(1.0 + 0.3 * r.standard_normal(F)))
    U = np.linalg.qr(r.standard_normal((T, 3)))[0]   # three orthogonal modes at the same strength
    V = np.linalg.qr(r.standard_normal((F, 3)))[0]
    return strength * edge * (U @ V.T)


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], "little")


def _sensitivity_cell(args):
    kind, noise, T, F, strength = args
    found = exact = 0
    for i in range(N_REC):
        r = np.random.default_rng(_seed("sens", kind, noise, T, F, strength, i))
        W = NOISE[noise](r, T, F) + _signal(kind, r, T, F, strength)
        K = Projection(W, far=FAR, seed=i).K_signal
        found += K > 0
        exact += K == RANK[kind]
    return args, found / N_REC, exact / N_REC


def _level_cell(args):
    noise, T, F = args
    hits = 0
    for i in range(N_REC * 2):
        r = np.random.default_rng(_seed("level", noise, T, F, i))
        hits += Projection(NOISE[noise](r, T, F), far=FAR, seed=i).K_signal > 0
    return args, hits / (N_REC * 2), N_REC * 2


def _decay_cell(snr_db):
    import common as C
    specs = [(0.99, 0.4), (0.97, 1.2), (0.95, 2.3)]
    A = C.oscillator_operator(specs)
    mu = np.linalg.eigvals(A)
    W = C.linear_trajectory(A, 160, seed=202)
    sigma = math.sqrt(float(np.mean(W ** 2)) / 10.0 ** (snr_db / 10.0))
    ea, eb = [], []
    for s in range(40):
        g = np.random.default_rng(_seed("decay", snr_db, s))
        m = np.asarray(Aperture(W + sigma * g.standard_normal(W.shape)).rates().mu)
        rem, got = list(range(m.size)), []
        for t in mu:                            # nearest recovered eigenvalue to each true one
            if not rem:
                got.append(np.nan); continue
            j = min(rem, key=lambda k: abs(m[k] - t)); rem.remove(j); got.append(m[j])
        got = np.asarray(got)
        ea.append(float(np.nanmean(np.abs(-np.log(np.abs(got)) + np.log(np.abs(mu))))))
        eb.append(float(np.nanmean(np.abs(np.angle(got) - np.angle(mu)))))
    return snr_db, ea, eb


def _burst(T=64, F=256):
    t = np.arange(T)[:, None]; f = np.arange(F)[None, :]
    b1 = np.exp(-0.5 * ((t - 0.46 * T) / 3.0) ** 2) * np.exp(-0.5 * ((f - 0.50 * F) / (0.22 * F)) ** 2)
    b2 = np.exp(-0.5 * ((t - 0.54 * T) / 4.0) ** 2) * np.exp(-0.5 * ((f - 0.66 * F) / (0.16 * F)) ** 2)
    B = b1 + 0.7 * b2
    return B / B.max()


def _extract_cell(snr):
    B = _burst()
    errs = []
    for s in range(20):
        W = B + np.random.default_rng(_seed("extract", snr, s)).standard_normal(B.shape) / snr
        clean, _ = Aperture(W, window=None).extract()
        errs.append(float(np.linalg.norm(clean - B) / np.linalg.norm(B)))
    return snr, errs


def _mean_se(xs):
    xs = np.asarray(xs, float)
    xs = xs[np.isfinite(xs)]
    return float(xs.mean()), float(xs.std(ddof=1) / math.sqrt(xs.size)) if xs.size > 1 else 0.0


def _binom_se(p, n):
    return math.sqrt(max(p * (1 - p), 1.0 / n) / n)   # a rate of 0 or 1 still carries 1/n


def main():
    sens = [(k, n, T, F, s) for k in SIGNALS for n in NOISE for (T, F) in SHAPES for s in RUNGS]
    level = [(n, T, F) for n in NOISE for (T, F) in SHAPES]
    metrics = {}
    with Pool(int(os.environ.get("BASELINE_PROCS", os.cpu_count() or 1))) as pool:
        for (k, n, T, F, s), found, exact in pool.imap_unordered(_sensitivity_cell, sens):
            key = f"{k}.{n}.{T}x{F}.x{s:.3f}"
            metrics[f"sensitivity.{key}"] = dict(value=found, se=_binom_se(found, N_REC), better="higher")
            if s >= 1.0:
                metrics[f"accuracy.count.{key}"] = dict(value=exact, se=_binom_se(exact, N_REC),
                                                        better="higher")
        for (n, T, F), rate, cnt in pool.imap_unordered(_level_cell, level):
            metrics[f"accuracy.level.{n}.{T}x{F}"] = dict(value=rate, se=_binom_se(FAR, cnt),
                                                          better="bound", bound=FAR)
        for snr_db, ea, eb in pool.imap_unordered(_decay_cell, (60, 40, 30, 20, 10)):
            v, se = _mean_se(ea)
            metrics[f"accuracy.decay_rate.snr{snr_db}dB"] = dict(value=v, se=se, better="lower")
            v, se = _mean_se(eb)
            metrics[f"accuracy.frequency.snr{snr_db}dB"] = dict(value=v, se=se, better="lower")
        for snr, errs in pool.imap_unordered(_extract_cell, (5, 10, 50)):
            v, se = _mean_se(errs)
            metrics[f"accuracy.extract.snr{snr}"] = dict(value=v, se=se, better="lower")
    src = Path(E.__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(src.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    out = dict(meta=dict(kind="sensitivity+accuracy", far=FAR, records=N_REC,
                         source_sha256=h.hexdigest(), numpy=np.__version__),
               metrics=dict(sorted(metrics.items())))
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
