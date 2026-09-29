"""screen_null.py -- does the screen's floor hold its level on noise, and does it still find signal?

    python screen_null.py                   # the library on sys.path
    python screen_null.py <path to a src/>  # a specific copy, e.g. v0.2.3's

  null     the share of noise-only records in which the default read claims a mode (K_signal > 0),
           at far = 0.05, for noise families x shapes: Gaussian at equal and unequal channel levels,
           heavy-tailed, skewed, and counts, real and complex, narrow, wide and F >> T;
  detect   the share of records in which a planted signal is found (K_signal > 0), and the share in
           which its rank is counted exactly (K_signal equal to the planted rank): a persistent mode
           across all channels, a narrow broadband burst, a narrowband line, three planted modes
           near the edge, and two strong modes, each over Gaussian noise at equal and at unequal
           channel levels.

Seeded and single-threaded.  The first line stamps the SHA-256 of the library source that ran
(LF-normalised, so a clean checkout reproduces it).  Every line is one JSON record.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import hashlib  # noqa: E402
import json     # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import entroptics as E  # noqa: E402
from entroptics.projection import Projection  # noqa: E402

N_NULL = 200
N_DET = 100


def emit(**kw):
    print(json.dumps(kw), flush=True)


def stamp():
    h = hashlib.sha256()
    for p in sorted(Path(E.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    emit(file=E.__file__, source_sha256=h.hexdigest())


def levels(F, spread):
    return np.exp(np.linspace(-spread, spread, F))


NOISE = {
    "gaussian": lambda r, T, F: r.standard_normal((T, F)),
    "gaussian, levels e^-1..e^1": lambda r, T, F: r.standard_normal((T, F)) * levels(F, 1.0),
    "gaussian, levels e^-2..e^2": lambda r, T, F: r.standard_normal((T, F)) * levels(F, 2.0),
    "complex gaussian, levels e^-1..e^1": lambda r, T, F: (r.standard_normal((T, F))
                                                          + 1j * r.standard_normal((T, F))) * levels(F, 1.0),
    "student t (5)": lambda r, T, F: r.standard_t(5, (T, F)),
    "exponential": lambda r, T, F: r.exponential(1.0, (T, F)),
    "lognormal": lambda r, T, F: r.lognormal(0.0, 1.0, (T, F)),
    "lognormal (sigma 1.5)": lambda r, T, F: r.lognormal(0.0, 1.5, (T, F)),
    "pareto (3)": lambda r, T, F: r.pareto(3.0, (T, F)),
    "chi-square (1)": lambda r, T, F: r.chisquare(1, (T, F)),
    "poisson (1)": lambda r, T, F: r.poisson(1.0, (T, F)).astype(float),
    "poisson (5)": lambda r, T, F: r.poisson(5.0, (T, F)).astype(float),
    "poisson (5), levels 1..20": lambda r, T, F: r.poisson(np.linspace(1, 20, F), (T, F)).astype(float),
}
SHAPES = ((256, 8), (256, 32), (1024, 64), (64, 128), (32, 512), (20, 1000))


def null():
    for name, make in NOISE.items():
        for T, F in SHAPES:
            k = [Projection(make(np.random.default_rng(1000 * T + F + i), T, F)).K_signal > 0
                 for i in range(N_NULL)]
            emit(null=name, T=T, F=F, records=N_NULL, false_alarm_rate=float(np.mean(k)))


def planted(kind, r, T, F, lv):
    x = r.standard_normal((T, F)) * lv
    if kind == "persistent mode":
        x += np.outer(r.standard_normal(T), lv / np.sqrt(F)) * 1.6 * (1 + np.sqrt(F / T))
    elif kind == "narrow broadband burst":
        w = max(4, T // 64)
        t0 = T // 2 - w // 2
        x[t0:t0 + w] += 2.5 * np.hanning(w)[:, None] * lv
    elif kind == "narrowband line":
        c = F // 3
        x[:, c:c + 2] += 0.9 * np.sin(2 * np.pi * 0.05 * np.arange(T))[:, None] * lv[c:c + 2]
    elif kind == "three modes near the edge":
        U = np.linalg.qr(r.standard_normal((T, 3)))[0]
        V = np.linalg.qr(r.standard_normal((F, 3)))[0]
        edge = np.sqrt(T) + np.sqrt(F)
        x += (U * (edge * np.array([1.6, 1.3, 1.15]))) @ V.T * lv
    elif kind == "two strong modes":
        x += 2.0 * (r.standard_normal((T, 2)) @ r.standard_normal((2, F))) * lv
    return x


RANK = {"persistent mode": 1, "narrow broadband burst": 1, "narrowband line": 1,
        "three modes near the edge": 3, "two strong modes": 2}


def detect():
    for kind in RANK:
        for T, F in ((256, 32), (1024, 64), (32, 512)):
            for spread in (0.0, 1.0):
                lv = levels(F, spread)
                K = [Projection(planted(kind, np.random.default_rng(7 * T + F + i), T, F, lv)).K_signal
                     for i in range(N_DET)]
                emit(detect=kind, T=T, F=F, levels=f"e^-{spread:g}..e^{spread:g}", records=N_DET,
                     found_rate=float(np.mean(np.array(K) > 0)),
                     exact_rate=float(np.mean(np.array(K) == RANK[kind])), mean_K=float(np.mean(K)))


if __name__ == "__main__":
    stamp()
    null()
    detect()
