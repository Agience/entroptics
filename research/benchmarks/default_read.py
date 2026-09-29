"""default_read.py -- the default read on records with flat channels and count noise, and the
moment pencil's cut.

Every figure the 0.2.5 CHANGELOG quotes for these fixes comes from this script.  Run it on the
current source and on 0.2.3's, the last release (``git archive v0.2.3 src/entroptics``), to get the
before and after:

    python default_read.py                    # the library on sys.path
    python default_read.py <path to a src/>   # a specific copy, e.g. 0.2.3's

  flat      false alarms (K > 0 on noise-only records, far = 0.05) with constant channels or sparse
            count channels beside Gaussian ones, and with no such channel;
  counts    false alarms on Poisson noise under the default floor, and the detection of a planted
            mode;
  pencil    the moment pencil's leading eigenvalue against the truth, median over records, with
            ``rcond=1e-6`` (0.2.3's fixed cut) and with the default (where the library derives it).

Seeded and single-threaded.  The first line stamps the SHA-256 of the library source that ran
(LF-normalised, so a clean checkout reproduces it).
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
if len(sys.argv) > 1:
    sys.path.insert(0, sys.argv[1])

import hashlib  # noqa: E402
import inspect  # noqa: E402
import json     # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import entroptics as E  # noqa: E402
from entroptics.projection import Projection  # noqa: E402

T = 256


def emit(**kw):
    print(json.dumps(kw), flush=True)


def stamp():
    h = hashlib.sha256()
    for p in sorted(Path(E.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    emit(file=E.__file__, source_sha256=h.hexdigest())


def rate(make, n=200, null=None):
    return float(np.mean([Projection(make(np.random.default_rng(i)), null=null).K_signal > 0
                          for i in range(n)]))


def flat():
    cases = [
        ("8 gaussian channels", lambda r: r.standard_normal((T, 8))),
        ("8 gaussian + 2 constant", lambda r: np.concatenate([r.standard_normal((T, 8)), np.full((T, 2), 3.0)], 1)),
        ("8 gaussian + 8 constant", lambda r: np.concatenate([r.standard_normal((T, 8)), np.full((T, 8), 3.0)], 1)),
        ("8 gaussian + 8 poisson(0.2)", lambda r: np.concatenate([r.standard_normal((T, 8)),
                                                                    r.poisson(0.2, (T, 8)).astype(float)], 1)),
        ("64 gaussian + 8 constant", lambda r: np.concatenate([r.standard_normal((T, 64)), np.full((T, 8), 3.0)], 1)),
    ]
    for name, mk in cases:
        emit(flat=name, records=200, false_alarm_rate=rate(mk))


def counts():
    for lam, F in ((3.0, 8), (1.0, 8), (3.0, 32)):
        mk = (lambda r, lam=lam, F=F: r.poisson(lam, (T, F)).astype(float))
        emit(counts=f"poisson({lam:g}), {F} channels", records=120, mp=rate(mk, 120))

    def planted(r):
        x = r.poisson(3.0, (T, 8)).astype(float)
        return x + np.outer(r.standard_normal(T) * 1.2, np.ones(8) / np.sqrt(8)) * np.sqrt(3.0) * 3
    emit(counts="poisson(3) + a planted mode", records=40, detection_mp=rate(planted, 40))


def pencil():
    derived = inspect.signature(E.matrix_pencil).parameters["rcond"].default is None \
        if hasattr(E, "matrix_pencil") else False

    def acov(x, L):
        x = x - x.mean()
        return np.array([x[:x.size - k] @ x[k:] / x.size for k in range(L)])

    rng = np.random.default_rng(0)
    for label, mus, TT in (("AR(0.9), T=4000", [0.9], 4000), ("two modes 0.95/0.6, T=4000", [0.95, 0.6], 4000),
                           ("AR(0.9), T=500", [0.9], 500)):
        errs = {"rcond_1e-6": {}, "default": {}}
        for _ in range(40):
            comps = []
            for mu in mus:
                z = np.zeros(TT)
                e = rng.standard_normal(TT)
                for t in range(1, TT):
                    z[t] = mu * z[t - 1] + e[t]
                comps.append(z)
            c = acov(sum(comps), 30)
            for n in (2, 5, 8, 12):
                for key, kw in (("rcond_1e-6", {"rcond": 1e-6}), ("default", {})):
                    ev = E.hankel_spectrum(c, n, **kw).evals
                    errs[key].setdefault(n, []).append(abs(ev[0] - max(mus)))
        emit(pencil=label, derived_default=derived,
             median_abs_error={k: {str(n): float(np.median(v)) for n, v in d.items()} for k, d in errs.items()})


if __name__ == "__main__":
    stamp()
    flat()
    counts()
    pencil()
