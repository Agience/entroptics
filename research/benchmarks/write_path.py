"""write_path.py -- the write path's numbers: the basis round trip, drift, and extract's recovery.

Every write-path figure the README and the CHANGELOG quote comes from this script:
  drift     over 80 seeded records of a rank-3 process on 40 channels with unequal channel noise:
            the rate at which ``Basis.drift`` reads structure in a later record of the SAME process
            (a false alarm, at far = 0.05), and in one with a planted extra mode or a local transient
            (a detection); the same for equal channel noise, for a basis read from noise alone, and
            for a complex rank-2 process;
  round     the largest round-trip certificate figure over those 80 records;
  verdict   ``Drift.identified`` against the truth: over 4000 records whose sample noise covariance is
            exactly diagonal (so the moments are exact), how often each kind of span is identified,
            and how often an identified channel's noise is off by more than 0.1%; and drift's false
            alarms on later records of the same process, split by the verdict, on eight real and
            complex configurations;
  extract   ``Aperture.extract`` on an exactly rank-1 record with no noise, its concentration in time
            varied, and on a sine that fills the record: the relative recovery error ||clean - W|| / ||W||.

    python write_path.py            # one JSON line per result

Seeded and single-threaded, so it reproduces.  The first line stamps the SHA-256 of the library source
that ran, since an installed distribution's version metadata can be stale.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import json     # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import entroptics as E  # noqa: E402


def emit(**kw):
    print(json.dumps(kw), flush=True)


def source_stamp():
    """The SHA-256 over the imported package's source files, in name order, with line endings in
    git's form (LF), so a clean checkout of the release reproduces it."""
    h = hashlib.sha256()
    for p in sorted(Path(E.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))   # git's form: a checkout reproduces it
    return dict(file=E.__file__, source_sha256=h.hexdigest())


def drift_rates(n=80):
    out = {k: [] for k in ("same_unequal", "same_equal", "own", "extra_mode", "local_transient",
                           "noise_basis", "complex_same")}
    cert = []
    for seed in range(n):
        rng = np.random.default_rng(4000 + seed)
        F, T = 40, 600
        Bt = np.linalg.qr(rng.standard_normal((F, 3)))[0].T
        nzh = np.exp(rng.uniform(-0.5, 0.5, F))
        u = np.linalg.qr(rng.standard_normal((F, 1)))[0][:, 0]

        def rec(nz, extra=None):
            W = (rng.standard_normal((T, 3)) * [6, 4, 3]) @ Bt + rng.standard_normal((T, F)) * nz + 5.0
            if extra == "mode":
                W += np.outer(rng.standard_normal(T) * 1.5, u)
            if extra == "local":
                W[200:260, 10:16] += 2.0 * np.hanning(60)[:, None] * nz[10:16]
            return W

        W = rec(nzh)
        b = E.Aperture(W).basis()
        out["own"].append(b.drift(W).K > 0)
        out["same_unequal"].append(b.drift(rec(nzh)).K > 0)
        out["extra_mode"].append(b.drift(rec(nzh, "mode")).K > 0)
        out["local_transient"].append(b.drift(rec(nzh, "local")).K > 0)
        be = E.Aperture(rec(np.ones(F))).basis()
        out["same_equal"].append(be.drift(rec(np.ones(F))).K > 0)
        bn = E.Aperture(rng.standard_normal((T, F)) * nzh).basis()
        out["noise_basis"].append(bn.drift(rng.standard_normal((T, F)) * nzh).K > 0)
        Bc = np.linalg.qr(rng.standard_normal((12, 2)) + 1j * rng.standard_normal((12, 2)))[0].T

        def crec():
            return ((rng.standard_normal((500, 2)) + 1j * rng.standard_normal((500, 2))) * [5, 3]) @ Bc \
                + rng.standard_normal((500, 12)) + 1j * rng.standard_normal((500, 12))
        out["complex_same"].append(E.Aperture(crec()).basis().drift(crec()).K > 0)
        c = b.certify(W)
        cert.append(max(c.defect, c.inverse, c.idempotent, c.lossless))
    emit(drift="rate at which drift reads structure (K > 0), far = 0.05", records=n,
         **{k: float(np.mean(v)) for k, v in out.items()})
    emit(round_trip="largest certificate figure over the records", records=n, max=float(max(cert)))


def _white(rng, T, F, cx):
    """A (T, F) record whose sample covariance is exactly the identity: the moments are exact."""
    Z = rng.standard_normal((T, F)) + (1j * rng.standard_normal((T, F)) if cx else 0)
    Z = Z - Z.mean(0)
    return Z @ np.linalg.inv(np.linalg.cholesky(Z.conj().T @ Z / T).conj().T)


def verdict_exact(n=4000):
    from entroptics.basis import _noise_scale
    kinds = ("real span", "complex span", "phase-rotated real span", "a channel nearly inside")
    ident = [0] * 4
    total = [0] * 4
    wrong = [0] * 4
    for t in range(n):
        rng = np.random.default_rng(50_000 + t)
        K = int(rng.integers(1, 6))
        F = int(rng.integers(K + 1, min(K + 30, 60)))
        kind = t % 4
        B = rng.standard_normal((K, F))
        if kind == 1:
            B = B + 1j * rng.standard_normal((K, F))
        elif kind == 2:
            B = B * np.exp(1j * rng.uniform(0, 2 * np.pi, F))
        elif kind == 3:
            B = B + 1j * rng.standard_normal((K, F))
            B[:, 1:] *= 10 ** rng.uniform(-9, -3)
            B[0, 0] = 1.0
        T = int(rng.integers(max(3 * F, 200), 4000))
        Z = _white(rng, T, F, kind in (1, 2, 3))
        sig = np.exp(rng.uniform(-0.8, 0.8, F))
        s0 = sig * np.exp(rng.uniform(-1, 1, F))
        s, idf = _noise_scale(Z * sig, B, s0)
        total[kind] += 1
        if idf:
            ident[kind] += 1
            wrong[kind] += bool(np.max(np.abs((s / sig) ** 2 - 1)) > 1e-3)
    for k in range(4):
        emit(verdict="exact moments", span=kinds[k], records=total[k], identified=ident[k],
             identified_but_off_by_more_than_0p1pct=wrong[k])


def drift_by_verdict():
    for (T, F, K) in ((300, 6, 2), (500, 12, 2), (600, 40, 3), (200, 4, 1)):
        for cx in (False, True):
            fa = {True: [0, 0], False: [0, 0]}
            for bi in range(10):
                rng = np.random.default_rng(bi + 100 * F)
                M = rng.standard_normal((F, K)) + (1j * rng.standard_normal((F, K)) if cx else 0)
                Bt = np.linalg.qr(M)[0].T
                amp = np.array([6, 4, 3][:K])
                nz = np.exp(rng.uniform(-0.5, 0.5, F))

                def rec():
                    S = rng.standard_normal((T, K)) + (1j * rng.standard_normal((T, K)) if cx else 0)
                    N = rng.standard_normal((T, F)) + (1j * rng.standard_normal((T, F)) if cx else 0)
                    return (S * amp) @ Bt + N * nz
                b = E.Aperture(rec()).basis()
                if b.K != K:
                    continue
                for _ in range(15):
                    d = b.drift(rec())
                    fa[d.identified][0] += int(d.K > 0)
                    fa[d.identified][1] += 1
            emit(drift_by_verdict="false alarms on later records of the same process, far = 0.05",
                 T=T, F=F, K=K, complex=cx, identified=fa[True], not_identified=fa[False])


def extract_recovery():
    rng = np.random.default_rng(5)
    T, F = 256, 64
    t = np.arange(T)
    c = np.abs(rng.standard_normal(F)) + 0.5
    rows = []
    import warnings
    for w in (0.5, 2.0, 8.0, 32.0, 128.0):
        g = np.exp(-0.5 * ((t - T / 2) / w) ** 2)
        W = np.outer(g, c)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean, info = E.Aperture(W).extract()
        rows.append(dict(time_width=w, contrast=float(info["contrast"]),
                         error=float(np.linalg.norm(clean - W) / np.linalg.norm(W)),
                         warnings=sorted({str(x.message) for x in caught})))
    W = np.outer(np.sin(2 * np.pi * 3 * t / T), c)
    clean, info = E.Aperture(W).extract()
    rows.append(dict(sine_filling_the_record=True, contrast=float(info["contrast"]),
                     error=float(np.linalg.norm(clean - W) / np.linalg.norm(W))))
    for r in rows:
        emit(extract="rank-1, no noise", **r)


if __name__ == "__main__":
    emit(**source_stamp())
    drift_rates()
    verdict_exact()
    drift_by_verdict()
    extract_recovery()
