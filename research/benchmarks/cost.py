"""cost.py -- the release baseline's resources and throughput: instructions and memory per read.

    VALGRIND=/path/to/valgrind python cost.py [path to a src/] > cost_metrics.json

Throughput is measured as retired instructions per read, counted by valgrind's callgrind: every
operation the read performs, the ``@`` products and the LAPACK calls included, independent of the
host's load and clock.  A read's count is the difference between a run that performs it ``REPS``
times and the same run without it, divided by ``REPS``; each is taken twice, and the spread of the
two differences is the metric's uncertainty.  Memory is the peak of what the read allocates, from
``tracemalloc`` (numpy reports its buffers to it).  Single-threaded BLAS throughout.

The reads, their records and their sizes are the measurement protocol -- fixed here, never tuned.
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else \
    os.path.join(_HERE, "..", "..", "src")
sys.path.insert(0, _SRC)

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402

import numpy as np  # noqa: E402

REPS = 3


def _record(T, F, seed=0):
    r = np.random.default_rng(seed)
    u = r.standard_normal(T); v = r.standard_normal(F)
    return r.standard_normal((T, F)) + 2.0 * (math.sqrt(T) + math.sqrt(F)) * np.outer(
        u / np.linalg.norm(u), v / np.linalg.norm(v))


def _burst(T=64, F=256):
    t = np.arange(T)[:, None]; f = np.arange(F)[None, :]
    b = np.exp(-0.5 * ((t - 0.46 * T) / 3.0) ** 2) * np.exp(-0.5 * ((f - 0.5 * F) / (0.22 * F)) ** 2)
    return b / b.max() + np.random.default_rng(0).standard_normal((T, F)) / 10


def _reads():
    """name -> (setup() -> inputs, read(inputs))."""
    import entroptics as E
    from entroptics.batch import ResolvedScreen
    from entroptics.reads import spectral_optics

    def proj(T, F):
        return (lambda: _record(T, F)), (lambda W: E.Projection(W).K_signal)

    def stream():
        X = _record(400, 32)

        def run(X):
            rs = ResolvedScreen(32)
            for i in range(0, 400, 16):
                rs.update(X[i:i + 16])
            return rs.K_signal
        return (lambda: X), run
    return {
        "projection.256x32": proj(256, 32),
        "projection.1024x64": proj(1024, 64),
        "projection.32x512": proj(32, 512),
        "spectral.1024x64": ((lambda: _record(1024, 64)), (lambda W: spectral_optics(W))),
        "rates.2000x32": ((lambda: _record(2000, 32)), (lambda W: E.Aperture(W).rates())),
        "extract.64x256": ((lambda: _burst()), (lambda W: E.Aperture(W, window=None).extract())),
        "resolved_batch.32x128x16": ((lambda: np.stack([_record(128, 16, s) for s in range(32)])),
                                     (lambda X: E.resolved_batch(X))),
        "resolved_screen.400x32": stream(),
    }


def _one(name, reps):
    """The child: set up ``name``'s inputs, warm the library, then read ``reps`` times."""
    reads = _reads()
    setup, read = reads[name]
    x = setup()
    import entroptics as E
    E.Projection(_record(64, 8, 1))                       # warm caches either way
    for _ in range(reps):
        read(x)


def _instructions(name, reps, valgrind):
    out = subprocess.run([valgrind, "--tool=callgrind", "--callgrind-out-file=/dev/null",
                          sys.executable, __file__, _SRC, "--one", name, str(reps)],
                         capture_output=True, text=True)
    m = re.search(r"Collected : (\d+)", out.stderr)
    if not m:
        raise RuntimeError(out.stderr[-2000:])
    return int(m.group(1))


def _peak_bytes(name):
    import tracemalloc
    setup, read = _reads()[name]
    x = setup()
    read(x)                                               # warm, outside the measurement
    tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    read(x)
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return peak


def main():
    if "--one" in sys.argv:
        i = sys.argv.index("--one")
        _one(sys.argv[i + 1], int(sys.argv[i + 2]))
        return
    valgrind = os.environ.get("VALGRIND", "valgrind")
    metrics = {}
    for name in _reads():
        d = [(_instructions(name, REPS, valgrind) - _instructions(name, 0, valgrind)) / REPS
             for _ in range(2)]
        metrics[f"throughput.instructions.{name}"] = dict(
            value=float(np.mean(d)), se=float(abs(d[0] - d[1]) / 2), better="lower")
        p = [_peak_bytes(name) for _ in range(2)]
        metrics[f"resources.peak_bytes.{name}"] = dict(
            value=float(np.mean(p)), se=float(abs(p[0] - p[1]) / 2), better="lower")
        print(name, metrics[f"throughput.instructions.{name}"]["value"],
              metrics[f"resources.peak_bytes.{name}"]["value"], file=sys.stderr, flush=True)
    json.dump(dict(meta=dict(kind="resources+throughput", reps=REPS, numpy=np.__version__),
                   metrics=metrics), sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
