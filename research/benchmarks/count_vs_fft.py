"""count_vs_fft.py -- the like-for-like reads beside the FFT, in retired instructions.

    VALGRIND=/path/to/valgrind python count_vs_fft.py [path to a src/] > count_vs_fft.jsonl

The same records and the same reads as ``operator_vs_fft.py``'s like-for-like timings -- two
persistent tones in white noise, one channel and 16, T = 1024 to 1048576 -- each costed as
``cost.py`` costs a read: the instructions valgrind's callgrind counts for a call, the difference
between a run that makes it ``REPS`` times and the same run without it, divided by ``REPS``.  Each is
taken twice, and the spread of the two differences is its uncertainty.  An instruction count does not depend on the host's load
or clock, so two libraries, or a library and the FFT, are compared on what each computes; the
milliseconds in ``operator_vs_fft.jsonl`` also carry the host.  Single-threaded throughout.

  fft       the bare FFT of every channel (``scipy.fft.rfft``, one worker): a spectrum only;
  pipeline  ``operator_vs_fft.fft_pipeline`` unpadded: a count at ``far``, each peak's frequency,
            decay rate and power;
  read      the library's read of the same: ``Aperture(X).dynamics()`` on 16 channels, the
            fixed-depth ``koopman_lift(X, 16)`` on one, each ``resolved()`` and ``modes()``.
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
sys.path.insert(0, _HERE)

import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402

REPS = 3                                   # cost.py's protocol
CELLS = [(F, T) for F in (1, 16) for T in (1024, 4096, 16384, 65536, 262144, 1048576)]
METHODS = ("fft", "pipeline", "read")


def _record(F, T):
    """The like-for-like record at ``(F, T)``: ``operator_vs_fft.run_like_for_like``'s first sweep,
    drawn in its order from its seed."""
    rng = np.random.default_rng(5)
    for F_, T_ in CELLS:
        t = np.arange(T_)[:, None]
        ph = rng.uniform(0, 2 * np.pi, (2, F_))
        X = (np.cos(2 * np.pi * 0.0402 * t + ph[0]) + 0.6 * np.cos(2 * np.pi * 0.0926 * t + ph[1])
             + 0.05 * rng.standard_normal((T_, F_)))
        if (F_, T_) == (F, T):
            return X
    raise KeyError((F, T))


def _call(method, F):
    import entroptics as E
    from scipy import fft as sfft
    from operator_vs_fft import fft_pipeline
    if method == "fft":
        return lambda X: sfft.rfft(X, axis=0, workers=1)
    if method == "pipeline":
        return lambda X: fft_pipeline(X, pad=1)
    if F == 1:
        return lambda X: (lambda D: (D.resolved(), D.modes()))(E.koopman_lift(X, 16))
    return lambda X: (lambda D: (D.resolved(), D.modes()))(E.Aperture(X).dynamics())


def _one(method, F, T, reps):
    """The child: build the record, make the call once to warm it -- every cache a first call
    fills, in this run and in the run it is differenced against -- then make it ``reps`` times.  The
    difference is the cost of a call, not of a first call."""
    X = _record(F, T)
    f = _call(method, F)
    f(X)
    for _ in range(reps):
        f(X)


def _instructions(method, F, T, reps):
    out = subprocess.run([os.environ.get("VALGRIND", "valgrind"), "--tool=callgrind",
                          "--callgrind-out-file=/dev/null", sys.executable, __file__, _SRC,
                          "--one", method, str(F), str(T), str(reps)], capture_output=True, text=True)
    m = re.search(r"Collected : (\d+)", out.stderr)
    if not m:
        raise RuntimeError(out.stderr[-2000:])
    return int(m.group(1))


def _cell(args):
    method, F, T = args
    d = [(_instructions(method, F, T, REPS) - _instructions(method, F, T, 0)) / REPS for _ in range(2)]
    return dict(method=method, F=F, T=T, instructions=float(np.mean(d)), se=float(abs(d[0] - d[1]) / 2))


def main():
    if "--one" in sys.argv:
        i = sys.argv.index("--one")
        _one(sys.argv[i + 1], int(sys.argv[i + 2]), int(sys.argv[i + 3]), int(sys.argv[i + 4]))
        return
    import hashlib
    from pathlib import Path
    import entroptics as E
    h = hashlib.sha256()
    for p in sorted(Path(E.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    print(json.dumps(dict(file=E.__file__, source_sha256=h.hexdigest())), flush=True)
    jobs = [(m, F, T) for (F, T) in CELLS for m in METHODS]
    jobs.sort(key=lambda j: -j[1] * j[2])                    # the longest first
    with Pool(int(os.environ.get("COUNT_PROCS", "4"))) as pool:
        for row in pool.imap_unordered(_cell, jobs):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
