"""Entroptics -- read any 2-D signal as a finite optical aperture whose resolution
is fixed by the signal's OWN Shannon entropy (entropy + optics).

    from entroptics import Aperture, Projection

    ap = Aperture(W)          # W: (T, F), real or complex, numpy or torch
    ap.etendue, ap.a_delta    # optics ABOUT the structure
    ap.rates()                # exact per-mode decay rates (the streaming operator)
    sc = ap.projection()      # the companion PROJECTION (info WITHIN it)

ONE PATH
--------
Every reading has exactly one public way to reach it.  A measurement is reached through the
front door that owns it -- never through both a front door and a free function of the same
name, and never through two free functions of different names.  The frame-level functions
(``phi``, ``etendue``, ``decay``, ``geometry``, ...) are the IMPLEMENTATION and live in their
own modules (``entroptics.reads``, ``entroptics.entropy``); they are not a second API.

    ap = Aperture(W)                # a finite record is read whole, commensurate across signals
    ap.phi, ap.etendue, ap.decay    # the reads -- one way each
    entroptics.reads.phi(W)         # the implementation, if you want the primitive

Four entry points, one job each
-------------------------------
    Aperture(W)          ONE signal, the optics ABOUT it -- fill, etendue, strehl, the matched
                         geometry, the decay and its diffraction limit; batch or streaming.  It
                         OWNS a projection and an operator, surfaces their headline reads as thin
                         delegations (`ap.rates()`, `ap.footprints`), and hands you the objects
                         themselves (`ap.projection()`, `ap.dynamics()`) for the rest.
    Projection(W)        ONE signal, the projection WITHIN it -- the kernel every other
                         entry point is built on (its modes, count above the floor, footprints).
    Screen()             TWO OR MORE signals meeting on a shared basis -- coupling, transfer,
                         what crosses.  It earns its place only when the question is BETWEEN
                         signals; for one signal an Aperture or Projection answers directly.
    resolved_batch(X)    A STACK of signals, read at once, backend-optimal (numpy CPU / torch
    ResolvedScreen(F)    GPU).  ``ResolvedScreen``/``ResolvedScreenBatch`` are its stateful
    sweep(W)             siblings for a revisited screen; ``sweep`` scans one wide field by
                         patch.  All read exactly what a per-frame ``Projection`` reads.

``Projection`` is the kernel: ``Aperture``, ``Screen``, ``sweep`` and ``reads.scale_profile``
all construct one.  ``batch`` is the single exception -- it is built on the three primitives
``projection.fold_target_batch`` / ``normalize_batch`` / ``project_batch`` instead, which is
the declared contract that keeps the batched read identical to the per-frame one.

Every module name resolves to the module
----------------------------------------
``entroptics.<name>`` is always the module -- no exceptions, nothing to remember.  ``dynamics``,
``extract`` and ``sweep`` once resolved to a function here; each was a second path to a reading a
front door already owns (``Aperture.dynamics()``, ``Aperture.extract()``, ``Aperture.sweep()``),
so none is exported.

Standalone: numpy only; scipy and torch are optional.
"""

# ── the BLAS thread pin — must run before numpy is imported, or it is inert ──────────────────────
#
# This package is the eigh caller (`dynamics.py`, `reads.py`), and two threads inside
# `numpy.linalg.eigh` fault OpenBLAS on the reference box (independent 64x64 matrices, 400
# iterations, numpy 2.4.4 / scipy-openblas 0.3.31, DYNAMIC_ARCH Haswell):
#
#     1 thread,  400 iters                     0/1 crash
#     2 threads, 400 iters                     3/3 crash   (exit 139, access violation)
#     2 threads, OPENBLAS_NUM_THREADS=1        0/3 crash
#     2 threads, N=8 (below the MT threshold)  0/2 crash
#
# The faulthandler dumps show more threads faulting than the process has Python threads, so the
# fault is in the OpenBLAS worker pool itself, and the same race also hangs instead of faulting --
# an apparently clean run is not evidence the pool is safe.
#
# The line must sit above the imports below: OpenBLAS sizes its pool once, when the library loads
# under `import numpy`, and setting the variable after that does nothing (unset -> 8 threads,
# set-before -> 1, set-after -> 8, read back through `threadpoolctl.threadpool_info()`). At
# package scope it also covers direct submodule imports, because Python initialises parent
# packages before submodules.
#
# `setdefault`, so an operator who exported the variable keeps
# their own value, including one that reinstates the fault, because a caller who has measured
# their own hardware outranks a default derived from this one. This makes the guard weaker than
# an unconditional set, and it cannot help a process that imported numpy before entroptics.
import os as _os

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
del _os

from . import aperture as _aperture
from . import fields             # noqa: F401 -- N-D field reduction (entroptics.fields)
from .aperture import *          # noqa: F401,F403 -- the full front-door surface
from .environment import set_precision, precision   # ENVIRONMENTAL compute precision (64 default / 32 fast)
from .projection import read_batch, BatchRead           # batched monitor (bit-identical ensemble read)
from .reads import spectral_batch                    # batched correlation-eigvalsh read (bit-identical)
from .dynamics import hankel_spectrum, jackknife, HankelSpectrum  # scalar-sequence moment pencil + jackknife
from .batch import (resolved_batch, ResolvedBatch,   # the ONE batched resolved-screen read (numpy CPU / torch GPU)
                    ResolvedScreen, ResolvedScreenBatch, ResourceLimits, recommend_backend)
from .lift import koopman_lift, delay_embed          # Koopman observable lift (nonlinear -> linear operator)

__version__ = "0.2.1"
__all__ = list(_aperture.__all__) + ["set_precision", "precision", "read_batch", "BatchRead",
                                     "spectral_batch",
                                     "hankel_spectrum", "jackknife", "HankelSpectrum",
                                     "resolved_batch", "ResolvedBatch", "ResolvedScreen",
                                     "ResolvedScreenBatch", "ResourceLimits", "recommend_backend",
                                     "koopman_lift", "delay_embed"]
