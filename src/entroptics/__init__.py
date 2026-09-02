"""Entroptics -- read any 2-D signal as a finite optical aperture whose resolution
is fixed by the signal's OWN Shannon entropy (entropy + optics).

    from entroptics import Aperture, Screen

    ap = Aperture(W)          # W: (T, F), real or complex, numpy or torch
    ap.etendue, ap.a_delta    # optics ABOUT the structure
    ap.rates()                # exact per-mode decay rates (streaming operator)
    sc = ap.screen()          # the companion PROJECTION (info WITHIN it)

``Aperture`` is the single front door (batch or streaming); ``Screen`` is the
projection.  The full optics surface is re-exported here; the submodules
(``entroptics.reads``, ``entroptics.screen``, ...) hold the individual reads.
Standalone: numpy only; scipy and torch are optional.
"""
from . import aperture as _aperture
from . import fields             # noqa: F401 -- N-D field reduction (entroptics.fields)
from .aperture import *          # noqa: F401,F403 -- the full front-door surface
from .environment import set_precision, precision   # ENVIRONMENTAL compute precision (64 default / 32 fast)
from .screen import read_batch, BatchRead           # batched monitor (bit-identical ensemble read)
from .reads import spectral_batch                    # batched correlation-eigvalsh read (bit-identical)
from .sweep import sweep                             # the aperture swept where there's coherence
from .extract import extract                         # the read-side FILTER (project onto resolved modes)

__version__ = "0.1.0"
__all__ = list(_aperture.__all__) + ["set_precision", "precision", "read_batch", "BatchRead",
                                     "spectral_batch", "sweep", "extract"]
