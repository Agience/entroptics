"""
environment.py -- the numpy/torch dispatch that keeps Entroptics ONE code path.

Every numeric kernel (entropy, reads, dynamics) is written once against ``xp`` (the
array namespace of its input) plus the handful of helpers here whose numpy/torch
spellings differ.  Feed numpy -> runs on CPU; feed a torch tensor -> runs on its
device (GPU).  No torch shadow, no second implementation.

``xp = ns(x)`` returns numpy or torch; the ``_`` helpers wrap only the ops that
differ between them.  Ops with identical spelling (``xp.linalg.eigh``, ``xp.abs``,
``xp.sum``, ``xp.where``, ``xp.log2``, ``xp.conj``, ``xp.angle``, ``xp.diag``,
``xp.outer``, ``@`` ...) are used directly.
"""
from __future__ import annotations

import os

import numpy as np

# ── compute precision: an ENVIRONMENTAL setting, not a per-read knob ───────────────────
# The float width the projection/monitor path computes in.  Default 64 (bit-perfect -- the
# reference results, and the only mode the test suite runs).  Set 32 for throughput on a
# GPU deployment (fp32 SVD is ~20x fp64 on these cards); it perturbs singular values at the
# ~1e-4 level (an exact SVD in fp32, NOT an approximation or a dropped-data filter), so
# K_signal can differ only for a mode sitting within that band of the floor.  Choose it by
# DEPLOYMENT (env var ENTROPTICS_PRECISION=32, or set_precision(32)), like a device choice --
# a system that must be exact leaves it at 64.  Does NOT touch the certified DMD/dynamics
# path (dynamics.py stays float64 for eig stability).
_PRECISION = 32 if str(os.environ.get("ENTROPTICS_PRECISION", "64")).strip() in ("32", "single", "float32") else 64


def set_precision(bits: int) -> None:
    """Set the ENVIRONMENTAL compute precision: 64 (default, bit-perfect) or 32 (fast, GPU)."""
    global _PRECISION
    _PRECISION = 32 if int(bits) == 32 else 64


def precision() -> int:
    """The current compute precision (64 or 32)."""
    return _PRECISION


def rdtype(xp):
    """The real compute dtype for the current precision (float64 default, float32 if set)."""
    return xp.float32 if _PRECISION == 32 else xp.float64


def cdtype(xp):
    """The complex compute dtype for the current precision (complex128 default, complex64 if set)."""
    return xp.complex64 if _PRECISION == 32 else xp.complex128


def as_compute(xp, x):
    """Cast x to the compute precision (complex if x is complex, else real).  Index arithmetic
    inside the fold can transiently promote (e.g. float32 edges - int64 indices -> float64); a
    final cast pins the screen to the set precision.  At the default (64) it is float64 -> a
    no-op cast, so results stay bit-identical."""
    xc = xp.is_complex(x) if is_torch(xp) else np.iscomplexobj(x)
    dt = cdtype(xp) if xc else rdtype(xp)
    return x.to(dt) if is_torch(xp) else np.asarray(x).astype(dt, copy=False)


def ns(x):
    """The array namespace (numpy or torch) for x -- no hard torch import."""
    if type(x).__module__.partition(".")[0] == "torch":
        import torch
        return torch
    return np


def is_torch(xp) -> bool:
    """True if xp is the torch namespace rather than numpy."""
    return xp is not np


# ── dtype / construction ──────────────────────────────────────────────────────

def asnum(x, *, complex=None):
    """Cast x to the real compute dtype, or the complex one if complex (or if x is already
    complex and complex is not False).  Precision follows the environmental setting
    (float64/complex128 by default; float32/complex64 when ``set_precision(32)``)."""
    xp = ns(x)
    xc = xp.is_complex(x) if is_torch(xp) else np.iscomplexobj(x)
    want_c = xc if complex is None else complex
    dt = (cdtype(xp) if want_c else rdtype(xp))
    return x.to(dt) if is_torch(xp) else np.asarray(x).astype(dt)


def zeros(xp, shape, *, complex=False, ref=None):
    """A zero array of ``shape`` at the compute precision (complex dtype if ``complex``; on
    ``ref``'s device for torch)."""
    dt = cdtype(xp) if complex else rdtype(xp)
    if is_torch(xp):
        return xp.zeros(shape, dtype=dt, device=(ref.device if ref is not None else None))
    return np.zeros(shape, dtype=dt)


def arange_int(xp, n, *, ref=None):
    """The integer range ``[0, n)`` (on ``ref``'s device for torch)."""
    if is_torch(xp):
        return xp.arange(n, device=(ref.device if ref is not None else None))
    return np.arange(n)


def ones(xp, n, *, ref=None):
    """A real ones vector of length ``n`` (on ``ref``'s device for torch)."""
    if is_torch(xp):
        return xp.ones(n, dtype=rdtype(xp), device=(ref.device if ref is not None else None))
    return np.ones(n)


# ── ops that differ between numpy and torch ───────────────────────────────────

def clampmin(xp, x, lo):
    """Clamp x below at ``lo`` (numpy clip / torch clamp)."""
    return x.clamp(min=lo) if is_torch(xp) else np.clip(x, lo, None)


def cliprange(xp, x, lo, hi):
    """Clamp x to ``[lo, hi]`` (numpy clip / torch clamp)."""
    return x.clamp(min=lo, max=hi) if is_torch(xp) else np.clip(x, lo, hi)


def argsort_desc(xp, v):
    """Indices that sort v in descending order."""
    return v.argsort(descending=True) if is_torch(xp) else np.argsort(v)[::-1]


def flip(xp, v):
    """Reverse a 1-D array along axis 0."""
    return xp.flip(v, [0]) if is_torch(xp) else v[::-1]


def sum_ax(xp, x, ax=None, keep=False):
    """Sum over axis ``ax`` (None = all).  Wraps numpy ``axis=`` / torch ``dim=``."""
    if is_torch(xp):
        return xp.sum(x) if ax is None else xp.sum(x, dim=ax, keepdim=keep)
    return np.sum(x) if ax is None else np.sum(x, axis=ax, keepdims=keep)


def to_numpy(x):
    """Materialise any array (numpy or torch, CPU/GPU) to a numpy array."""
    if type(x).__module__.partition(".")[0] == "torch":
        t = x.detach().cpu()
        try:
            import torch
            if torch.is_complex(t):
                t = t.to(torch.complex128)
        except Exception:  # pragma: no cover
            pass
        return t.numpy()
    return np.asarray(x)


def mean0(xp, x):
    """Mean over axis 0, keeping the axis (shape (1, ...))."""
    return xp.mean(x, dim=0, keepdim=True) if is_torch(xp) else np.mean(x, axis=0, keepdims=True)


def nanmean0(xp, x):
    """NaN-ignoring mean over axis 0.  torch's nanmean rejects complex, so split
    real/imag there (numpy's nanmean is already complex-safe)."""
    if is_torch(xp):
        if xp.is_complex(x):
            return xp.nanmean(xp.real(x), dim=0) + 1j * xp.nanmean(xp.imag(x), dim=0)
        return xp.nanmean(x, dim=0)
    return np.nanmean(x, axis=0)


def median0(xp, x):
    """Per-column median over axis 0 (numpy-style interpolated median)."""
    return xp.quantile(x, 0.5, dim=0) if is_torch(xp) else np.median(x, axis=0)


def median1d(xp, v):
    """Median of a 1-D array (numpy-style interpolated)."""
    return xp.quantile(v, 0.5) if is_torch(xp) else np.median(v)


def movedim(xp, x, src, dst):
    """Move axis ``src`` to position ``dst`` (numpy moveaxis / torch movedim)."""
    return xp.movedim(x, src, dst) if is_torch(xp) else np.moveaxis(x, src, dst)


def stack(xp, arrs, axis=0):
    """Stack arrays along a new ``axis`` (numpy stack / torch stack)."""
    return xp.stack(arrs, dim=axis) if is_torch(xp) else np.stack(arrs, axis=axis)


def permute(xp, x, dims):
    """Reorder all axes of x (numpy transpose / torch permute)."""
    return x.permute(*dims) if is_torch(xp) else np.transpose(x, dims)


def asdtype_of(ref, arr_np):
    """Bring a numpy array onto ref's backend/device with ref's EXACT dtype --
    used to inject a seeded (deterministic) random matrix into either backend."""
    if is_torch(ns(ref)):
        import torch
        return torch.as_tensor(arr_np, device=ref.device).to(ref.dtype)
    return arr_np.astype(ref.dtype)


def cumsum0(xp, x):
    """Cumulative sum along axis 0."""
    return xp.cumsum(x, dim=0) if is_torch(xp) else np.cumsum(x, axis=0)


def cat0(xp, parts):
    """Concatenate arrays along axis 0 (numpy concatenate / torch cat)."""
    return xp.cat(parts, dim=0) if is_torch(xp) else np.concatenate(parts, axis=0)


def linspace(xp, a, b, n, *, ref=None):
    """``n`` evenly spaced points on ``[a, b]`` at the compute precision (on ``ref``'s device
    for torch)."""
    # dtype follows the compute precision so the fold's area-weighted resample keeps the
    # screen in float32 under set_precision(32) (a float64 edge vector would upcast the data
    # back to float64 and forfeit the fp32 SVD speedup); float64 by default -> bit-identical.
    if is_torch(xp):
        return xp.linspace(a, b, n, dtype=rdtype(xp), device=(ref.device if ref is not None else None))
    return np.linspace(a, b, n, dtype=rdtype(np))


def floor_int(xp, x):
    """Elementwise floor, cast to integer."""
    return xp.floor(x).long() if is_torch(xp) else np.floor(x).astype(int)


def std0(xp, x):
    """Population standard deviation (ddof = 0)."""
    return xp.std(x, unbiased=False) if is_torch(xp) else np.std(x)


def svdvals(xp, M):
    """Singular values of M in descending order."""
    return xp.linalg.svdvals(M)          # both numpy>=2 and torch expose this


def vnorm(xp, x):
    """Euclidean (Frobenius) norm of x."""
    return xp.linalg.norm(x)
