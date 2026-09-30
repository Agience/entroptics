"""basis.py -- the write path: a record's resolved modes as an exact encode / decode pair (KLT).

A :class:`Basis` is ``K`` orthonormal rows ``B`` over a record's channels, in the record's own
whitened (noise) coordinates, with the per-channel ``centre`` and ``scale`` that define them.  The
adjoint of an orthonormal basis is its exact inverse on the span, so

    encode(W) = ((W - centre) / scale) @ B^H          (T, K)   the mode coordinates
    decode(A) = (A @ B) * scale + centre              (T, F)   back in W's own units
    split(W)  = (decode(encode(W)),  W - decode(encode(W)))

and ``resolved + residual == W``: nothing is lost and nothing is synthesised.  Each row of a frame
is coded on its own, so a basis read once can be shared ahead of time and applied to every later
frame -- the Karhunen-Loeve transform of the process that produced the record.

The rows are the record's KLT restricted to the modes its read resolved: the resolved modes'
ordered-axis profiles (read on the entropy-matched screen and carried to the record's grid by the
fold's adjoint, as :func:`extract.filter_projection` carries them) project the record in its noise
metric, and the right singular vectors of that projection are ``B``, strongest first.  The time
profiles are the screen's and the metric is the noise's, so ``B`` is close to, but not the same as,
the top ``K`` right singular vectors of the noise-whitened record.  Channels are never folded, so
``B`` is at the record's native channel resolution.

:meth:`Basis.certify` is the round trip's certificate, measured on a frame; :meth:`Basis.drift`
reads, against the existing noise floor, what a new record holds that the basis does not span.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import environment as _env

__all__ = ["Basis", "RoundTrip", "Drift"]


def _mode_indices(modes, K: int) -> np.ndarray:
    """``modes`` as distinct integer indices into ``K`` resolved modes, in the order first given.
    A repeated index names no further mode and is dropped; an index outside ``0 .. K-1`` (a
    negative one included) names no mode of this read and is refused, as is a non-integer one."""
    a = np.asarray(modes)
    if a.size == 0:
        return np.zeros(0, dtype=int)
    if a.dtype == bool or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"modes are integer indices into the {K} resolved modes; got {modes!r}")
    a = a.ravel().astype(int)
    bad = a[(a < 0) | (a >= K)]
    if bad.size:
        raise ValueError(f"modes {sorted(set(bad.tolist()))} are outside the {K} resolved modes "
                         f"(indices 0 .. {K - 1})")
    first = np.unique(a, return_index=True)[1]
    return a[np.sort(first)]


def _rank(s: np.ndarray, shape) -> int:
    # the round-off rank of an SVD: numpy.linalg.matrix_rank's default tolerance
    return int((s > s.max(initial=0.0) * max(shape) * np.finfo(float).eps).sum())


@dataclass(frozen=True)
class RoundTrip:
    """The write path's certificate on one frame.  Every figure is measured, none is a verdict.

    ``defect`` bounds ``inverse`` and ``idempotent`` (for ``BB^H = I + E``, ``P^2 - P = B^H E B``),
    so the two sitting at or under it, to round-off, certifies the round trip; ``lossless`` is the
    identity ``resolved + residual == W`` itself."""
    defect:     float   # ||B B^H - I||_2 -- the orthonormality of the rows
    inverse:    float   # ||encode(decode(A)) - A|| / ||A||, A = encode(frame): decode inverted
    idempotent: float   # ||split(resolved).residual|| / ||resolved||, whitened: a projection
    lossless:   float   # max |resolved + residual - W| / max |W| over the measured cells
    captured:   float   # ||A||^2 / ||whitened frame||^2 -- the share of the frame in the span


@dataclass(frozen=True)
class Drift:
    """What a record holds that a basis does not span, read against the existing noise floor.
    ``K`` modes above the floor are structure the basis misses; ``K == 0`` says it still spans
    the record at the level ``far``."""
    K:          int      # resolved modes in the residual
    contrast:   float    # the residual's top singular value over its floor
    share:      float    # the residual's share of the whitened record's energy
    projection: object   # the residual's :class:`projection.Projection`; None when the span fills
                         # every channel and nothing outside it is left to read
    identified: bool = True   # the record's noise was identified channel by channel; when False
                              # (few channels outside the span), the metric is partly the basis's
                              # and the floor's level is not held


@dataclass(frozen=True)
class Basis:
    """``K`` orthonormal rows ``B`` over ``F`` channels, with the per-channel ``centre`` and noise
    ``scale`` that define the coordinates they are orthonormal in, the ``power`` (whitened variance
    per row) of the record they were read from, and that read's ``far``.  numpy arrays.  A channel
    the record never measured has a zero column in ``B`` and a NaN ``centre``: it carries nothing
    to encode, and it decodes to NaN."""
    B:      np.ndarray    # (K, F) orthonormal rows, strongest first
    centre: np.ndarray    # (F,)  per-channel centre (NaN on a channel never measured)
    scale:  np.ndarray    # (F,)  per-channel noise scale (1 on a channel never measured)
    power:  np.ndarray    # (K,)  whitened variance of the source record along each row
    far:    float = 0.05  # the false-alarm level of the read the basis came from
    source: np.ndarray | None = None   # (T, F) the record the basis was read from, in W's units:
                                       # what ``drift`` compares a new record against

    @property
    def K(self) -> int:
        return int(self.B.shape[0])

    @property
    def F(self) -> int:
        return int(self.B.shape[1])

    @property
    def _active(self) -> np.ndarray:
        # a channel never measured has no centre; a channel with no noise is its centre exactly
        return np.isfinite(self.centre) & (self.scale > 0)

    def _frame(self, W, mask=None) -> np.ndarray:
        W = np.asarray(_env.to_numpy(W))
        if W.ndim != 2 or int(W.shape[1]) != self.F:
            raise ValueError(f"a basis over {self.F} channels codes (T, {self.F}) frames; "
                             f"got shape {W.shape}")
        if mask is not None:
            W = np.where(np.asarray(_env.to_numpy(mask), bool), np.nan, W)
        return W

    def _whiten(self, W, mask=None) -> np.ndarray:
        W = self._frame(W, mask)
        with np.errstate(divide="ignore", invalid="ignore"):
            X = (W - self.centre[None, :]) / self.scale[None, :]
        return np.where(np.isfinite(np.abs(X)), X, np.nan)

    def encode(self, W, mask=None) -> np.ndarray:
        """The mode coordinates ``A`` ``(T, K)`` of a frame ``(T, F)``.  A row with an unmeasured
        cell (non-finite, or ``True`` in ``mask``) on a channel the basis carries is solved by least
        squares over its measured cells: a missing cell constrains nothing."""
        X = self._whiten(W, mask)
        act = self._active
        Xa, Ba = X[:, act], self.B[:, act]
        obs = np.isfinite(Xa)
        Bh = Ba.conj().T
        A = np.zeros((X.shape[0], self.K), dtype=np.result_type(Xa, Ba))
        full = obs.all(axis=1)
        A[full] = Xa[full] @ Bh
        for t in np.flatnonzero(~full):
            o = obs[t]
            if o.any() and self.K:
                A[t] = np.linalg.lstsq(Ba[:, o].T, Xa[t, o], rcond=None)[0]
        return A

    def decode(self, A) -> np.ndarray:
        """The frame ``(T, F)``, in the original units, that the coordinates ``A`` describe."""
        A = np.asarray(_env.to_numpy(A))
        return (A @ self.B) * self.scale[None, :] + self.centre[None, :]

    def split(self, W, mask=None):
        """``(resolved, residual)`` with ``resolved + residual == W``: the frame's projection onto
        the basis, and what the basis does not span.  Both are NaN where ``W`` was not measured
        and on a channel the basis never measured."""
        Wn = np.asarray(_env.to_numpy(W))
        if mask is not None:
            Wn = np.where(np.asarray(_env.to_numpy(mask), bool), np.nan, Wn)
        resolved = self.decode(self.encode(Wn))
        resolved = np.where(np.isfinite(np.abs(Wn)), resolved, np.nan)
        return resolved, Wn - resolved

    def select(self, modes) -> "Basis":
        """The basis restricted to rows ``modes`` (indices ``0 .. K-1`` into ``B``, in the order
        given; a repeat is dropped, so the rows stay orthonormal)."""
        i = _mode_indices(modes, self.K)
        return Basis(B=self.B[i], centre=self.centre, scale=self.scale, power=self.power[i],
                     source=self.source,
                     far=self.far)

    def certify(self, W, mask=None) -> RoundTrip:
        """The round trip's certificate on the frame ``W``: see :class:`RoundTrip`."""
        act = self._active
        Ba = self.B[:, act]
        E = Ba @ Ba.conj().T - np.eye(self.K)
        defect = float(np.linalg.norm(E, 2)) if self.K else 0.0
        Wn = np.asarray(_env.to_numpy(W))
        if mask is not None:
            Wn = np.where(np.asarray(_env.to_numpy(mask), bool), np.nan, Wn)
        A = self.encode(Wn)
        nA = float(np.linalg.norm(A))
        inverse = float(np.linalg.norm(self.encode(self.decode(A)) - A)) / nA if nA > 0 else 0.0
        resolved, residual = self.split(Wn)
        R2 = self.split(resolved)[1][:, act] / self.scale[None, act]      # a residual has no centre
        Rw = self._whiten(resolved)[:, act]
        nR = float(np.sqrt(np.nansum(np.abs(Rw) ** 2)))
        idem = float(np.sqrt(np.nansum(np.abs(R2) ** 2))) / nR if nR > 0 else 0.0
        meas = np.isfinite(np.abs(Wn)) & np.isfinite(np.abs(resolved))
        scale_W = float(np.max(np.abs(Wn[meas]))) if meas.any() else 0.0
        err = float(np.max(np.abs((resolved + residual - Wn)[meas]))) if meas.any() else 0.0
        X = self._whiten(Wn)[:, act]
        nX2 = float(np.nansum(np.abs(X) ** 2))
        captured = nA ** 2 / nX2 if nX2 > 0 else 0.0
        return RoundTrip(defect=defect, inverse=inverse, idempotent=idem,
                         lossless=err / scale_W if scale_W > 0 else 0.0, captured=captured)

    def drift(self, W, mask=None, *, far: float | None = None, null=None, seed: int = 0) -> Drift:
        """What the record ``W`` holds that this basis does not span, read against the existing
        noise floor at ``far`` (default: the level the basis was read at).

        The span is the basis's; the noise metric is the record's own.  A basis carried to a new
        record carries its source's noise-scale estimates, and their sampling error, mixed across
        channels by any rotation, is correlated structure the floor would count.  So each channel's
        noise is solved from the record's own residual off the span (:func:`noise_scale`, exact in
        any metric), the span is re-expressed in that metric, and the residual is rotated into the ``F - K``
        dimensions the span leaves -- coordinates of its complement, built from the span's own
        Householder reflectors, so no ``F x F`` matrix is formed -- and read there by
        :class:`projection.Projection`, whose floor is sized for that ensemble when the record's
        noise is identified channel by channel.  When the channels outnumber what the residual can
        tell apart, the metric is partly the basis's, the floor's level is not held, and
        ``Drift.identified`` is False.  A channel neither record measured, or with no noise, is left
        out; a row with a gap is masked.

        The floor answers the question drift asks -- is this record the same process as the one
        the basis was read from? -- by comparing the two, not by testing the residual against
        independent channels: a process whose modes the basis did not all resolve leaves a
        correlated bulk in every record's residual, the source's included, and an independent-
        channel floor reads that bulk as drift in every record (70% of same-process records, on a
        42-mode process read at F/T = 0.5).  The source's rows are carried into the same complement
        coordinates, pooled with the record's, and the floor is the exact rank over reads of random
        subsets of the pool the record's size: under the null that both records are one process
        of exchangeable rows the record is one such subset, so its level is ``far`` whatever the
        process's correlation.  A caller's ``null`` replaces that floor, and a basis without its
        source record takes the independent-channel floor.  Rows with a gap are left out of the
        comparison."""
        from .projection import Projection
        far = self.far if far is None else float(far)
        Wf = self._frame(W, mask)
        X = self._whiten(Wf)
        on = self._active & np.isfinite(X).any(axis=0)
        gap = ~np.isfinite(X[:, on]).all(axis=1)
        full = ~gap
        Bs = self.B[:, on] * self.scale[None, on]            # the span, in W's units
        D = np.where(np.isfinite(Wf[:, on]), Wf[:, on] - self.centre[None, on], 0.0)
        s_new, identified = _noise_scale(D[full], Bs, self.scale[on], far)
        live = s_new > 0
        D, Bs, s_new = D[:, live], Bs[:, live], s_new[live]
        Xa = D / s_new[None, :]
        Fo = int(Xa.shape[1])
        r = 0
        Ba = np.zeros((0, Fo), dtype=np.result_type(Bs, float))
        if self.K:
            _, sv, Vh = np.linalg.svd(Bs / s_new[None, :], full_matrices=False)
            r = _rank(sv, Bs.shape)
            Ba = Vh[:r]                                       # the span, orthonormal in this metric
        h = tau = None
        if r:
            # R Ba^H = 0 row-wise, i.e. each residual row, as a column, is orthogonal to the
            # columns of Ba^T: those are the span to factor out (Ba^H would be its conjugate)
            h, tau = np.linalg.qr(Ba.T, mode="raw")           # Q = H_1 ... H_r, reflectors in h

        def complement(Y):
            """Rows ``Y`` (in this metric) off the span, in the complement's coordinates."""
            Rt = (Y - (Y @ Ba.conj().T) @ Ba).T if r else Y.T.copy()
            Rt = Rt.astype(np.result_type(Rt, Ba), copy=True)
            for i in range(r):
                v = np.zeros(Fo, dtype=h.dtype)
                v[i] = 1.0
                v[i + 1:] = h[i, i + 1:]
                Rt -= np.conj(tau[i]) * np.outer(v, v.conj() @ Rt)      # apply H_i^H
            return Rt[r:].T

        R = complement(Xa)                                    # the complement's coordinates
        if R.shape[1] == 0:                                   # the span fills every channel left:
            return Drift(K=0, contrast=0.0, share=0.0, projection=None,   # nothing outside to read
                         identified=identified)
        m = np.repeat(gap[:, None], R.shape[1], axis=1) if gap.any() else None
        if null is None and self.source is not None:
            # the source's rows, measured in full, carried into the same coordinates
            Sw = self._frame(self.source)
            Ds = Sw[:, on] - self.centre[None, on]
            Ds = Ds[np.isfinite(Ds).all(axis=1)][:, live] / s_new[None, :]
            if Ds.shape[0]:
                null = _two_sample_floor(R[full], complement(Ds), far, seed)
                R, m = R[full], None
        sc = Projection(R, mask=m, far=far, null=null, seed=seed)
        tot = float(np.sum(np.abs(Xa[full]) ** 2))
        share = float(np.sum(np.abs(R[full]) ** 2)) / tot if tot > 0 else 0.0
        contrast = float(sc.sigma_top / sc.noise_floor) if sc.noise_floor > 0 else 0.0
        return Drift(K=int(sc.K_signal), contrast=contrast, share=share, projection=sc,
                     identified=identified)


def _two_sample_floor(R_new: np.ndarray, R_src: np.ndarray, far: float, seed: int):
    """A floor provider for the screen of ``R_new``: the exact rank over the same read of random
    subsets, of ``R_new``'s size, of the pooled rows of both records -- each subset read as a
    :class:`projection.Projection` (whitened, its fold decided on it) and scored as its screen's
    standardized deviate, as the exact permutation floor scores its draws."""
    from .projection import Projection
    from .null_providers import (fewest_draws, _exact_rank_score, _scored, _observed_floor, mp)
    pool = np.vstack([R_new, R_src])
    n = int(R_new.shape[0])

    def provider(ctx):
        rng = np.random.default_rng(seed)
        draws = fewest_draws(ctx.far)

        def draw():
            sub = pool[rng.permutation(pool.shape[0])[:n]]
            return _scored(np.asarray(_env.to_numpy(Projection(sub, null=mp).screen)), "projection")
        score = _exact_rank_score(draw, draws, ctx.far)
        return _observed_floor(score, np.asarray(_env.to_numpy(ctx.data)), "projection")
    provider.__name__ = "two_sample_null"
    return provider


def noise_scale(D: np.ndarray, Bs: np.ndarray, s0: np.ndarray, far: float = 0.05) -> np.ndarray:
    """Each channel's noise scale, from what a span leaves of the centred record ``D`` ``(T, F)``;
    see :func:`_noise_scale`, which also says whether the data resolved every channel's noise."""
    return _noise_scale(D, Bs, s0, far)[0]


def _noise_scale(D: np.ndarray, Bs: np.ndarray, s0: np.ndarray, far: float = 0.05):
    """``(scale, identified)``: each channel's noise scale, from what a span leaves of the centred
    record ``D`` ``(T, F)``, and whether the data resolved every channel's noise at level ``far``.

    The span is the rows ``Bs`` (in ``D``'s units), removed orthogonally in the metric ``s0``.  With
    noise independent across channels, variance ``sigma_g^2``, and ``x_g = sigma_g^2 / s0_g^2``, the
    residual's mean power in channel ``f`` is exactly

        m_f = sum_g A_fg x_g,        A = |I - P|^2 (elementwise),

    ``P`` the span's projector in the ``s0`` metric -- whatever ``s0`` is.  ``A`` is the Schur square
    of a projector, so it is symmetric and positive semidefinite, ``A = V diag(lambda) V^T``.

    **How well the data fix each direction.**  Each ``m_f`` is a mean over the ``T`` rows, and for
    Gaussian noise ``Cov(m_f, m_g) = (kappa / T) |Sigma_R,fg|^2`` (``kappa = 2`` real, 1 complex).
    Under the metric (``x = 1``) the residual's covariance ``Sigma_R`` is ``I - P`` itself, so the
    moments' noise is ``(kappa / T) A`` -- the system's own matrix -- and the estimate of ``x`` along
    eigendirection ``v_i`` has standard error ``sqrt(kappa / (T lambda_i))``.  A direction is
    **resolved** when the data would tell its variance apart from zero at the reader's level:

        lambda_i > kappa z^2 / T,     z the two-sided normal quantile at ``far``,

    paid for over the ``n`` directions (``far_n = 1 - (1 - far)^(1/n)``), as every look in this
    library is.  ``far`` is the only level; there is no other cut.

    **The estimate** is the exact moment solution along every direction ``A`` carries above its own
    round-off (at most the structural count of them), and the metric stands along the rest:

        x = 1 + sum_i v_i v_i^T (m - A 1) / lambda_i.

    Pinning a weakly resolved direction to the metric would trade its variance for the metric's
    bias, so the estimate keeps it; the **verdict** carries the resolution.  ``identified`` is True
    only when every channel's noise is fixed to within ``1 / z`` of itself: each channel's standard
    error, from the moments' covariance evaluated at the solved ``x`` (``(kappa/T) |M diag(x) M^H|^2``,
    ``M = I - P``, carried through ``A^-1``), is below ``x_f / z``.  On the Woodbury path the test is
    a rigorous sufficient bound, which can only say no more often.  A direction goes unresolved because the system is short of rank (the
    residual lives in ``F - K`` dimensions, and its covariance carries at most ``(F-K)(F-K+1)/2``
    numbers, ``(F-K)^2`` complex) or because a channel lies nearly inside the span, which leaves
    almost nothing of its noise to see.  A channel lying in the span has no residual at all, and a
    variance solved at or below zero is scatter around one the data does not resolve: the metric
    stands for both, and ``identified`` is False.

    ``A`` is never formed when it need not be: ``A`` is at least ``diag(1 - 2 p)`` (``p_f = P_ff``),
    so when every ``1 - 2 p_f`` clears the resolution bound, every direction is resolved and the
    system is solved exactly by Woodbury in O(F K^4) (``|P|^2`` has rank at most ``K^2``) -- the
    cheaper path wherever ``n > K^2``.  A channel that never moved (``s0 = 0``) reads 0.  With no
    span it is the plain RMS."""
    from .null_providers import _norm_ppf
    if not D.shape[0]:
        return np.zeros(D.shape[1]), False
    T = int(D.shape[0])
    live = s0 > 0
    X = D[:, live] / s0[None, live]
    Fl = int(X.shape[1])
    kappa = 1.0 if (np.iscomplexobj(D) and np.any(np.imag(D) != 0)) else 2.0
    Ba = np.zeros((0, Fl), dtype=np.result_type(X, Bs, float))
    if Bs.shape[0]:
        _, sv, Vh = np.linalg.svd(Bs[:, live] / s0[None, live], full_matrices=False)
        Ba = Vh[:_rank(sv, Bs[:, live].shape)]
    r = int(Ba.shape[0])
    R = X - (X @ Ba.conj().T) @ Ba if r else X
    m = np.mean(np.abs(R) ** 2, axis=0)
    identified = True
    if not r:
        x = m
    else:
        eps = np.finfo(float).eps
        p = np.sum(np.abs(Ba) ** 2, axis=0).real                        # P_ff
        use = (1.0 - p) > Fl * eps                                      # channels with a residual
        x = np.ones(Fl)
        Bu, mu, n = Ba[:, use], m[use], int(use.sum())
        room = Fl - r
        # |I - P|^2 has rank at most room(room+1)/2 for a real span and room^2 for a complex one; a
        # span is real exactly when its real and imaginary parts span nothing more than it does
        cx = bool(np.iscomplexobj(Ba) and np.any(np.imag(Ba) != 0))
        if cx:
            RI = np.concatenate([np.real(Ba), np.imag(Ba)], axis=0)
            cx = _rank(np.linalg.svd(RI, compute_uv=False), RI.shape) > r
        count = room * room if cx else room * (room + 1) // 2        # an upper bound on the rank
        xu = np.ones(0)
        if n:
            z = _norm_ppf(1.0 - 0.5 * (1.0 - (1.0 - far) ** (1.0 / n)))
            cut = kappa * z * z / T                                     # the resolution bound
            d = 1.0 - 2.0 * p[use]
            if n > r * r and np.min(d) > cut:
                # A >= diag(d) > cut I: full rank -- the exact solution, by Woodbury on
                # A = diag(d) + C C^H.  The verdict there is a rigorous sufficient bound, since the
                # plug-in covariance below would need n x n matrices: ||A^-1|| <= 1/min(d) and
                # ||Cov m|| <= (kappa/T) max(x)^2, so every channel's standard error is at most
                # sqrt(kappa/T) max(x) / min(d); it vouches for the record only when that is
                # below min(x) / z.  It never calls a record identified that the exact test
                # would not.
                C = (Bu[:, None, :] * Bu.conj()[None, :, :]).reshape(r * r, n).T
                Cd = C / d[:, None]
                inner = np.eye(r * r) + C.conj().T @ Cd
                xu = np.real(mu / d - Cd @ np.linalg.solve(inner, Cd.conj().T @ mu))
                if np.all(xu > 0.0):
                    identified = bool(z * np.sqrt(kappa / T) * xu.max() / xu.min() < np.min(d))
                else:
                    identified = False
            else:
                A = np.abs(np.eye(n) - Bu.conj().T @ Bu) ** 2
                lam, V = np.linalg.eigh(A)
                lam, V = lam[::-1], V[:, ::-1]                          # largest first
                # the estimate uses every direction A carries above its own round-off (2 r eps
                # relative per entry, over n entries), at most the structural count of them
                k = min(count, int(np.sum(lam > lam[0] * 2 * r * n * eps)))
                g = (V[:, :k].T @ (mu - A @ np.ones(n))) / lam[:k]
                xu = 1.0 + V[:, :k] @ g
                # the verdict: every channel's noise fixed to within 1/z of itself, with the
                # moments' covariance evaluated at the solved x -- (kappa/T)|M diag(x) M^H|^2,
                # M = I - P -- not at the metric, so a mis-set metric cannot flatter it
                identified = False
                if k == n and np.all(xu > 0.0):
                    M = np.eye(n) - Bu.conj().T @ Bu
                    Cm = (kappa / T) * np.abs((M * xu[None, :]) @ M.conj().T) ** 2
                    Ainv = (V / lam[None, :]) @ V.T
                    se = np.sqrt(np.clip(np.diag(Ainv @ Cm @ Ainv), 0.0, None))
                    identified = bool(np.all(se < xu / z))
            x[use] = np.where(xu > 0.0, xu, 1.0)
        identified = bool(identified and use.all() and np.all(xu > 0.0))
    s = np.zeros(D.shape[1])
    s[live] = np.sqrt(x) * s0[live]
    return s, identified


def basis_of(sc, modes=None) -> Basis:
    """The :class:`Basis` of the projection ``sc``'s resolved modes (all ``K_signal`` by default;
    ``modes`` selects screen modes by index).  The implementation of :meth:`Aperture.basis`."""
    from .extract import _adjoint_lift, _orth, _whitened_live
    U = np.asarray(_env.to_numpy(sc.U))
    N = int(sc.screen.shape[0])
    k_idx = (np.arange(int(sc.K_signal)) if modes is None
             else _mode_indices(modes, int(sc.K_signal)))
    W, rows, cols, Z, centre, scale, missing = _whitened_live(sc)
    F = int(W.shape[1])
    QL = _orth(_adjoint_lift(Z.shape[0], N, U[:, k_idx]))
    # The screen's scale is each channel's robust spread, signal included.  The basis codes frames
    # in the NOISE metric: the channel span is read on the screen's metric (its directions average
    # over every row, so they are precise), and each channel's noise is solved from what that span
    # leaves (``noise_scale``, exact in any metric), over the rows measured in full.
    G = QL.conj().T @ Z
    Bs = np.zeros((0, Z.shape[1]), dtype=Z.dtype)
    if G.shape[0]:
        _, s, Vh = np.linalg.svd(G, full_matrices=False)
        Bs = Vh[:_rank(s, G.shape)] * scale[None, :]
    D = np.where(missing, 0.0, W[np.ix_(rows, cols)] - centre[None, :])
    scale = noise_scale(D[~missing.any(axis=1)], Bs, scale, float(sc._far))
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(missing | (scale[None, :] == 0), 0.0, D / scale[None, :])
    G = QL.conj().T @ Z                                        # the resolved part's channel images
    if G.shape[0]:
        _, s, Vh = np.linalg.svd(G, full_matrices=False)
        r = _rank(s, G.shape)
        Bl, pw = Vh[:r], s[:r] ** 2 / Z.shape[0]
    else:
        Bl, pw = np.zeros((0, Z.shape[1]), dtype=Z.dtype), np.zeros(0)
    B = np.zeros((Bl.shape[0], F), dtype=Bl.dtype)
    B[:, cols] = Bl
    centre_f = np.full(F, np.nan, dtype=centre.dtype)
    scale_f = np.ones(F)
    centre_f[cols], scale_f[cols] = centre, scale
    from .extract import _flat_centres
    for f, c in _flat_centres(sc, W).items():    # no noise scale: not coded, decodes to its centre
        centre_f[f], scale_f[f] = c, 0.0
    return Basis(B=B, centre=centre_f, scale=scale_f, power=pw, far=float(sc._far),
                 source=np.array(_env.to_numpy(sc.W)))
