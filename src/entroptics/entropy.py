"""
entropy.py -- Entropy geometry: the matched SCALE read from a signal's own
Shannon entropy, plus the fold (fractional resample) and normalize.

This is the ENTROPY side of Entroptics.  The OPTICS side is reads.py / aperture.py;
the PROJECTION side is screen.py.  Standalone: numpy only.

Axis convention for every 2-D input W (shape (T, F)):
  * axis-0 (rows, "T") is the ORDERED / evolution axis   -> subscript _T
  * axis-1 (cols, "F") is the feature / channel axis      -> subscript _F
"time"/"frequency" are ROLES, not literal physics -- any 2-D array with one
ordered axis works (spectrogram, waterfall, embedding stack, market panel, image).

Per-axis geometry symbols (a = T or F):
  H_a      Shannon entropy (bits) of that axis' power marginal
  n_a      effective mode count = round(2^H_a)
  delta_a  matched cell scale  = len_a / 2^H_a   (delta_T window width, delta_F bin width)
"""
from __future__ import annotations

import numpy as np

from . import environment as _env

try:
    from scipy.stats import norm as _scipy_norm
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False
    _scipy_norm = None  # type: ignore[assignment]

# MAD_SCALE: median-absolute-deviation -> Gaussian sigma.  Exact 1/Phi^{-1}(0.75)
#   from scipy.stats.norm; falls back to 1.4826 (5-dp) when scipy is unavailable.
MAD_SCALE: float = (
    float(1.0 / _scipy_norm.ppf(0.75)) if _SCIPY_AVAILABLE else 1.4826
)

# MAD_LOGVAR: the asymptotic sampling variance of log(MAD-hat) from N Gaussian samples is
# ~ MAD_LOGVAR / N.  This is the analytic influence-function variance 1/(16 f(D)^2 D^2) =
# 1.36046, evaluated at D = Phi^{-1}(3/4) = 0.67449 with f the standard-normal density
# (equivalently CV(MAD/sigma) = sqrt(1.36046)/sqrt(N) = 1.166/sqrt(N)).  DERIVED like
# MAD_SCALE, not fitted; sets the shrinkage of noisy small-N per-channel scales toward the
# pooled one (see normalize / _shrink_mad).
MAD_LOGVAR: float = 1.36046


def shannon_bits(weights) -> float:
    """Shannon entropy (bits) of a non-negative weight array -- the ONE definition
    used across Entroptics (geometry marginals, screen mode weights, decay/axis
    spectra).  Normalises internally; returns a Python float.  Backend-agnostic
    (numpy or torch).  H(w) = -sum p log2 p, p = w / sum w."""
    xp = _env.ns(weights)
    s = float(_env.sum_ax(xp, weights))
    if s <= 1e-30:
        return 0.0
    p = weights / s
    return -float(_env.sum_ax(
        xp, xp.where(p > 0, p * xp.log2(_env.cliprange(xp, p, 1e-12, 1.0)), xp.zeros_like(p))))


def geometry(W: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Read the natural matched scale of a 2-D observation W (T x F) from its own
    Shannon entropy.

    H_T (ordered) and H_F (feature) are the Shannon entropies of the global power
    marginals of P = |W|^2:

        p_t[t] = sum_f P[t,f] / sum P        (ordered marginal)
        p_f[f] = sum_t P[t,f] / sum P        (feature marginal)

    Power-weighting lets bright (on-signal) rows dominate regardless of what
    fraction of the capture they occupy.

    Returns
    -------
    dict with per-axis keys (a = T, F):
        H_T, H_F          : marginal Shannon entropies (bits)
        n_T, n_F          : effective mode counts = round(2^H_a)
        delta_T, delta_F  : matched cell scales.  delta_T := 1.0 ALWAYS (the ordered axis
                            is never folded, see FOLD POLICY below); only delta_F carries the
                            entropy ratio len_F / 2^H_F (float >= 1, the feature bin width).

    delta_F is the REAL (un-floored) matched-scale ratio -- one parameter-free scale derived
    from the signal's own Shannon entropy.  The feature fold (normalize/project)
    resamples fractionally to round(2^H_F) cells, or takes the exact-integer reshape fast
    path when the ratio is whole.

    FOLD POLICY: only the FEATURE axis folds; the ORDERED axis is kept at native resolution.
    The ordered reads -- coherence (adjacent-row similarity), the decay/OTF (lag structure),
    the exact rates (the ordered trajectory) -- all require native ordered spacing, and
    folding the ordered axis would blend adjacent rows into spurious correlation.  For the
    feature axis, a near-uniform marginal (structureless noise) sits below its max log2(F)
    only by a finite-sample deficit, so we snap to NO FOLD (delta_F = 1.0) whenever H_F lies
    within a band of log2(F).  The band is the conservative Miller-Madow uniform-null bias
    (F-1)/(2 T ln2), CAPPED at log2(F)/2: deliberately large (neither noise nor a low-rank
    signal's delocalised marginal folds -- folding either would blur the SVD modes) but capped
    so it never exceeds log2(F) and disables the fold vacuously for wide-short data.  The cap
    means the guard always folds once power concentrates below sqrt(F) effective channels.
    Closed-form; the only choice is the sqrt(F) concentration floor.  See
    research/validation/miller_madow_check.py.
    """
    xp = _env.ns(W)                      # numpy or torch -- ONE code path (GPU when fed a tensor)
    T, F = int(W.shape[0]), int(W.shape[1])

    P = _env.asnum(xp.abs(W))           # |W| as float (real, complex, negative all fine)
    P = xp.where(xp.isfinite(P), P, xp.zeros_like(P))   # NaN/Inf -> 0 (handles gaps)
    if mask is not None:
        P = xp.where(mask, xp.zeros_like(P), P)
    P = P * P                           # weight peaks, suppress noise floor

    P_total = float(_env.sum_ax(xp, P))
    if P_total <= 0:                     # no signal -> maximal entropy (fully spread; no fold)
        H_F = float(np.log2(F))
        H_T = float(np.log2(T))
    else:
        H_F = shannon_bits(_env.sum_ax(xp, P, 0))   # entropy of the feature power marginal
        H_T = shannon_bits(_env.sum_ax(xp, P, 1))   # entropy of the ordered power marginal

    ln2 = float(np.log(2.0))
    # FEATURE-axis fold guard.  A structureless (uniform-power) feature marginal sits below its
    # max log2(F) only by the finite-sample deficit; we snap to NO FOLD when H_F lies within a
    # band of log2(F).  The band is the CONSERVATIVE Miller-Madow uniform-null bias
    # (F-1)/(2 T ln2) -- deliberately large (it exceeds the mean deficit) so that neither
    # structureless noise NOR a genuine low-rank signal's delocalised feature marginal folds
    # (folding either would blur the SVD modes the feature reads resolve).  That band, however,
    # grows without bound in F/T and EXCEEDS log2(F) for wide-short data (F >~ 2 T ln F),
    # disabling the fold vacuously (a fully redundant marginal could not fold).  We therefore
    # CAP it at half of log2(F): the guard never suppresses a fold once power concentrates below
    # sqrt(F) effective channels (deficit > log2(F)/2), keeping it operative at every shape while
    # leaving the conservative band unchanged wherever it is already below the cap (all tall/
    # square shapes -- so noise and low-rank signals still never fold there).  See
    # research/validation/miller_madow_check.py.  Closed-form; the only choice is the sqrt(F)
    # concentration floor (a stated criterion, not fitted).
    band_F = min((F - 1) / (2.0 * max(1, T) * ln2), 0.5 * float(np.log2(F)))
    if H_F >= np.log2(F) - band_F:               # indistinguishable from uniform -> no fold
        n_F, delta_F = F, 1.0
    else:
        n_F_real = min(float(F), 2.0 ** H_F)
        n_F = max(1, min(F, int(round(n_F_real))))
        delta_F = max(1.0, F / n_F_real)
    # The ORDERED axis is kept at native resolution (never folded): the ordered reads --
    # coherence (adjacent-row similarity, section 5), the decay/OTF (lag structure, section
    # 4), and the exact rates (the ordered trajectory, section 9) -- all require native
    # ordered spacing; folding it would blend adjacent rows into spurious correlation
    # (corrupting the coherence read) and blur the SVD modes.  Only the feature axis folds.
    n_T, delta_T = T, 1.0

    return {
        "H_T": H_T, "n_T": n_T, "delta_T": delta_T,   # ordered axis
        "H_F": H_F, "n_F": n_F, "delta_F": delta_F,   # feature axis
    }


def downsample(A: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """COARSEN ``A`` along ``axis`` to ``n_out`` cells (the scale > 1 regime,
    n_out <= n_in): area-weighted MEAN resample.  Exact block-average when the
    factor is an integer (byte-identical); area-weighted (cumsum-interpolated)
    otherwise.  Level-preserving (the per-cell value, not the summed energy).
    Real- and complex-safe; vectorised.  ``n_out == n_in`` returns ``A`` unchanged.
    Backend-agnostic (numpy or torch)."""
    xp = _env.ns(A)
    A = _env.movedim(xp, A, axis, 0)
    n_in = int(A.shape[0])
    if n_out == n_in or n_in == 0:
        return _env.movedim(xp, A, 0, axis)
    cplx = xp.is_complex(A) if _env.is_torch(xp) else np.iscomplexobj(A)
    sh = tuple(int(s) for s in A.shape[1:])
    flat = _env.asnum(A.reshape(n_in, -1), complex=cplx)
    ref = flat if _env.is_torch(xp) else None
    z = _env.zeros(xp, (1, int(flat.shape[1])), complex=cplx, ref=ref)
    cs = _env.cat0(xp, [z, _env.cumsum0(xp, flat)])
    edges = _env.linspace(xp, 0.0, float(n_in), n_out + 1, ref=ref)
    lo = _env.cliprange(xp, _env.floor_int(xp, edges), 0, n_in)
    fr = (edges - lo)[:, None]
    cs_lo = cs[lo]
    interp = cs_lo + fr * (cs[_env.cliprange(xp, lo + 1, 0, n_in)] - cs_lo)
    binned = (interp[1:] - interp[:-1]) / (edges[1:] - edges[:-1])[:, None]
    out = binned.reshape((n_out,) + sh)
    return _env.movedim(xp, out if cplx else xp.real(out), 0, axis)


def upsample(A: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """REFINE ``A`` along ``axis`` to ``n_out`` cells (the scale < 1 regime,
    n_out >= n_in): nearest-block HOLD resample (== ``np.repeat`` for an integer
    factor, so the integer-matched inverse fold is unchanged).  Level-preserving.
    ``n_out == n_in`` returns ``A`` unchanged.  Backend-agnostic (numpy or torch)."""
    xp = _env.ns(A)
    A = _env.movedim(xp, A, axis, 0)
    n_in = int(A.shape[0])
    if n_out == n_in or n_in == 0:
        return _env.movedim(xp, A, 0, axis)
    idx = _env.cliprange(xp, (_env.arange_int(xp, n_out, ref=A) * n_in) // n_out, 0, n_in - 1)
    return _env.movedim(xp, A[idx], 0, axis)


def live_view(W, mask: np.ndarray | None = None):
    """The read-side analogue of ``Screen``'s ignore-missing: DROP fully-dead rows/cols
    (every cell missing or masked) and fill any remaining scattered missing cell with
    its column mean, returning a clean, NaN-free array so the correlation / SVD reads
    never see a gap.  No missing data -> returns ``W`` unchanged (backend preserved);
    otherwise returns a numpy array."""
    xp = _env.ns(W)
    bad = ~xp.isfinite(xp.abs(W))
    if mask is not None:
        bad = bad | mask
    if not bool(bad.any()):
        return W
    Wn = np.asarray(_env.to_numpy(W))
    b = np.asarray(_env.to_numpy(bad), dtype=bool)
    if Wn.ndim == 2:
        live_r, live_c = ~b.all(axis=1), ~b.all(axis=0)
        if live_r.any() and live_c.any() and not (live_r.all() and live_c.all()):
            Wn, b = Wn[np.ix_(live_r, live_c)], b[np.ix_(live_r, live_c)]
    if b.any():                                  # scattered gaps -> column mean (0 after centring)
        with np.errstate(invalid="ignore", divide="ignore"):
            col = np.nanmean(np.where(b, np.nan, Wn), axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        Wn = np.where(b, col[None, :] if Wn.ndim == 2 else col, Wn)
    return Wn


def _shrink_mad(xp, mad, pos, typical: float, N: int, eps: float):
    """James-Stein shrinkage of the per-channel MAD toward the pooled scale ``typical``.

    The per-channel MAD from ``N`` rows has log-sampling-variance ``V_samp ~
    MAD_LOGVAR/N``.  Shrink each channel's log-MAD toward ``log(typical)`` by the
    DATA-DERIVED weight ``w = max(0, 1 - V_samp/V_obs)`` (``V_obs`` = observed
    cross-channel variance of the log-MADs, James-Stein / empirical Bayes).  When the
    channels are homoscedastic (``V_obs ~ V_samp``, e.g. iid noise with few rows)
    ``w -> 0`` and every channel gets the SAME pooled scale, so a noisy small-N MAD
    cannot disperse the whitened screen; when they genuinely differ (``V_obs >>
    V_samp``) ``w -> 1`` and full per-channel whitening is preserved (each channel is
    equalised to unit noise, which the asymmetric below-cap could not do).  Parameter-
    free.  Backend-agnostic (numpy or torch)."""
    if typical <= eps:
        return mad
    lm = xp.log(_env.clampmin(xp, mad, eps))
    lm0 = float(np.log(typical))
    if int(pos.sum()) > 1:
        V_obs = float(_env.std0(xp, lm[pos])) ** 2
        w = 0.0 if V_obs <= 0.0 else max(0.0, min(1.0, 1.0 - (MAD_LOGVAR / max(N, 1)) / V_obs))
    else:
        w = 0.0
    return xp.exp(lm0 + w * (lm - lm0))


def normalize(W: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-channel robust (MAD) WHITENING at native resolution -- give each feature
    channel a common, unit noise scale so the screen's noise floor is a clean iid
    reference.  This is normalization ONLY; the entropy-matched RESCALING of BOTH
    axes is done together by ``screen.project`` (feature and ordered in one call).

    Each channel's median is subtracted and it is divided by MAD * MAD_SCALE
    (robust sigma).  Because a per-channel MAD from few rows is noisy, each channel's
    scale is SHRUNK toward the pooled cross-channel scale by a data-derived weight
    (``_shrink_mad``): homoscedastic noise pools to one stable scale so a small-N MAD
    cannot disperse the screen and inflate the floor, while genuinely different
    channels are each equalised to unit noise.  No analyst-chosen floor.  Masked /
    non-finite cells are excluded from the statistics and marked MISSING (NaN), so ``project``'s fold averages only valid cells (a zero
    would drag the average toward the noise mean).

    Returns a (T, F) array (same shape as W) of whitened channels; masked cells NaN.
    """
    xp = _env.ns(W)
    is_complex = xp.is_complex(W) if _env.is_torch(xp) else np.iscomplexobj(W)
    data = _env.asnum(W, complex=is_complex)
    bad = ~xp.isfinite(xp.abs(data))
    if mask is not None:
        bad = bad | mask
    if bool(bad.any()):
        # robust masked / non-finite path (np.ma, numpy) -- the rare batch-with-gaps case.
        out = _normalize_masked_np(_env.to_numpy(W), _env.to_numpy(bad))
        if _env.is_torch(xp):
            import torch
            return torch.as_tensor(out, device=W.device)
        return out

    # clean path -- backend-agnostic (numpy on CPU, torch on its device).
    eps = 1e-12
    if is_complex:
        med = _env.median0(xp, xp.real(data)) + 1j * _env.median0(xp, xp.imag(data))
    else:
        med = _env.median0(xp, data)
    centered = data - med[None, :]
    mad = _env.median0(xp, xp.abs(centered)) * MAD_SCALE
    pos = mad > eps
    typical = float(_env.median1d(xp, mad[pos])) if bool(pos.any()) else 1.0
    mad_eff = _shrink_mad(xp, mad, pos, typical, int(data.shape[0]), eps)
    safe = mad_eff > eps
    zero = _env.zeros(xp, tuple(int(s) for s in data.shape), complex=is_complex,
                     ref=(data if _env.is_torch(xp) else None))
    return xp.where(safe[None, :], centered / xp.where(safe[None, :], mad_eff[None, :], 1.0), zero)


def _normalize_masked_np(W, bad):
    """Robust per-channel MAD whitening in numpy with a bad-cell mask (np.ma);
    masked / non-finite cells -> NaN.  The rare batch-with-gaps path (see normalize)."""
    is_complex = np.iscomplexobj(W)
    data = W.astype(np.complex128 if is_complex else np.float64).copy()
    mask = np.asarray(bad, bool)
    data[mask] = 0.0
    eps = 1e-12
    data_ma = np.ma.array(data, mask=mask)
    if is_complex:
        med = (np.ma.median(data_ma.real, axis=0, keepdims=True).filled(0.0)
               + 1j * np.ma.median(data_ma.imag, axis=0, keepdims=True).filled(0.0))
    else:
        med = np.ma.median(data_ma, axis=0, keepdims=True).filled(0.0)
    centered = data - med
    centered_ma = np.ma.array(centered, mask=mask)
    mad_raw = (np.ma.median(np.abs(centered_ma), axis=0, keepdims=True).filled(0.0)) * MAD_SCALE
    posm = mad_raw > eps
    typical_mad = float(np.median(mad_raw[posm])) if np.any(posm) else 1.0
    # James-Stein shrink toward the pooled scale (see _shrink_mad), with a PER-CHANNEL
    # sampling variance: channel j with n_j VALID cells has V_samp = MAD_LOGVAR/n_j, so a
    # heavily-gapped channel (noisier MAD, fewer valid cells) is shrunk harder toward the
    # pooled scale -- unlike a single row count, which would under-shrink gapped channels.
    n_valid = np.maximum((~mask).sum(axis=0, keepdims=True).astype(float), 1.0)   # (1, F)
    if typical_mad > eps and int(np.count_nonzero(posm)) > 1:
        lm = np.log(np.maximum(mad_raw, eps)); lm0 = float(np.log(typical_mad))
        V_obs = float(np.var(lm[posm]))
        w = (np.zeros_like(mad_raw) if V_obs <= 0.0
             else np.clip(1.0 - (MAD_LOGVAR / n_valid) / V_obs, 0.0, 1.0))
        mad_eff = np.exp(lm0 + w * (lm - lm0))
    else:
        mad_eff = np.maximum(mad_raw, typical_mad)
    safe = mad_eff > eps
    out = np.where(safe, centered / np.where(safe, mad_eff, 1.0), 0.0)
    out[mask] = np.nan
    return out
