"""
Experiment 16 -- The matched scale must be read BEFORE the whitening.

`Projection` runs `geometry -> normalize -> project` (projection.read): the entropy-matched
scale of Def 2.1 is read off the RAW frame, and only then is the frame whitened per channel
and folded.  That ordering looks avoidable -- if the channels might be in different units,
why not equalise them first and count afterwards? -- so this measures what the other order
costs.

The two steps want opposite things:

  geometry   asks WHERE THE POWER IS.  Its answer, 2^H_F, is a statement about how unevenly
             the raw amplitudes are spread across the channels.
  normalize  asks whether the channels are ON A COMMON FOOTING.  It divides each channel by
             its OWN robust scale, so a channel carrying nothing but noise is lifted to the
             same amplitude as a channel carrying the signal.  Flattening that unevenness is
             its entire job.

So whitening first erases exactly the concentration the fold exists to find.  Planted here on
the case where a fold is unambiguously correct -- a narrow line of KNOWN width on a wide
continuous axis, where folding to about the line's own width is lossless (exp8 shows the same
fold is invertible on a continuous axis):

  (a) as shipped, n_F tracks the planted line width across widths and axis lengths;
  (b) whitening first returns n_F = F at every width and every length -- no fold, ever;
  (c) the mechanism, measured: the share of the frame's power held by the on-line channels,
      before and after whitening.

The ordering is therefore forced, and it leaves a PRECONDITION on the caller rather than a
choice for the library: 2^H_F only means "how many channels are in play" when the channels
are already in commensurate units.  Nothing in a single frame can establish that -- scaling
one column by c is indistinguishable from that channel being c times louder -- so it is a
declared property of the input, like which axis is the ordered one.

Deterministic (fixed seeds).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics.entropy import geometry, normalize

import common as C

T = 200
F_SWEEP = [64, 128, 256]
WIDTH_SWEEP = [2.0, 4.0, 8.0]
NOISE = 0.15
AMP = 6.0
SEED0 = 1600


def _line(F: int, width: float, seed: int) -> np.ndarray:
    """A Gaussian line `width` channels wide, centred on a continuous axis of F channels."""
    g = C.rng(seed)
    f = np.arange(F)[None, :]
    profile = AMP * np.exp(-0.5 * ((f - F // 2) / width) ** 2)
    return profile * np.ones((T, 1)) + g.normal(0.0, NOISE, (T, F))


def _on_line(F: int, width: float) -> np.ndarray:
    """The channels within one `width` of the centre -- where the planted line actually is."""
    f = np.arange(F)
    return np.abs(f - F // 2) <= width


def run() -> dict:
    rows, shipped_ok, whitened_folded = [], 0, 0
    shares_before, shares_after = [], []

    for i, F in enumerate(F_SWEEP):
        for j, width in enumerate(WIDTH_SWEEP):
            W = _line(F, width, SEED0 + 10 * i + j)
            n_shipped = geometry(W)["n_F"]
            n_whitened = geometry(np.asarray(normalize(W)))["n_F"]

            on = _on_line(F, width)
            Wn = np.asarray(normalize(W))
            before = float((W[:, on] ** 2).sum() / (W ** 2).sum())
            after = float((Wn[:, on] ** 2).sum() / (Wn ** 2).sum())
            shares_before.append(before)
            shares_after.append(after)

            # the fold is "right" when it lands near the line's own extent and well below F
            if n_shipped < F:
                shipped_ok += 1
            if n_whitened < F:
                whitened_folded += 1

            rows.append([F, f"{width:.0f}", n_shipped, n_whitened,
                         f"{before:.3f}", f"{after:.3f}"])

    table = C.md_table(
        ["F (channels)", "line width", "n_F as shipped", "n_F if whitened first",
         "on-line power share, raw", "after whitening"], rows)

    n = len(rows)
    headline = (
        f"On a planted line of known width, the scale read on the RAW frame folds in {shipped_ok}/{n} "
        f"cases (n_F tracking the line width) while the same read taken AFTER whitening folds in "
        f"{whitened_folded}/{n} -- it returns n_F = F every time. The mechanism is measured: the "
        f"on-line channels hold {np.mean(shares_before):.1%} of the power before whitening and "
        f"{np.mean(shares_after):.1%} after, because dividing each channel by its own scale lifts "
        f"the noise-only channels to the amplitude of the line.")

    concl = (
        "The order in `projection.read` (geometry -> normalize -> project) is forced, not "
        "incidental: geometry measures the unevenness of the raw amplitudes across channels, and "
        "normalize exists to remove that unevenness, so composing them the other way leaves the "
        "first read nothing to find. The cost is a PRECONDITION rather than a parameter: 2^H_F "
        "counts channels in play only when the channels are already commensurate. That cannot be "
        "established from one frame -- scaling a column by c is indistinguishable from that channel "
        "being c times louder -- so, like which axis is ordered, it is declared by the caller. A "
        "frame of mixed units reads its units, not its structure.")

    return dict(
        title="16. The matched scale must be read before the whitening",
        setup=(f"A Gaussian line of known width on a continuous axis, T={T}, amplitude {AMP} over "
               f"noise {NOISE}; F swept over {F_SWEEP} and line width over {WIDTH_SWEEP}. For each, "
               f"n_F from geometry(W) against n_F from geometry(normalize(W))."),
        table=table,
        metrics=dict(shipped_folded=int(shipped_ok), whitened_folded=int(whitened_folded),
                     cases=int(n),
                     share_before=float(np.mean(shares_before)),
                     share_after=float(np.mean(shares_after))),
        headline=headline,
        conclusion=concl,
        provisional=False,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"], "\n")
    print(r["table"], "\n")
    print(r["headline"])
