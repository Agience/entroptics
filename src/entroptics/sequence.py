"""
sequence.py -- the ordered-axis reads on a symbol sequence.

The complement of `entropy.py`, which reads an ordered `(T, F)` frame of amplitudes.  Here the
ordered object is a string of symbols from a finite alphabet, and the question is not "how many
modes rise above the floor" but "how much of the next symbol does the past already determine".

    H_n        the block entropy of order n: the Shannon entropy of the n-word distribution.
               H_1 sees only symbol frequencies; H_n captures every correlation shorter than n.
    h_n        = H_{n+1} - H_n = H(X_{n+1} | X_1..X_n), the conditional entropy of the next symbol.
    h_n_av     = H_n / n.
    h          the entropy rate, the common limit.  h <= H_1 <= log2|X|, with h < H_1 exactly when
               correlations are present.

Both intermediate quantities decrease monotonically to `h` and bound it from above in the
infinite-data limit (Lesne, MSCS 2014, eq. 45):

    h_n_av  >=  h_n  >=  h

That ordering does not hold on a finite sequence: once the number of possible n-words exceeds the
number of observed windows, `H_n` saturates at `log2(N-n+1)` and the ladder turns over and dives
well before it reaches `h`, so no function here returns a single `h`.  See :func:`entropy_rate` for
the collapse this produces and :func:`lempel_ziv_rate` for the single-sequence estimate of `h`;
:func:`surrogate_test` never needs either.

`h` on its own means almost nothing, which is what the rest of this module is shaped around.  Lesne
states it flatly (section 4.2): *"there is no one-to-one correspondence between entropy rates and
processes.  There is no way to directly infer any insights into the underlying process from the
value of h itself: only a differential study makes sense."*  Only `h = H_1` is directly meaningful,
and it says there are no correlations at all.  So the read this module exists for is not `h` but
:func:`surrogate_test` -- the sequence against its own shuffles.

That comparison is why no word-length threshold appears anywhere here.  A plug-in block entropy is
biased downward at large `n`, badly, once the observed words stop repeating: with `N` samples over
an alphabet of `k` there are `k^n` possible n-words and the estimate degenerates into
`log2(N-n+1)`.  The usual response is to fit a maximum word length.  It is not needed: a shuffle of
the sequence has the same length, the same alphabet and the same symbol frequencies, so it carries
the same finite-size bias at every `n`.  The bias cancels in the difference, and what survives is
the only thing that was ever evidence -- the correlations the shuffle destroyed.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "alphabet_size", "block_entropies", "entropy_rate", "lempel_ziv_rate",
    "surrogate_test", "redundancy_rate", "effective_length",
]

from .entropy import shannon_bits


def _as_symbols(seq) -> np.ndarray:
    """Any sequence of hashables -> a contiguous integer coding, and the alphabet size.

    The symbols themselves never matter to any read here: entropy is a property of the probability
    VALUES and is invariant under relabelling (Lesne section 2.1). Coding to integers is therefore
    lossless for this purpose and makes the word-counting exact."""
    a = np.asarray(list(seq))
    _, inv = np.unique(a, return_inverse=True)
    return inv.astype(np.int64)


def alphabet_size(seq) -> int:
    """The number of distinct symbols actually observed.

    Observed, never declared: `log2|X|` is the ceiling every rate here is compared against, and a
    declared alphabet wider than the sequence actually uses would raise that ceiling with symbols
    that never occurred -- the same discipline `geometry`'s measured extent applies, one dimension
    down."""
    return int(np.unique(_as_symbols(seq)).size)


def block_entropies(seq, n_max: int) -> list:
    """`[H_1, H_2, ..., H_n_max]` in bits -- the entropy of the n-word distribution for each order.

    Words are the `N - n + 1` overlapping windows of length `n`. Returns plug-in estimates, biased
    downward at large `n` by construction; see the module docstring for why that is left uncorrected
    and how :func:`surrogate_test` cancels it."""
    x = _as_symbols(seq)
    N, k = int(x.size), int(np.unique(x).size)
    if N == 0 or n_max < 1:
        return []
    out = []
    for n in range(1, int(n_max) + 1):
        if n > N:
            break
        # pack each n-word into one integer in base k -- exact, and avoids string keys
        w = np.zeros(N - n + 1, dtype=object) if k ** n > 2 ** 62 else np.zeros(N - n + 1, dtype=np.int64)
        for j in range(n):
            w = w * k + x[j:N - n + 1 + j]
        _, counts = np.unique(w, return_counts=True)
        out.append(float(shannon_bits(counts.astype(float))))
    return out


def entropy_rate(seq, n_max: int = 8) -> dict:
    """The entropy-rate ladder: `H_n`, `h_n = H_{n+1} - H_n`, `h_n_av = H_n / n`.

    No single `h` is returned, because collapsing the ladder to one number is unsafe on a finite
    sequence.  `h_n_av >= h_n >= h` holds in the infinite-data limit (Lesne eq. 45), but on a finite
    sequence the ladder turns over and dives once `k^n` exceeds the number of windows: almost every
    n-word becomes unique, `H_n` saturates at `log2(N-n+1)`, and the increments collapse toward
    zero.  Taking `min(h_n)` as an upper bound on `h` can then read well below the true rate -- on
    an i.i.d. uniform sequence over 4 symbols with `N = 3000`, whose true rate is `h = H_1 = 2.0`
    bits exactly, `min(h_n)` at `n_max = 6` reads 0.43.

    The bias is not symmetric between a sequence and its shuffle, either: a shuffled sticky Markov
    chain has more distinct words than the original, so it saturates first, and a verdict built on
    `min(h_n)` can invert sign for a strongly correlated source.

    So the ladder is returned whole and the caller reads it. :func:`surrogate_test` compares `H_n`
    at each order against the shuffle null, which is where the theorem is actually stated and where
    the bias cancels; :func:`lempel_ziv_rate` is the single-sequence estimate of `h` itself."""
    x = _as_symbols(seq)
    N, k = int(x.size), int(np.unique(x).size)
    H = block_entropies(x, n_max + 1)
    h_n = [H[i + 1] - H[i] for i in range(len(H) - 1)]
    h_av = [H[i] / (i + 1) for i in range(len(H))]
    return {
        "H_n": H[:n_max], "h_n": h_n[:n_max], "h_n_av": h_av[:n_max],
        "H_1": (H[0] if H else 0.0), "alphabet": k, "N": N,
    }


def lempel_ziv_rate(seq) -> float:
    """`L = N_w * log2(N) / N` -- the Lempel-Ziv estimate of the entropy rate (Lesne eq. 62), where
    `N_w` is the number of words in the LZ-76 parsing.

    Model-free, from ONE sequence, no fitting and no word-length choice. The Ziv-Lempel theorem
    (eq. 61) makes this asymptotically equal to both the algorithmic complexity rate and the Shannon
    entropy rate for a stationary ergodic source, so it estimates the same `h` the block ladder
    bounds from above -- by a completely different route, which is what makes agreement between them evidence.

    Convergence is slow, and this is an estimate on a finite sequence, not a bound. Its value on a
    short sequence is dominated by `log2(N)/N`; use it differentially, as with everything else here."""
    x = _as_symbols(seq)
    N = int(x.size)
    if N < 2:
        return 0.0
    seen, n_w, i = set(), 0, 0
    while i < N:
        j = i + 1
        while j <= N and tuple(x[i:j]) in seen:
            j += 1
        seen.add(tuple(x[i:min(j, N)]))
        n_w += 1
        i = j if j <= N else N
    return float(n_w * np.log2(N) / N)


def surrogate_test(seq, *, draws: int = 200, n_max: int = 8, seed: int = 0) -> dict:
    """The read this module exists for: does this sequence carry ordered structure at all?

    A random shuffle can only increase block entropy -- `H_n(sigma.X) >= H_n(X)`, with equality iff
    the source is uncorrelated (Lesne section 4.2, citing Karlin and Taylor 1975). The test is that
    inequality, read at each order `n` against a shuffle ensemble. A `z_n` significantly below zero
    is the finding: less block entropy than chance means the order carries structure.

    The comparison is on `H_n`, never on a derived `h` -- that is what makes it correct.  A
    permutation preserves length, alphabet and every symbol frequency, so at a fixed `n` the real
    sequence and its shuffles carry identical finite-size bias and it cancels exactly in the
    difference. Collapsing the ladder to one number first does not have that property: the bias
    lands differently on the two sides and can invert the verdict on a sticky Markov chain (see
    :func:`entropy_rate`). No word-length limit, no bias correction and no threshold is needed here,
    and none is present.

    `n = 1` is a free internal control, asserted: a permutation cannot change
    the symbol histogram, so `H_1` is identical between the sequence and every shuffle. If `z_1` is
    not zero the shuffle is not a permutation and every other row is meaningless. It is returned as
    `control_H1_exact` so a caller cannot skip it.

    `onset` is the first order at which the sequence departs from its shuffle -- the shortest
    window in which any structure is visible at all. It is not the Markov order: a lag-2 chain
    induces real, weak one-step correlation as a side effect, so it can report `onset = 2` while
    its dominant signal sits at `n = 3`. Both readings are true and answer different questions;
    treating the onset as the order would overstate what was measured.

    Recovering the order needs the other half of Lesne section 4.2 -- for a Markov chain of order q,
    `H_n = H_q + (n-q)h` for `n >= q`, so the order is where the ladder becomes linear in n. That is
    a further read on `entropy_rate`'s `h_n` and is not taken here.

    Returns per-order `z` and one-sided empirical p-values (the fraction of shuffles at or below the
    real value). No verdict: the reader supplies the false-alarm level, as everywhere else."""
    x = _as_symbols(seq)
    real = block_entropies(x, n_max)
    rng = np.random.default_rng(seed)
    null = np.array([block_entropies(rng.permutation(x), n_max) for _ in range(int(draws))])
    z, p = [], []
    for i in range(len(real)):
        col = null[:, i]
        sd = float(col.std(ddof=1)) if col.size > 1 else 0.0
        # A degenerate null reads as z = 0; the test is against float noise, not against zero.
        # At n = 1 every shuffle gives the same H_1 by construction, so `sd` is not 0.0 but ~1e-16
        # of accumulated rounding. Dividing on `sd > 0` alone would divide a tiny numerator by a
        # tiny denominator and produce an arbitrary z for the one rung whose answer is known exactly.
        floor = 1e-12 * max(1.0, abs(float(col.mean())))
        z.append(float((real[i] - col.mean()) / sd) if sd > floor else 0.0)
        p.append(float((col <= real[i]).sum() + 1) / float(col.size + 1))
    onset = next((i + 1 for i, v in enumerate(z) if i >= 1 and v < 0 and p[i] < 0.05), None)
    return {
        "H_n_real": real, "H_n_null_mean": [float(v) for v in null.mean(axis=0)],
        "z": z, "p_value": p,
        "control_H1_exact": bool(abs(real[0] - float(null[:, 0].mean())) < 1e-12) if real else False,
        "onset": onset,
        "draws": int(draws), "n_max": int(n_max), "N": int(x.size),
        "alphabet": int(np.unique(x).size),
    }


def redundancy_rate(seq) -> float:
    """`1 - h / log2|X|` in [0, 1] -- Shannon's redundancy of a source (Lesne section 5.2), using the entropy rate, so it counts both an uneven symbol
    distribution and the temporal correlations.

    Stands on `lempel_ziv_rate`, never on the block ladder: the ladder's finite-sample bias makes
    `min(h_n)` fall below the true rate (see :func:`entropy_rate`), which would report a source as
    more redundant than it is -- an overstatement of compressibility, in a number whose whole job is
    to say how much can safely be thrown away.

    Not the same quantity as `extract.mode_fill` or an axis fill, which are functions of `H_1`
    alone on a weight vector. This one is a property of an ordered sequence and is strictly larger
    whenever correlations exist, which is exactly the information a fill cannot see."""
    k = alphabet_size(seq)
    if k < 2:
        return 0.0
    return float(min(1.0, max(0.0, 1.0 - lempel_ziv_rate(seq) / np.log2(k))))


def effective_length(seq) -> float:
    """`N_eff = N * h / log2|X|` -- the effective number of independent samples in a correlated
    sequence (Lesne section 4.3).

    Why a caller outside this module wants it: every finite-size bound in the instrument is written
    in the number of ordered samples, and the certified band goes as `sqrt(F/T) + F/T` with `T` a
    count of rows. When those rows are correlated, the evidence they carry is worth fewer
    independent samples than there are rows, and a band computed on the raw count is optimistic. This
    is the correction, and it is a measurement.

    Equals `N` exactly when the sequence is uncorrelated and uniform, and falls as either correlation
    or unevenness rises."""
    x = _as_symbols(seq)
    k = int(np.unique(x).size)
    if k < 2:
        return 0.0
    return float(min(float(x.size), x.size * lempel_ziv_rate(x) / np.log2(k)))
