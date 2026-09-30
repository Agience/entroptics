"""``lempel_ziv_rate`` counts the LZ-76 parsing it names.

It counted an LZ-78 parsing (a dictionary of earlier phrases) while documenting LZ-76 (any earlier
substring, overlaps allowed), and in quadratic time.  The count is now the LZ-76 word count, by an
online suffix automaton, checked here against a direct search."""
import numpy as np

from entroptics.sequence import _lz76_words, lempel_ziv_rate


def _brute(x):
    x = list(x)
    words, i = 0, 0
    while i < len(x):
        l = 1
        while i + l <= len(x) and any(x[p:p + l] == x[i:i + l] for p in range(i)):
            l += 1
        words, i = words + 1, i + l
    return words


def test_lempel_ziv_1976_worked_example():
    # 0 | 001 | 10 | 100 | 1000 | 101 -- six words
    x = [int(c) for c in "0001101001000101"]
    assert _lz76_words(x) == 6
    assert lempel_ziv_rate(x) == 6 * np.log2(16) / 16


def test_the_count_matches_a_direct_search():
    rng = np.random.default_rng(0)
    for trial in range(400):
        N = int(rng.integers(1, 50))
        if trial % 2:
            x = rng.integers(0, 3, N)
        else:
            x = np.tile(rng.integers(0, 3, int(rng.integers(1, 5))), N)[:N]
        assert _lz76_words(x.tolist()) == _brute(x.tolist())
