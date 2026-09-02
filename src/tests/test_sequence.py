"""The ordered-axis reads on a symbol sequence -- block entropies, the shuffle surrogate, LZ.

Lesne (MSCS 2014) §4.2: a shuffle can only raise block entropy, `H_n(σ.X) ≥ H_n(X)`, with equality
iff the source is uncorrelated. That inequality is the test; everything else here supports it.

The failure modes these tests watch for:
  - the test reports structure in an i.i.d. sequence (a false positive on the one source that
    provably has none);
  - the test reports no structure in a periodic or Markov sequence;
  - the verdict inverts on a correlated source;
  - the shuffle is not a permutation, so the null does not preserve the symbol histogram and every
    comparison is against the wrong ensemble;
  - `onset` is read as the Markov order, which it is not.
"""
import numpy as np
import pytest

from entroptics.sequence import (
    alphabet_size, block_entropies, effective_length, entropy_rate, lempel_ziv_rate,
    redundancy_rate, surrogate_test,
)

N = 3000


def _sticky(n=N, k=4, p=0.85, seed=0):
    rng = np.random.default_rng(seed)
    s = [0]
    for _ in range(n - 1):
        s.append(s[-1] if rng.random() < p else int(rng.integers(0, k)))
    return np.array(s)


def test_an_iid_sequence_shows_no_structure():
    """The negative control, and the one that matters most. An i.i.d. source provably has no
    temporal organisation, so a test that finds some here finds it anywhere."""
    seq = np.random.default_rng(0).integers(0, 4, N)
    r = surrogate_test(seq, draws=100, n_max=5, seed=1)
    assert r["onset"] is None
    assert all(abs(v) < 4.0 for v in r["z"]), r["z"]


def test_periodic_and_markov_sources_ARE_detected():
    """The positive controls. Without these, a function returning 'no structure' always would pass
    the negative control perfectly."""
    per = surrogate_test(np.tile([0, 1, 2, 3], N // 4), draws=100, n_max=5, seed=1)
    assert per["z"][1] < -50, per["z"]
    assert per["onset"] == 2

    mk = surrogate_test(_sticky(), draws=100, n_max=5, seed=1)
    assert mk["z"][1] < -50, mk["z"]


def test_the_verdict_is_NOT_inverted_on_a_correlated_source():
    """Pins the sign on a correlated source.

    Comparing `H_n` at a fixed order is what makes this work: at fixed `n` both sides have the same
    length, alphabet and symbol frequencies, so the plug-in bias is identical and cancels.
    Collapsing the ladder to `h_upper = min(h_n)` first does not have that property -- the shuffle
    has more distinct n-words at a given `n`, so it saturates at `log2(N-n+1)` first and its rate
    collapses further than the real sequence's, which can invert the sign. A sticky chain must read
    negative."""
    r = surrogate_test(_sticky(p=0.85), draws=100, n_max=4, seed=1)
    assert r["z"][1] < 0, f"sign inverted on a correlated source: z_2={r['z'][1]}"


def test_the_shuffle_null_preserves_the_histogram_exactly():
    """`H_1` is a function of the symbol frequencies alone, and a permutation cannot change them.
    If this drifts, the null is not a permutation and every other row is meaningless — which is why
    the test reports it."""
    for seq in (np.random.default_rng(2).integers(0, 5, 500), _sticky(500), np.tile([0, 1, 2], 300)):
        r = surrogate_test(seq, draws=50, n_max=3, seed=3)
        assert r["control_H1_exact"], "the shuffle changed the symbol histogram"
        assert abs(r["z"][0]) < 1e-6


def test_onset_is_the_onset_and_not_the_markov_order():
    """Pins the modest claim, because the ambitious one is false.

    A lag-2 chain induces real weak one-step correlation as a side effect, so `onset = 2` even
    though the dependency needs 3-words to be seen and the dominant signal sits at `n = 3`
    (`z_2 = -3.7` against `z_3 = -512`). `onset` is the shortest window carrying any structure;
    reading it as the Markov order would overstate what was measured."""
    rng = np.random.default_rng(4)
    s = [0, 1]
    for _ in range(N - 2):
        s.append((s[-1] + s[-2]) % 4 if rng.random() < 0.9 else int(rng.integers(0, 4)))
    r = surrogate_test(np.array(s), draws=100, n_max=5, seed=1)
    assert r["onset"] == 2, (r["onset"], r["z"])                  # the onset, weak but real
    assert r["z"][2] < 20 * r["z"][1], (r["z"],)                  # the order is where it dominates


def test_a_sequence_of_unique_symbols_carries_NO_orderable_information():
    """The result that explains a standing null finding.

    When every symbol occurs exactly once, every n-word is unique, so `H_n = log2(N-n+1)` — a
    function of length alone. Two such sequences over the same multiset are therefore
    information-theoretically identical at the symbol level, and so is every shuffle of either.

    This is exactly the shape of the `drift` / `jumble` arms that the reasoning work compared: the
    same twelve concepts in two orders. No instrument can separate them, because at the level
    compared there is nothing to separate. The signal, if any, lives after a coarse graining to a
    smaller alphabet (Lesne §2.5) — regions, coarser than concepts."""
    drift = ["dog", "wolf", "pack", "hunt", "prey", "deer",
             "forest", "tree", "leaf", "plant", "soil", "earth"]
    jumble = ["leaf", "hunt", "earth", "dog", "soil", "tree",
              "prey", "plant", "wolf", "forest", "deer", "pack"]
    assert block_entropies(drift, 4) == block_entropies(jumble, 4)
    assert surrogate_test(drift, draws=100, n_max=3, seed=1)["onset"] is None

    # and coarse-graining to regions makes them differ, in the direction structure predicts
    region = {"dog": "a", "wolf": "a", "pack": "a", "prey": "a", "deer": "a", "hunt": "c",
              "forest": "p", "soil": "p", "earth": "p", "tree": "l", "leaf": "l", "plant": "l"}
    d2 = [region[c] for c in drift]
    j2 = [region[c] for c in jumble]
    assert block_entropies(d2, 2)[1] < block_entropies(j2, 2)[1]


def test_lempel_ziv_separates_structure_from_noise():
    """The single-sequence estimate of `h`, by a route independent of the block ladder — which is
    what makes the two agreeing evidence."""
    iid = np.random.default_rng(5).integers(0, 4, N)
    per = np.tile([0, 1, 2, 3], N // 4)
    assert lempel_ziv_rate(per) < 0.5 * lempel_ziv_rate(iid)


def test_redundancy_and_effective_length_bracket_their_ranges():
    """An i.i.d. uniform source is incompressible and every sample is independent; a periodic one is
    neither."""
    iid = np.random.default_rng(6).integers(0, 4, N)
    per = np.tile([0, 1, 2, 3], N // 4)
    assert redundancy_rate(iid) == pytest.approx(0.0, abs=0.05)
    assert redundancy_rate(per) > 0.5
    assert effective_length(iid) == pytest.approx(N, rel=0.15)
    assert effective_length(per) < 0.5 * N


def test_the_alphabet_is_observed_never_declared():
    assert alphabet_size(["a", "b", "a", "c"]) == 3
    assert entropy_rate(["a", "a", "a"], 2)["alphabet"] == 1
    assert redundancy_rate(["a", "a", "a"]) == 0.0      # one symbol: no capacity to spend


def test_block_entropies_are_the_shannon_definition():
    """`H_1` of a known histogram, against `shannon_bits` directly."""
    from entroptics.entropy import shannon_bits
    seq = ["a"] * 8 + ["b"] * 4 + ["c"] * 2 + ["d"] * 2
    assert block_entropies(seq, 1)[0] == pytest.approx(
        shannon_bits(np.array([8.0, 4.0, 2.0, 2.0])), abs=1e-12)
