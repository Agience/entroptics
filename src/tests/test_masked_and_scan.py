"""Masked-input handling: drop fully-dead rows/cols (ignore), don't zero-fill."""
import numpy as np
import pytest

from entroptics.screen import Screen


# ── fully-dead rows/cols are IGNORED (dropped), not zero-imputed ──────────────

def test_clean_input_takes_unchanged_path():
    W = np.random.default_rng(0).standard_normal((60, 30))
    sc = Screen(W)
    assert sc._live_rows is None and sc._live_cols is None
    assert (sc.T, sc.F) == (60, 30)


def test_fully_dead_columns_dropped_not_zero_filled():
    W = np.random.default_rng(1).standard_normal((40, 20))
    W[:, 5] = np.nan; W[:, 12] = np.nan                 # two fully-dead feature channels
    sc = Screen(W)
    assert sc._live_cols is not None and int(sc._live_cols.sum()) == 18
    assert (sc.T, sc.F) == (40, 20)                      # native shape preserved
    assert sc.F_eff <= 18                                # screen excludes the dead channels
    assert not sc._live_cols[5] and not sc._live_cols[12]   # the two dead channels are dropped


def test_fully_dead_rows_dropped():
    W = np.random.default_rng(2).standard_normal((30, 25))
    W[7, :] = np.nan                                     # one fully-dead ordered row
    sc = Screen(W)
    assert sc._live_rows is not None and int(sc._live_rows.sum()) == 29


def test_scattered_nan_not_dropped():
    W = np.random.default_rng(3).standard_normal((30, 20))
    W[4, 7] = np.nan; W[10, 3] = np.nan                  # scattered, no fully-dead line
    sc = Screen(W)
    assert sc._live_rows is None and sc._live_cols is None   # nothing dropped


def test_dead_channels_do_not_inflate_k_signal():
    # pad pure noise with 40% fully-dead channels; K_signal must not jump vs the clean read
    rng = np.random.default_rng(7)
    core = rng.standard_normal((64, 40))
    padded = np.concatenate([core, np.full((64, 26), np.nan)], axis=1)
    assert Screen(padded).K_signal == Screen(core).K_signal


def test_optics_reads_tolerate_missing_data():
    """Every optics read ignores missing data (drops fully-dead lines, cleans scattered
    gaps) -- feeding NaN-laden input to the front door no longer raises."""
    from entroptics import Aperture
    W = np.random.default_rng(0).standard_normal((50, 30))
    W[:, 5] = np.nan; W[:, 11] = np.nan          # fully-dead feature channels
    W[3, 7] = np.nan                              # a scattered gap
    W[9, :] = np.nan                             # a fully-dead ordered row
    o = Aperture(W).optics()
    assert all(np.isfinite(v) for v in o.values() if isinstance(v, float))
