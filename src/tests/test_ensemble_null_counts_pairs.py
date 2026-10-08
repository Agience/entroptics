"""The streaming operator's null is the null of what it reads.

``Dynamics`` accumulates one outer product per transition pair, and an ensemble of short
independent runs (``update_block(..., adjacent=False)``) has far fewer pairs than frames.  The count,
the floor contrast and the significance sized the null by frames, which set the edge too low: on
pure noise, 60 runs of 3 frames each reported structure in 26.5% of records at ``far = 0.05``."""
import numpy as np

from entroptics.dynamics import Dynamics

FAR = 0.05


def _ensemble(rng, runs, run_len, F):
    D = Dynamics(F, far=FAR)
    for _ in range(runs):
        D.update_block(rng.standard_normal((run_len, F)), adjacent=False)
    return D


def test_an_ensemble_of_short_runs_holds_the_level_on_noise():
    """The read is the exact test (or the closed form where it holds) on the window the operator
    keeps, so its rate is ``far`` up to the records' binomial error (1000 records read 0.049)."""
    rng = np.random.default_rng(0)
    records = [_ensemble(rng, 60, 3, 16) for _ in range(400)]
    assert records[0].n_frames == 180 and records[0].n_pairs == 120
    rate = np.mean([D.resolved(seed=i) > 0 for i, D in enumerate(records)])
    assert rate <= FAR + 3 * np.sqrt(FAR * (1 - FAR) / len(records)), rate


def test_significance_is_the_count():
    D = _ensemble(np.random.default_rng(1), 60, 3, 16)
    sig = D.significance()
    assert D.resolved() in (int(np.sum(sig.pvalue < FAR)), int(np.sum(sig.pvalue <= FAR)))
