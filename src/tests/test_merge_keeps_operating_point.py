"""A merge keeps the caller's operating point.

``Dynamics.merge`` with an empty side rebuilt the other side from its state without ``far`` or
``null``, so the merged operator counted at the default 0.05 against the default null, whatever the
caller had set."""
import numpy as np

from entroptics.dynamics import Dynamics, dynamics
from entroptics.null_providers import robust


def test_an_empty_merge_keeps_far_and_null():
    W = np.random.default_rng(0).standard_normal((300, 8))
    full = dynamics(W)
    for empty_first in (True, False):
        empty = Dynamics(8, far=0.01, null=robust)
        if empty_first:
            out = empty.merge(full)
        else:
            full_c = Dynamics(8, far=0.01, null=robust).update_block(W)
            out = full_c.merge(Dynamics(8))
        assert out._far == 0.01 and out._null is robust
