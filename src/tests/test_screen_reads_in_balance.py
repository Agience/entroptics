"""Every screen read runs on balanced frames and scores a side with that side's own null.

Two reads broke the rule:
- ``coupling`` read the raw placed frames, so a side that balances somewhere other than its column
  mean (a declared ``zero``) was coupled on structure its own zero removes;
- ``linear`` decided "resolves nothing" with the screen's null, while ``certify`` -- which the
  docstring says it agrees with -- uses the side's own."""
import numpy as np

from entroptics.screen import Screen


def test_coupling_is_read_where_each_side_balances():
    rng = np.random.default_rng(0)
    T, D = 400, 6
    trend = np.outer(np.linspace(-3.0, 3.0, T), np.ones(D))      # shared, and each side's own zero
    sc = Screen()
    for name in ("a", "b"):
        sc.register(name, entry=lambda s: s, zero=lambda X, tr=trend: tr + np.mean(X - tr, axis=0))
    sc.place("a", trend + rng.standard_normal((T, D)))
    sc.place("b", trend + rng.standard_normal((T, D)))
    # balanced, the two sides are independent noise: nothing couples them
    assert sc.couple("a", "b") == 0.0


def test_linearity_is_scored_with_the_sides_own_null():
    rng = np.random.default_rng(1)
    # a shared mode, so the cube's departure from linearity is structure the default null resolves
    X = np.outer(rng.standard_normal(300), np.linspace(1.0, 2.0, 8)) + 0.1 * rng.standard_normal((300, 8))

    def resolves_nothing(ctx):
        return float("inf")

    sc = Screen()
    sc.register("cube", entry=lambda s: s ** 3, inverse=np.cbrt, null=resolves_nothing)
    sc.place("cube", X)
    # the side's own null resolves nothing, so its departure from linearity is not seen
    assert sc.linear("cube", X).linear
