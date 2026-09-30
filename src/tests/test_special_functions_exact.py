"""The special functions behind the nulls are evaluated to float precision, not approximated.

The normal quantile was Acklam's fitted rational (1.2e-9); digamma and trigamma were a five-term
asymptotic series (8.8e-12, 1.9e-10).  Each is now evaluated from its own law to round-off.
References: mpmath at 40 digits (the 1e-300 quantile checked by evaluating Q there)."""
import math

import pytest

from entroptics.entropy import _digamma, _trigamma
from entroptics.null_providers import _norm_isf, _norm_ppf

PSI = {0.01: (-100.56088545786867242, 10001.621213528312804),
       1.0: (-0.57721566490153286061, 1.6449340668482264365),
       2.5: (0.70315664064524318723, 0.49035775610023486497),
       100.0: (4.6001618527380874002, 0.010050166663333571395)}
ISF = {0.3: 0.52440051270804081597, 0.05: 1.644853626951472688, 1e-9: 5.9978070150076868614,
       1e-300: 37.0470962993612}


@pytest.mark.parametrize("x", sorted(PSI))
def test_digamma_and_trigamma(x):
    psi, psi1 = PSI[x]
    assert abs(_digamma(x) / psi - 1.0) < 1e-15
    assert abs(_trigamma(x) / psi1 - 1.0) < 1e-14


@pytest.mark.parametrize("p", sorted(ISF))
def test_the_normal_quantile(p):
    assert abs(_norm_isf(p) / ISF[p] - 1.0) < 1e-14
    assert _norm_ppf(p) == -_norm_isf(p)
    q = 1.0 - p                                   # symmetry, on the complement actually represented
    assert _norm_isf(q) == -_norm_isf(1.0 - q)


def test_normal_quantile_edges():
    assert _norm_isf(0.5) == pytest.approx(0.0, abs=1e-16)
    assert _norm_isf(0.0) == math.inf and _norm_isf(1.0) == -math.inf
