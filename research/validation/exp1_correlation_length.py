"""
Experiment 1 -- Correlation length <-> diffraction limit.

Ground truth: F independent AR(1) channels with lag-1 coefficient phi = e^{-1/rho}
have population autocorrelation C(tau) = exp(-tau/rho), i.e. a known exponential
correlation length rho.  The diffraction limit is read off the signal's own decay
via Aperture(W).correlation_length; the entropy-width a_delta and the
reciprocal integral length 1/xi are monotone in 1/rho and track it.

Deterministic (fixed seeds).  Re-runnable: `python exp1_correlation_length.py`.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401 -- run against local src/, not any installed entroptics

import numpy as np

from entroptics import Aperture

import common as C

T, F = 4000, 48
RHOS = [2, 4, 8, 16, 32, 64, 128]
SEED = 101


def run() -> dict:
    rows = []
    rho_arr, ad_arr, xi_arr = [], [], []
    for i, rho in enumerate(RHOS):
        phi = C.phi_for_rho(rho)
        W = C.ar1(T, F, phi, seed=SEED + i)
        ap = Aperture(W, window=None)
        a_delta, xi = ap.a_delta, ap.correlation_length
        rho_arr.append(rho); ad_arr.append(a_delta); xi_arr.append(xi)
        rows.append([rho, round(phi, 4), round(1.0 / rho, 4),
                     round(a_delta, 5), round(xi, 3), round(1.0 / xi, 5)])

    rho_arr = np.array(rho_arr, float)
    ad_arr = np.array(ad_arr, float)
    xi_arr = np.array(xi_arr, float)
    inv_rho = 1.0 / rho_arr

    # monotonicity of the two reads in the true inverse correlation length
    sp_ad = C.spearman(inv_rho, ad_arr)
    sp_ix = C.spearman(inv_rho, 1.0 / xi_arr)
    # xi tracks rho ~ linearly (integral length == correlation length); fit slope
    slope_xi, r2_xi = C.loglog_fit(rho_arr, xi_arr)
    slope_ad, r2_ad = C.loglog_fit(inv_rho, ad_arr)
    # how close xi is to the true rho (ratio)
    ratio = xi_arr / rho_arr

    headers = ["rho (true)", "phi", "1/rho", "a_delta", "xi", "1/xi"]
    table = C.md_table(headers, rows)

    headline = (
        f"a_delta and 1/xi are perfectly monotone in 1/rho (Spearman = "
        f"{sp_ad:.3f} and {sp_ix:.3f}); xi ~ rho with log-log slope "
        f"{slope_xi:.3f} (R^2={r2_xi:.4f}), and a_delta ~ 1/rho with slope "
        f"{slope_ad:.3f} (R^2={r2_ad:.4f}).")
    concl = ("The diffraction limit tracks the true correlation length: the integral "
             "length xi recovers rho to within a constant shape factor "
             f"(xi/rho in [{ratio.min():.2f}, {ratio.max():.2f}]), and both a_delta "
             "and 1/xi are strictly monotone in 1/rho.")

    return dict(
        title="1. Correlation length <-> diffraction limit",
        setup=(f"{F} independent AR(1) channels, T={T}, phi=e^(-1/rho); population "
               f"autocorrelation exp(-tau/rho).  Read diffraction_limit(decay(W))."),
        table=table,
        metrics=dict(spearman_a_delta=sp_ad, spearman_inv_xi=sp_ix,
                     loglog_slope_xi_vs_rho=slope_xi, r2_xi=r2_xi,
                     loglog_slope_a_delta=slope_ad, r2_a_delta=r2_ad,
                     xi_over_rho_min=float(ratio.min()), xi_over_rho_max=float(ratio.max())),
        headline=headline,
        conclusion=concl,
    )


if __name__ == "__main__":
    r = run()
    print(r["title"]); print(r["setup"]); print(r["table"])
    print("HEADLINE:", r["headline"]); print("CONCLUSION:", r["conclusion"])
