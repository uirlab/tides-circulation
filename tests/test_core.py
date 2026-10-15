"""Quick checks of the core estimator on synthetic tables (no data needed): run with  python -m pytest tests/
1. A tide (quasi-symmetric flows) gives a lack-of-fit deviance close to its degrees of freedom.
2. A planted loop field (a cycle around a ring of stations) inflates it far beyond that expectation.
3. The fitted tide reproduces each station's departures within its pairs up to the ridge term lambda * u."""
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import irrev as I  # noqa: E402

K, RNG = 25, np.random.default_rng(0)


def trips(loop=0.0):
    u = RNG.normal(0, 1, K)
    lo, hi = np.triu_indices(K, 1)
    t = RNG.poisson(40, lo.size)
    eta = u[lo] - u[hi]
    if loop:  # cyclic field on the ring a -> a+1: equal and opposite on every edge of the cycle
        ring = (hi - lo == 1).astype(float)
        ring[(lo == 0) & (hi == K - 1)] = -1.0  # closing edge K-1 -> 0 makes the ring a cycle (a path alone is a gradient)
        eta = eta + loop * ring
    p = expit(eta)
    n1 = RNG.binomial(t, p)
    L = np.repeat(lo, t); H = np.repeat(hi, t)
    y = np.concatenate([np.r_[np.ones(a), np.zeros(b - a)] for a, b in zip(n1, t)])
    return L, H, y, lo.size


def test_tide_is_quasi_symmetric():
    L, H, y, npair = trips()
    g_qs, g_s, _ = I.deviances(L, H, y, K)
    df = npair - (K - 1)
    assert abs(g_qs - df) / np.sqrt(2 * df) < 4  # within sampling noise of its chi-square expectation
    assert g_s > 3 * g_qs  # the tide carries most of the asymmetry


def test_planted_loop_is_detected():
    L, H, y, npair = trips(loop=1.2)
    g_qs, _, _ = I.deviances(L, H, y, K)
    df = npair - (K - 1)
    assert (g_qs - df) / np.sqrt(2 * df) > 5  # far above its tide-only expectation


def test_station_balance_up_to_ridge():
    L, H, y, _ = trips(loop=0.8)
    plo, phi, n1, t, _ = I.aggregate(L, H, y, K)
    u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
    r = n1 - t * expit(u[plo] - u[phi])
    resid = np.bincount(plo, r, K) - np.bincount(phi, r, K)
    assert np.allclose(resid, I.LAM * u, atol=1e-5)
