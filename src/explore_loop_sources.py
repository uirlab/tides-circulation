"""EXPLORATORY (NYC May 2026 only; June held out): what else explains the loop field?
Features: D (legal one-way detour asymmetry, length) and BL (bike-lane-weighted directed path-cost asymmetry).
Reports shrinkage-corrected in-sample explained shares, by distance band. Writes results/exp_loop_sources.json"""
import json
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, "src")
import mechanism_share as M  # noqa: E402

may = M.city_set("citibike", "citibike")
a, b = may["a"], may["b"]
BL, okb, _ = M.covariate("citibike_lane", a, b)
keep = okb & np.isfinite(BL)
eps, se, D, BL, und = may["eps"][keep], may["se"][keep], may["D"][keep], BL[keep], may["und"][keep]
w = 1 / se**2; n = eps.size

# shrinkage delta for this pair set (2 null replicates; same method as src/shrinkage_calibration.py)
os.environ["CITY"] = "citibike"
for m_ in ("irrev", "irrev_strat", "loop_map"):
    sys.modules.pop(m_, None)
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
import loop_map as L  # noqa: E402
Dd = S.load(); K = Dd["K"]
p_trip = np.zeros(Dd["lo"].size)
for k in range(I.NB):
    for s in range(S.NS):
        m = (Dd["b"] == k) & (Dd["st"] == s)
        if m.sum() < 2:
            continue
        plo, phi, n1, t, inv = I.aggregate(Dd["lo"][m], Dd["hi"][m], Dd["y"][m], K)
        u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
        p_trip[np.flatnonzero(m)] = expit(u[plo] - u[phi])[inv]
sid = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
keys = set(zip(a[keep], b[keep]))
rng = np.random.default_rng(9); deltas = []
for _ in range(2):
    Dn = dict(Dd); Dn["y"] = (rng.random(Dd["lo"].size) < p_trip).astype(float)
    lo, hi, e0, s0, t0, _ = L.pair_residuals(Dn)
    k_ = t0 >= M.MIN_T
    sel = np.array([(x, y) in keys for x, y in zip(sid[lo[k_]], sid[hi[k_]])])
    deltas.append(float(sel.sum() - ((e0[k_][sel] / s0[k_][sel]) ** 2).sum()))
dl = float(np.mean(deltas))


def wls(X, y, w):
    W = np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
    return beta


def es_corr(eps, se, pred, delta):
    w = 1 / se**2; n = eps.size
    return float(1 - ((w * (eps - pred) ** 2).sum() - n + delta) / ((w * eps**2).sum() - n + delta))


out = dict(n_pairs=n, delta=dl, corr_D_BL=float(np.corrcoef(D, BL)[0, 1]))
for name, X in (("D", D[:, None]), ("BL", BL[:, None]), ("D+BL", np.c_[D, BL])):
    beta = wls(X, eps, w)
    out[name] = dict(beta=beta.tolist(), ES_in_sample_corrected=es_corr(eps, se, X @ beta, dl))
# by distance band (delta apportioned by pair count, crude)
bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 1e9)]
out["by_distance"] = []
bD, bDB = wls(D[:, None], eps, w), wls(np.c_[D, BL], eps, w)
for lo_, hi_ in bands:
    mm = (und >= lo_) & (und < hi_)
    frac = mm.sum() / n
    out["by_distance"].append(dict(m=[lo_, hi_], n=int(mm.sum()),
                                   excess_per_pair=float(((w[mm] * eps[mm] ** 2).sum() - mm.sum() + dl * frac) / mm.sum()),
                                   ES_D=es_corr(eps[mm], se[mm], D[mm] * bD[0], dl * frac),
                                   ES_D_BL=es_corr(eps[mm], se[mm], np.c_[D, BL][mm] @ bDB, dl * frac)))
json.dump(out, open("results/exp_loop_sources.json", "w"), indent=1)
print(json.dumps(out, indent=1))
