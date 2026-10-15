"""Descriptive correction for step 1 (not part of the decision rule): pseudo-count shrinkage in pair residuals
deflates sum(w*eps^2) below n under a no-loop null, which inflates ES. Estimate delta = n - E_null[sum w eps^2]
by simulating directions from each (bin, stratum) Bradley-Terry fit (pure quasi-symmetry, no loops), recomputing the
pooled residuals, and report ES with numerator and denominator corrected by delta. Usage: python src/shrinkage_calibration.py"""
import json
import os
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, "src")
os.environ["CITY"] = "citibike_jun"
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
import loop_map as L  # noqa: E402
import mechanism_share as M  # noqa: E402

D = S.load(); K = D["K"]
p_trip = np.zeros(D["lo"].size)
for k in range(I.NB):
    for s in range(S.NS):
        m = (D["b"] == k) & (D["st"] == s)
        if m.sum() < 2:
            continue
        plo, phi, n1, t, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
        u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
        p_trip[np.flatnonzero(m)] = expit(u[plo] - u[phi])[inv]
real = M.city_set("citibike_jun", "citibike")
may = M.city_set("citibike", "citibike")
beta = M.fit_beta(may["eps"], may["se"], may["D"])
keys = set(zip(real["a"], real["b"]))
sid = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
rng = np.random.default_rng(8)
deltas = []
for rep in range(5):
    Dn = dict(D); Dn["y"] = (rng.random(D["lo"].size) < p_trip).astype(float)
    lo, hi, eps, se, t, _ = L.pair_residuals(Dn)
    k_ = t >= M.MIN_T
    sel = np.array([(a, b) in keys for a, b in zip(sid[lo[k_]], sid[hi[k_]])])
    e, s_ = eps[k_][sel], se[k_][sel]
    deltas.append(float(e.size - (e**2 / s_**2).sum()))
    print("rep", rep, "n", e.size, "delta", round(deltas[-1]), flush=True)
dl = float(np.mean(deltas))
w = 1 / real["se"] ** 2; n = real["eps"].size
num = (w * (real["eps"] - beta * real["D"]) ** 2).sum() - n + dl
den = (w * real["eps"] ** 2).sum() - n + dl
out = dict(delta_mean=dl, delta_reps=deltas, ES_out_june_uncorrected=M.es(real["eps"], real["se"], real["D"], beta),
           ES_out_june_shrinkage_corrected=float(1 - num / den), beta_may=beta, n=n)
json.dump(out, open("results/1b_shrinkage_calibration.json", "w"), indent=1); print(json.dumps(out, indent=1))
