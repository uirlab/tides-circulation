"""Methods review (2026-10-13): how far the ridge-penalised Bradley-Terry fit departs from exact station balance.
At the ridge optimum each station's residual departures equal lam * u_a (gradient of the penalised log-likelihood).
Run from the repository root (cluster recommended): python src/rb_ridge_check.py [city]   (default citibike, May 2026, 15-min, strata)."""
import json
import os
import sys

import numpy as np

os.environ["CITY"] = sys.argv[1] if len(sys.argv) > 1 else "citibike"
sys.path.insert(0, "src")
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402

D = S.load(); K = D["K"]; lam = I.LAM
absr, rel, net = [], [], []
for k in range(I.NB):
    for s in range(S.NS):
        m = (D["b"] == k) & (D["st"] == s)
        if m.sum() < 2:
            continue
        plo, phi, n1, t, _ = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
        u = I.fit_bt(plo, phi, n1, t, K, lam)
        r = lam * u
        tot = np.bincount(plo, t, K) + np.bincount(phi, t, K)
        dep = np.bincount(plo, n1, K) + np.bincount(phi, t - n1, K); arr = tot - dep
        act = tot > 0
        absr.append(np.abs(r[act])); rel.append(np.abs(r[act]) / tot[act]); net.append(np.abs(dep - arr)[act])
a, q, nt = np.concatenate(absr), np.concatenate(rel), np.concatenate(net)
out = dict(city=os.environ["CITY"], cells=int(a.size), abs_residual_trips=dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
           p99=float(np.percentile(a, 99)), max=float(a.max())), share_cells_above_1_trip=float((a > 1).mean()),
           residual_over_station_trips=dict(median=float(np.median(q)), p95=float(np.percentile(q, 95))),
           sum_abs_residual_over_sum_abs_net_imbalance=float(a.sum() / nt.sum()))
print(json.dumps(out, indent=1))
os.makedirs("results/rb", exist_ok=True)
json.dump(out, open(f"results/rb/ridge_check_{os.environ['CITY']}.json", "w"), indent=1)
