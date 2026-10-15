"""ES_dev restricted to pairs with >= 30 June trips (follow-up to src/dev_es.py).
Uses beta_hat from results/devES.json unchanged. Per-pair deviance contributions; E0 per pair from 20 null replicates."""
import json
import sys

import numpy as np
from scipy.special import expit, xlogy

sys.path.insert(0, "src")
import dev_es as E  # noqa: E402

bh = json.load(open("results/devES.json"))["real"]["beta_hat_may"]
jun, lam = E.month_cells("citibike_jun")
import irrev as I  # noqa: E402


def contrib(c, n1, t, off):
    ok = t > 0
    u = I.fit_bt(c["plo"][ok], c["phi"][ok], n1[ok], t[ok], c["K"], lam, off=off[ok])
    p = np.clip(expit(u[c["plo"]] - u[c["phi"]] + off), 1e-12, 1 - 1e-12)
    n0 = t - n1
    g = 2 * (xlogy(n1, n1) - xlogy(n1, t * p) + xlogy(n0, n0) - xlogy(n0, t * (1 - p)))
    return np.where(ok, g, 0.0), u


tot = {}
for c in jun:
    t = c["T"].sum(0).astype(float)
    for k, v in zip(c["ids"], t):
        tot[k] = tot.get(k, 0) + v
keep_ids = {k for k, v in tot.items() if v >= 30}
rng = np.random.default_rng(83)
gq = gd = e0 = 0.0
for c in jun:
    t = c["T"].sum(0).astype(float); n1 = c["N1"].sum(0).astype(float)
    m = np.array([x in keep_ids for x in c["ids"]])
    g0, u0 = contrib(c, n1, t, np.zeros(t.size)); g1, _ = contrib(c, n1, t, bh * c["D"])
    gq += g0[m].sum(); gd += g1[m].sum()
    p = expit(u0[c["plo"]] - u0[c["phi"]])
    e0 += np.mean([contrib(c, rng.binomial(t.astype(int), p).astype(float), t, np.zeros(t.size))[0][m].sum() for _ in range(20)])
out = dict(beta_hat=bh, n_pairs_ge30=len(keep_ids), G2QS=gq, G2D=gd, E0=e0, loop_excess=gq - e0, ES_dev_ge30=(gq - gd) / (gq - e0))
json.dump(out, open("results/devES_ge30.json", "w"), indent=1); print(out)
