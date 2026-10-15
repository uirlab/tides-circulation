"""Step 5 (descriptive; prediction pre-stated): tidal vs loop share of pairwise irreversibility across time windows. Usage (project root): NWORKERS=.. python src/tidal_share.py   (CITY env as elsewhere)
Writes results/5_tidal_share_<CITY>.json.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import expit, xlogy

sys.path.insert(0, os.path.dirname(__file__))
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402

WINDOWS = {"15min": 15, "1h": 60, "4h": 240, "16h": 960}
if os.environ.get("ONLY_15"):
    WINDOWS = {"15min": 15}
B0 = 20
_G = {}


SIM_SEED = os.environ.get("SIM_TIDE")  # coarse-graining check: tide-only simulated directions


def _init():
    _G.update(S.load())
    if SIM_SEED:
        D = _G; K = D["K"]; rng = np.random.default_rng(int(SIM_SEED)); y = D["y"].copy()
        for k in range(I.NB):
            for s in range(S.NS):
                m = (D["b"] == k) & (D["st"] == s)
                if m.sum() < 2:
                    continue
                plo, phi, n1, t, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
                u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
                y[np.flatnonzero(m)] = (rng.random(m.sum()) < expit(u[plo] - u[phi])[inv]).astype(float)
        D["y"] = y


def _task(args):
    wname, k, s = args
    W = WINDOWS[wname]
    K = _G["K"]
    win = ((_G["tm"] - I.T0) // W).astype(np.int64)
    m = (win == k) & (_G["st"] == s)
    if m.sum() < 2:
        return None
    lo, hi, y = _G["lo"][m], _G["hi"][m], _G["y"][m]
    g_qs, g_s, (plo, phi, n1, t, p, inv) = I.deviances(lo, hi, y, K)
    rng = np.random.default_rng([list(WINDOWS).index(wname), k, s, 55])  # stable seed
    null = [I.deviances(lo, hi, (rng.random(lo.size) < p[inv]).astype(float), K)[0] for _ in range(B0)]
    n0 = t - n1
    both = (n1 > 0) & (n0 > 0)
    ep = float(((n1 - n0) * (np.log(np.where(both, n1, 1)) - np.log(np.where(both, n0, 1))))[both].sum())
    active = np.unique(np.r_[plo, phi]).size
    return dict(g_qs=g_qs, g_s=g_s, e0=float(np.mean(null)), active=active, ep_plugin=ep, n_trips=int(lo.size))


def main():
    tasks = [(w, k, s) for w, W in WINDOWS.items() for k in range(960 // W) for s in range(S.NS)]
    with Pool(I.NWORKERS, initializer=_init) as pool:
        res = pool.map(_task, tasks)
    out = {}
    for w in WINDOWS:
        rs = [r for (ww, _, _), r in zip(tasks, res) if ww == w and r is not None]
        g_s = sum(r["g_s"] for r in rs); g_qs = sum(r["g_qs"] for r in rs); e0 = sum(r["e0"] for r in rs)
        tidal = g_s - g_qs - sum(r["active"] - 1 for r in rs)
        loop = g_qs - e0
        out[w] = dict(G2_S=g_s, G2_QS=g_qs, E0_QS=e0, tidal=tidal, loop=loop, loop_fraction=loop / (loop + tidal),
                      ep_plugin=sum(r["ep_plugin"] for r in rs), n_tables=len(rs))
    # per-stratum loop fraction at 15 min (exploratory add-on)
    out["per_stratum_15min"] = {}
    for s_ in range(S.NS):
        rs = [r for (ww, _, ss), r in zip(tasks, res) if ww == "15min" and ss == s_ and r is not None]
        if rs:
            tid = sum(r["g_s"] - r["g_qs"] - (r["active"] - 1) for r in rs); lp = sum(r["g_qs"] - r["e0"] for r in rs)
            out["per_stratum_15min"][["classic-member", "classic-casual", "ebike-member", "ebike-casual"][s_]] = dict(
                tidal=tid, loop=lp, loop_fraction=lp / (lp + tid), trips=sum(r["n_trips"] for r in rs))
    if "16h" in out:
        fr = [out[w]["loop_fraction"] for w in WINDOWS]
        out["prediction"] = dict(monotone_increasing=bool(all(a <= b for a, b in zip(fr[:-1], fr[1:]))),
                                 loop_fraction_16h_ge_0_5=bool(out["16h"]["loop_fraction"] >= 0.5))
    os.makedirs("results/cas", exist_ok=True)
    path = (f"results/cas/{I.CITY}.json" if os.environ.get("ONLY_15") else
            f"results/5_tidal_share_{I.CITY}{'_simtide' + SIM_SEED if SIM_SEED else ''}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
