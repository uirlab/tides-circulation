"""EXPLORATORY (NYC May 2026 only; June held out): magnitude predictors of the pair loop excess.
Pair excess e_p = w*eps^2 - 1 + delta_p (delta_p from count-level no-loop null replicates).
Direction-free predictors: habit regularity R (from trip TIMES only, both directions pooled), greenway share G,
river crossing C, plus controls log t, distance, |D|. Writes results/exp_loop_sources2.json"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import expit

sys.path.insert(0, "src")
os.environ["CITY"] = "citibike"
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
import loop_map as L  # noqa: E402
import mechanism_share as M  # noqa: E402

D = S.load(); K = D["K"]
d = np.load(I.DATA, allow_pickle=True)
sid, lat, lon = d["station_id"].astype(str), d["lat"], d["lon"]
lo, hi, eps, se, t, _ = L.pair_residuals(D)
keep = t >= 30
lo, hi, eps, se, t = lo[keep], hi[keep], eps[keep], se[keep], t[keep]

# null contributions per pair (2 replicates)
p_trip = np.zeros(D["lo"].size)
for k in range(I.NB):
    for s in range(S.NS):
        m = (D["b"] == k) & (D["st"] == s)
        if m.sum() < 2:
            continue
        plo, phi, n1, tt, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
        u = I.fit_bt(plo, phi, n1, tt, K, I.LAM)
        p_trip[np.flatnonzero(m)] = expit(u[plo] - u[phi])[inv]
key = lo * K + hi
acc = np.zeros(key.size); rng = np.random.default_rng(12)
for _ in range(2):
    Dn = dict(D); Dn["y"] = (rng.random(D["lo"].size) < p_trip).astype(float)
    l0, h0, e0, s0, t0, _ = L.pair_residuals(Dn)
    pos = {kk: i for i, kk in enumerate(l0 * K + h0)}
    for i, kk in enumerate(key):
        j = pos.get(kk)
        acc[i] += (1 - (e0[j] / s0[j]) ** 2) if j is not None else 0.0
delta_p = acc / 2
e_p = (eps / se) ** 2 - 1 + delta_p

# habit regularity: per pair, max over 30-min slots of #distinct weekdays with >= 1 trip (directions pooled) / #weekdays
slot = (D["tm"] // 30).astype(np.int64)
pk = D["lo"] * K + D["hi"]
sel = np.isin(pk, key)
code = (pk[sel] * 64 + slot[sel]) * 32 + D["day"][sel]
u_code = np.unique(code)
pair_slot = u_code // 32
ps, cnt = np.unique(pair_slot, return_counts=True)  # distinct days per (pair, slot)
pair_of = ps // 64
order = np.argsort(pair_of)
mx = {}
for p_, c_ in zip(pair_of, cnt):
    if c_ > mx.get(p_, 0):
        mx[p_] = c_
R = np.array([mx.get(kk, 0) for kk in key]) / D["ndays"]

# greenway share of the straight segment (within 100 m), river crossing
import json as _j
F = _j.load(open("data/raw/nycdot/bike_routes_mzxg-pwib.geojson"))["features"]
lat0 = np.radians(40.73)
toxy = lambda la, lo_: np.c_[np.radians(lo_) * np.cos(lat0) * 6371000, np.radians(la) * 6371000]
pts = []
for f in F:
    p = f["properties"]
    if p["status"] == "Current" and p.get("grnwy") == "Greenway":
        g = f["geometry"]; lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
        for ln in lines:
            arr = np.array(ln)
            for a_, b_ in zip(arr[:-1], arr[1:]):  # densify every ~20 m
                n = max(2, int(np.hypot(*(toxy(b_[1], b_[0]) - toxy(a_[1], a_[0]))[0]) // 20) + 1)
                pts.append(np.c_[np.linspace(a_[1], b_[1], n), np.linspace(a_[0], b_[0], n)])
GP = np.vstack(pts); gtree = cKDTree(toxy(GP[:, 0], GP[:, 1]))
XY = toxy(lat, lon)
G = np.zeros(key.size)
for i, (a, b) in enumerate(zip(lo, hi)):
    if not (np.isfinite(XY[a]).all() and np.isfinite(XY[b]).all()):
        G[i] = np.nan; continue
    q = XY[a][None, :] + np.linspace(0, 1, 21)[:, None] * (XY[b] - XY[a])[None, :]
    G[i] = float(np.mean(gtree.query(q)[0] <= 100))
C = json.load(open("data/raw/nycdot/centerline_inkn-q76z.geojson"))["features"]
mid, bor = [], []
for f in C:
    g = f.get("geometry")
    if not g:
        continue
    cs = g["coordinates"][0] if g["type"] == "MultiLineString" else g["coordinates"]
    if not cs:
        continue
    c0 = cs[len(cs) // 2]; mid.append((c0[1], c0[0])); bor.append(f["properties"].get("boroughcode"))
mid = np.array(mid); bor = np.array(bor)
okst = np.isfinite(lat)
b_st = np.full(K, None, object)
b_st[okst] = bor[cKDTree(toxy(mid[:, 0], mid[:, 1])).query(XY[okst])[1]]
Cx = np.array([(b_st[a] != b_st[b]) and (b_st[a] == "1" or b_st[b] == "1") for a, b in zip(lo, hi)], float)

Dv, okd, und = M.covariate("citibike", sid[lo], sid[hi])
ok = okd & np.isfinite(G)
X = np.c_[np.log(t), np.log(und), np.abs(Dv), R, G, Cx][ok]
names = ["log_t", "log_dist", "absD", "habit_R", "greenway_G", "river_cross"]
y = e_p[ok]
Z = (X - X.mean(0)) / X.std(0)
beta = np.linalg.lstsq(np.c_[np.ones(len(y)), Z], y, rcond=None)[0]
resid = y - np.c_[np.ones(len(y)), Z] @ beta
s2 = resid @ resid / (len(y) - Z.shape[1] - 1)
se_b = np.sqrt(np.diag(s2 * np.linalg.inv(np.c_[np.ones(len(y)), Z].T @ np.c_[np.ones(len(y)), Z])))
out = dict(n_pairs=int(ok.sum()), total_excess=float(y.sum()), mean_excess=float(y.mean()),
           ols_std_beta={n: [float(b), float(s)] for n, b, s in zip(names, beta[1:], se_b[1:])})
# concentration: share of excess in top decile / flagged groups
for n, col in zip(names, X.T):
    if n == "river_cross":
        m = col > 0.5
    elif n == "greenway_G":
        m = col >= 0.3
    else:
        m = col >= np.quantile(col, 0.9)
    out[f"conc_{n}"] = dict(pair_share=float(m.mean()), excess_share=float(y[m].sum() / y.sum()), mean_excess=float(y[m].mean()))
out["R_quartiles"] = np.quantile(X[:, 3], [0.25, 0.5, 0.75, 0.9]).tolist()
json.dump(out, open("results/exp_loop_sources2.json", "w"), indent=1)
print(json.dumps(out, indent=1))
