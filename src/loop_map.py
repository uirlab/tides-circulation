"""Exploratory (descriptive, no decision rule): where is the loop irreversibility in Citi Bike May 2026?

Per (15-min bin, stratum) Bradley-Terry fit -> residual log-odds beyond the tidal potential for each
station pair; pooled over bins/strata with Fisher weights. Then: triangle affinities (potential-free),
station loop intensity, and correlates (distance, net imbalance). Run from project root:
    python src/loop_map.py
Outputs: results/0cd_loops.json, data/processed/loops_pairs.npz, data/processed/loops_stations.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402

STATIONS = "data/processed/citibike_202605_stations.csv"
MIN_T = 30  # pair needs >= 30 trips (all bins/strata) to enter the loop map


def pair_residuals(D):
    K = D["K"]
    num = {}  # key -> [sum w*eps, sum w, sum t]
    grp = D.get("month", np.zeros(D["lo"].size, np.int64))  # step 3b: fits per month x bin x stratum
    for g in np.unique(grp):
      for k in range(I.NB):
        for s in range(S.NS):
            m = (grp == g) & (D["b"] == k) & (D["st"] == s)
            if m.sum() < 2:
                continue
            plo, phi, n1, t, _ = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
            u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
            eta = u[plo] - u[phi]
            p = expit(eta)
            eps = logit((n1 + 0.5) / (t + 1.0)) - eta
            w = t * p * (1 - p)
            key = plo * K + phi
            for kk, a, b_, c in zip(key, w * eps, w, t):
                v = num.setdefault(int(kk), [0.0, 0.0, 0.0])
                v[0] += a; v[1] += b_; v[2] += c
    keys = np.fromiter(num.keys(), np.int64)
    arr = np.array(list(num.values()))
    eps_bar = arr[:, 0] / np.maximum(arr[:, 1], 1e-12)
    se = 1 / np.sqrt(np.maximum(arr[:, 1], 1e-12))
    return keys // K, keys % K, eps_bar, se, arr[:, 2], arr[:, 1]


def main():
    D = S.load()
    K = D["K"]
    d = np.load(I.DATA, allow_pickle=True)
    sid, names = d["station_id"].astype(str), d["station_name"].astype(str)
    lo, hi, eps, se, t, wsum = pair_residuals(D)
    keep = t >= MIN_T
    lo, hi, eps, se, t, wsum = lo[keep], hi[keep], eps[keep], se[keep], t[keep], wsum[keep]
    z = eps / se
    np.savez_compressed("data/processed/loops_pairs.npz", lo=lo, hi=hi, eps=eps, se=se, t=t)

    # coordinates
    coord = {}
    with open(STATIONS) as f:
        for r in csv.DictReader(f):
            coord[r["id"]] = (float(r["lat"]), float(r["lon"]))
    lat = np.array([coord.get(x, (np.nan, np.nan))[0] for x in sid]); lon = np.array([coord.get(x, (np.nan, np.nan))[1] for x in sid])
    R = 6371.0
    p1, p2 = np.radians(lat[lo]), np.radians(lat[hi])
    dist = 2 * R * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon[hi] - lon[lo]) / 2) ** 2))

    # triangles among kept pairs (oriented eps: lo->hi positive)
    E = {}
    for a, b, e, s_ in zip(lo, hi, eps, se):
        E[(int(a), int(b))] = (e, s_)
    adj = [set() for _ in range(K)]
    for a, b in E:
        adj[a].add(b); adj[b].add(a)

    def ori(i, j):
        if i < j:
            e, s_ = E[(i, j)]; return e, s_
        e, s_ = E[(j, i)]; return -e, s_

    tri = []
    for i in range(K):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[i] & adj[j]:
                if k <= j:
                    continue
                e1, s1 = ori(i, j); e2, s2 = ori(j, k); e3, s3 = ori(k, i)
                A = e1 + e2 + e3
                sA = np.sqrt(s1 * s1 + s2 * s2 + s3 * s3)
                tri.append((i, j, k, A, A / sA))
    tri = np.array(tri, dtype=float) if tri else np.zeros((0, 5))
    tz = tri[:, 4]
    pvals = 2 * norm.sf(np.abs(tz))
    order = np.argsort(pvals)
    m = pvals.size
    bh = pvals[order] <= 0.05 * np.arange(1, m + 1) / m
    n_fdr = int(np.max(np.nonzero(bh)[0]) + 1) if bh.any() else 0

    # station loop intensity: excess of Fisher-weighted squared residuals over chi2_1 expectation
    excess = z * z - 1
    L = np.bincount(lo, excess, K) + np.bincount(hi, excess, K)
    npairs = np.bincount(lo, minlength=K) + np.bincount(hi, minlength=K)
    dep = np.bincount(D["lo"][D["y"] == 1], minlength=K) + np.bincount(D["hi"][D["y"] == 0], minlength=K)
    arr_ = np.bincount(D["hi"][D["y"] == 1], minlength=K) + np.bincount(D["lo"][D["y"] == 0], minlength=K)
    vol = dep + arr_
    imb = np.abs(dep - arr_) / np.maximum(vol, 1)
    act = npairs >= 5
    rho_imb = spearmanr(L[act] / npairs[act], imb[act])
    rho_vol = spearmanr(L[act] / npairs[act], vol[act])
    with open("data/processed/loops_stations.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["station_id", "name", "lat", "lon", "loop_excess", "n_pairs", "volume", "net_imbalance_frac"])
        for i in range(K):
            w.writerow([sid[i], names[i], lat[i], lon[i], round(float(L[i]), 3), int(npairs[i]), int(vol[i]), round(float(imb[i]), 4)])

    # distance profile
    qs = np.nanpercentile(dist, [0, 20, 40, 60, 80, 100])
    prof = []
    for a, b in zip(qs[:-1], qs[1:]):
        mm = (dist >= a) & (dist <= b)
        prof.append(dict(km=[round(float(a), 2), round(float(b), 2)], n=int(mm.sum()), mean_z2_minus1=float(np.mean(z[mm] ** 2 - 1)),
                         frac_abs_z_gt3=float(np.mean(np.abs(z[mm]) > 3))))
    top = []
    for idx in order[:20]:
        i, j, k, A, zz = tri[idx]
        i, j, k = int(i), int(j), int(k)
        cyc = (i, j, k) if A > 0 else (i, k, j)
        top.append(dict(cycle=[names[c] for c in cyc], log_affinity=round(abs(float(A)), 3), z=round(abs(float(zz)), 1),
                        odds_ratio=round(float(np.exp(abs(A))), 2)))
    out = dict(
        pairs_kept=int(lo.size), min_trips=MIN_T,
        pairs_abs_z_gt3=float(np.mean(np.abs(z) > 3)), pairs_mean_z2=float(np.mean(z * z)),
        triangles=int(m), triangles_fdr05=n_fdr, triangles_abs_z_gt4=int((np.abs(tz) > 4).sum()),
        median_abs_log_affinity_fdr=float(np.median(np.abs(tri[order[:n_fdr], 3]))) if n_fdr else None,
        distance_profile=prof,
        station_loop_vs_imbalance_spearman=[float(rho_imb.correlation), float(rho_imb.pvalue)],
        station_loop_vs_volume_spearman=[float(rho_vol.correlation), float(rho_vol.pvalue)],
        top_triangles=top,
    )
    json.dump(out, open("results/0cd_loops.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "top_triangles"}, indent=1))
    for r in top[:10]:
        print(r)


if __name__ == "__main__":
    main()
