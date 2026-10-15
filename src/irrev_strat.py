"""Phase 0c-b: housekeeping irreversibility stratified by bike type x rider type. Reuses the fitting/deviance code of src/irrev.py.
Usage (project root):  python src/irrev_strat.py all   # potentials -> synthetic -> real (real only if synthetic passes)
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import expit
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import irrev as I  # noqa: E402

POT_S = f"data/processed/{I.CITY}{I.SENS}_202605_potentials_strata.npz"
NS = 4  # strata 0..3 = classic/e-bike x member/casual
OUT = ("results/0cb_strata" if I.CITY == "citibike" else
       f"results/3_units/{I.CITY}" if I.CITY.startswith("u_") else f"results/1_replication_{I.CITY}") + I.SENS
_cnt = np.bincount(np.load(I.DATA)["stratum"].clip(0), minlength=NS) if os.path.exists(I.DATA) else np.zeros(NS)
BIG = int(np.argmax(_cnt))  # largest stratum in this city (spec: e-bike x member for Citi Bike)
B_NULL, R_SYN = 50, 20
S2_SCALES = [0.4, 0.8, 1.6]  # power at the smallest scale reaching the h' thresholds
NW = I.NWORKERS


def load():
    D = I.load()
    d = np.load(I.DATA, allow_pickle=True)
    tm = d["tmin"].astype(float)
    keep = (tm >= I.T0) & (tm < I.T0 + I.W * I.NB)
    D["st"] = d["stratum"].astype(np.int64)[keep]
    if os.environ.get("BY_MONTH"):  # step 3b pooled multi-month units: month index per trip
        mon = np.array([x[:7] for x in d["days"].astype(str)])
        D["month"] = np.unique(mon, return_inverse=True)[1][d["day"].astype(np.int64)][keep]
    return D


def _pot_stratum(s):
    D = load()
    K, NB = D["K"], I.NB
    m_s = D["st"] == s
    lo, hi, y, b, tm, day, hour = (D[k][m_s] for k in ("lo", "hi", "y", "b", "tm", "day", "hour"))
    U = np.zeros((NB, K))
    for k in range(NB):
        m = b == k
        plo, phi, n1, t, _ = I.aggregate(lo[m], hi[m], y[m], K)
        U[k] = I.fit_bt(plo, phi, n1, t, K, I.LAM)
    x = np.clip((tm - (I.T0 + I.W / 2)) / I.W, 0, NB - 1)
    b0 = np.floor(x).astype(np.int64); b1 = np.minimum(b0 + 1, NB - 1); w = x - b0
    eta_bar = (1 - w) * (U[b0, lo] - U[b0, hi]) + w * (U[b1, lo] - U[b1, hi])
    eta = eta_bar.copy()
    for di in range(D["ndays"]):
        for h in range(I.T0_H, I.T1_H):
            m = (day == di) & (hour == h)
            if m.any():
                dl = I.fit_bt(lo[m], hi[m], y[m], np.ones(m.sum()), K, I.LAM_D, off=eta_bar[m])
                eta[m] = eta_bar[m] + dl[lo[m]] - dl[hi[m]]
    return s, np.flatnonzero(m_s), eta


def potentials():
    t = time.time()
    D = load()
    eta = np.full(D["lo"].size, np.nan)
    with Pool(NS) as pool:
        for s, idx, e in pool.map(_pot_stratum, range(NS)):
            eta[idx] = e
    assert np.isfinite(eta).all()
    np.savez_compressed(POT_S, eta_null=eta.astype(np.float32))
    print(dict(stage="potentials", runtime_s=round(time.time() - t)), flush=True)


_G = {}


def _init():
    _G.update(load())
    _G["eta"] = np.load(POT_S)["eta_null"].astype(float)


def strat_stats(lo, hi, y, day, st, K, split=True, arrays=False):
    gqs = gs = 0.0
    big = None
    sums = np.zeros(6); arr = ([], [])
    for s in range(NS):
        m = st == s
        if m.sum() < 2:
            continue
        g1, g2, _ = I.deviances(lo[m], hi[m], y[m], K)
        gqs += g1; gs += g2
        if split:
            if arrays:
                a, b = I.split_half_sums(lo[m], hi[m], y[m], day[m], K, return_arrays=True)
                arr[0].append(a); arr[1].append(b)
                if s == BIG:
                    big = (g1, g2, a, b)
            else:
                ss = I.split_half_sums(lo[m], hi[m], y[m], day[m], K)
                sums += ss
                if s == BIG:
                    big = (g1, g2, ss)
        elif s == BIG:
            big = (g1, g2)
    if arrays:
        return gqs, gs, (np.concatenate(arr[0]), np.concatenate(arr[1])), big
    return gqs, gs, sums, big


def _syn_bin(k):
    rng = np.random.default_rng(30_000 + k)
    K = _G["K"]
    m = _G["b"] == k
    lo, hi, day, st, eta = _G["lo"][m], _G["hi"][m], _G["day"][m], _G["st"][m], _G["eta"][m]
    draw = lambda ex=0.0: (rng.random(lo.size) < expit(eta + ex)).astype(float)
    nulls, nulls_big = [], []
    for _ in range(B_NULL):
        g, _, _, big = strat_stats(lo, hi, draw(), day, st, K, split=False)
        nulls.append(g); nulls_big.append(big[0])
    out = dict(bin=k, null_mean=float(np.mean(nulls)), null_sd=float(np.std(nulls, ddof=1)),
               big_null_mean=float(np.mean(nulls_big)), big_null_sd=float(np.std(nulls_big, ddof=1)))

    def rep(ex, pooled=False):
        y = draw(ex)
        g, gs, sums, big = strat_stats(lo, hi, y, day, st, K)
        r = dict(g=g, gs=gs, sums=sums, big_g=big[0], big_gs=big[1], big_sums=big[2])
        if pooled:
            pg, pgs, _ = I.deviances(lo, hi, y, K)
            r.update(pooled_g=pg, pooled_gs=pgs)
        return r

    out["S1"] = [rep(0.0, pooled=True) for _ in range(R_SYN)]
    plo, phi, _, _, inv = I.aggregate(lo, hi, np.zeros(lo.size), K)
    out["S2"] = {str(sc): [rep(sc * I._cyclic_field(plo, phi, K, rng)[inv]) for _ in range(R_SYN)] for sc in S2_SCALES}
    return out


def decide(z, h, h_big, rho, p, rho_big, p_big):
    # z >= 5 in >= 50% of bins (= 32/64 at baseline; fraction for other bin counts, as in the sensitivity analysis)
    go = (int((z >= 5).sum()) >= 0.5 * len(z) and np.median(h) >= 0.10 and np.median(h_big) >= 0.10
          and rho > 0 and p < 0.01 and rho_big > 0 and p_big < 0.01)
    het = np.median(h) < 0.03 or int((z < 3).sum()) >= 0.75 * len(z)  # = 48/64 at baseline
    return dict(verdict="GO" if go else ("HETEROGENEITY" if het else "PARTIAL"), n_z_ge5=int((z >= 5).sum()),
                n_z_lt3=int((z < 3).sum()), median_h=float(np.median(h)), median_h_big=float(np.median(h_big)),
                rho=rho, p_split=p, rho_big=rho_big, p_split_big=p_big)


def _eval(res, key, sub=None):
    mean = np.array([r["null_mean"] for r in res]); sd = np.array([r["null_sd"] for r in res])
    bm = np.array([r["big_null_mean"] for r in res])
    verdicts, zs, het_h, mh, mhb, fails = [], [], [], [], [], []
    for i in range(R_SYN):
        reps = [(r[key] if sub is None else r[key][sub])[i] for r in res]
        g = np.array([x["g"] for x in reps]); gs = np.array([x["gs"] for x in reps])
        bg = np.array([x["big_g"] for x in reps]); bgs = np.array([x["big_gs"] for x in reps])
        z = (g - mean) / sd; h = (g - mean) / (gs - mean); hb = (bg - bm) / (bgs - bm)
        rho, zr = I.corr_from_sums(np.sum([x["sums"] for x in reps], axis=0))
        rb, zrb = I.corr_from_sums(np.sum([x["big_sums"] for x in reps], axis=0))
        d = decide(z, h, hb, rho, float(norm.sf(zr)), rb, float(norm.sf(zrb)))
        verdicts.append(d["verdict"]); mh.append(float(np.median(h))); mhb.append(float(np.median(hb)))
        fails.append([c for c, ok in [("n_z_ge5", d["n_z_ge5"] >= 32), ("median_h", d["median_h"] >= 0.10),
                                      ("median_h_big", d["median_h_big"] >= 0.10), ("split", d["p_split"] < 0.01 and d["rho"] > 0),
                                      ("split_big", d["p_split_big"] < 0.01 and d["rho_big"] > 0)] if not ok])
        zs.append(z)
        if "pooled_g" in reps[0] and _G_pooled_mean is not None:
            pm = _G_pooled_mean
            pg = np.array([x["pooled_g"] for x in reps]); pgs = np.array([x["pooled_gs"] for x in reps])
            het_h.append(float(np.median((pg - pm) / (pgs - pm))))
    zs = np.concatenate(zs)
    out = dict(verdicts={v: verdicts.count(v) for v in set(verdicts)}, frac_abs_z_gt3=float((np.abs(zs) > 3).mean()),
               median_z=float(np.median(zs)), median_h=float(np.median(mh)), median_h_big=float(np.median(mhb)),
               failing_criteria={c: sum(c in f for f in fails) for c in ("n_z_ge5", "median_h", "median_h_big", "split", "split_big")})
    if het_h:
        out["heterogeneity_artifact_unstratified_median_h"] = float(np.median(het_h))
    return out


_G_pooled_mean = np.array([r["null_mean"] for r in json.load(open("results/0c_irrev_real.json"))["per_bin"]]) \
    if (I.CITY == "citibike" and not I.SENS and os.path.exists("results/0c_irrev_real.json")) else None  # 0c pooled null: Citi Bike baseline (15-min) only


def synthetic():
    t = time.time()
    with Pool(NW, initializer=_init) as pool:
        res = pool.map(_syn_bin, range(I.NB))
    summ = dict(S1=_eval(res, "S1"), S2={str(sc): _eval(res, "S2", str(sc)) for sc in S2_SCALES})
    ok1 = summ["S1"]["verdicts"].get("GO", 0) == 0 and summ["S1"]["frac_abs_z_gt3"] <= 0.05
    sc = next((str(x) for x in S2_SCALES if summ["S2"][str(x)]["median_h"] >= 0.10 and summ["S2"][str(x)]["median_h_big"] >= 0.10), None)
    ok2 = sc is not None and summ["S2"][sc]["verdicts"].get("GO", 0) >= 16
    summ["requirements"] = dict(S1=ok1, S2=ok2, S2_scale_used=sc, all=bool(ok1 and ok2))
    summ["runtime_s"] = round(time.time() - t)
    json.dump(summ, open(f"{OUT}_synthetic.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


def _real_bin(k):
    rng = np.random.default_rng(40_000 + k)
    K = _G["K"]
    m = _G["b"] == k
    lo, hi, y, day, st, eta = (_G[x][m] for x in ("lo", "hi", "y", "day", "st", "eta"))
    nulls, nb = [], []
    for _ in range(B_NULL):
        g, _, _, big = strat_stats(lo, hi, (rng.random(lo.size) < expit(eta)).astype(float), day, st, K, split=False)
        nulls.append(g); nb.append(big[0])
    g, gs, (a, b), big = strat_stats(lo, hi, y, day, st, K, arrays=True)
    per_s = {}
    for s in range(NS):
        ms = st == s
        g1, g2, _ = I.deviances(lo[ms], hi[ms], y[ms], K)
        per_s[s] = (g1, g2)
    return dict(bin=k, g=g, gs=gs, null_mean=float(np.mean(nulls)), null_sd=float(np.std(nulls, ddof=1)),
                big_g=big[0], big_gs=big[1], big_null_mean=float(np.mean(nb)), big_null_sd=float(np.std(nb, ddof=1)),
                a=a, b=b, big_a=big[2], big_b=big[3], per_stratum=per_s)


def _perm(a, b, rng, n=1000):
    rho = float(np.corrcoef(a, b)[0, 1])
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(n)])
    return rho, float((1 + (perm >= rho).sum()) / (n + 1))


def real():
    if os.environ.get("SKIP_SYN_GATE"):  # bootstrap replicates reuse the gate of the unit they resample (R2a)
        req = {"all": True}
    else:
        req = json.load(open(f"{OUT}_synthetic.json"))["requirements"]
    if not req["all"]:
        print("Synthetic requirements FAILED; real statistic not run (amend spec first).", req, flush=True)
        return
    with Pool(NW, initializer=_init) as pool:
        res = pool.map(_real_bin, range(I.NB))
    g = np.array([r["g"] for r in res]); gs = np.array([r["gs"] for r in res])
    mean = np.array([r["null_mean"] for r in res]); sd = np.array([r["null_sd"] for r in res])
    bg = np.array([r["big_g"] for r in res]); bgs = np.array([r["big_gs"] for r in res])
    bm = np.array([r["big_null_mean"] for r in res]); bsd = np.array([r["big_null_sd"] for r in res])
    z = (g - mean) / sd; h = (g - mean) / (gs - mean); hb = (bg - bm) / (bgs - bm); zb = (bg - bm) / bsd
    rng = np.random.default_rng(5)
    rho, p = _perm(np.concatenate([r["a"] for r in res]), np.concatenate([r["b"] for r in res]), rng)
    rb, pb = _perm(np.concatenate([r["big_a"] for r in res]), np.concatenate([r["big_b"] for r in res]), rng)
    dec = decide(z, h, hb, rho, p, rb, pb)
    if I.CITY == "citibike":
        dec["retention_vs_0c"] = dec["median_h"] / 0.37761082437720916
    dec["big_stratum"] = BIG
    dec["big_n_z_ge5"] = int((zb >= 5).sum())
    out = dict(decision=dec, per_bin=[dict(bin=r["bin"], z=float(z[i]), h=float(h[i]), z_big=float(zb[i]), h_big=float(hb[i]),
                                          g=r["g"], gs=r["gs"], null_mean=r["null_mean"],
                                          per_stratum_raw={str(s): v for s, v in r["per_stratum"].items()})
                                     for i, r in enumerate(res)])
    json.dump(out, open(f"{OUT}_real.json", "w"), indent=1)
    print(json.dumps(dec, indent=1), flush=True)


def main():
    """Run each stage in its own interpreter: a second fork-Pool in the same process deadlocked
    on a cluster (workers inherited a held queue lock from the first pool)."""
    import subprocess
    for stage in ("potentials", "synthetic", "real"):
        subprocess.run([sys.executable, __file__, stage], check=True)


if __name__ == "__main__":
    {"all": main, "potentials": potentials, "synthetic": synthetic, "real": real}[sys.argv[1]]()
