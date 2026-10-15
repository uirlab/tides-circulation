"""Tidal vs housekeeping irreversibility of OD trips (Phase 0c; ).

Quasi-symmetry (instantaneous detailed balance with arbitrary occupancy) conditional on
pair totals is a Bradley-Terry model: P(lo->hi | pair) = expit(u_lo - u_hi). Its lack of
fit G2_QS is the housekeeping (cycle-affinity) part; G2_S (symmetry) the total asymmetry.

Usage (project root):
    python src/irrev.py potentials   # fit u-bar (per 15-min bin) and delta (day x hour); no G2 computed
    python src/irrev.py synthetic    # S1 calibration, S3 fast-drift null, S2 power
    python src/irrev.py real         # real statistic + decision (only after synthetic checks pass)
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, cg, LinearOperator
from scipy.special import expit, log_expit, xlogy

CITY = os.environ.get("CITY", "citibike")  # citibike (0c/0c-b) | divvy | bluebikes (step 1)
_D0 = f"data/processed/{CITY}_202605_weekday_trips.npz"
DATA = _D0 if os.path.exists(_D0) else f"data/processed/{CITY}_weekday_trips.npz"  # step-3 units use the plain name
SENS = os.environ.get("SENS", "")  # sensitivity-variant tag
POT = f"data/processed/{CITY}{SENS}_202605_potentials.npz"
T0_H, T1_H = int(os.environ.get("T0_H", "6")), int(os.environ.get("T1_H", "22"))
W = float(os.environ.get("BIN_MIN", "15"))
T0, NB = 60.0 * T0_H, int(60 * (T1_H - T0_H) / W)  # baseline: 06:00, 15-min bins, 64 bins to 22:00
LAM, LAM_D = float(os.environ.get("LAM", "0.1")), float(os.environ.get("LAM_D", "1.0"))
B_NULL, R_SYN = 50, 20
S2_SCALES = [0.1, 0.2, 0.4, 0.8, 1.6]
NWORKERS = int(os.environ.get("NWORKERS", "10"))  # Slurm: set to cpus-per-task


# ---------------- core ----------------

def fit_bt(lo, hi, n1, t, K, lam, off=None, iters=50, tol=1e-8):
    """Ridge-penalised Bradley-Terry by damped Newton. Units: 'edges' (pairs or trips) with
    n1 successes (lo->hi) out of t, optional per-edge offset."""
    u = np.zeros(K)
    off = np.zeros(lo.size) if off is None else off
    n0 = t - n1

    def obj(u):
        eta = u[lo] - u[hi] + off
        return (n1 * log_expit(eta) + n0 * log_expit(-eta)).sum() - 0.5 * lam * (u @ u)

    f = obj(u)
    for _ in range(iters):
        eta = u[lo] - u[hi] + off
        p = expit(eta)
        r = n1 - t * p
        g = np.bincount(lo, r, K) - np.bincount(hi, r, K) - lam * u
        w = t * p * (1 - p)
        d = np.bincount(lo, w, K) + np.bincount(hi, w, K) + lam
        H = sp.coo_matrix((np.concatenate([d, -w, -w]),
                           (np.concatenate([np.arange(K), lo, hi]), np.concatenate([np.arange(K), hi, lo]))),
                          shape=(K, K)).tocsr()
        M = LinearOperator((K, K), matvec=lambda v, d=d: v / d)
        step, _ = cg(H, g, M=M, tol=1e-10, atol=0, maxiter=5000)
        a = 1.0
        while True:
            un = u + a * step
            fn = obj(un)
            if fn >= f - 1e-12 or a < 1e-4:
                break
            a *= 0.5
        u, f = un, fn
        if np.max(np.abs(a * step)) < tol:
            break
    return u


def aggregate(lo, hi, y, K):
    key = lo.astype(np.int64) * K + hi
    uk, inv = np.unique(key, return_inverse=True)
    t = np.bincount(inv).astype(float)
    n1 = np.bincount(inv, y).astype(float)
    return (uk // K).astype(np.int64), (uk % K).astype(np.int64), n1, t, inv


def deviances(lo, hi, y, K):
    """G2_QS (BT lack of fit), G2_S (vs symmetry), plus pair arrays and fitted p."""
    plo, phi, n1, t, inv = aggregate(lo, hi, y, K)
    u = fit_bt(plo, phi, n1, t, K, LAM)
    p = expit(u[plo] - u[phi])
    n0 = t - n1
    g_qs = 2 * (xlogy(n1, n1) - xlogy(n1, t * p) + xlogy(n0, n0) - xlogy(n0, t * (1 - p))).sum()
    g_s = 2 * (xlogy(n1, n1) - xlogy(n1, t / 2) + xlogy(n0, n0) - xlogy(n0, t / 2)).sum()
    return g_qs, g_s, (plo, phi, n1, t, p, inv)


def std_resid(lo, hi, y, K):
    plo, phi, n1, t, inv = aggregate(lo, hi, y, K)
    u = fit_bt(plo, phi, n1, t, K, LAM)
    p = expit(u[plo] - u[phi])
    r = (n1 - t * p) / np.sqrt(np.maximum(t * p * (1 - p), 1e-12))
    return plo * K + phi, r


def split_half_sums(lo, hi, y, day, K, return_arrays=False):
    odd = day % 2 == 1
    ko, ro = std_resid(lo[odd], hi[odd], y[odd], K)
    ke, re_ = std_resid(lo[~odd], hi[~odd], y[~odd], K)
    common, io, ie = np.intersect1d(ko, ke, return_indices=True)
    a, b = ro[io], re_[ie]
    if return_arrays:
        return a, b
    return np.array([a.size, a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])


def corr_from_sums(s):
    n, sa, sb, saa, sbb, sab = s
    cov = sab / n - sa * sb / n**2
    va, vb = saa / n - (sa / n) ** 2, sbb / n - (sb / n) ** 2
    rho = cov / np.sqrt(va * vb)
    return float(rho), float(rho * np.sqrt(n))  # rho, Fisher-normal z (independence null)


def decide(z, h, rho, p_split):
    n_z5 = int((z >= 5).sum())
    n_z3_below = int((z < 3).sum())
    med_h = float(np.median(h))
    passed = n_z5 >= 32 and med_h >= 0.10 and rho > 0 and p_split < 0.01
    kill = n_z3_below >= 48 or med_h < 0.03
    return dict(verdict="PASS" if passed else ("KILL" if kill else "WEAK"), n_z_ge5=n_z5,
                n_z_lt3=n_z3_below, median_h=med_h, rho=rho, p_split=p_split)


# ---------------- data ----------------

def load():
    d = np.load(DATA, allow_pickle=True)
    o, e, day, tm = d["o"].astype(np.int64), d["e"].astype(np.int64), d["day"].astype(np.int64), d["tmin"].astype(float)
    keep = (tm >= T0) & (tm < T0 + W * NB)
    o, e, day, tm = o[keep], e[keep], day[keep], tm[keep]
    K = int(d["station_id"].size)
    lo, hi = np.minimum(o, e), np.maximum(o, e)
    y = (o == lo).astype(float)
    b = ((tm - T0) // W).astype(np.int64)
    hour = (tm // 60).astype(np.int64)
    return dict(lo=lo, hi=hi, y=y, day=day, tm=tm, b=b, hour=hour, K=K,
                names=d["station_name"], ndays=int(d["days"].size))


def potentials():
    D = load()
    K, lo, hi, y, b = D["K"], D["lo"], D["hi"], D["y"], D["b"]
    U = np.zeros((NB, K))
    for k in range(NB):
        m = b == k
        plo, phi, n1, t, _ = aggregate(lo[m], hi[m], y[m], K)
        U[k] = fit_bt(plo, phi, n1, t, K, LAM)
    x = np.clip((D["tm"] - (T0 + W / 2)) / W, 0, NB - 1)
    b0 = np.floor(x).astype(np.int64)
    b1 = np.minimum(b0 + 1, NB - 1)
    w = x - b0
    eta_bar = (1 - w) * (U[b0, lo] - U[b0, hi]) + w * (U[b1, lo] - U[b1, hi])
    eta = eta_bar.copy()
    hours = np.arange(T0_H, T1_H)
    Dl = np.zeros((D["ndays"], hours.size, K))
    for di in range(D["ndays"]):
        for hj, h in enumerate(hours):
            m = (D["day"] == di) & (D["hour"] == h)
            if not m.any():
                continue
            # trip-level fit (t=1 each) with offset eta_bar: fit_bt uses y as successes
            delta = fit_bt(lo[m], hi[m], y[m], np.ones(m.sum()), K, LAM_D, off=eta_bar[m])
            Dl[di, hj] = delta
            eta[m] = eta_bar[m] + delta[lo[m]] - delta[hi[m]]
    # S3 amplitude: SD of adjacent-bin potential changes over stations active in both bins
    act = np.abs(U) > 1e-9
    dif = [(U[k + 1] - U[k])[act[k] & act[k + 1]] for k in range(NB - 1)]
    A = float(np.std(np.concatenate(dif)))
    np.savez_compressed(POT, U=U, eta_null=eta.astype(np.float32), eta_bar=eta_bar.astype(np.float32), A_s3=A)
    print(dict(bins=NB, stations=K, trips=int(lo.size), S3_amplitude=A,
               sd_eta_bar=float(eta_bar.std()), sd_delta_part=float((eta - eta_bar).std())))


# ---------------- synthetic ----------------

_G = {}


def _init():
    D = load()
    P = np.load(POT)
    _G.update(D)
    _G["eta"] = P["eta_null"].astype(float)
    _G["A"] = float(P["A_s3"])


def _cyclic_field(plo, phi, K, rng):
    m = plo.size
    e = rng.standard_normal(m)
    div = np.bincount(plo, e, K) - np.bincount(phi, e, K)
    one = np.ones(m)
    deg = np.bincount(plo, one, K) + np.bincount(phi, one, K) + 1e-6
    L = sp.coo_matrix((np.concatenate([deg, -one, -one]),
                       (np.concatenate([np.arange(K), plo, phi]), np.concatenate([np.arange(K), phi, plo]))),
                      shape=(K, K)).tocsr()
    # CG instead of SuperLU spsolve: spsolve segfaulted on a cluster scipy build
    M = LinearOperator((K, K), matvec=lambda v: v / deg)
    x, _ = cg(L, div, M=M, tol=1e-10, atol=0, maxiter=20000)
    return e - (x[plo] - x[phi])


def _syn_bin(k):
    rng = np.random.default_rng(10_000 + k)
    K = _G["K"]
    m = _G["b"] == k
    lo, hi, day, tm, eta = _G["lo"][m], _G["hi"][m], _G["day"][m], _G["tm"][m], _G["eta"][m]
    draw = lambda ex=0.0: (rng.random(lo.size) < expit(eta + ex)).astype(float)
    null = np.array([deviances(lo, hi, draw(), K)[0] for _ in range(B_NULL)])
    out = dict(bin=k, null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)))

    def rep(ex):
        y = draw(ex)
        g_qs, g_s, _ = deviances(lo, hi, y, K)
        return g_qs, g_s, split_half_sums(lo, hi, y, day, K)

    out["S1"] = [rep(0.0) for _ in range(R_SYN)]
    ph = rng.uniform(0, 2 * np.pi, K)
    out["S3"] = []
    for _ in range(R_SYN):
        ph = rng.uniform(0, 2 * np.pi, K)
        ex = _G["A"] * (np.sin(2 * np.pi * tm / 30 + ph[lo]) - np.sin(2 * np.pi * tm / 30 + ph[hi]))
        out["S3"].append(rep(ex))
    plo, phi, _, _, inv = aggregate(lo, hi, np.zeros(lo.size), K)
    out["S2"] = {}
    for s in S2_SCALES:
        reps = []
        for _ in range(R_SYN):
            c = _cyclic_field(plo, phi, K, rng)
            reps.append(rep(s * c[inv]))
        out["S2"][s] = reps
    return out


def _evaluate(res, key, sub=None):
    """Apply the decision rule to each synthetic replicate r across all bins."""
    mean = np.array([r["null_mean"] for r in res])
    sd = np.array([r["null_sd"] for r in res])
    reps = [(r[key] if sub is None else r[key][sub]) for r in res]
    verdicts, zs_all, hs = [], [], []
    for i in range(R_SYN):
        g_qs = np.array([reps[k][i][0] for k in range(NB)])
        g_s = np.array([reps[k][i][1] for k in range(NB)])
        sums = np.sum([reps[k][i][2] for k in range(NB)], axis=0)
        z = (g_qs - mean) / sd
        h = (g_qs - mean) / (g_s - mean)  # share of the signal above noise
        rho, zr = corr_from_sums(sums)
        from scipy.stats import norm
        dcs = decide(z, h, rho, float(norm.sf(zr)))
        verdicts.append(dcs["verdict"]); zs_all.append(z); hs.append(float(np.median(h)))
    zs_all = np.concatenate(zs_all)
    return dict(verdicts={v: verdicts.count(v) for v in set(verdicts)}, frac_abs_z_gt3=float((np.abs(zs_all) > 3).mean()),
                median_h_over_reps=float(np.median(hs)), median_z=float(np.median(zs_all)))


def synthetic():
    t = time.time()
    with Pool(NWORKERS, initializer=_init) as pool:
        res = pool.map(_syn_bin, range(NB))
    import pickle
    pickle.dump(res, open("data/processed/0c_irrev_synthetic_raw.pkl", "wb"))
    summ = dict(S1=_evaluate(res, "S1"), S3=_evaluate(res, "S3"), S2={str(s): _evaluate(res, "S2", s) for s in S2_SCALES})
    ok_s1 = summ["S1"]["verdicts"].get("PASS", 0) == 0 and summ["S1"]["frac_abs_z_gt3"] <= 0.05
    ok_s3 = summ["S3"]["verdicts"].get("PASS", 0) <= 2
    power_scale = next((s for s in S2_SCALES if summ["S2"][str(s)]["median_h_over_reps"] >= 0.10), None)
    ok_s2 = power_scale is not None and summ["S2"][str(power_scale)]["verdicts"].get("PASS", 0) >= 16
    summ["requirements"] = dict(S1=ok_s1, S3=ok_s3, S2=ok_s2, S2_scale_used=power_scale, all=bool(ok_s1 and ok_s3 and ok_s2))
    summ["runtime_s"] = round(time.time() - t)
    os.makedirs("results", exist_ok=True)
    json.dump(summ, open("results/0c_irrev_synthetic.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


# ---------------- real ----------------

def _real_bin(k):
    rng = np.random.default_rng(20_000 + k)
    K = _G["K"]
    m = _G["b"] == k
    lo, hi, y, day, eta = _G["lo"][m], _G["hi"][m], _G["y"][m], _G["day"][m], _G["eta"][m]
    null = np.array([deviances(lo, hi, (rng.random(lo.size) < expit(eta)).astype(float), K)[0] for _ in range(B_NULL)])
    g_qs, g_s, (plo, phi, n1, t, p, inv) = deviances(lo, hi, y, K)
    p_trip = p[inv]
    naive = np.array([deviances(lo, hi, (rng.random(lo.size) < p_trip).astype(float), K)[0] for _ in range(B_NULL)])
    a, b = split_half_sums(lo, hi, y, day, K, return_arrays=True)
    return dict(bin=k, g_qs=float(g_qs), g_s=float(g_s), null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                naive_mean=float(naive.mean()), naive_sd=float(naive.std(ddof=1)), n_trips=int(lo.size),
                n_pairs=int(plo.size), res_odd=a, res_even=b)


def real():
    with Pool(NWORKERS, initializer=_init) as pool:
        res = pool.map(_real_bin, range(NB))
    g_qs = np.array([r["g_qs"] for r in res]); g_s = np.array([r["g_s"] for r in res])
    mean = np.array([r["null_mean"] for r in res]); sd = np.array([r["null_sd"] for r in res])
    z = (g_qs - mean) / sd
    h = (g_qs - mean) / (g_s - mean)  # share of the signal above noise
    a = np.concatenate([r["res_odd"] for r in res]); b = np.concatenate([r["res_even"] for r in res])
    rho = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(3)
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(1000)])
    p_split = float((1 + (perm >= rho).sum()) / 1001)
    dec = decide(z, h, rho, p_split)
    naive_z = (g_qs - np.array([r["naive_mean"] for r in res])) / np.array([r["naive_sd"] for r in res])
    out = dict(decision=dec, per_bin=[dict(bin=r["bin"], start=f"{int((T0 + W * r['bin']) // 60):02d}:{int((T0 + W * r['bin']) % 60):02d}",
                                          n_trips=r["n_trips"], n_pairs=r["n_pairs"], g_qs=r["g_qs"], g_s=r["g_s"],
                                          null_mean=r["null_mean"], null_sd=r["null_sd"], z=float(z[i]), h=float(h[i]),
                                          tidal_share_raw=float(1 - r["g_qs"] / r["g_s"]), naive_z=float(naive_z[i]))
                                     for i, r in enumerate(res)],
               split_half=dict(n=int(a.size), rho=rho, p_perm=p_split))
    json.dump(out, open("results/0c_irrev_real.json", "w"), indent=1)
    print(json.dumps(dict(decision=dec, split_half=out["split_half"]), indent=1))


if __name__ == "__main__":
    {"potentials": potentials, "synthetic": synthetic, "real": real}[sys.argv[1]]()
