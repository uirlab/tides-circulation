"""H-FARE test: did the transit-linked enrichment of circulation change after Taipei made
the first 30 minutes of YouBike free (28 Feb 2024)? Difference-in-differences against the same months a year earlier.
Run from the repository root, in this order:
    python src/fare.py power            # synthetic power and calibration check (template month g_f2309)
    python src/fare.py month g_f2403    # one outcome month: enrichment, ride share, 100 delete-d jackknife subsets
    python src/fare.py combine          # T1 verdict, S2, event-study data
Outputs: results/fare2024/power.json, results/fare2024/<unit>.json, results/fare2024.json, results/fare2024.md,
results/figdata_fare2024.csv. Seeds are fixed."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from scipy.stats import t as tdist

MODE = sys.argv[1]
UNIT = sys.argv[2] if MODE == "month" else "g_f2309"
os.environ["CITY"] = UNIT; os.environ["BIN_MIN"] = "60"
sys.path.insert(0, "src")
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
import loop_map as L  # noqa: E402

OUT = "results/fare2024"
UNITS = ["g_f2212", "g_f2301", "g_f2302", "g_f2303", "g_f2304", "g_f2305",
         "g_f2312", "g_f2401", "g_f2402", "g_f2403", "g_f2404", "g_f2405"]
CELLS = [UNITS[0:3], UNITS[3:6], UNITS[6:9], UNITS[9:12]]  # Y0 pre, Y0 post, Y1 pre, Y1 post
N_JK, MARGIN = int(os.environ.get("FARE_NJK", "100")), 0.163  # FARE_NJK: smoke tests only
ARR = ("lo", "hi", "y", "day", "tm", "b", "hour", "st")


def labels(D, stable_only=False):
    """Map each station-index pair (lo*K+hi) to its fixed transit-linked label (1, 0) or -1 if unlabelled."""
    X = np.load("data/processed/fare_tau_2309_2311.npz", allow_pickle=True)
    lab = dict(zip(X["pair"], X["T"].astype(int)))
    names = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
    stab = set(np.load("data/processed/fare_stable_stations.npy", allow_pickle=True).astype(str)) if stable_only else None

    def get(lo, hi):
        a, b = names[lo], names[hi]
        if stab is not None and (a not in stab or b not in stab):
            return -1
        return lab.get("\t".join(sorted((a, b))), -1)
    return get


def e_p(D, seed):
    """Debiased per-pair loop excess, exactly as src/xfer.py (pairs with >= 30 rides)."""
    K = D["K"]
    lo, hi, eps, se, t, _ = L.pair_residuals(D)
    k = t >= 30; lo, hi, eps, se, t = lo[k], hi[k], eps[k], se[k], t[k]
    key = lo * K + hi
    p_trip = np.zeros(D["lo"].size)
    for b in range(I.NB):
        for s in range(S.NS):
            m = (D["b"] == b) & (D["st"] == s)
            if m.sum() < 2:
                continue
            plo, phi, n1, tt, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
            u = I.fit_bt(plo, phi, n1, tt, K, I.LAM)
            p_trip[np.flatnonzero(m)] = expit(u[plo] - u[phi])[inv]
    acc = np.zeros(key.size); rng = np.random.default_rng(seed)
    for _ in range(2):
        Dn = dict(D); Dn["y"] = (rng.random(D["lo"].size) < p_trip).astype(float)
        l0, h0, e0, s0, t0, _ = L.pair_residuals(Dn)
        pos = {kk: i for i, kk in enumerate(l0 * K + h0)}
        for i, kk in enumerate(key):
            j = pos.get(kk)
            acc[i] += (1 - (e0[j] / s0[j]) ** 2) if j is not None else 0.0
    return lo, hi, (eps / se) ** 2 - 1 + acc / 2, t


def summarise(lo, hi, ep, t, get):
    lab = np.array([get(a, b) for a, b in zip(lo, hi)])
    m = lab >= 0; T = lab[m] == 1; y = ep[m]; tt = t[m]
    return dict(logE=float(np.log((y[T].sum() / y.sum()) / T.mean())), log_share=float(np.log(tt[T].sum() / tt.sum())),
                n_pairs=int(m.sum()), n_T=int(T.sum()))


_G = {}


def _sub(args):
    """One delete-d jackknife subset: keep the drawn weekdays only."""
    seed, draw = args
    D = _G["D"]; m = np.isin(D["day"], draw)
    Ds = dict(D); Ds.update({k: D[k][m] for k in ARR})
    lo, hi, ep, t = e_p(Ds, seed)
    return {tag: summarise(lo, hi, ep, t, g) for tag, g in _G["get"].items()}


def month():
    D = S.load(); _G["D"] = D
    _G["get"] = {"all": labels(D), "stable": labels(D, stable_only=True)}
    lo, hi, ep, t = e_p(D, 13)
    full = {tag: summarise(lo, hi, ep, t, g) for tag, g in _G["get"].items()}
    n = D["ndays"]; mm = int(round(0.75 * n))
    jobs = [(1000 + s, np.sort(np.random.default_rng(s).choice(n, size=mm, replace=False))) for s in range(N_JK)]
    with Pool(I.NWORKERS) as pool:
        reps = pool.map(_sub, jobs)
    out = dict(unit=UNIT, n_days=n, m_days=mm, factor=mm / (n - mm), full=full,
               jk={tag: {k: [r[tag][k] for r in reps] for k in ("logE", "log_share")} for tag in full})
    json.dump(out, open(f"{OUT}/{UNIT}{os.environ.get('FARE_TAG', '')}.json", "w"), indent=1)
    print(json.dumps(dict(unit=UNIT, n_days=n, pairs=full["all"]["n_pairs"]), indent=1), flush=True)


def did(vals, var_jk):
    """Delta and its conservative variance: max of jackknife-based and between-month estimates."""
    v = [np.asarray(vals[i * 3:(i + 1) * 3]) for i in range(4)]
    delta = (v[3].mean() - v[2].mean()) - (v[1].mean() - v[0].mean())
    vi = float(np.sum(var_jk) / 9)
    s2 = float(sum(((x - x.mean()) ** 2).sum() for x in v) / 8); vii = 4 * s2 / 3
    if vii > vi:
        se = np.sqrt(vii); q = tdist.ppf(0.975, 8); p = float(2 * tdist.sf(abs(delta) / se, 8)); src = "between-month (t, 8 df)"
    else:
        se = np.sqrt(vi); q = 1.96; p = float(2 * norm.sf(abs(delta) / se)); src = "jackknife (normal)"
    return dict(delta=float(delta), se=float(se), ci95=[float(delta - q * se), float(delta + q * se)], p=p, var_source=src,
                var_jackknife=vi, var_between_month=vii, ratio=float(np.exp(delta)),
                ratio_ci95=[float(np.exp(delta - q * se)), float(np.exp(delta + q * se))])


def verdict(r):
    if r["p"] < 0.01 and abs(r["ratio"] - 1) >= 0.15:
        return "CHANGE (" + ("weakened" if r["delta"] < 0 else "strengthened") + ")"
    if -MARGIN < r["ci95"][0] and r["ci95"][1] < MARGIN:
        return "STABLE"
    return "INCONCLUSIVE"


# ---------------- power and calibration (synthetic; template month Sep 2023) ----------------

def _syn_month(args):
    seed, kfac = args
    D = _G["D"]; rng = np.random.default_rng(seed)
    p = expit(_G["eta"] + _G["g"] * np.where(_G["Ttrip"], kfac, 1.0))
    Dn = dict(D); Dn["y"] = (rng.random(D["lo"].size) < p).astype(float)
    if _G.get("jk"):
        n = D["ndays"]; mm = int(round(0.75 * n)); vals = []
        for s in range(N_JK):
            draw = np.random.default_rng(s).choice(n, size=mm, replace=False); m = np.isin(D["day"], draw)
            Ds = dict(Dn); Ds.update({k: Dn[k][m] for k in ARR})
            lo, hi, ep, t = e_p(Ds, seed + s)
            vals.append(summarise(lo, hi, ep, t, _G["get"])["logE"])
        vals = np.asarray(vals); return float(mm / (n - mm) * vals.var())
    lo, hi, ep, t = e_p(Dn, seed)
    return summarise(lo, hi, ep, t, _G["get"])["logE"]


def power():
    D = S.load(); K = D["K"]; _G["D"] = D; get = labels(D); _G["get"] = get
    # tide: per-bin potentials fitted on the template month; circulation: the template's own pair residuals
    eta = np.zeros(D["lo"].size)
    for b in range(I.NB):
        m = D["b"] == b
        plo, phi, n1, tt, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
        u = I.fit_bt(plo, phi, n1, tt, K, I.LAM); eta[np.flatnonzero(m)] = (u[plo] - u[phi])[inv]
    lo, hi, eps, se, t, _ = L.pair_residuals(D); keep = t >= 30
    gpair = dict(zip(lo[keep] * K + hi[keep], eps[keep]))
    tk = D["lo"] * K + D["hi"]
    _G["eta"] = eta; _G["g"] = np.array([gpair.get(k, 0.0) for k in tk])
    lab = {k: get(k // K, k % K) for k in gpair}
    _G["Ttrip"] = np.array([lab.get(k, -1) == 1 for k in tk])
    res = dict(template=UNIT, n_planted_pairs=int(sum(v == 1 for v in lab.values())))
    with Pool(I.NWORKERS) as pool:
        base = np.array(pool.map(_syn_month, [(20000 + i, 1.0) for i in range(50 * 12)])).reshape(50, 12)
        res["baseline_E"] = float(np.exp(base.mean()))
        # E ratio produced by each planted factor (Y1 post months), 20 draws each
        grid = [0.5, 0.6, 0.7, 0.8, 0.9]
        ratio = {k: float(np.exp(np.mean(pool.map(_syn_month, [(30000 + i, k) for i in range(20)])) - base.mean())) for k in grid}
        res["E_ratio_by_k"] = ratio
        k25 = min(grid, key=lambda k: abs(ratio[k] - 0.75)); res["k_for_25pct"] = k25
        post = np.array(pool.map(_syn_month, [(40000 + i, k25) for i in range(50 * 3)])).reshape(50, 3)
    _G["jk"] = True
    with Pool(min(I.NWORKERS, 120)) as pool:
        vjk = np.array(pool.map(_syn_month, [(50000 + i, 1.0) for i in range(10 * 12)])).reshape(10, 12)
    _G["jk"] = False
    vbar = vjk.mean(0)
    null = [did(base[i], vbar) for i in range(50)]
    alt = [did(np.r_[base[i, :9], post[i]], vbar) for i in range(50)]
    emp_sd = float(np.std([r["delta"] for r in null], ddof=1))
    res.update(empirical_sd_delta_null=emp_sd, mean_jackknife_se=float(np.sqrt(vbar.sum() / 9)),
               mean_used_se_null=float(np.mean([r["se"] for r in null])),
               false_change_rate=float(np.mean([verdict(r).startswith("CHANGE") for r in null])),
               stable_rate_null=float(np.mean([verdict(r) == "STABLE" for r in null])),
               power_change_25pct=float(np.mean([verdict(r).startswith("CHANGE") for r in alt])),
               mean_ci_halfwidth_null=float(np.mean([(r["ci95"][1] - r["ci95"][0]) / 2 for r in null])))
    cal = abs(res["mean_jackknife_se"] / emp_sd - 1) <= 0.30
    res["calibration_ok"] = bool(cal)
    res["pass"] = bool(cal and res["false_change_rate"] <= 0.05 and res["power_change_25pct"] >= 0.8)
    res["note"] = "synthetic months contain no month-to-month fluctuation; power is an upper bound"
    json.dump(res, open(f"{OUT}/power.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def combine():
    pw = json.load(open(f"{OUT}/power.json")); val = json.load(open(f"{OUT}/validity.json"))
    R = {u: json.load(open(f"{OUT}/{u}.json")) for u in UNITS}
    tag = "stable" if val["check2_flag"] else "all"
    out = dict(power=pw, validity=val, pairs_used=tag)
    if os.environ.get("Y0_FARE_CHANGE"):
        out["note"] = "Y0 had a fare change (validity check 1): Y1 pre-post only, exploratory"
    for key, name in (("logE", "T1_enrichment"), ("log_share", "S2_ride_share")):
        vals = [R[u]["full"][tag][key] for u in UNITS]
        vjk = [R[u]["factor"] * np.var(R[u]["jk"][tag][key]) for u in UNITS]
        r = did(vals, vjk); r["monthly"] = dict(zip(UNITS, vals)); r["monthly_se"] = dict(zip(UNITS, np.sqrt(vjk).tolist()))
        if name == "T1_enrichment":
            r["verdict"] = verdict(r) if pw["pass"] else "NOT CONFIRMATORY (power/calibration check failed; reported descriptively)"
        out[name] = r
    json.dump(out, open("results/fare2024.json", "w"), indent=1)
    with open("results/figdata_fare2024.csv", "w") as f:
        f.write("unit,year,period,logE,se_logE,log_share,se_log_share\n")
        for i, u in enumerate(UNITS):
            f.write(f"{u},{'Y0' if i < 6 else 'Y1'},{'pre' if i % 6 < 3 else 'post'},{out['T1_enrichment']['monthly'][u]:.5f},"
                    f"{out['T1_enrichment']['monthly_se'][u]:.5f},{out['S2_ride_share']['monthly'][u]:.5f},"
                    f"{out['S2_ride_share']['monthly_se'][u]:.5f}\n")
    t1 = out["T1_enrichment"]
    with open("results/fare2024.md", "w") as f:
        f.write("# H-FARE: transit-linked circulation around Taipei's 28 Feb 2024 fare change\n\n"
                "Run: `python src/fare.py power`, `month <unit>` for the twelve units, `combine`.\n\n"
                f"- Power check: pass = {pw['pass']} (power {pw['power_change_25pct']:.2f}, false change {pw['false_change_rate']:.2f}).\n"
                f"- Pairs used: {tag}.\n"
                f"- T1 verdict: **{t1['verdict']}**; ratio of changes exp(Δ) = {t1['ratio']:.3f} "
                f"(95% CI {t1['ratio_ci95'][0]:.3f}–{t1['ratio_ci95'][1]:.3f}), P = {t1['p']:.2g}; variance: {t1['var_source']}.\n"
                f"- S2 ride share on transit-linked pairs: exp(Δ) = {out['S2_ride_share']['ratio']:.3f} "
                f"(95% CI {out['S2_ride_share']['ratio_ci95'][0]:.3f}–{out['S2_ride_share']['ratio_ci95'][1]:.3f}).\n")
    print(json.dumps({k: out[k] for k in ("T1_enrichment", "S2_ride_share")}, indent=1), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"power": power, "month": month, "combine": combine}[MODE]()
