"""H-ACT: do antisymmetric land-use interactions explain the loop field beyond one-way geometry? Run from the repository root (cluster recommended):
    NWORKERS=32 python src/hact.py
Writes results/hact.json. The real statistic runs only if the count-level synthetic gate passes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from scipy.special import expit, logit

sys.path.insert(0, "src")
R_M = 250.0
WORK_COLS = ["CNS09", "CNS10", "CNS11", "CNS12", "CNS13", "CNS14", "CNS20"]
FOOD = {"food_and_drink", "shopping"}
LEIS = {"arts_and_entertainment", "cultural_and_historic", "sports_and_recreation"}
TYPES = ["RES", "WORK", "FOOD", "LEIS"]
PAIRS_IJ = [(i, j) for i in range(4) for j in range(i + 1, 4)]
NSIM = 50
KAPPA = float(os.environ.get("KAPPA", "2.39"))  # calibrated planted-effect scale
NW = int(os.environ.get("NWORKERS", "8"))


# ---------------- profiles ----------------

def xy(lat, lon, lat0):
    return np.c_[np.radians(lon) * np.cos(np.radians(lat0)) * 6371000, np.radians(lat) * 6371000]


def profiles(npz, state, ov):
    d = np.load(npz, allow_pickle=True)
    sid, lat, lon = d["station_id"].astype(str), d["lat"], d["lon"]
    ok = np.isfinite(lat) & np.isfinite(lon)
    lat0 = float(np.nanmedian(lat))
    S = xy(lat[ok], lon[ok], lat0)
    xw = pd.read_csv(f"data/raw/lodes/{state}_xwalk.csv.gz", usecols=["tabblk2020", "blklatdd", "blklondd"], dtype={"tabblk2020": str})
    rac = pd.read_csv(f"data/raw/lodes/{state}_rac_S000_JT00_2023.csv.gz", usecols=["h_geocode", "C000"], dtype={"h_geocode": str})
    wac = pd.read_csv(f"data/raw/lodes/{state}_wac_S000_JT00_2023.csv.gz", usecols=["w_geocode"] + WORK_COLS, dtype={"w_geocode": str})
    xw = xw.set_index("tabblk2020")
    out = np.full((sid.size, 4), np.nan)
    cols = []
    for df, key, val in ((rac, "h_geocode", rac["C000"].values), (wac, "w_geocode", wac[WORK_COLS].sum(1).values)):
        g = xw.reindex(df[key].values)
        m = g.blklatdd.notna().values
        tree = cKDTree(xy(g.blklatdd.values[m], g.blklondd.values[m], lat0))
        w = val[m].astype(float)
        nb = tree.query_ball_point(S, R_M)
        cols.append(np.array([w[ix].sum() for ix in nb]))
    P = pq.read_table(f"data/raw/overture/places_{ov}.parquet", columns=["tax_top", "operating_status", "confidence", "lat", "lon"]).to_pandas()
    P = P[(P.operating_status != "permanently_closed") & (P.confidence >= 0.5)]
    for cats in (FOOD, LEIS):
        q = P[P.tax_top.isin(cats)]
        tree = cKDTree(xy(q.lat.values, q.lon.values, lat0))
        cols.append(np.array([len(ix) for ix in tree.query_ball_point(S, R_M)], float))
    X = np.log1p(np.c_[tuple(cols)])
    X = (X - X.mean(0)) / X.std(0)
    out[ok] = X
    return {s: out[i] for i, s in enumerate(sid) if ok[i]}


def feats(prof, a_ids, b_ids):
    Xa = np.array([prof.get(s, [np.nan] * 4) for s in a_ids]); Xb = np.array([prof.get(s, [np.nan] * 4) for s in b_ids])
    F = np.c_[tuple(Xa[:, i] * Xb[:, j] - Xa[:, j] * Xb[:, i] for i, j in PAIRS_IJ)]
    return F


# ---------------- residual machinery ----------------

def load_month(city, by_month=False):
    os.environ["CITY"] = city
    if by_month:
        os.environ["BY_MONTH"] = "1"
    else:
        os.environ.pop("BY_MONTH", None)
    for m in ("irrev", "irrev_strat", "loop_map", "mechanism_share"):
        sys.modules.pop(m, None)
    import irrev as I
    import irrev_strat as S
    D = S.load()
    sid = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
    K = D["K"]
    p_trip = np.zeros(D["lo"].size)
    grp = D.get("month", np.zeros(D["lo"].size, np.int64))
    for g in np.unique(grp):
        for k in range(I.NB):
            for s in range(S.NS):
                m = (grp == g) & (D["b"] == k) & (D["st"] == s)
                if m.sum() < 2:
                    continue
                plo, phi, n1, t, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
                u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
                p_trip[np.flatnonzero(m)] = expit(u[plo] - u[phi])[inv]
    return D, sid, p_trip


def residuals_from(D, sid, y=None, subset=None):
    import loop_map as L
    Dn = dict(D)
    if y is not None:
        Dn["y"] = y
    if subset is not None:
        Dn = {k: (v[subset] if isinstance(v, np.ndarray) and v.shape == D["lo"].shape else v) for k, v in Dn.items()}
    lo, hi, eps, se, t, _ = L.pair_residuals(Dn)
    k = t >= 30
    return sid[lo[k]], sid[hi[k]], eps[k], se[k]


def dcov(osm, a, b):
    P = np.load(f"data/processed/legal_paths_{osm}.npz")
    pos = {s: i for i, s in enumerate(P["station_id"].astype(str))}
    L = P["L"]
    ia = np.array([pos.get(s, -1) for s in a]); ib = np.array([pos.get(s, -1) for s in b])
    Dv = np.full(a.size, np.nan); ok = (ia >= 0) & (ib >= 0)
    lab, lba = np.full(a.size, np.inf), np.full(a.size, np.inf)
    lab[ok] = L[ia[ok], ib[ok]]; lba[ok] = L[ib[ok], ia[ok]]
    ok &= np.isfinite(lab) & np.isfinite(lba) & (lab > 0) & (lba > 0)
    Dv[ok] = np.log(lba[ok] / lab[ok])
    und = np.where(ok, np.minimum(lab, lba), np.nan)
    return Dv, und


def wls(X, y, w):
    W = np.sqrt(w)[:, None]
    return np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)[0]


def es_c(eps, se, pred, delta):
    w = 1 / se**2; n = eps.size
    return float(1 - ((w * (eps - pred) ** 2).sum() - n + delta) / ((w * eps**2).sum() - n + delta))


def delta_contrib(D, sid, p_trip, keyset, reps, seed, subset=None):
    """Per-pair null contributions (1 - w e^2) on the given pair set; returns dict key -> mean contribution."""
    rng = np.random.default_rng(seed)
    acc = {}
    for _ in range(reps):
        y = (rng.random(D["lo"].size) < p_trip).astype(float)
        a, b, e, s_ = residuals_from(D, sid, y=y, subset=subset)
        for k_, c in zip(zip(a, b), 1 - (e / s_) ** 2):
            if k_ in keyset:
                acc.setdefault(k_, []).append(c)
    return {k_: float(np.mean(v)) for k_, v in acc.items()}


def assemble(a, b, eps, se, prof, osm):
    Dv, und = dcov(osm, a, b)
    F = feats(prof, a, b)
    ok = np.isfinite(Dv) & np.isfinite(F).all(1)
    return dict(a=a[ok], b=b[ok], eps=eps[ok], se=se[ok], D=Dv[ok], F=F[ok], und=und[ok])


def dES(fit, ev, delta, sel=None):
    """Fit [D] and [D,F] on `fit`; out-of-sample ΔES on `ev` (optionally a subset)."""
    w = 1 / fit["se"] ** 2
    b1 = wls(fit["D"][:, None], fit["eps"], w)
    b2 = wls(np.c_[fit["D"], fit["F"]], fit["eps"], w)
    e = ev if sel is None else {k: v[sel] for k, v in ev.items()}
    es1 = es_c(e["eps"], e["se"], e["D"] * b1[0], delta)
    es2 = es_c(e["eps"], e["se"], np.c_[e["D"], e["F"]] @ b2, delta)
    return es2 - es1, b2


# ---------------- synthetic gate ----------------

_W = {}


def _sim_init(payload):
    _W.update(payload)


def _sim(args):
    i, planted = args
    rng = np.random.default_rng(1000 + i)
    keys = _W["keys"]  # union of station-id pairs (a<b by station index order handled via orient)
    G = dict(zip(keys, rng.standard_normal(len(keys)) * _W["sdG"]))  # keys canonical: (min id, max id)
    u = rng.standard_normal(6); u /= np.linalg.norm(u)
    cu = np.nan_to_num(_W["jun_F"] @ u)
    beta = KAPPA * float(np.sqrt(0.08 * _W["E_tot"] / (_W["wj"] * cu**2).sum())) if planted else 0.0  # per-sim; KAPPA: calibrated scale
    out = {}
    for tag in ("may", "jun"):
        D, sid, p, prof = _W[tag + "_D"], _W[tag + "_sid"], _W[tag + "_p"], _W["prof_nyc"]
        a_ids, b_ids = sid[D["lo"]], sid[D["hi"]]
        uk, inv = np.unique(np.char.add(np.char.add(a_ids.astype("U12"), "|"), b_ids.astype("U12")), return_inverse=True)
        ua = np.array([x.split("|")[0] for x in uk]); ub = np.array([x.split("|")[1] for x in uk])
        g = np.array([G.get((x, y_), 0.0) if x < y_ else -G.get((y_, x), 0.0) for x, y_ in zip(ua, ub)])
        Fu = feats(prof, ua, ub) @ u
        Fu = np.nan_to_num(Fu)
        off = (g + beta * Fu)[inv]
        y = (rng.random(D["lo"].size) < expit(logit(np.clip(p, 1e-9, 1 - 1e-9)) + off)).astype(float)
        a, b, e, s_ = residuals_from(D, sid, y=y)
        out[tag] = assemble(a, b, e, s_, prof, "citibike")
    d, _ = dES(out["may"], out["jun"], _W["delta_jun_total"] * out["jun"]["eps"].size / _W["n_jun_real"])
    # bootstrap CI lower bound (B = 100 for the gate)
    w = 1 / out["may"]["se"] ** 2
    b1 = wls(out["may"]["D"][:, None], out["may"]["eps"], w); b2 = wls(np.c_[out["may"]["D"], out["may"]["F"]], out["may"]["eps"], w)
    J = out["jun"]; n = J["eps"].size; dl = _W["delta_jun_total"] * n / _W["n_jun_real"]
    v = []
    for _ in range(100):
        ix = rng.integers(0, n, n)
        v.append(es_c(J["eps"][ix], J["se"][ix], np.c_[J["D"], J["F"]][ix] @ b2, dl) - es_c(J["eps"][ix], J["se"][ix], J["D"][ix] * b1[0], dl))
    lo_ci = float(np.percentile(v, 2.5))
    return dict(i=i, planted=planted, dES=float(d), ci_lo=lo_ci, supported=bool(d >= 0.05 and lo_ci > 0))


def main():
    t0 = time.time()
    prof = {"nyc": profiles("data/processed/citibike_202605_weekday_trips.npz", "ny", "citibike")}
    prof["nyc"].update({k: v for k, v in profiles("data/processed/citibike_jun_202605_weekday_trips.npz", "ny", "citibike").items() if k not in prof["nyc"]})
    Dm, sm, pm = load_month("citibike"); Dj, sj, pj = load_month("citibike_jun")
    may = assemble(*residuals_from(Dm, sm), prof["nyc"], "citibike")
    jun = assemble(*residuals_from(Dj, sj), prof["nyc"], "citibike")
    keyset = set(zip(jun["a"], jun["b"]))
    dc = delta_contrib(Dj, sj, pj, keyset, reps=2, seed=21)
    delta_vec = np.array([dc.get(k, np.nan) for k in zip(jun["a"], jun["b"])])
    delta_vec = np.nan_to_num(delta_vec, nan=np.nanmean(delta_vec))
    delta_tot = float(delta_vec.sum())
    wj = 1 / jun["se"] ** 2
    E_tot = float((wj * jun["eps"] ** 2).sum() - jun["eps"].size + delta_tot)
    sdG = np.sqrt(0.92 * E_tot / wj.sum())
    c_ref = np.nan_to_num(jun["F"] @ (np.ones(6) / np.sqrt(6)))
    beta_unit = float(np.sqrt(0.08 * E_tot / (wj * c_ref**2).sum()))
    out = dict(n_may=int(may["eps"].size), n_jun=int(jun["eps"].size), delta_jun=delta_tot, E_tot_jun=E_tot, sdG=float(sdG))
    print("prep done", round(time.time() - t0), "s", out, flush=True)
    keys = list({(min(x, y_), max(x, y_)) for x, y_ in list(zip(may["a"], may["b"])) + list(keyset)})
    payload = dict(keys=keys, sdG=sdG, beta_unit=beta_unit, may_D=Dm, may_sid=sm, may_p=pm, jun_D=Dj, jun_sid=sj, jun_p=pj,
                   prof_nyc=prof["nyc"], delta_jun_total=delta_tot, n_jun_real=jun["eps"].size,
                   jun_F=jun["F"], wj=wj, E_tot=E_tot)
    with Pool(NW, initializer=_sim_init, initargs=(payload,)) as pool:
        sims = pool.map(_sim, [(i, True) for i in range(NSIM)] + [(NSIM + i, False) for i in range(NSIM)])
    pw = np.mean([s["supported"] for s in sims if s["planted"]]); fp = np.mean([s["supported"] for s in sims if not s["planted"]])
    out["synthetic"] = dict(power=float(pw), false_pos=float(fp), median_dES_planted=float(np.median([s["dES"] for s in sims if s["planted"]])),
                            kappa=KAPPA)
    med = out["synthetic"]["median_dES_planted"]
    out["synthetic"]["pass_"] = bool(pw >= 0.8 and fp <= 0.05 and 0.06 <= med <= 0.10)  # calibration window
    print("synthetic", out["synthetic"], round(time.time() - t0), "s", flush=True)
    if not out["synthetic"]["pass_"]:
        out["note"] = "synthetic gate failed; real statistic not computed (amend spec first)"
        json.dump(out, open("results/hact.json", "w"), indent=1); return
    # ---------- real ----------
    d, b2 = dES(may, jun, delta_tot)
    rng = np.random.default_rng(5)
    w = 1 / may["se"] ** 2
    b1 = wls(may["D"][:, None], may["eps"], w)
    n = jun["eps"].size; v = []
    for _ in range(1000):
        ix = rng.integers(0, n, n); dl = delta_vec[ix].sum()
        v.append(es_c(jun["eps"][ix], jun["se"][ix], np.c_[jun["D"], jun["F"]][ix] @ b2, dl) - es_c(jun["eps"][ix], jun["se"][ix], jun["D"][ix] * b1[0], dl))
    ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    verdict = "SUPPORTED" if (d >= 0.05 and ci[0] > 0) else ("WEAK" if (d > 0 and ci[0] > 0) else "NOT SUPPORTED")
    A = np.zeros((4, 4))
    for c, (i, j) in zip(b2[1:], PAIRS_IJ):
        A[i, j], A[j, i] = c, -c
    R, Wk, Fd, Ls = 0, 1, 2, 3
    out["primary"] = dict(dES=float(d), ci95=ci, verdict=verdict, coef_D=float(b2[0]), A=A.tolist(),
                          curl_RES_WORK_LEIS=float(A[R, Wk] + A[Wk, Ls] + A[Ls, R]),
                          curl_RES_WORK_FOOD=float(A[R, Wk] + A[Wk, Fd] + A[Fd, R]))
    long_ = jun["und"] > 1000
    for nm, sel in (("gt_1km", long_), ("le_1km", ~long_)):
        out[f"dES_{nm}"] = float(dES(may, jun, float(delta_vec[sel].sum()), sel=sel)[0])
    for nm, strata in (("casual", (1, 3)), ("member", (0, 2))):
        sub = np.isin(Dj["st"], strata)
        a, b, e, s_ = residuals_from(Dj, sj, subset=sub)
        J = assemble(a, b, e, s_, prof["nyc"], "citibike")
        dsub = delta_contrib(Dj, sj, pj, set(zip(J["a"], J["b"])), reps=1, seed=31, subset=sub)
        dl = float(np.nansum([dsub.get(k, np.nan) for k in zip(J["a"], J["b"])]))
        out[f"dES_{nm}"] = float(dES(may, J, dl)[0])
    for city, unit, st, ov, osm in (("chicago", "u_chi_jo25", "il", "divvy", "divvy"), ("boston", "u_bos_jo25", "ma", "bluebikes", "bluebikes")):
        try:
            Dc, sc, pc = load_month(unit, by_month=True)
            pr = profiles(f"data/processed/{unit}_weekday_trips.npz", st, ov)
            C = assemble(*residuals_from(Dc, sc), pr, osm)
            dcc = delta_contrib(Dc, sc, pc, set(zip(C["a"], C["b"])), reps=1, seed=41)
            dl = float(np.nansum([dcc.get(k, np.nan) for k in zip(C["a"], C["b"])]))
            out[f"transfer_{city}"] = dict(n=int(C["eps"].size), dES_nyc_coefs=float(dES(may, C, dl)[0]),
                                           excess_per_pair=float(((C["eps"] / C["se"]) ** 2).mean() - 1 + dl / max(C["eps"].size, 1)))
        except Exception as ex:  # noqa: BLE001
            out[f"transfer_{city}"] = dict(error=str(ex))
    out["runtime_s"] = round(time.time() - t0)
    json.dump(out, open("results/hact.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
