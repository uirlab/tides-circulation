"""H-MODE: mode substitution (bike one leg, rapid transit back).
Run from the repository root (cluster recommended):  NWORKERS=32 python src/modesub.py   -> results/mode.json
M1 (magnitude, permutation) on NYC June + Chicago/Boston Jul-Oct 2025; M2 (direction, May->June) on NYC.
Gates run before each real statistic. Reuses helpers from src/hact.py.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from scipy.stats import norm

sys.path.insert(0, "src")
import hact as H  # noqa: E402

NW = int(os.environ.get("NWORKERS", "8"))
RADIUS = float(os.environ.get("RADIUS", "400")); SENS = os.environ.get("SENS", ""); M1_ONLY = os.environ.get("M1_ONLY") == "1"
GTFS = {"nyc": "mta_subway", "chi": "cta", "bos": "mbta"}
ELEV = {}
for r in csv.DictReader(open("data/processed/station_elevation_all.csv")):
    if r["z"] not in ("", "nan"):
        ELEV[(r["city"], r["id"])] = float(r["z"])


# ---------------- transit indicator ----------------

def rapid_stops(feed):
    z = zipfile.ZipFile(f"data/raw/gtfs/{feed}.zip")
    rd = lambda n, cols: pd.read_csv(io.BytesIO(z.read(n)), usecols=cols, dtype=str)
    routes = rd("routes.txt", ["route_id", "route_type"])
    rapid = set(routes.route_id[routes.route_type.astype(int).isin([0, 1])])
    trips = rd("trips.txt", ["route_id", "trip_id"])
    trips = trips[trips.route_id.isin(rapid)]
    st = rd("stop_times.txt", ["trip_id", "stop_id"])
    st = st[st.trip_id.isin(set(trips.trip_id))].merge(trips, on="trip_id")[["stop_id", "route_id"]].drop_duplicates()
    hdr = pd.read_csv(io.BytesIO(z.read("stops.txt")), nrows=0).columns
    stops = rd("stops.txt", ["stop_id", "stop_lat", "stop_lon"] + (["parent_station"] if "parent_station" in hdr else []))
    if "parent_station" in stops:
        par = stops.set_index("stop_id").parent_station.fillna("")
        st["stop_id"] = [p if isinstance(p, str) and p else s for s, p in zip(st.stop_id, par.reindex(st.stop_id).values)]
    st = st.drop_duplicates()
    pos = stops.set_index("stop_id")[["stop_lat", "stop_lon"]].astype(float)
    st = st[st.stop_id.isin(pos.index)]
    ll = pos.loc[st.stop_id].values
    return st.stop_id.values, st.route_id.values, ll[:, 0], ll[:, 1]


def transit_sets(city, sid, lat, lon):
    s_id, r_id, sla, slo = rapid_stops(GTFS[city])
    lat0 = float(np.nanmedian(lat))
    tree = cKDTree(H.xy(sla, slo, lat0))
    out = {}
    ok = np.isfinite(lat)
    P = H.xy(lat[ok], lon[ok], lat0)
    for s, p in zip(sid[ok], P):
        ix = tree.query_ball_point(p, RADIUS)
        best = {}
        for i in ix:
            dd = np.hypot(*(H.xy(np.array([sla[i]]), np.array([slo[i]]), lat0)[0] - p))
            r = r_id[i]
            if r not in best or dd < best[r][0]:
                best[r] = (dd, s_id[i])
        out[s] = {r: v[1] for r, v in best.items()}
    return out


def T_pairs(sets, a, b):
    T = np.zeros(a.size)
    for i, (x, y) in enumerate(zip(a, b)):
        A, B = sets.get(x, {}), sets.get(y, {})
        T[i] = float(any(A[r] != B[r] for r in set(A) & set(B)))
    return T


# ---------------- helpers ----------------

def habit(D, sid, a, b):
    K = D["K"]
    pos = {s: i for i, s in enumerate(sid)}
    key = np.array([min(pos[x], pos[y]) * K + max(pos[x], pos[y]) for x, y in zip(a, b)])
    grp = D.get("month", np.zeros(D["lo"].size, np.int64))
    pk = D["lo"] * K + D["hi"]; slot = (D["tm"] // 30).astype(np.int64)
    sel = np.isin(pk, key); Rs = []
    for g in np.unique(grp):
        mg = sel & (grp == g)
        code = np.unique((pk[mg] * 64 + slot[mg]) * 64 + D["day"][mg])
        ps, cnt = np.unique(code // 64, return_counts=True)
        mx = {}
        for p_, c_ in zip(ps // 64, cnt):
            mx[p_] = max(mx.get(p_, 0), c_)
        Rs.append(np.array([mx.get(k_, 0) for k_ in key]) / max(np.unique(D["day"][mg]).size, 1))
    return np.mean(Rs, axis=0)


def pair_table(unit, city, osm, by_month=False, delta_reps=2, seed=51):
    D, sid, p = H.load_month(unit, by_month=by_month)
    import loop_map as L
    lo, hi, eps, se, t, _ = L.pair_residuals(D)
    k = t >= 30
    a, b, eps, se, t = sid[lo[k]], sid[hi[k]], eps[k], se[k], t[k]
    Dv, und = H.dcov(osm, a, b)
    d = np.load([f for f in (f"data/processed/{unit}_202605_weekday_trips.npz", f"data/processed/{unit}_weekday_trips.npz") if os.path.exists(f)][0], allow_pickle=True)
    sets = transit_sets(city, d["station_id"].astype(str), d["lat"], d["lon"])
    T = T_pairs(sets, a, b)
    ha = np.array([ELEV.get((city, x), np.nan) for x in a]); hb = np.array([ELEV.get((city, y), np.nan) for y in b])
    ok = np.isfinite(Dv) & np.isfinite(ha) & np.isfinite(hb)
    tab = dict(a=a[ok], b=b[ok], eps=eps[ok], se=se[ok], t=t[ok].astype(float), D=Dv[ok], und=und[ok], T=T[ok], dh=(ha - hb)[ok])
    dc = H.delta_contrib(D, sid, p, set(zip(tab["a"], tab["b"])), reps=delta_reps, seed=seed)
    tab["delta_p"] = np.array([dc.get(k_, np.nan) for k_ in zip(tab["a"], tab["b"])])
    tab["delta_p"] = np.nan_to_num(tab["delta_p"], nan=np.nanmean(tab["delta_p"]))
    tab["e"] = (tab["eps"] / tab["se"]) ** 2 - 1 + tab["delta_p"]
    tab["R"] = habit(D, sid, tab["a"], tab["b"])
    return tab, (D, sid, p)


# ---------------- M1 ----------------

def m1_design(tab):
    w = tab["t"]
    dec = np.minimum((np.argsort(np.argsort(w)) * 10) // len(w), 9)
    ter = np.minimum((np.argsort(np.argsort(tab["und"])) * 3) // len(w), 2)
    cell = dec * 3 + ter
    Xc = np.c_[np.eye(10)[dec], np.log(tab["und"]), np.abs(tab["D"]), (tab["R"] - tab["R"].mean()) / tab["R"].std()]
    return Xc, cell


def beta_T(y, T, Xc):
    z = (T - T.mean()) / (T.std() + 1e-12)
    b = np.linalg.lstsq(np.c_[Xc, z], y, rcond=None)[0]
    return b[-1] / y.std()


def perm_cells(T, cell, rng):
    out = T.copy()
    for c in np.unique(cell):
        m = cell == c
        out[m] = rng.permutation(T[m])
    return out


def m1_test(y, T, Xc, cell, rng, B=2000, normal=False):
    obs = beta_T(y, T, Xc)
    null = np.array([beta_T(y, perm_cells(T, cell, rng), Xc) for _ in range(B)])
    p = float(norm.sf((obs - null.mean()) / null.std())) if normal else float((1 + (null >= obs).sum()) / (B + 1))
    return obs, p


def m1(tab, rng):
    Xc, cell = m1_design(tab); y, T = tab["e"], tab["T"]
    hit = fp = 0
    for _ in range(100):
        Tp = perm_cells(T, cell, rng)
        o, p = m1_test(y + 0.15 * y.std() * (Tp - Tp.mean()) / (Tp.std() + 1e-12), Tp, Xc, cell, rng, B=200, normal=True)
        hit += (o > 0 and p < 0.001)
        o2, p2 = m1_test(y, perm_cells(T, cell, rng), Xc, cell, rng, B=200, normal=True)
        fp += (o2 > 0 and p2 < 0.001)
    power = dict(planted=hit / 100, false_pos=fp / 100, pass_=bool(hit >= 80 and fp <= 5))
    out = dict(n=int(y.size), frac_T1=float(T.mean()), power=power)
    if power["pass_"]:
        o, p = m1_test(y, T, Xc, cell, rng)
        enr = float((y[T > 0.5].sum() / y.sum()) / max(T.mean(), 1e-12))
        out.update(beta_T=float(o), p=p, enrichment_T1=enr,
                   verdict="SUPPORTED" if (o > 0 and p < 0.001) else ("REJECTED" if (o <= 0 or p >= 0.05) else "INCONCLUSIVE"))
    return out


# ---------------- M2 ----------------

def m2_fit(tab, cols):
    X = np.c_[tuple(tab[c] if c != "dhT" else tab["dh"] * tab["T"] for c in cols)]
    return H.wls(X, tab["eps"], 1 / tab["se"] ** 2), X


def m2_stats(may, jun, rng, B=1000):
    bF, _ = m2_fit(may, ["D", "dh", "dhT"]); bR, _ = m2_fit(may, ["D", "dh"])
    XF = np.c_[jun["D"], jun["dh"], jun["dh"] * jun["T"]]; XR = np.c_[jun["D"], jun["dh"]]
    dl = jun["delta_p"].sum()
    d = H.es_c(jun["eps"], jun["se"], XF @ bF, dl) - H.es_c(jun["eps"], jun["se"], XR @ bR, dl)
    n = jun["eps"].size; dv, cm, cj = [], [], []
    nm = may["eps"].size
    for _ in range(B):
        ix = rng.integers(0, n, n); im = rng.integers(0, nm, nm)
        dv.append(H.es_c(jun["eps"][ix], jun["se"][ix], XF[ix] @ bF, jun["delta_p"][ix].sum()) - H.es_c(jun["eps"][ix], jun["se"][ix], XR[ix] @ bR, jun["delta_p"][ix].sum()))
        cm.append(m2_fit({k: v[im] for k, v in may.items()}, ["D", "dh", "dhT"])[0][2])
        cj.append(m2_fit({k: v[ix] for k, v in jun.items()}, ["D", "dh", "dhT"])[0][2])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    gj = float(m2_fit(jun, ["D", "dh", "dhT"])[0][2])
    res = dict(dES=float(d), dES_ci=q(dv), coef_may=float(bF[2]), coef_may_ci=q(cm), coef_jun=gj, coef_jun_ci=q(cj))
    sup = res["coef_may_ci"][0] > 0 and res["coef_jun_ci"][0] > 0 and res["dES_ci"][0] > 0
    rej = res["coef_jun_ci"][1] <= 0 or d <= 0
    res["verdict"] = "SUPPORTED" if sup else ("REJECTED" if rej else "INCONCLUSIVE")
    return res


_W = {}


def _init(payload):
    _W.update(payload)


def _sim(args):
    i, gamma = args
    rng = np.random.default_rng(5000 + i)
    keys = _W["keys"]
    G = dict(zip(keys, rng.standard_normal(len(keys)) * _W["sdG"]))
    out = {}
    for tag in ("may", "jun"):
        D, sid, p = _W[tag + "_raw"]
        ua_ub = np.char.add(np.char.add(sid[D["lo"]].astype("U16"), "|"), sid[D["hi"]].astype("U16"))
        uk, inv = np.unique(ua_ub, return_inverse=True)
        ua = np.array([x.split("|")[0] for x in uk]); ub = np.array([x.split("|")[1] for x in uk])
        g = np.array([G.get((x, y), 0.0) if x < y else -G.get((y, x), 0.0) for x, y in zip(ua, ub)])
        feat = np.array([_W["dhT"].get((x, y), 0.0) for x, y in zip(ua, ub)])
        y = (rng.random(D["lo"].size) < expit(logit(np.clip(p, 1e-9, 1 - 1e-9)) + (g + gamma * feat)[inv])).astype(float)
        a, b, e, s_ = H.residuals_from(D, sid, y=y)
        base = _W[tag + "_tab"]
        idx = {k: j for j, k in enumerate(zip(base["a"], base["b"]))}
        sel = [(j, idx[k]) for j, k in enumerate(zip(a, b)) if k in idx]
        jj = np.array([x for x, _ in sel]); kk = np.array([x for _, x in sel])
        out[tag] = {k: (v[kk] if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        out[tag]["eps"], out[tag]["se"] = e[jj], s_[jj]
    r = m2_stats(out["may"], out["jun"], rng, B=100)
    return dict(i=i, gamma=gamma, dES=r["dES"], supported=r["verdict"] == "SUPPORTED")


def m2_gate(may, jun, raw_may, raw_jun):
    wj = 1 / jun["se"] ** 2
    E = float((wj * jun["eps"] ** 2).sum() - jun["eps"].size + jun["delta_p"].sum())
    f = jun["dh"] * jun["T"]
    gamma0 = float(np.sqrt(0.01 * E / (wj * f**2).sum()))
    keys = list({(min(x, y), max(x, y)) for x, y in list(zip(may["a"], may["b"])) + list(zip(jun["a"], jun["b"]))})
    dhT = {}
    for t in (may, jun):
        for x, y, v in zip(t["a"], t["b"], t["dh"] * t["T"]):
            dhT[(x, y)] = v; dhT[(y, x)] = -v
    payload = dict(keys=keys, sdG=float(np.sqrt(0.99 * E / wj.sum())), dhT=dhT, may_raw=raw_may, jun_raw=raw_jun, may_tab=may, jun_tab=jun)
    steps, kappa = [], 1.0
    with Pool(NW, initializer=_init, initargs=(payload,)) as pool:
        for step in range(4):  # initial + at most 3 calibration steps (spec)
            sims = pool.map(_sim, [(100 * step + i, kappa * gamma0) for i in range(50)] + [(100 * step + 50 + i, 0.0) for i in range(50)])
            pl = [s for s in sims if s["gamma"] > 0]; nu = [s for s in sims if s["gamma"] == 0]
            med = float(np.median([s["dES"] for s in pl]))
            st = dict(kappa=kappa, median_dES=med, power=float(np.mean([s["supported"] for s in pl])), false_pos=float(np.mean([s["supported"] for s in nu])))
            steps.append(st); print("M2 gate step", st, flush=True)
            if 0.005 <= med <= 0.02:
                break
            kappa *= float(np.sqrt(0.01 / max(med, 1e-4)))  # proportional recalibration toward the window centre
    final = steps[-1]
    final["pass_"] = bool(0.005 <= final["median_dES"] <= 0.02 and final["power"] >= 0.8 and final["false_pos"] <= 0.05)
    return dict(steps=steps, final=final)


def main():
    t0 = time.time(); rng = np.random.default_rng(61)
    out = {}
    may, raw_may = pair_table("citibike", "nyc", "citibike")
    jun, raw_jun = pair_table("citibike_jun", "nyc", "citibike")
    out["nyc_n"] = dict(may=int(may["eps"].size), jun=int(jun["eps"].size), T1_frac_jun=float(jun["T"].mean()))
    print("tables", out["nyc_n"], round(time.time() - t0), flush=True)
    out["M1_nyc_june"] = m1(jun, rng); print("M1 nyc", out["M1_nyc_june"], flush=True)
    if M1_ONLY:
        json.dump(out, open(f"results/mode{SENS}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)); return
    for city, unit, osm in (("chi", "u_chi_jo25", "divvy"), ("bos", "u_bos_jo25", "bluebikes")):
        tab, _ = pair_table(unit, city, osm, by_month=True, delta_reps=1)
        out[f"M1_{city}"] = m1(tab, rng); print("M1", city, out[f"M1_{city}"], flush=True)
    gate = m2_gate(may, jun, raw_may, raw_jun); out["M2_gate"] = gate
    if gate["final"]["pass_"]:
        out["M2"] = m2_stats(may, jun, rng)
    else:
        out["M2"] = dict(note="synthetic gate failed; real M2 not computed")
    print("M2", out["M2"], flush=True)
    out["runtime_s"] = round(time.time() - t0)
    json.dump(out, open("results/mode.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
