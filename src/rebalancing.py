"""Step 4: do tides or loops drive rebalancing?
Run from the repository root (cluster recommended) after units u_nyc_2025-05 and u_nyc_2025-06 exist:  python src/rebalancing.py
Writes results/4_rebalancing.json.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.spatial import cKDTree

sys.path.insert(0, "src")
MONTHS = {"fit": "2025-05", "test": "2025-06"}
HOL = set(USFederalHolidayCalendar().holidays("2024-01-01", "2026-12-31").date)
TZ = "America/New_York"
U_THR = float(os.environ.get("U_THR", "3")); GAP_S = float(os.environ.get("GAP_S", "600")); SENS = os.environ.get("SENS", "")


def norm(x):
    return " ".join(str(x).lower().split())


def trip_events(ym):
    z = glob.glob(f"data/raw/citibike/{ym.replace('-', '')}-*.zip")[0]
    parts = []
    with zipfile.ZipFile(z) as zz:
        for n in sorted(n for n in zz.namelist() if n.endswith(".csv") and "MACOSX" not in n):
            with zz.open(n) as f:
                parts.append(pd.read_csv(f, usecols=["started_at", "ended_at", "start_station_name", "end_station_name"]))
    df = pd.concat(parts, ignore_index=True).dropna()
    dep = pd.DataFrame(dict(st=df.start_station_name.map(norm), t=pd.to_datetime(df.started_at), k=-1))
    arr = pd.DataFrame(dict(st=df.end_station_name.map(norm), t=pd.to_datetime(df.ended_at), k=+1))
    return pd.concat([dep, arr], ignore_index=True)


def status(ym):
    y, m = map(int, ym.split("-"))
    lo = pd.Timestamp(year=y, month=m, day=1, tz=TZ).tz_convert("UTC")
    hi = (pd.Timestamp(year=y, month=m, day=1, tz=TZ) + pd.offsets.MonthBegin(1)).tz_convert("UTC")
    dset = ds.dataset(f"data/raw/station_status/citi_bike_stations_{y}.parquet")
    f = (ds.field("fetched_at") >= lo) & (ds.field("fetched_at") < hi)
    tb = dset.to_table(filter=f, columns=["station_name", "lat", "lon", "capacity", "num_bikes_available", "num_bikes_disabled",
                                          "fetched_at", "is_renting", "is_returning"])
    s = tb.to_pandas()
    s["t"] = s.fetched_at.dt.tz_convert(TZ).dt.tz_localize(None)
    s["st"] = s.station_name.map(norm)
    s["B"] = s.num_bikes_available.fillna(0) + s.num_bikes_disabled.fillna(0)
    return s


def operator_moves(s, ev):
    """Per station: R+ and R- (|U| >= 3 per interval, gap <= 10 min), weekdays 06-22 local."""
    out = {}
    ev = ev.sort_values("t")
    evg = {k: (g.t.values, np.cumsum(g.k.values)) for k, g in ev.groupby("st")}
    for st, g in s.sort_values("t").groupby("st"):
        if g.capacity.nunique() > 1:  # capacity changes within month ~ valet / reconfiguration -> exclude
            continue
        t = g.t.values; B = g.B.values.astype(float)
        dt = np.diff(t).astype("timedelta64[s]").astype(float)
        dB = np.diff(B)
        tt = pd.to_datetime(t[1:])
        ok = (dt <= GAP_S) & (tt.weekday < 5) & ~pd.Index(tt.date).isin(HOL) & (tt.hour >= 6) & (tt.hour < 22)
        if st in evg:
            et, cs = evg[st]
            c0 = np.searchsorted(et, t[:-1], side="right"); c1 = np.searchsorted(et, t[1:], side="right")
            csum = np.r_[0, cs]
            N = csum[c1] - csum[c0]
        else:
            N = np.zeros(dB.size)
        U = (dB - N)[ok]
        out[st] = dict(Rp=float(U[U >= U_THR].sum()), Rm=float(-U[U <= -U_THR].sum()), n_int=int(ok.sum()),
                       lat=float(g.lat.iloc[0]), lon=float(g.lon.iloc[0]))
    return out


def predictors(unit):
    os.environ["CITY"] = unit
    for mname in ("irrev", "irrev_strat", "loop_map"):
        sys.modules.pop(mname, None)
    import irrev as I
    import irrev_strat as S
    import loop_map as L
    D = S.load(); K = D["K"]
    d = np.load(I.DATA, allow_pickle=True)
    names = np.array([norm(x) for x in d["station_name"].astype(str)])
    lat, lon = d["lat"], d["lon"]
    vol = np.bincount(D["lo"], minlength=K) + np.bincount(D["hi"], minlength=K)
    # tidal imbalance: per (day, 15-min bin) |dep - arr| summed, / volume
    o = np.where(D["y"] == 1, D["lo"], D["hi"]); e = np.where(D["y"] == 1, D["hi"], D["lo"])
    key_o = (D["day"] * I.NB + D["b"]) * K + o; key_e = (D["day"] * I.NB + D["b"]) * K + e
    u, inv = np.unique(np.r_[key_o, key_e], return_inverse=True)
    net = np.bincount(inv, np.r_[np.ones(o.size), -np.ones(e.size)])
    T = np.bincount(u % K, np.abs(net), minlength=K) / np.maximum(vol, 1)
    lo, hi, eps, se, t, _ = L.pair_residuals(D)
    k = t >= 30
    z2m1 = (eps[k] / se[k]) ** 2 - 1
    lam = (np.bincount(lo[k], z2m1, K) + np.bincount(hi[k], z2m1, K))
    npair = np.bincount(lo[k], minlength=K) + np.bincount(hi[k], minlength=K)
    Lam = np.where(npair >= 5, lam / np.maximum(npair, 1), np.nan)
    return pd.DataFrame(dict(st=names, lat=lat, lon=lon, V=np.log(np.maximum(vol, 1)), T=T, Lam=Lam, vol=vol))


def join(P, R):
    rs = pd.DataFrame([dict(st=k, **v) for k, v in R.items()])
    m = P.merge(rs.drop(columns=["lat", "lon"]), on="st", how="left")
    miss = m.Rp.isna() & np.isfinite(m.lat)
    if miss.any():  # fallback: nearest status station within 30 m
        lat0 = np.radians(40.75)
        Q = np.c_[np.radians(rs.lon) * np.cos(lat0), np.radians(rs.lat)] * 6371000
        X = np.c_[np.radians(m.lon[miss]) * np.cos(lat0), np.radians(m.lat[miss])] * 6371000
        dd, ii = cKDTree(Q).query(X)
        for col in ("Rp", "Rm", "n_int"):
            m.loc[miss, col] = np.where(dd <= 30, rs[col].values[ii], np.nan)
    m["RB"] = m.Rp + m.Rm
    return m


def ols(X, y):
    X1 = np.c_[np.ones(len(y)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ b
    s2 = r @ r / (len(y) - X1.shape[1])
    cov = s2 * np.linalg.inv(X1.T @ X1)
    return b, np.sqrt(np.diag(cov)), 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))


def r2(y, yhat):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    sets = {}
    for role, ym in MONTHS.items():
        P = predictors(f"u_nyc_{ym}")
        R = operator_moves(status(ym), trip_events(ym))
        m = join(P, R).dropna(subset=["RB", "Lam", "T", "V"])
        m = m[m.vol >= 100]
        sets[role] = m
        print(role, ym, "stations", len(m), "median RB", float(m.RB.median()), flush=True)
    f, te = sets["fit"], sets["test"]
    if os.environ.get("DUMP_STATIONS"):  # descriptive dump for figures only
        te[["st", "V", "T", "Lam", "RB", "vol"]].to_csv("results/figdata_rebal_june2025.csv", index=False); print("dumped", len(te)); return
    mu = {c: (f[c].mean(), f[c].std()) for c in ("V", "T", "Lam")}
    Z = lambda d, cs: np.c_[[(d[c] - mu[c][0]) / mu[c][1] for c in cs]].T
    yf, yt = np.log1p(f.RB.values), np.log1p(te.RB.values)
    out = {"n_fit": len(f), "n_test": len(te)}
    # synthetic check (from May only; spec)
    rng = np.random.default_rng(4)
    b_vt, _, _ = ols(Z(f, ["V", "T"]), yf)
    sd = float(np.std(yf - np.c_[np.ones(len(yf)), Z(f, ["V", "T"])] @ b_vt))
    gains_null, fire = [], 0
    for _ in range(200):
        base = 0.5 * Z(f, ["T"])[:, 0]
        y0 = base + rng.standard_normal(len(f)) * sd
        y0t = 0.5 * Z(te, ["T"])[:, 0] + rng.standard_normal(len(te)) * sd
        b2, _, _ = ols(Z(f, ["V", "T"]), y0); b3, _, _ = ols(Z(f, ["V", "T", "Lam"]), y0)
        gains_null.append(r2(y0t, np.c_[np.ones(len(te)), Z(te, ["V", "T", "Lam"])] @ b3) - r2(y0t, np.c_[np.ones(len(te)), Z(te, ["V", "T"])] @ b2))
        y1 = base + 0.2 * Z(f, ["Lam"])[:, 0] + rng.standard_normal(len(f)) * sd
        y1t = 0.5 * Z(te, ["T"])[:, 0] + 0.2 * Z(te, ["Lam"])[:, 0] + rng.standard_normal(len(te)) * sd
        b2, _, _ = ols(Z(f, ["V", "T"]), y1); b3, _, _ = ols(Z(f, ["V", "T", "Lam"]), y1)
        bt, set_, _ = ols(Z(te, ["V", "T", "Lam"]), y1t)
        g = r2(y1t, np.c_[np.ones(len(te)), Z(te, ["V", "T", "Lam"])] @ b3) - r2(y1t, np.c_[np.ones(len(te)), Z(te, ["V", "T"])] @ b2)
        fire += (bt[3] > 0.1) and (bt[3] - 1.96 * set_[3] > 0) and (g >= 0.01)
    out["synthetic"] = dict(null_gain_lt_0_01=float(np.mean(np.array(gains_null) < 0.01)), loops_cost_power=fire / 200)
    out["synthetic"]["pass"] = out["synthetic"]["null_gain_lt_0_01"] >= 0.95 and out["synthetic"]["loops_cost_power"] >= 0.80
    if not out["synthetic"]["pass"]:
        out["note"] = "synthetic check failed; real test not run"
        json.dump(out, open(f"results/4_rebalancing{SENS}.json", "w"), indent=1); print(json.dumps(out, indent=1)); return
    res = {}
    for role, d, y in (("fit", f, yf), ("test", te, yt)):
        b, s_, rr = ols(Z(d, ["V", "T", "Lam"]), y)
        res[role] = dict(beta=dict(V=b[1], T=b[2], Lam=b[3]), ci=dict(T=[b[2] - 1.96 * s_[2], b[2] + 1.96 * s_[2]], Lam=[b[3] - 1.96 * s_[3], b[3] + 1.96 * s_[3]]), r2=float(rr))
    b2, _, _ = ols(Z(f, ["V", "T"]), yf); b3, _, _ = ols(Z(f, ["V", "T", "Lam"]), yf)
    gain = r2(yt, np.c_[np.ones(len(te)), Z(te, ["V", "T", "Lam"])] @ b3) - r2(yt, np.c_[np.ones(len(te)), Z(te, ["V", "T"])] @ b2)
    h4a = all(res[r]["ci"]["T"][0] > 0 for r in res)
    lam_t = res["test"]["beta"]["Lam"]; ci_l = res["test"]["ci"]["Lam"]
    h4b = gain < 0.01 and (abs(lam_t) < 0.1 or (ci_l[0] <= 0 <= ci_l[1]))
    cost = lam_t > 0.1 and ci_l[0] > 0 and gain >= 0.01
    out.update(fits=res, oos_r2_gain_lambda=gain, H4a_tides_cost=bool(h4a), H4b_loops_neutral=bool(h4b),
               verdict="SUPPORTED" if (h4a and h4b) else ("LOOPS COST" if cost else "INCONCLUSIVE"),
               spearman_T_RB_test=float(pd.Series(te["T"]).corr(pd.Series(yt), method="spearman")),
               spearman_Lam_RB_test=float(pd.Series(te["Lam"]).corr(pd.Series(yt), method="spearman")))
    json.dump(out, open(f"results/4_rebalancing{SENS}.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
