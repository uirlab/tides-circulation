"""H-FARE prep: twelve monthly YouBike units around the 28 Feb 2024 fare change, and the
fixed transit-linked pair labels from Sep-Nov 2023 transfer rides. Same preprocessing as H-XFER (src/prep_xfer.py):
hour cells, mid-hour stamps, weekdays 06-22 h (cut later by the pipeline), no self-loops, no group-ride collapse.
Run from the repository root:  python src/prep_fare.py
Outputs: data/processed/g_f<YYMM>_weekday_trips.npz (12 units, plus g_f2309 as the power-check template), data/processed/fare_tau_2309_2311.npz,
results/fare2024/validity.json"""
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, "src")
import prep_global as P  # noqa: E402  (reuses finish(); the dispatcher ignores "none")

R = "data/raw/youbike/"
# excluded days, fixed before the data were opened (public holidays; 28-29 Feb 2024 is the policy transition)
HOL = {2022: ["2022-12-01", "2022-12-02"],  # YouBike 1.0 still running in Taipei until 3 Dec 2022
       2023: ["2023-01-02", "2023-01-20", "2023-01-23", "2023-01-24", "2023-01-25", "2023-01-26", "2023-01-27",
                        "2023-02-27", "2023-02-28", "2023-04-03", "2023-04-04", "2023-04-05", "2023-05-01",
                        "2023-09-29", "2023-10-09", "2023-10-10"],
       2024: ["2024-01-01", "2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-28", "2024-02-29",
              "2024-04-04", "2024-04-05", "2024-05-01"]}
OUTCOME = [(2022, 12), (2023, 1), (2023, 2), (2023, 3), (2023, 4), (2023, 5),
           (2023, 12), (2024, 1), (2024, 2), (2024, 3), (2024, 4), (2024, 5)]
TAU_M = [(2023, 9), (2023, 10), (2023, 11)]


def full(y, m):
    """As src/prep_xfer.py full(), with the year as an argument."""
    with zipfile.ZipFile(f"{R}{y}{m:02d}_youbike.zip") as z:
        cols = ["rent_time", "rent_station", "return_time", "return_station", "rent", "infodate"]
        has_header = z.open(z.namelist()[0]).read(9) == b"rent_time"
        df = pd.read_csv(z.open(z.namelist()[0]), dtype=str, header=0 if has_header else None, names=None if has_header else cols,
                         usecols=["rent_time", "rent_station", "return_station", "rent"],
                         encoding="utf-8-sig", encoding_errors="replace")
    df = df.rename(columns={"rent_time": "t", "rent_station": "s", "return_station": "e", "rent": "dur"})
    df = df.dropna(subset=["t", "s", "e"])
    for c in ("s", "e"):
        df[c] = df[c].str.rstrip("�")
    return df


def xfer(y, m):
    df = pd.read_csv(f"{R}xfer_{y}{m:02d}.csv", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={"借車時間": "t", "借車站": "s", "還車站": "e", "租借時數": "dur"})[["t", "s", "e", "dur"]]
    df = df.dropna(subset=["t", "s", "e"])
    df["t"] = pd.to_datetime(df.t.str[:19])
    return df


def canon(xdf, names):
    """As src/prep_xfer.py canon(): exact match, else the longest full-file name that is a prefix of the transfer name."""
    ns = sorted((n for n in names if isinstance(n, str)), key=len, reverse=True); mp = {}
    for n in set(xdf.s) | set(xdf.e):
        mp[n] = n if n in names else next((c for c in ns if len(c) >= 4 and n.startswith(c)), n)
    xdf = xdf.copy(); xdf["s"] = xdf.s.map(mp); xdf["e"] = xdf.e.map(mp)
    print("transfer station names:", len(mp), "unmatched:", sum(v not in names for v in mp.values()), flush=True)
    return xdf


def weekday(df, y):
    t = pd.to_datetime(df.t); hol = {pd.Timestamp(h).date() for h in HOL[y]}
    return df[(t.dt.weekday < 5) & (~t.dt.date.isin(hol)) & (df.s != df.e)]


def und(d):
    return pd.Series([tuple(sorted(p)) for p in zip(d.s, d.e)])


os.makedirs("results/fare2024", exist_ok=True)
validity = {}
stations = {}
for y, m in [(2023, 9)] + OUTCOME:  # Sep 2023: template for the synthetic power check only (not an outcome month)
    f = full(y, m); t = pd.to_datetime(f.t)
    c = pd.DataFrame(dict(h=t.dt.floor("h"), s=f.s, e=f.e)).value_counts().rename("n")
    key = f"f{y % 100:02d}"; name = f"g_f{y % 100:02d}{m:02d}"
    P.HOL[key], P.YEAR[key], P.MONTH[key] = HOL[y], y, m
    c = c[c > 0].reset_index(); ts = c.h + pd.Timedelta(minutes=30)
    P.finish(name, ts, ts, c.s, c.e, np.zeros(len(c), int), weight=c.n.values, group=False)
    d = np.load(f"data/processed/{name}_weekday_trips.npz", allow_pickle=True)
    tm = d["tmin"]; w = (tm >= 360) & (tm < 1320)
    act = np.unique(np.r_[d["o"][w], d["e"][w]])
    if (y, m) == (2023, 9):
        continue
    stations[name] = set(d["station_id"][act].astype(str))
    validity[name] = dict(rides_06_22=int(w.sum()), active_stations=int(act.size), weekdays=int(d["days"].size))

# fixed transit-linked labels from Sep-Nov 2023 (no overlap with any outcome month)
fm = pd.concat([weekday(full(y, m), y) for y, m in TAU_M])
names = set(fm.s) | set(fm.e)
xm = pd.concat([weekday(canon(xfer(y, m), names), y) for y, m in TAU_M])
nm = und(fm).value_counts(); nx = und(xm).value_counts()
tau = (nx.reindex(nm.index).fillna(0) / nm)[nm >= 30]
thr = float(tau.quantile(0.9)); T = (tau >= thr).values
keys = np.array(["\t".join(k) for k in tau.index])
np.savez_compressed("data/processed/fare_tau_2309_2311.npz", pair=keys, tau=tau.values, T=T, n=nm.reindex(tau.index).values)

units = [f"g_f{y % 100:02d}{m:02d}" for y, m in OUTCOME]
stable = set.intersection(*stations.values())
np.save("data/processed/fare_stable_stations.npy", np.array(sorted(stable)))
pre0, post0, pre1, post1 = units[0:3], units[3:6], units[6:9], units[9:12]
mean_st = lambda us: float(np.mean([validity[u]["active_stations"] for u in us]))
chg = {"Y0": mean_st(post0) / mean_st(pre0) - 1, "Y1": mean_st(post1) / mean_st(pre1) - 1}
out = dict(months=validity, tau=dict(n_pairs=int(tau.size), threshold_top_decile=thr, n_T=int(T.sum()),
                                     median_tau=float(tau.median()), transfer_share_of_rides=float(nx.sum() / nm.sum())),
           stations_active_in_all_12=len(stable), station_change_pre_to_post=chg,
           check2_flag=bool(max(abs(v) for v in chg.values()) > 0.10))
print(json.dumps(out, indent=1), flush=True)
json.dump(out, open("results/fare2024/validity.json", "w"), indent=1)
