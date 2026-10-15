"""H-XFER prep: June 2023 YouBike, all rides (g_xj6) and non-transfer rides (g_xn6),
plus pair-level covariates from May 2023 (transfer share tau) and June (median ride duration).
Run from the repository root (cluster recommended): python src/prep_xfer.py
Outputs: data/processed/g_xj6_weekday_trips.npz, g_xn6_weekday_trips.npz, data/processed/xfer_pairs.npz"""
import sys
import zipfile

import numpy as np
import pandas as pd

sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, "src")
import prep_global as P  # noqa: E402  (reuses finish(); the dispatcher ignores "none")

R = "data/raw/youbike/"


def full(month):
    with zipfile.ZipFile(f"{R}2023{month:02d}_youbike.zip") as z:
        cols = ["rent_time", "rent_station", "return_time", "return_station", "rent", "infodate"]
        has_header = z.open(z.namelist()[0]).read(9) == b"rent_time"  # June 2023 file ships without a header row
        df = pd.read_csv(z.open(z.namelist()[0]), dtype=str, header=0 if has_header else None, names=None if has_header else cols,
                         usecols=["rent_time", "rent_station", "return_station", "rent"],
                         encoding="utf-8-sig", encoding_errors="replace")  # some names are byte-truncated mid-character
    df = df.rename(columns={"rent_time": "t", "rent_station": "s", "return_station": "e", "rent": "dur"})
    df = df.dropna(subset=["t", "s", "e"])
    for c in ("s", "e"):
        df[c] = df[c].str.rstrip("\ufffd")
    return df


def canon(xdf, names):
    """Map transfer-file station names onto (possibly truncated) full-file names: exact match, else the longest full-file
    name that is a prefix of the transfer name."""
    ns = sorted((n for n in names if isinstance(n, str)), key=len, reverse=True); m = {}
    for n in set(xdf.s) | set(xdf.e):
        if n in names:
            m[n] = n
        else:
            hit = next((c for c in ns if len(c) >= 4 and n.startswith(c)), None)
            m[n] = hit if hit is not None else n
    xdf = xdf.copy(); xdf["s"] = xdf.s.map(m); xdf["e"] = xdf.e.map(m)
    miss = [n for n, v in m.items() if v not in names]
    print("transfer station names:", len(m), "unmatched:", len(miss), miss[:10])
    return xdf


def xfer(month):
    df = pd.read_csv(f"{R}xfer_2023{month:02d}.csv", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={"借車時間": "t", "借車站": "s", "還車站": "e", "租借時數": "dur"})[["t", "s", "e", "dur"]]
    df = df.dropna(subset=["t", "s", "e"])
    df["t"] = pd.to_datetime(df.t.str[:19])  # ISO with +08:00 offset -> naive local
    return df


def cells(df):
    t = pd.to_datetime(df.t)
    return pd.DataFrame(dict(h=t.dt.floor("h"), s=df.s, e=df.e)).value_counts().rename("n")


# June: all rides and non-transfer rides (count-level subtraction per hour x origin x destination)
fj = full(6); xj = canon(xfer(6), set(fj.s) | set(fj.e))
cf, cx = cells(fj), cells(xj)
cn = (cf - cx.reindex(cf.index).fillna(0)).clip(lower=0)
print("June rides", int(cf.sum()), "transfer", int(cx.sum()), "transfer rides without a matching full-data cell", int((cx - cf.reindex(cx.index).fillna(0)).clip(lower=0).sum()))
for name, c in (("g_xj6", cf), ("g_xn6", cn)):
    c = c[c > 0].reset_index(); t = c.h + pd.Timedelta(minutes=30)
    P.finish(name, t, t, c.s, c.e, np.zeros(len(c), int), weight=c.n.values, group=False)

# pair covariates
fm = full(5); xm = canon(xfer(5), set(fm.s) | set(fm.e))
tm = pd.to_datetime(fm.t); keep = (tm.dt.weekday < 5) & (tm.dt.date != pd.Timestamp("2023-05-01").date()) & (fm.s != fm.e)
fm = fm[keep]; xt = pd.to_datetime(xm.t); xm = xm[(xt.dt.weekday < 5) & (xt.dt.date != pd.Timestamp("2023-05-01").date()) & (xm.s != xm.e)]
und = lambda d: pd.Series([tuple(sorted(p)) for p in zip(d.s, d.e)])
nm = und(fm).value_counts(); nx = und(xm).value_counts()
tau = (nx.reindex(nm.index).fillna(0) / nm)[nm >= 30]
# directional May transfers for T3
dx = pd.Series(list(zip(xm.s, xm.e))).value_counts()
sec = pd.to_timedelta(fj.dur, errors="coerce").dt.total_seconds()
dur = pd.Series(sec.values, index=und(fj).values).groupby(level=0).median()
keys = np.array(["\t".join(k) for k in tau.index])
np.savez_compressed("data/processed/xfer_pairs.npz", pair=keys, tau=tau.values, n_may=nm.reindex(tau.index).values,
                    x_ab=np.array([dx.get(k, 0) for k in tau.index]), x_ba=np.array([dx.get((k[1], k[0]), 0) for k in tau.index]),
                    dur=dur.reindex(tau.index).values)
print("May pairs with >=30 rides", len(tau), "median tau", float(tau.median()), "share of May rides that are transfers", float(nx.sum() / nm.sum()))
