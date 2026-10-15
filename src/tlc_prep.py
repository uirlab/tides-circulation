"""Step 3 prep: NYC TLC Yellow + HVFHV -> compact weekday trip arrays per (year, period).

Run from the repository root (cluster recommended):  python src/tlc_prep.py
Writes data/processed/tlc_<y22|y23|y24>_<pre|early|late>.npz and data/processed/tlc_cbd_zones.json.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.tseries.holiday import USFederalHolidayCalendar

RAW = "data/raw/tlc"
YEARS = {"y22": 2022, "y23": 2023, "y24": 2024}  # year of the October that starts the Oct-Jun window
PERIODS = {"pre": [10, 11, 12], "early": [2, 3], "late": [5, 6]}
EXCL = {1, 264, 265}  # EWR, unknown, outside NYC
T0, W, NB = 360, 15, 64
HOL = set(USFederalHolidayCalendar().holidays("2022-01-01", "2026-12-31").date)
COLS = {"yellow": ("tpep_pickup_datetime", 0), "fhvhv": ("pickup_datetime", 1)}  # (time column, stratum)


def month_file(kind, y0, m):
    yy = y0 if m >= 10 else y0 + 1
    return f"{RAW}/{kind}_tripdata_{yy}-{m:02d}.parquet"


def load_month(kind, y0, m):
    tcol, st = COLS[kind]
    df = pq.read_table(month_file(kind, y0, m), columns=[tcol, "PULocationID", "DOLocationID"]).to_pandas()
    t = pd.to_datetime(df[tcol])
    o, e = df.PULocationID.to_numpy(np.int64), df.DOLocationID.to_numpy(np.int64)
    yy = y0 if m >= 10 else y0 + 1
    mins = t.dt.hour.to_numpy() * 60 + t.dt.minute.to_numpy()
    date = t.dt.date
    keep = ((t.dt.year.to_numpy() == yy) & (t.dt.month.to_numpy() == m) & (t.dt.weekday.to_numpy() < 5)
            & ~date.isin(HOL).to_numpy() & (mins >= T0) & (mins < T0 + W * NB)
            & (o != e) & ~np.isin(o, list(EXCL)) & ~np.isin(e, list(EXCL)))
    return o[keep], e[keep], mins[keep], date[keep].to_numpy(), np.full(keep.sum(), st, np.int8)


def cbd_zones():
    """Zones with >= 90% of Feb 2025 HVFHV same-zone trips charged the CBD fee (spec finalisation)."""
    df = pq.read_table(f"{RAW}/fhvhv_tripdata_2025-02.parquet", columns=["PULocationID", "DOLocationID", "cbd_congestion_fee"]).to_pandas()
    same = df[df.PULocationID == df.DOLocationID]
    share = same.groupby("PULocationID").cbd_congestion_fee.apply(lambda x: float((x > 0).mean()))
    n = same.groupby("PULocationID").size()
    z = sorted(int(k) for k, v in share.items() if v >= 0.90 and n[k] >= 50)
    return z, {int(k): [round(float(share[k]), 3), int(n[k])] for k in share.index}


def main():
    z, detail = cbd_zones()
    json.dump(dict(cbd_zones=z, n=len(z), share_and_n=detail), open("data/processed/tlc_cbd_zones.json", "w"))
    print("CBD zones:", len(z), z, flush=True)
    for ys, y0 in YEARS.items():
        for per, months in PERIODS.items():
            parts = [load_month(k, y0, m) for m in months for k in ("yellow", "fhvhv")]
            o, e, mins, date, st = (np.concatenate([p[i] for p in parts]) for i in range(5))
            days = np.array(sorted(set(date)))
            dmap = {d: i for i, d in enumerate(days)}
            day = np.array([dmap[d] for d in date], np.int16)
            lo, hi = np.minimum(o, e).astype(np.int16), np.maximum(o, e).astype(np.int16)
            y = (o == lo).astype(np.int8)
            b = ((mins - T0) // W).astype(np.int8)
            np.savez(f"data/processed/tlc_{ys}_{per}.npz", lo=lo, hi=hi, y=y, b=b, st=st, day=day,
                     days=np.array([str(d) for d in days]))
            print(ys, per, "trips", lo.size, "days", days.size, "yellow", int((st == 0).sum()), "hvfhv", int((st == 1).sum()), flush=True)


if __name__ == "__main__":
    main()
