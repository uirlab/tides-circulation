"""Prepare Lyft-format bike-share trips (May 2026 weekdays) for the irreversibility tests.

Identical filters to src/prep_citibike.py (which produced the 0c/0c-b inputs), plus station coordinates.
Usage (project root): python src/prep_lyft.py <citibike|divvy|bluebikes>
"""
import zipfile

import numpy as np
import pandas as pd

import sys
CITY = sys.argv[1]
ZIPS = {"citibike": ["data/raw/citibike/202605-citibike-tripdata.zip"],
        "divvy": ["data/raw/divvy/202605-divvy-tripdata.zip"],
        "bluebikes": ["data/raw/bluebikes/202605-bluebikes-tripdata.zip"],
        # NYC held-out-in-time H-OW test; output file keeps the 202605 name pattern
        "citibike_jun": ["data/raw/citibike/202606-citibike-tripdata.zip"],
        # Chicago power fallback: May + June 2026
        "divvy_mj": ["data/raw/divvy/202605-divvy-tripdata.zip", "data/raw/divvy/202606-divvy-tripdata.zip"]}[CITY]
MONTHS = {"divvy_mj": (5, 6), "citibike_jun": (6,)}.get(CITY, (5,))
OUT = f"data/processed/{CITY}_202605_weekday_trips.npz"
HOLIDAYS = {pd.Timestamp("2026-05-25").date(), pd.Timestamp("2026-06-19").date()}  # Memorial Day, Juneteenth

cols = ["started_at", "ended_at", "start_station_id", "end_station_id", "start_station_name", "rideable_type", "member_casual", "start_lat", "start_lng"]
parts = []
for ZIP in ZIPS:
    with zipfile.ZipFile(ZIP) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv") and "MACOSX" not in n):
            with z.open(name) as f:
                parts.append(pd.read_csv(f, usecols=cols, dtype={"start_station_id": str, "end_station_id": str}))
df = pd.concat(parts, ignore_index=True)
n_raw = len(df)
df["started_at"] = pd.to_datetime(df["started_at"])
df["ended_at"] = pd.to_datetime(df["ended_at"])
df = df.dropna(subset=["start_station_id", "end_station_id"])
df = df[df.start_station_id != df.end_station_id]
d = df.started_at.dt
df = df[(d.weekday < 5) & (~d.date.isin(HOLIDAYS)) & (d.month.isin(MONTHS))]
n_filt = len(df)

# group-ride collapse: same (start,end), start within 60 s and end within 60 s of the previous kept-or-dropped trip
df = df.sort_values(["start_station_id", "end_station_id", "started_at"]).reset_index(drop=True)
same = (df.start_station_id.eq(df.start_station_id.shift())) & (df.end_station_id.eq(df.end_station_id.shift()))
ds = (df.started_at - df.started_at.shift()).dt.total_seconds()
de = (df.ended_at - df.ended_at.shift()).dt.total_seconds().abs()
dup = same & (ds <= 60) & (de <= 60)
df = df[~dup]
n_final = len(df)

ids = np.array(sorted(set(df.start_station_id) | set(df.end_station_id)))
idx = {s: i for i, s in enumerate(ids)}
names = df.groupby("start_station_id").start_station_name.first().reindex(ids).fillna("").to_numpy()
geo = df.groupby("start_station_id")[["start_lat", "start_lng"]].median().reindex(ids)
o = df.start_station_id.map(idx).to_numpy(np.int32)
e = df.end_station_id.map(idx).to_numpy(np.int32)
days = sorted(df.started_at.dt.date.unique())
dmap = {x: i for i, x in enumerate(days)}
day = df.started_at.dt.date.map(dmap).to_numpy(np.int16)
tmin = (df.started_at.dt.hour * 60 + df.started_at.dt.minute + df.started_at.dt.second / 60).to_numpy(np.float32)
# stratum for the heterogeneity check: bike type x rider type; -1 = other
bike = df.rideable_type.map({"classic_bike": 0, "electric_bike": 1})
rider = df.member_casual.map({"member": 0, "casual": 1})
stratum = (bike * 2 + rider).fillna(-1).to_numpy(np.int8)
np.savez_compressed(OUT, o=o, e=e, day=day, tmin=tmin, stratum=stratum, station_id=ids, station_name=names.astype(str),
                    days=np.array([str(x) for x in days]), lat=geo.start_lat.to_numpy(float), lon=geo.start_lng.to_numpy(float))
print(dict(strata=np.unique(stratum, return_counts=True), raw=n_raw, weekday_nonloop=n_filt, group_collapsed=int(dup.sum()), final=n_final,
           stations=len(ids), weekdays=len(days)))
