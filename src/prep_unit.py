"""Generic per-unit prep for step 3. Same filters as src/prep_lyft.py, except that
weekday exclusion uses the full US federal holiday calendar.
Usage (project root): python src/prep_unit.py <unit_name> <city: citibike|divvy|bluebikes> <YYYY-MM>[,<YYYY-MM>...]
Writes data/processed/<unit_name>_weekday_trips.npz
"""
import glob
import sys
import zipfile

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

name, city, months = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
HOL = set(USFederalHolidayCalendar().holidays("2023-01-01", "2026-12-31").date)
cols = ["started_at", "ended_at", "start_station_id", "end_station_id", "start_station_name", "rideable_type", "member_casual",
        "start_lat", "start_lng"]
parts = []
for ym in months:
    zs = [z for z in glob.glob(f"data/raw/{city}/{ym.replace('-', '')}-*.zip")]
    if not zs and city == "citibike" and ym in ("2026-05",):
        zs = ["data/raw/citibike/202605-citibike-tripdata.zip"]
    assert len(zs) == 1, (ym, zs)
    with zipfile.ZipFile(zs[0]) as z:
        for n in sorted(n for n in z.namelist() if n.endswith(".csv") and "MACOSX" not in n):
            with z.open(n) as f:
                parts.append(pd.read_csv(f, usecols=cols, dtype={"start_station_id": str, "end_station_id": str}))
df = pd.concat(parts, ignore_index=True)
n_raw = len(df)
df["started_at"] = pd.to_datetime(df["started_at"]); df["ended_at"] = pd.to_datetime(df["ended_at"])
df = df.dropna(subset=["start_station_id", "end_station_id"])
df = df[df.start_station_id != df.end_station_id]
d = df.started_at.dt
ym_ok = d.strftime("%Y-%m").isin(months)
df = df[(d.weekday < 5) & (~d.date.isin(HOL)) & ym_ok]
df = df.sort_values(["start_station_id", "end_station_id", "started_at"]).reset_index(drop=True)
same = df.start_station_id.eq(df.start_station_id.shift()) & df.end_station_id.eq(df.end_station_id.shift())
GS = float(__import__("os").environ.get("GROUP_S", "60"))  # group-ride window (baseline 60 s)
dup = same & ((df.started_at - df.started_at.shift()).dt.total_seconds() <= GS) & ((df.ended_at - df.ended_at.shift()).dt.total_seconds().abs() <= GS)
df = df[~dup]
ids = np.array(sorted(set(df.start_station_id) | set(df.end_station_id)))
idx = {s: i for i, s in enumerate(ids)}
names = df.groupby("start_station_id").start_station_name.first().reindex(ids).fillna("").to_numpy()
geo = df.groupby("start_station_id")[["start_lat", "start_lng"]].median().reindex(ids)
days = sorted(df.started_at.dt.date.unique()); dmap = {x: i for i, x in enumerate(days)}
bike = df.rideable_type.map({"classic_bike": 0, "electric_bike": 1}); rider = df.member_casual.map({"member": 0, "casual": 1})
stratum = (bike * 2 + rider).fillna(-1).to_numpy(np.int8)
np.savez_compressed(f"data/processed/{name}_weekday_trips.npz", o=df.start_station_id.map(idx).to_numpy(np.int32),
                    e=df.end_station_id.map(idx).to_numpy(np.int32), day=df.started_at.dt.date.map(dmap).to_numpy(np.int16),
                    tmin=(df.started_at.dt.hour * 60 + df.started_at.dt.minute + df.started_at.dt.second / 60).to_numpy(np.float32),
                    stratum=stratum, station_id=ids, station_name=names.astype(str), days=np.array([str(x) for x in days]),
                    lat=geo.start_lat.to_numpy(float), lon=geo.start_lng.to_numpy(float))
print(dict(unit=name, raw=n_raw, final=len(df), collapsed=int(dup.sum()), stations=len(ids), weekdays=len(days),
           strata=np.unique(stratum, return_counts=True)[1].tolist()), flush=True)
