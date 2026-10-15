"""Citi Bike station table for the descriptive loop map (src/loop_map.py): one row per start/end station id with its
median coordinates and trip counts, derived from the May 2026 trip file. (The original analysis read an equivalent
table produced in a sibling project.) Run from the repository root:
    python src/station_table.py   -> data/processed/citibike_202605_stations.csv
"""
import os
import zipfile

import pandas as pd

ZIP = "data/raw/citibike/202605-citibike-tripdata.zip"
parts = []
with zipfile.ZipFile(ZIP) as z:
    for n in z.namelist():
        if n.endswith(".csv") and not n.startswith("__MACOSX"):
            with z.open(n) as f:
                parts.append(pd.read_csv(f, usecols=["start_station_id", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng"],
                                         dtype={"start_station_id": str, "end_station_id": str}))
d = pd.concat(parts, ignore_index=True)
s = pd.DataFrame({"id": d.start_station_id, "lat": d.start_lat, "lon": d.start_lng, "dep": 1, "arr": 0})
e = pd.DataFrame({"id": d.end_station_id, "lat": d.end_lat, "lon": d.end_lng, "dep": 0, "arr": 1})
a = pd.concat([s, e]).dropna(subset=["id", "lat", "lon"])
t = a.groupby("id").agg(lat=("lat", "median"), lon=("lon", "median"), n_obs=("lat", "size"), dep=("dep", "sum"), arr=("arr", "sum")).reset_index()
os.makedirs("data/processed", exist_ok=True)
t.to_csv("data/processed/citibike_202605_stations.csv", index=False)
print("stations", len(t))
