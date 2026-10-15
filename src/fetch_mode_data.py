"""Downloads for H-MODE: MTA subway, CTA and MBTA GTFS;
USGS EPQS elevations for stations not in the sibling-project elevation file. Run from the repository root (cluster recommended)."""
import csv
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.makedirs("data/raw/gtfs", exist_ok=True)
for name, url in (("mta_subway", "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"),
                  ("cta", "https://www.transitchicago.com/downloads/sch_data/google_transit.zip"),
                  ("mbta", "https://cdn.mbta.com/MBTA_GTFS.zip")):
    f = f"data/raw/gtfs/{name}.zip"
    if not os.path.exists(f):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; mobility-irreversibility)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(f + ".part", "wb") as o:
            o.write(r.read())
        os.rename(f + ".part", f)
    print(name, os.path.getsize(f), flush=True)

known = {}
if os.path.exists("data/processed/citibike_202605_station_elevation_sibling.csv"):
    for r in csv.DictReader(open("data/processed/citibike_202605_station_elevation_sibling.csv")):
        if r["z"].strip():  # blanks are re-queried
            known[("nyc", r["id"])] = float(r["z"])
todo = []
for city, files in (("nyc", ["citibike_202605", "citibike_jun_202605"]), ("chi", ["u_chi_jo25"]), ("bos", ["u_bos_jo25"])):
    for f in files:
        p = f"data/processed/{f}_weekday_trips.npz"
        d = np.load(p, allow_pickle=True)
        for s, la, lo in zip(d["station_id"].astype(str), d["lat"], d["lon"]):
            if np.isfinite(la) and (city, s) not in known and (city, s, la, lo) not in todo:
                todo.append((city, s, float(la), float(lo)))
todo = list({(c, s): (c, s, la, lo) for c, s, la, lo in todo}.values())
print("elevation queries needed", len(todo), flush=True)


def q(item):
    c, s, la, lo = item
    u = f"https://epqs.nationalmap.gov/v1/json?x={lo}&y={la}&units=Meters&wkid=4326"
    for _ in range(4):
        try:
            with urllib.request.urlopen(u, timeout=30) as r:
                return c, s, float(json.loads(r.read())["value"])
        except Exception:  # noqa: BLE001
            continue
    return c, s, float("nan")


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(q, todo))
with open("data/processed/station_elevation_all.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["city", "id", "z", "source"])
    for (c, s), z in known.items():
        w.writerow([c, s, z, "sibling_epqs"])
    for c, s, z in res:
        w.writerow([c, s, z, "epqs_2026-10-02"])
print("done; failed", sum(np.isnan(z) for _, _, z in res), flush=True)
