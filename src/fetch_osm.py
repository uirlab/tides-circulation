"""Fetch named OSM street ways (tags + geometry) in each city's station bounding box, tiled.
OpenStreetMap data are under the ODbL. Output: data/raw/osm/<city>_ways.json. Usage: python src/fetch_osm.py"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

EPS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
       "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
HW = "primary|secondary|tertiary|residential|unclassified|living_street|primary_link|secondary_link|tertiary_link|trunk"


def fetch(q, att_max=9):
    for att in range(att_max):
        ep = EPS[att % len(EPS)]
        try:
            req = urllib.request.Request(ep, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "research-mobility-irreversibility (academic)"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as ex:  # noqa: BLE001
            print(f"    retry {att} {ep.split('/')[2]}: {ex}", flush=True)
            time.sleep(5 * (att + 1))
    raise RuntimeError("overpass failed")


for c, nt in [("divvy", 5), ("bluebikes", 3), ("citibike", 8)]:  # confirmatory cities first
    out = f"data/raw/osm/{c}_ways.json"
    if os.path.exists(out):
        print(c, "exists", flush=True); continue
    d = np.load(f"data/processed/{c}_202605_weekday_trips.npz", allow_pickle=True)
    la, lo = d["lat"], d["lon"]; ok = np.isfinite(la) & np.isfinite(lo)
    s, n = np.percentile(la[ok], [0.5, 99.5]); w, e = np.percentile(lo[ok], [0.5, 99.5]); m = 0.005
    s, n, w, e = s - m, n + m, w - m, e + m
    ways = {}
    os.makedirs("data/raw/osm/tiles", exist_ok=True)
    for i in range(nt):
        for j in range(nt):
            bb = f"{s+(n-s)*i/nt:.4f},{w+(e-w)*j/nt:.4f},{s+(n-s)*(i+1)/nt:.4f},{w+(e-w)*(j+1)/nt:.4f}"
            tf = f"data/raw/osm/tiles/{c}_{nt}_{i}_{j}.json"
            if os.path.exists(tf):
                r = json.load(open(tf))
            else:
                r = fetch(f'[out:json][timeout:90];way["highway"~"^({HW})$"]["name"]({bb});out tags geom;')
                json.dump(r, open(tf, "w"))
            for el in r.get("elements", []):
                ways[el["id"]] = el
            print(f"  {c} tile {i*nt+j+1}/{nt*nt}: {len(ways)} ways", flush=True)
            time.sleep(2)
    json.dump(dict(bbox=[s, w, n, e], elements=list(ways.values())), open(out, "w"))
    print(c, len(ways), "ways", round(os.path.getsize(out) / 1e6, 1), "MB", flush=True)
