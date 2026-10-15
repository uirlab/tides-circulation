"""R1 spatial units. Run from the repository root (cluster recommended).
  python src/rb_spatial.py catch <city> <radius_m>   # R1a: bike stations -> nearest rail station; tpe | bay
  python src/rb_spatial.py hex <unit>                # R1b: bike stations -> 1 km hexagonal cells
Writes data/processed/rb/<name>_weekday_trips.npz in the pipeline's format and prints the share of trips kept."""
import os
import sys

import numpy as np

COORDS = {"g_ybk": "results/figdata_coords_g_ybk.npz", "g_lon": "results/figdata_coords_g_lon.npz",
          "g_seo": "results/figdata_coords_g_seo.npz", "g_mex": "results/figdata_coords_g_mex.npz",
          "g_tpex": "results/figdata_coords_g_tpex.npz", "g_bar": "results/figdata_coords_g_bar.npz"}  # others: lat/lon in the trip file


def data_path(c):
    p = f"data/processed/{c}_202605_weekday_trips.npz"
    return p if os.path.exists(p) else f"data/processed/{c}_weekday_trips.npz"


def load(c):
    d = dict(np.load(data_path(c), allow_pickle=True))
    if c in COORDS:
        g = np.load(COORDS[c]); d["lat"], d["lon"] = g["lat"].astype(float), g["lon"].astype(float)
    assert d["lat"].size == d["station_id"].size, (c, d["lat"].size, d["station_id"].size)
    return d


def hav(lat1, lon1, lat2, lon2):
    p = np.radians
    a = np.sin((p(lat2) - p(lat1)) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin((p(lon2) - p(lon1)) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(a))


def relabel(d, node_of, node_ids, node_names, node_lat, node_lon, out):
    """Map station i -> node node_of[i] (-1 = drop); drop trips with a dropped end or both ends in one node."""
    o, e = node_of[d["o"]], node_of[d["e"]]
    keep = (o >= 0) & (e >= 0) & (o != e)
    used = np.unique(np.r_[o[keep], e[keep]]); new = -np.ones(node_ids.size, int); new[used] = np.arange(used.size)
    res = {k: v[keep] for k, v in d.items() if isinstance(v, np.ndarray) and v.shape[:1] == d["o"].shape}
    res["o"], res["e"] = new[o[keep]].astype(np.int32), new[e[keep]].astype(np.int32)
    res.update(station_id=np.asarray(node_ids)[used], station_name=np.asarray(node_names).astype(str)[used],
               lat=np.asarray(node_lat, float)[used], lon=np.asarray(node_lon, float)[used], days=d["days"])
    os.makedirs("data/processed/rb", exist_ok=True)
    np.savez_compressed(f"data/processed/rb/{out}_weekday_trips.npz", **res)
    print(dict(unit=out, trips_in=int(d["o"].size), trips_kept=int(keep.sum()), share_kept=round(float(keep.mean()), 4),
               nodes=int(used.size)), flush=True)
    return used


def catch(city, r):
    bike_c, rail_c, tag = {"tpe": ("g_ybk", "g_tpex", "tpe"), "bay": ("g_baw", "g_bar", "bay")}[city]
    B, R = load(bike_c), load(rail_c)
    has_trips = np.zeros(R["station_id"].size, bool); has_trips[np.r_[R["o"], R["e"]]] = True  # excluded interchanges have none
    elig = np.flatnonzero(has_trips & np.isfinite(R["lat"]))
    node = -np.ones(B["station_id"].size, int)
    ok = np.isfinite(B["lat"]) & np.isfinite(B["lon"])
    for i in np.flatnonzero(ok):
        dist = hav(B["lat"][i], B["lon"][i], R["lat"][elig], R["lon"][elig]); j = int(np.argmin(dist))
        if dist[j] <= r:
            node[i] = elig[j]
    out = f"r1a_{tag}_r{r}"
    used = relabel(B, node, R["station_id"], R["station_name"], R["lat"], R["lon"], out + "_bike")
    rail_node = -np.ones(R["station_id"].size, int); rail_node[used] = used  # same node set as the bike table
    relabel(R, rail_node, R["station_id"], R["station_name"], R["lat"], R["lon"], out + "_rail")


def hexcells(unit, size_km=1.0):
    d = load(unit)
    ok = np.isfinite(d["lat"]) & np.isfinite(d["lon"])
    lat0, lon0 = np.nanmedian(d["lat"]), np.nanmedian(d["lon"])
    x = (d["lon"] - lon0) * np.cos(np.radians(lat0)) * 111.32; y = (d["lat"] - lat0) * 110.57
    s = size_km / np.sqrt(3)  # pointy-top hexagons with centre-to-centre distance size_km
    q = (np.sqrt(3) / 3 * x - y / 3) / s; rr = (2 / 3 * y) / s
    cx, cz = q, rr; cy = -cx - cz
    rx, ry, rz = np.round(cx), np.round(cy), np.round(cz)
    dx, dy, dz = np.abs(rx - cx), np.abs(ry - cy), np.abs(rz - cz)
    fx = (dx > dy) & (dx > dz); fz = ~fx & (dz >= dy)
    rx = np.where(fx, -ry - rz, rx); rz = np.where(fz, -rx - ry, rz)
    key = np.where(ok, (rx.astype(np.int64) + 10_000) * 100_000 + (rz.astype(np.int64) + 10_000), -1)
    cells, node = np.unique(key[ok], return_inverse=True)
    node_of = -np.ones(key.size, int); node_of[ok] = node
    clat = np.array([np.nanmean(d["lat"][node_of == k]) for k in range(cells.size)])
    clon = np.array([np.nanmean(d["lon"][node_of == k]) for k in range(cells.size)])
    relabel(d, node_of, cells.astype(str), cells.astype(str), clat, clon, f"r1b_{unit}")


if __name__ == "__main__":
    {"catch": lambda: catch(sys.argv[2], int(sys.argv[3])), "hex": lambda: hexcells(sys.argv[2])}[sys.argv[1]]()
