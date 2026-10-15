"""Aggregates that the figure scripts need from the trip files, cached in results/ so that the figures can be rebuilt
without trip data. Each function reads its cached file if present; otherwise it builds it once from
data/processed/<key>_202605_weekday_trips.npz (which requires the raw data; see data/README.md) and saves it."""
import os

import numpy as np


def _trips(key):
    return np.load(f"data/processed/{key}_202605_weekday_trips.npz", allow_pickle=True)


def stations(key):
    """Station ids, names and coordinates in model order for a system (results/figdata_stations_<key>.npz)."""
    f = f"results/figdata_stations_{key}.npz"
    if not os.path.exists(f):
        d = _trips(key)
        np.savez_compressed(f, station_id=d["station_id"], station_name=d["station_name"],
                            lat=d["lat"].astype(float), lon=d["lon"].astype(float))
    return np.load(f, allow_pickle=True)


def netflow_citibike(hours=(8, 17)):
    """Station-to-station trip counts in each given hour (all weekdays of May 2026) for Fig. 1b, as sparse
    (i, j, count) triplets per hour (results/figdata_netflow_citibike.npz). Returns {hour: N x N count matrix}."""
    f = "results/figdata_netflow_citibike.npz"
    if not os.path.exists(f):
        d = _trips("citibike"); o, e, t = d["o"], d["e"], d["tmin"]; out = {}
        for h in hours:
            k = (t >= h * 60) & (t < (h + 1) * 60)
            key, c = np.unique(o[k].astype(np.int64) * len(d["lat"]) + e[k], return_counts=True)
            out[f"i{h}"], out[f"j{h}"], out[f"c{h}"] = key // len(d["lat"]), key % len(d["lat"]), c
        np.savez_compressed(f, n=len(d["lat"]), **out)
    z = np.load(f); N = int(z["n"]); C = {}
    for h in hours:
        M = np.zeros((N, N)); np.add.at(M, (z[f"i{h}"], z[f"j{h}"]), z[f"c{h}"]); C[h] = M
    return C
