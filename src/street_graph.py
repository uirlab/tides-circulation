"""Directed street graph from the OSM extract + legal shortest-path lengths between stations. Usage: python src/street_graph.py <osm_city> <data_city> [<data_city> ...]
Writes data/processed/legal_paths_<osm_city>.npz with station ids (union over data cities) and L[a, b] (metres, inf if
unreachable within 10 km).
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

FWD, REV = {"yes", "true", "1"}, {"-1", "reverse"}
LANE = os.environ.get("LANE") == "1"  # exploratory: bike-lane-weighted costs
FAC = {"track": 0.5, "lane": 0.75}


def lane_factor(t, forward, oneway):
    """Cost factor for travelling an edge forward/backward given cycleway tags (right-hand traffic)."""
    vals = []
    gen = t.get("cycleway", ""); both = t.get("cycleway:both", "")
    right, left = t.get("cycleway:right", ""), t.get("cycleway:left", "")
    if forward:
        vals = [right, both, gen] + ([left] if oneway else [])
    else:
        vals = [left, both, gen] if not oneway else [x.replace("opposite_", "") for x in (gen, left, right, both) if x.startswith("opposite")]
    f = [FAC[v] for v in vals if v in FAC]
    return min(f) if f else 1.0
LIMIT_M, SNAP_M = 10000.0, 100.0


def bike_both_ways(t):
    if t.get("oneway:bicycle", "").lower() == "no":
        return True
    return any(k.startswith("cycleway") and str(v).startswith("opposite") for k, v in t.items())


def build(osm):
    E = json.load(open(f"data/raw/osm/{osm}_ways.json"))["elements"]
    key = {}
    coords = []

    def nid(la, lo):
        k = (round(la, 7), round(lo, 7))
        if k not in key:
            key[k] = len(coords); coords.append(k)
        return key[k]

    src, dst, ln = [], [], []
    for w in E:
        g = w.get("geometry")
        if not g or len(g) < 2:
            continue
        t = w["tags"]
        ow = t.get("oneway", "no").lower()
        both = ow not in FWD | REV or bike_both_ways(t)
        rev = ow in REV
        ids = [nid(p["lat"], p["lon"]) for p in g]
        for a, b in zip(ids[:-1], ids[1:]):
            (la1, lo1), (la2, lo2) = coords[a], coords[b]
            p1, p2 = np.radians(la1), np.radians(la2)
            d = 2 * 6371000 * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2))
            d = max(d, 0.1)
            ow_flag = not both
            if both or not rev:
                src.append(a); dst.append(b); ln.append(d * (lane_factor(t, not rev, ow_flag) if LANE else 1.0))
            if both or rev:
                src.append(b); dst.append(a); ln.append(d * (lane_factor(t, rev, ow_flag) if LANE else 1.0))
    n = len(coords)
    G = sp.coo_matrix((ln, (src, dst)), shape=(n, n))  # parallel edges: keep the shortest (not the sum)
    order = np.lexsort((G.data, G.col, G.row))
    r, c, v = G.row[order], G.col[order], G.data[order]
    first = np.ones(r.size, bool); first[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    G = sp.csr_matrix((v[first], (r[first], c[first])), shape=(n, n))
    return G, np.array(coords)


_GG = _NODES = None


def _dij(ch):
    return dijkstra(_GG, directed=True, indices=_NODES[ch], limit=LIMIT_M)[:, _NODES].astype(np.float32)


def main():
    osm, cities = sys.argv[1], sys.argv[2:]
    G, xy = build(osm)
    sid, lat, lon = [], [], []
    seen = set()
    for c in cities:
        d = np.load(f"data/processed/{c}_202605_weekday_trips.npz", allow_pickle=True)
        for s_, la, lo in zip(d["station_id"].astype(str), d["lat"], d["lon"]):
            if s_ not in seen and np.isfinite(la) and np.isfinite(lo):
                seen.add(s_); sid.append(s_); lat.append(la); lon.append(lo)
    lat0 = np.radians(np.median(lat))
    P = np.c_[np.radians(xy[:, 1]) * np.cos(lat0), np.radians(xy[:, 0])] * 6371000
    Q = np.c_[np.radians(lon) * np.cos(lat0), np.radians(lat)] * 6371000
    dist, idx = cKDTree(P).query(Q)
    ok = dist <= SNAP_M
    nodes = idx[ok]
    L = np.full((len(sid), len(sid)), np.inf, np.float32)
    nw = int(os.environ.get("NWORKERS", "1"))
    if nw > 1:  # parallel over source chunks (identical result)
        from multiprocessing import Pool
        chunks = np.array_split(np.arange(nodes.size), nw * 4)
        global _GG, _NODES
        _GG, _NODES = G, nodes
        with Pool(nw) as pool:
            parts = pool.map(_dij, chunks)
        sub = np.vstack(parts)
    else:
        sub = dijkstra(G, directed=True, indices=nodes, limit=LIMIT_M)[:, nodes].astype(np.float32)
    okix = np.flatnonzero(ok)
    L[np.ix_(okix, okix)] = sub
    np.savez_compressed(f"data/processed/legal_paths_{osm}{'_lane' if LANE else ''}.npz", station_id=np.array(sid), L=L, snapped=ok, snap_dist=dist)
    fin = np.isfinite(L[np.ix_(okix, okix)])
    print(dict(osm=osm, graph_nodes=G.shape[0], graph_edges=G.nnz, stations=len(sid), snapped=int(ok.sum()),
               reachable_frac=float(fin.mean()), median_snap_m=float(np.median(dist))), flush=True)


if __name__ == "__main__":
    main()
