"""Step 1 Part B: one-way alignment test (H-OW).

Usage (project root):  CITY=<citibike|divvy_mj|bluebikes> OSM=<citibike|divvy|bluebikes> python src/oneway.py
Writes results/1_oneway_<CITY>.json. Citi Bike is in-sample (exploratory); Chicago/Boston are confirmatory.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from scipy.stats import binom, binomtest

sys.path.insert(0, os.path.dirname(__file__))
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
from loop_map import pair_residuals  # noqa: E402

CITY, OSM = I.CITY, os.environ.get("OSM", I.CITY)
MIN_T = int(os.environ.get("MIN_T", "30")); NEAR_M = float(os.environ.get("NEAR_M", "50")); MAX_ANG = float(os.environ.get("MAX_ANG", "45"))
SENS = os.environ.get("SENS", "")
ONEWAY_YES, ONEWAY_REV = {"yes", "true", "1"}, {"-1", "reverse"}

ABBR = [(r"\bavenue\b", "ave"), (r"\bav\b", "ave"), (r"\bstreet\b", "st"), (r"\bboulevard\b", "blvd"), (r"\bplace\b", "pl"),
        (r"\broad\b", "rd"), (r"\bdrive\b", "dr"), (r"\bparkway\b", "pkwy"), (r"\bterrace\b", "ter"), (r"\blane\b", "ln"),
        (r"\bsquare\b", "sq"), (r"\bhighway\b", "hwy"), (r"\bcourt\b", "ct")]
DIRS = r"^(north|south|east|west|n|s|e|w)\s+"


def norm_street(x: str) -> str:
    x = x.lower().replace(".", " ").replace("'", "")
    x = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", x)
    for a, b in ABBR:
        x = re.sub(a, b, x)
    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(DIRS, "", x)
    return x


def station_streets(name: str) -> set[str]:
    parts = re.split(r"\s*(?:&|/|\bat\b|\band\b|\(|\)|-)\s*", name)
    return {norm_street(p) for p in parts if p and len(norm_street(p)) > 2}


def to_xy(lat, lon, lat0):
    return (np.radians(lon) * 6371000 * np.cos(np.radians(lat0)), np.radians(lat) * 6371000)


def nearest_segment(px, py, xs, ys):
    """distance (m) and unit direction (in way-node order) of the way segment nearest to point (px, py)."""
    ax, ay, bx, by = xs[:-1], ys[:-1], xs[1:], ys[1:]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1)
    d = np.hypot(ax + t * dx - px, ay + t * dy - py)
    k = int(np.argmin(d))
    L = np.sqrt(L2[k])
    return float(d[k]), np.array([dx[k] / L, dy[k] / L])


def main():
    D = S.load()
    K = D["K"]
    d = np.load(I.DATA, allow_pickle=True)
    names, lat, lon = d["station_name"].astype(str), d["lat"], d["lon"]
    lo, hi, eps, se, t, _ = pair_residuals(D)
    keep = t >= MIN_T
    lo, hi, eps = lo[keep], hi[keep], eps[keep]
    lat0 = float(np.nanmedian(lat))
    X, Y = to_xy(lat, lon, lat0)

    osm = json.load(open(f"data/raw/osm/{OSM}_ways.json"))["elements"]
    by_name: dict[str, list] = {}
    for w in osm:
        nm = norm_street(w["tags"].get("name", ""))
        g = w.get("geometry")
        if not nm or not g or len(g) < 2:
            continue
        la_ = np.array([p["lat"] for p in g]); lo_ = np.array([p["lon"] for p in g])
        xs, ys = to_xy(la_, lo_, lat0)
        ow = w["tags"].get("oneway", "no").lower()
        kind = "fwd" if ow in ONEWAY_YES else ("rev" if ow in ONEWAY_REV else "two")
        if w["tags"].get("junction") == "roundabout":
            kind = "fwd"
        by_name.setdefault(nm, []).append((xs, ys, kind))

    streets = [station_streets(n) for n in names]
    rows = []
    for a, b, e in zip(lo, hi, eps):
        common = streets[a] & streets[b]
        if not common or not (np.isfinite(X[a]) and np.isfinite(X[b])):
            continue
        ab = np.array([X[b] - X[a], Y[b] - Y[a]])
        if np.hypot(*ab) < 1:
            continue
        ab /= np.hypot(*ab)
        for sname in common:
            ways = by_name.get(sname, [])
            near = []
            for xs, ys, kind in ways:
                da, ua = nearest_segment(X[a], Y[a], xs, ys)
                db, ub = nearest_segment(X[b], Y[b], xs, ys)
                if da <= NEAR_M or db <= NEAR_M:
                    near.append((da, db, ua, ub, kind))
            if not any(x[0] <= NEAR_M for x in near) or not any(x[1] <= NEAR_M for x in near):
                continue  # street not within 50 m of both stations
            kinds = {x[4] for x in near}
            if kinds <= {"two"}:
                cos_ok = all(min(abs(x[2] @ ab), abs(x[3] @ ab)) >= np.cos(np.radians(MAX_ANG)) for x in near)
                if cos_ok:
                    rows.append(dict(a=int(a), b=int(b), street=sname, kind="two", legal_ab=None, fav_ab=bool(e > 0)))
                continue
            if "two" in kinds:
                continue  # mixed one-way / two-way: excluded by spec
            # legal direction from each near segment, projected on the pair axis
            legal = []
            for da, db, ua, ub, kind in near:
                for dist, u in ((da, ua), (db, ub)):
                    if dist <= NEAR_M:
                        v = u if kind == "fwd" else -u
                        c = float(v @ ab)
                        if abs(c) < np.cos(np.radians(MAX_ANG)):
                            legal.append(None)
                        else:
                            legal.append(c > 0)
            if None in legal or len(set(legal)) != 1:
                continue  # inconsistent bearing
            rows.append(dict(a=int(a), b=int(b), street=sname, kind="one", legal_ab=bool(legal[0]), fav_ab=bool(e > 0)))
            break

    one = [r for r in rows if r["kind"] == "one"]
    two = [r for r in rows if r["kind"] == "two"]
    n = len(one)
    if os.environ.get("DUMP_PAIRS"):  # descriptive dump for figures only (no test, no verdict)
        tt = dict(zip((lo * K + hi).tolist(), t[keep].tolist()))
        A = np.array([r["a"] for r in rows]); B = np.array([r["b"] for r in rows])
        dist = np.hypot(X[B] - X[A], Y[B] - Y[A])
        geo = []
        for w in osm:
            ow = w["tags"].get("oneway", "no").lower(); g = w.get("geometry")
            if g and len(g) >= 2:
                geo.append((np.array([[q["lat"], q["lon"]] for q in g]), ow in ONEWAY_YES or ow in ONEWAY_REV or w["tags"].get("junction") == "roundabout"))
        np.savez_compressed(f"results/figdata_oneway_{CITY}.npz", a=A, b=B, kind=np.array([r["kind"] for r in rows]),
                            legal_ab=np.array([r["legal_ab"] if r["legal_ab"] is not None else False for r in rows]),
                            fav_ab=np.array([r["fav_ab"] for r in rows]), street=np.array([r["street"] for r in rows]),
                            t=np.array([tt[int(a_ * K + b_)] for a_, b_ in zip(A, B)]), dist_m=dist,
                            lat_a=lat[A], lon_a=lon[A], lat_b=lat[B], lon_b=lon[B],
                            way_lens=np.array([len(g_) for g_, _ in geo]), way_pts=np.concatenate([g_ for g_, _ in geo]),
                            way_oneway=np.array([o_ for _, o_ in geo]))
        print("dumped", len(rows), "pairs,", len(geo), "ways"); return
    if os.environ.get("N_ONLY"):  # blind step: n and power only
        PP = float(os.environ.get("POWER_P", "0.6"))  # step 3b spec: power at 0.70
        sims = np.random.default_rng(11).binomial(n, PP, 1000) if n else np.zeros(1000)
        pw = float(np.mean([(s_ / n >= 0.60) and (binom.sf(s_ - 1, n, 0.5) < 0.01) for s_ in sims])) if n else 0.0
        json.dump(dict(city=CITY, n_one_way_pairs=n, n_two_way_pairs=len(two), power_p=PP, power=pw),
                  open(f"results/1_oneway_{CITY}_n_only.json", "w"), indent=1)
        print(dict(n=n, n_two=len(two), power=pw)); return
    k = sum(r["legal_ab"] == r["fav_ab"] for r in one)
    f = k / n if n else float("nan")
    p = binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
    ci = binomtest(k, n, 0.5).proportion_ci(0.95) if n else None
    # pre-specified power: P(supported | true matching prob 0.6) with this n
    rng = np.random.default_rng(11)
    sims = rng.binomial(n, float(os.environ.get("POWER_P", "0.6")), 1000) if n else np.zeros(1000)
    supp = [(s_ / n >= 0.60) and (binom.sf(s_ - 1, n, 0.5) < 0.01) for s_ in sims] if n else [False]
    power = float(np.mean(supp))
    if n < 30:
        verdict = "INCONCLUSIVE (n < 30)"
    elif f >= 0.60 and p < 0.01:
        verdict = "SUPPORTED"
    elif f <= 0.55 or (n >= 100 and ci.low <= 0.5 <= ci.high):
        verdict = "REJECTED"
    else:
        verdict = "INCONCLUSIVE"
    # two-way control: favoured direction vs a fixed orientation (north/east-ward) -> should be ~0.5
    t_n = len(two)
    t_k = sum(((Y[r["b"]] - Y[r["a"]]) + (X[r["b"]] - X[r["a"]]) > 0) == r["fav_ab"] for r in two)
    out = dict(city=CITY, osm=OSM, confirmatory=CITY != "citibike", n_one_way_pairs=n, matches=k, f=f, p_one_sided=p,
               ci95=[ci.low, ci.high] if ci else None, power_at_0_6=power, verdict=verdict,
               underpowered=power < 0.8,
               two_way_control=dict(n=t_n, f_fixed_orientation=(t_k / t_n) if t_n else None,
                                    flag=bool(t_n and t_k / t_n >= 0.60)),
               streets_top=sorted({r["street"] for r in one})[:40])
    json.dump(out, open(f"results/1_oneway_{CITY}{SENS}.json", "w"), indent=1)
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "streets_top"}, indent=1))


if __name__ == "__main__":
    main()
