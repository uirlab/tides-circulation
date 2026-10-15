"""Attach coordinates to the global systems' station indices (for Fig 2a maps only).
Inputs: results/figdata_stations_<city>.npz (ids/names in model order, ) and the station lists in
data/raw/stations/ (see data/README.md). Output: results/figdata_coords_<city>.npz (lat, lon; NaN if unmatched).
Run from the project root: python3 src/station_coords.py
"""
import json
import re

import numpy as np
import pandas as pd

R = "data/raw/stations/"


def norm(s):
    return re.sub(r"\s+", "", str(s)).replace("台", "臺")


def save(city, lat, lon):
    np.savez(f"results/figdata_coords_{city}.npz", lat=lat, lon=lon)
    print(f"{city}: matched {np.isfinite(lat).sum()}/{lat.size}")


def london():
    s = np.load("results/figdata_stations_g_lon.npz"); ids, names = s["station_id"], s["station_name"]
    bp = json.load(open(R + "tfl_bikepoint.json"))
    by_term = {}; by_name = {}
    for p in bp:
        props = {q["key"]: q["value"] for q in p["additionalProperties"]}
        by_term[props.get("TerminalName", "")] = (p["lat"], p["lon"]); by_name[norm(p["commonName"])] = (p["lat"], p["lon"])
    ll = [by_term.get(i) or by_name.get(norm(n)) or (np.nan, np.nan) for i, n in zip(ids, names)]
    save("g_lon", *np.array(ll, float).T)


def mexico():
    s = np.load("results/figdata_stations_g_mex.npz"); ids = s["station_id"]
    st = json.load(open(R + "mex_station_information.json"))["data"]["stations"]
    m = {}
    for x in st:
        g = re.match(r"\d+", str(x.get("short_name", "")))
        if g:
            m[int(g.group())] = (x["lat"], x["lon"])
    ll = [m.get(int(re.match(r"\d+", i).group())) if re.match(r"\d+", i) else None for i in ids]
    save("g_mex", *np.array([v or (np.nan, np.nan) for v in ll], float).T)


def seoul():
    s = np.load("results/figdata_stations_g_seo.npz"); names = s["station_name"]
    x = pd.read_excel(R + "seoul_stations_2606.xlsx", header=None, skiprows=5).iloc[:, [0, 1, 4, 5]]
    x.columns = ["no", "name", "lat", "lon"]
    m = {norm(n): (a, b) for n, a, b in zip(x.name, x.lat, x.lon) if pd.notna(a)}
    ll = [m.get(norm(n), (np.nan, np.nan)) for n in names]
    save("g_seo", *np.array(ll, float).T)


def taipei():
    s = np.load("results/figdata_stations_g_tpe.npz"); names = s["station_name"]
    x = pd.read_csv(R + "taipei_mrt_exits.csv", encoding="big5")
    x["st"] = [norm(re.split(r"站", n)[0]) for n in x["出入口名稱"]]
    g = x.groupby("st")[["緯度", "經度"]].mean()
    m = {k: (r["緯度"], r["經度"]) for k, r in g.iterrows()}
    ll = [m.get(norm(re.sub(r"站$", "", re.sub(r"^[A-Za-z]+", "", n))), (np.nan, np.nan)) for n in names]  # "台北車站" -> "臺北車", as split from exit names
    lat, lon = np.array(ll, float).T
    miss = [n for n, a in zip(names, lat) if not np.isfinite(a)]
    print("taipei unmatched:", miss[:40])
    save("g_tpe", lat, lon)


def bart():
    import zipfile
    s = np.load("results/figdata_stations_g_bar.npz"); ids = s["station_id"]
    st = pd.read_csv(zipfile.ZipFile(R + "bart_google_transit.zip").open("stops.txt"), dtype={"stop_id": str})
    m = {r.stop_id: (r.stop_lat, r.stop_lon) for r in st.itertuples()}
    save("g_bar", *np.array([m.get(i, (np.nan, np.nan)) for i in ids], float).T)


def youbike():
    s = np.load("results/figdata_stations_g_ybk.npz"); names = s["station_id"]  # YouBike ids are the station names
    st = json.load(open(R + "youbike_immediate.json"))
    m = {norm(x["sna"].replace("YouBike2.0_", "")): (x["latitude"], x["longitude"]) for x in st}
    save("g_ybk", *np.array([m.get(norm(n), (np.nan, np.nan)) for n in names], float).T)


for f in (london, mexico, seoul, taipei, bart, youbike):
    f()
