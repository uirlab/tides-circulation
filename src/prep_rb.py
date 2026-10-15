"""Prep for the robustness checks.
Parameterised copy of src/prep_global.py: same parsing, strata and filters, but the month, input files, holidays and
group-ride window are arguments. Writes data/processed/rb/<out>_weekday_trips.npz. Run from the repository root (cluster recommended):
    python src/prep_rb.py <system> <YYYY-MM> <out> [GROUP_S]
systems: lon seo mex bue ybk tpe baw bar"""
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HOL = {  # fixed in the spec before any data were opened
    ("lon", "2026-04"): ["2026-04-03", "2026-04-06"], ("seo", "2026-04"): [],
    ("mex", "2026-04"): ["2026-04-02", "2026-04-03"], ("bue", "2026-04"): ["2026-04-02", "2026-04-03"],
    ("ybk", "2026-04"): ["2026-04-03", "2026-04-06"], ("tpe", "2026-04"): ["2026-04-03", "2026-04-06"],
    ("baw", "2025-09"): ["2025-09-01"], ("bar", "2025-09"): ["2025-09-01"],
    ("lon", "2026-05"): ["2026-05-04", "2026-05-25"],  # R4 group-window variants of the May London unit
}
FILES = {
    ("lon", "2026-04"): ["data/raw/tfl/441JourneyDataExtract01Apr2026-15Apr2026.csv", "data/raw/tfl/442JourneyDataExtract16Apr2026-30Apr2026.csv"],
    ("lon", "2026-05"): ["data/raw/tfl/443JourneyDataExtract01May2026-16May2026.csv", "data/raw/tfl/444JourneyDataExtract17May2026-31May2026.csv"],
    ("seo", "2026-04"): ["data/raw/seoul/seoul_bike_rental_2604.csv"],
    ("mex", "2026-04"): ["data/raw/ecobici_mx/2026-04.csv"],
    ("bue", "2026-04"): ["data/raw/ecobici_ba/recorridos-realizados-2026.zip"],
    ("ybk", "2026-04"): ["data/raw/youbike/202604_youbike.zip"],
    ("tpe", "2026-04"): ["data/raw/taipei_mrt/taipei_mrt_od_202604.csv"],
    ("baw", "2025-09"): ["data/raw/baywheels/202509-baywheels-tripdata.csv.zip"],
    ("bar", "2025-09"): ["data/raw/bart/date-hour-soo-dest-2025.csv.gz"],
}


def finish(out, ym, hol, start, end, s_id, e_id, stratum, s_name=None, lat=None, lon=None, weight=None, group=True, group_s=60):
    df = pd.DataFrame(dict(start=start, end=end, s=s_id.astype(str), e=e_id.astype(str), st=stratum))
    if s_name is not None: df["name"] = s_name
    if lat is not None: df["lat"], df["lon"] = lat, lon
    if weight is not None: df["w"] = weight
    df = df.dropna(subset=["start", "s", "e"])
    df = df[(df.s != df.e) & (df.s != "nan") & (df.e != "nan")]
    d = df.start.dt
    y, m = map(int, ym.split("-"))
    hol = {pd.Timestamp(x).date() for x in hol}
    df = df[(d.year == y) & (d.month == m) & (d.weekday < 5) & ~d.date.isin(hol)]
    n_pre = len(df)
    if group:
        df = df.sort_values(["s", "e", "start"]).reset_index(drop=True)
        same = df.s.eq(df.s.shift()) & df.e.eq(df.e.shift())
        dup = same & ((df.start - df.start.shift()).dt.total_seconds() <= group_s) & ((df.end - df.end.shift()).dt.total_seconds().abs() <= group_s)
        df = df[~dup]
    if weight is not None:
        df = df.loc[df.index.repeat(df.w.astype(int))]
    ids = np.array(sorted(set(df.s) | set(df.e))); idx = {x: i for i, x in enumerate(ids)}
    names = (df.groupby("s").name.first().reindex(ids).fillna("").to_numpy() if "name" in df else ids)
    if "lat" in df:
        geo = df.groupby("s")[["lat", "lon"]].median().reindex(ids); la, lo = geo.lat.to_numpy(float), geo.lon.to_numpy(float)
    else:
        la = lo = np.full(ids.size, np.nan)
    days = sorted(df.start.dt.date.unique()); dmap = {x: i for i, x in enumerate(days)}
    os.makedirs("data/processed/rb", exist_ok=True)
    res = dict(o=df.s.map(idx).to_numpy(np.int32), e=df.e.map(idx).to_numpy(np.int32),
               day=df.start.dt.date.map(dmap).to_numpy(np.int16),
               tmin=(df.start.dt.hour * 60 + df.start.dt.minute + df.start.dt.second / 60).to_numpy(np.float32),
               stratum=df.st.to_numpy(np.int8), station_id=ids, station_name=np.asarray(names).astype(str),
               days=np.array([str(x) for x in days]), lat=la, lon=lo)
    print(dict(unit=out, trips=len(df), before_group=n_pre, stations=ids.size, weekdays=len(days),
               strata=np.unique(df.st, return_counts=True)[1].tolist()), flush=True)
    return res


def save(out, res):
    np.savez_compressed(f"data/processed/rb/{out}_weekday_trips.npz", **res)


def tpe_clean(d):
    """Exactly src/tpe_merge.py then src/tpe_exclude.py."""
    names = d["station_name"].astype(str)
    plain = np.array([re.sub(r"^[A-Za-z]+", "", n) for n in names])
    uniq, new = np.unique(plain, return_inverse=True)
    o, e = new[d["o"]], new[d["e"]]
    keep = o != e
    out = {k: v[keep] if (isinstance(v, np.ndarray) and v.shape[:1] == keep.shape) else v for k, v in d.items()}
    out["o"], out["e"] = o[keep], e[keep]
    out["station_id"] = uniq; out["station_name"] = uniq
    for k in ("lat", "lon"):
        out[k] = np.full(uniq.size, np.nan)
    names = out["station_name"].astype(str)
    bad = np.isin(names, ["板橋", "大坪林", "景安", "頭前庄"]); assert bad.sum() == 4, bad.sum()
    keep = ~(bad[out["o"]] | bad[out["e"]])
    print("merged codes", plain.size, "->", uniq.size, "; interchange exclusion dropped", int((~keep).sum()), "of", keep.size)
    return {k: v[keep] if (isinstance(v, np.ndarray) and v.shape[:1] == keep.shape) else v for k, v in out.items()}


def main():
    which, ym, out = sys.argv[1], sys.argv[2], sys.argv[3]
    gs = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    hol, files = HOL[(which, ym)], FILES[(which, ym)]
    F = lambda **kw: finish(out, ym, hol, **kw)
    if which == "lon":
        df = pd.concat([pd.read_csv(f, usecols=["Start date", "End date", "Start station number", "End station number", "Start station", "Bike model"], dtype=str) for f in files])
        r = F(start=pd.to_datetime(df["Start date"]), end=pd.to_datetime(df["End date"]), s_id=df["Start station number"], e_id=df["End station number"],
              stratum=df["Bike model"].map({"CLASSIC": 0, "PBSC_EBIKE": 1}).fillna(-1).astype(int).values, s_name=df["Start station"].values, group_s=gs)
    elif which == "seo":
        df = pd.read_csv(files[0], encoding="cp949", dtype=str, usecols=["대여일시", "반납일시", "대여대여소ID", "반납대여소ID", "대여 대여소명", "이용자종류"])
        r = F(start=pd.to_datetime(df["대여일시"], errors="coerce"), end=pd.to_datetime(df["반납일시"], errors="coerce"), s_id=df["대여대여소ID"], e_id=df["반납대여소ID"],
              stratum=df["이용자종류"].map({"내국인": 0, "비회원": 1, "외국인": 1}).fillna(-1).astype(int).values, s_name=df["대여 대여소명"].values, group_s=gs)
    elif which == "mex":
        df = pd.read_csv(files[0], dtype=str)
        st = pd.to_datetime(df.Fecha_Retiro + " " + df.Hora_Retiro, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        en = pd.to_datetime(df.Fecha_Arribo + " " + df.Hora_Arribo, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        r = F(start=st, end=en, s_id=df.Ciclo_Estacion_Retiro, e_id=df.Ciclo_EstacionArribo, stratum=np.zeros(len(df), int), group_s=gs)
    elif which == "bue":
        with zipfile.ZipFile(files[0]) as z, z.open("Trips.csv") as f:
            df = pd.read_csv(f, dtype=str, usecols=["fecha_origen_recorrido", "fecha_destino_recorrido", "id_estacion_origen", "id_estacion_destino",
                                                   "nombre_estacion_origen", "lat_estacion_origen", "long_estacion_origen", "modelo_bicicleta"])
        models = {m: i for i, m in enumerate(df.modelo_bicicleta.value_counts().index[:4])}
        print("BA bike models -> strata", models)
        r = F(start=pd.to_datetime(df.fecha_origen_recorrido, errors="coerce"), end=pd.to_datetime(df.fecha_destino_recorrido, errors="coerce"),
              s_id=df.id_estacion_origen, e_id=df.id_estacion_destino, stratum=df.modelo_bicicleta.map(models).fillna(-1).astype(int).values,
              s_name=df.nombre_estacion_origen.values, lat=pd.to_numeric(df.lat_estacion_origen, errors="coerce").values,
              lon=pd.to_numeric(df.long_estacion_origen, errors="coerce").values, group_s=gs)
    elif which == "tpe":
        df = pd.read_csv(files[0], encoding="utf-8-sig", dtype={"時段": int, "人次": int})
        df = df[df["人次"] > 0]
        t = pd.to_datetime(df["日期"]) + pd.to_timedelta(df["時段"] * 60 + 30, unit="m")
        r = tpe_clean(F(start=t, end=t, s_id=df["進站"], e_id=df["出站"], stratum=np.zeros(len(df), int), weight=df["人次"].values, group=False))
    elif which == "bar":
        df = pd.read_csv(files[0], header=None, names=["date", "hour", "o", "d", "n"], dtype={"o": str, "d": str})
        df = df[df.date.str.startswith(ym) & (df.n > 0)]
        t = pd.to_datetime(df.date) + pd.to_timedelta(df.hour * 60 + 30, unit="m")
        r = F(start=t, end=t, s_id=df.o, e_id=df.d, stratum=np.zeros(len(df), int), weight=df.n.values, group=False)
    elif which == "baw":
        with zipfile.ZipFile(files[0]) as z:
            name = [n for n in z.namelist() if n.endswith(".csv") and "MACOSX" not in n][0]
            df = pd.read_csv(z.open(name), dtype={"start_station_id": str, "end_station_id": str})
        st = (df.rideable_type.map({"classic_bike": 0, "electric_bike": 1}) * 2 + df.member_casual.map({"member": 0, "casual": 1})).fillna(-1).astype(int)
        r = F(start=pd.to_datetime(df.started_at), end=pd.to_datetime(df.ended_at), s_id=df.start_station_id, e_id=df.end_station_id, stratum=st.values,
              s_name=df.start_station_name.values, lat=df.start_lat.values, lon=df.start_lng.values, group_s=gs)
    elif which == "ybk":
        with zipfile.ZipFile(files[0]) as z:
            df = pd.read_csv(z.open(z.namelist()[0]), encoding="utf-8-sig", dtype=str, usecols=["rent_time", "rent_station", "return_station", "bike_type"])
        t = pd.to_datetime(df.rent_time, errors="coerce") + pd.Timedelta(minutes=30)
        types = {k: i for i, k in enumerate(df.bike_type.value_counts().index[:4])}
        print("YouBike bike types -> strata", types)
        r = F(start=t, end=t, s_id=df.rent_station, e_id=df.return_station, stratum=df.bike_type.map(types).fillna(-1).astype(int).values,
              s_name=df.rent_station.values, group=False)
    save(out, r)


if __name__ == "__main__":
    main()
