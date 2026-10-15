"""Prep for the global replication: London, Seoul, Mexico City, Buenos Aires, Taipei MRT.
Writes data/processed/g_<scenario>_weekday_trips.npz in the same format as src/prep_unit.py. Run on a cluster."""
import sys
import zipfile

import numpy as np
import pandas as pd

HOL = {"lon": ["2026-05-04", "2026-05-25"], "seo": ["2026-05-01", "2026-05-05", "2026-05-25"], "mex": ["2026-05-01"],
       "bue": ["2026-05-01", "2026-05-25"], "tpe": ["2026-05-01"],
       # within-city bike vs rail
       "bar": ["2025-05-26"], "baw": ["2025-05-26"], "ybk": ["2026-05-01"],
       # H-XFER: June 2023
       "xj6": ["2023-06-22", "2023-06-23"], "xn6": ["2023-06-22", "2023-06-23"]}
YEAR = {"bar": 2025, "baw": 2025, "xj6": 2023, "xn6": 2023}
MONTH = {"xj6": 6, "xn6": 6}  # Bay Area uses May 2025 (BART 2026 file empty); all others May 2026


def finish(name, start, end, s_id, e_id, stratum, s_name=None, lat=None, lon=None, weight=None, group=True):
    df = pd.DataFrame(dict(start=start, end=end, s=s_id.astype(str), e=e_id.astype(str), st=stratum))
    if s_name is not None: df["name"] = s_name
    if lat is not None: df["lat"], df["lon"] = lat, lon
    if weight is not None: df["w"] = weight
    df = df.dropna(subset=["start", "s", "e"])
    df = df[(df.s != df.e) & (df.s != "nan") & (df.e != "nan")]
    d = df.start.dt
    hol = {pd.Timestamp(x).date() for x in HOL[name[2:5]]}
    df = df[(d.year == YEAR.get(name[2:5], 2026)) & (d.month == MONTH.get(name[2:5], 5)) & (d.weekday < 5) & ~d.date.isin(hol)]
    n_pre = len(df)
    if group:
        df = df.sort_values(["s", "e", "start"]).reset_index(drop=True)
        same = df.s.eq(df.s.shift()) & df.e.eq(df.e.shift())
        dup = same & ((df.start - df.start.shift()).dt.total_seconds() <= 60) & ((df.end - df.end.shift()).dt.total_seconds().abs() <= 60)
        df = df[~dup]
    if weight is not None:  # aggregated counts -> repeat rows
        df = df.loc[df.index.repeat(df.w.astype(int))]
    ids = np.array(sorted(set(df.s) | set(df.e))); idx = {x: i for i, x in enumerate(ids)}
    names = (df.groupby("s").name.first().reindex(ids).fillna("").to_numpy() if "name" in df else ids)
    if "lat" in df:
        geo = df.groupby("s")[["lat", "lon"]].median().reindex(ids); la, lo = geo.lat.to_numpy(float), geo.lon.to_numpy(float)
    else:
        la = lo = np.full(ids.size, np.nan)
    days = sorted(df.start.dt.date.unique()); dmap = {x: i for i, x in enumerate(days)}
    np.savez_compressed(f"data/processed/{name}_weekday_trips.npz", o=df.s.map(idx).to_numpy(np.int32), e=df.e.map(idx).to_numpy(np.int32),
                        day=df.start.dt.date.map(dmap).to_numpy(np.int16),
                        tmin=(df.start.dt.hour * 60 + df.start.dt.minute + df.start.dt.second / 60).to_numpy(np.float32),
                        stratum=df.st.to_numpy(np.int8), station_id=ids, station_name=np.asarray(names).astype(str),
                        days=np.array([str(x) for x in days]), lat=la, lon=lo)
    print(dict(unit=name, trips=len(df), before_group=n_pre, stations=ids.size, weekdays=len(days),
               strata=np.unique(df.st, return_counts=True)[1].tolist()), flush=True)


which = sys.argv[1]
if which == "lon":
    df = pd.concat([pd.read_csv(f"data/raw/tfl/{f}", usecols=["Start date", "End date", "Start station number", "End station number", "Start station", "Bike model"], dtype=str)
                    for f in ("443JourneyDataExtract01May2026-16May2026.csv", "444JourneyDataExtract17May2026-31May2026.csv")])
    finish("g_lon", pd.to_datetime(df["Start date"]), pd.to_datetime(df["End date"]), df["Start station number"], df["End station number"],
           df["Bike model"].map({"CLASSIC": 0, "PBSC_EBIKE": 1}).fillna(-1).astype(int).values, s_name=df["Start station"].values)
elif which == "seo":
    df = pd.read_csv("data/raw/seoul/seoul_bike_rental_2605.csv", encoding="cp949", dtype=str,
                     usecols=["대여일시", "반납일시", "대여대여소ID", "반납대여소ID", "대여 대여소명", "이용자종류"])
    finish("g_seo", pd.to_datetime(df["대여일시"], errors="coerce"), pd.to_datetime(df["반납일시"], errors="coerce"), df["대여대여소ID"], df["반납대여소ID"],
           df["이용자종류"].map({"내국인": 0, "비회원": 1, "외국인": 1}).fillna(-1).astype(int).values, s_name=df["대여 대여소명"].values)
elif which == "mex":
    df = pd.read_csv("data/raw/ecobici_mx/2026-05.csv", dtype=str)
    st = pd.to_datetime(df.Fecha_Retiro + " " + df.Hora_Retiro, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    en = pd.to_datetime(df.Fecha_Arribo + " " + df.Hora_Arribo, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    finish("g_mex", st, en, df.Ciclo_Estacion_Retiro, df.Ciclo_EstacionArribo, np.zeros(len(df), int))
elif which == "bue":
    with zipfile.ZipFile("data/raw/ecobici_ba/recorridos-realizados-2026.zip") as z, z.open("Trips.csv") as f:
        df = pd.read_csv(f, dtype=str, usecols=["fecha_origen_recorrido", "fecha_destino_recorrido", "id_estacion_origen", "id_estacion_destino",
                                               "nombre_estacion_origen", "lat_estacion_origen", "long_estacion_origen", "modelo_bicicleta"])
    models = {m: i for i, m in enumerate(df.modelo_bicicleta.value_counts().index[:4])}
    print("BA bike models -> strata", models)
    finish("g_bue", pd.to_datetime(df.fecha_origen_recorrido, errors="coerce"), pd.to_datetime(df.fecha_destino_recorrido, errors="coerce"),
           df.id_estacion_origen, df.id_estacion_destino, df.modelo_bicicleta.map(models).fillna(-1).astype(int).values,
           s_name=df.nombre_estacion_origen.values, lat=pd.to_numeric(df.lat_estacion_origen, errors="coerce").values,
           lon=pd.to_numeric(df.long_estacion_origen, errors="coerce").values)
elif which == "tpe":
    df = pd.read_csv("data/raw/taipei_mrt/taipei_mrt_od_202605.csv", encoding="utf-8-sig", dtype={"時段": int, "人次": int})
    df = df[df["人次"] > 0]
    t = pd.to_datetime(df["日期"]) + pd.to_timedelta(df["時段"] * 60 + 30, unit="m")  # mid-hour stamp
    finish("g_tpe", t, t, df["進站"], df["出站"], np.zeros(len(df), int), weight=df["人次"].values, group=False)
elif which == "bar":  # BART hourly OD, May 2025 (date, hour, origin, destination, count); mid-hour stamps as for Taipei
    df = pd.read_csv("data/raw/bart/date-hour-soo-dest-2025.csv.gz", header=None, names=["date", "hour", "o", "d", "n"], dtype={"o": str, "d": str})
    df = df[df.date.str.startswith("2025-05") & (df.n > 0)]
    t = pd.to_datetime(df.date) + pd.to_timedelta(df.hour * 60 + 30, unit="m")
    finish("g_bar", t, t, df.o, df.d, np.zeros(len(df), int), weight=df.n.values, group=False)
elif which == "baw":  # Bay Wheels, May 2025, Lyft format
    with zipfile.ZipFile("data/raw/baywheels/202505-baywheels-tripdata.csv.zip") as z:
        name = [n for n in z.namelist() if n.endswith(".csv") and "MACOSX" not in n][0]
        df = pd.read_csv(z.open(name), dtype={"start_station_id": str, "end_station_id": str})
    st = (df.rideable_type.map({"classic_bike": 0, "electric_bike": 1}) * 2 + df.member_casual.map({"member": 0, "casual": 1})).fillna(-1).astype(int)
    finish("g_baw", pd.to_datetime(df.started_at), pd.to_datetime(df.ended_at), df.start_station_id, df.end_station_id, st.values,
           s_name=df.start_station_name.values, lat=df.start_lat.values, lon=df.start_lng.values)
elif which == "ybk":  # YouBike 2.0 Taipei, May 2026; hour-truncated timestamps -> mid-hour, no group collapse
    with zipfile.ZipFile("data/raw/youbike/202605_youbike.zip") as z:
        df = pd.read_csv(z.open(z.namelist()[0]), encoding="utf-8-sig", dtype=str, usecols=["rent_time", "rent_station", "return_station", "bike_type"])
    t = pd.to_datetime(df.rent_time, errors="coerce") + pd.Timedelta(minutes=30)
    types = {k: i for i, k in enumerate(df.bike_type.value_counts().index[:4])}
    print("YouBike bike types -> strata", types)
    finish("g_ybk", t, t, df.rent_station, df.return_station, df.bike_type.map(types).fillna(-1).astype(int).values,
           s_name=df.rent_station.values, group=False)
