"""Taxi-zone unit for the decomposition: NYC yellow taxi + high-volume for-hire trips in one
month, weekdays 06:00-22:00, between taxi zones, in the weekday_trips format used by src/irrev.py.
Run from the repository root:  python src/prep_tlc_unit.py 2025 5   -> data/processed/tlc_2505_weekday_trips.npz
Strata: 0 = yellow, 1 = high-volume for-hire."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from tlc_prep import load_month  # noqa: E402

yy, m = int(sys.argv[1]), int(sys.argv[2])
y0 = yy if m >= 10 else yy - 1
parts = [load_month(k, y0, m) for k in ("yellow", "fhvhv")]
o = np.concatenate([p[0] for p in parts]); e = np.concatenate([p[1] for p in parts]); tmin = np.concatenate([p[2] for p in parts])
date = np.concatenate([p[3] for p in parts]); st = np.concatenate([p[4] for p in parts])
zones = np.unique(np.r_[o, e]); idx = {z: i for i, z in enumerate(zones)}
lk = pd.read_csv("data/raw/tlc/taxi_zone_lookup.csv").set_index("LocationID")
days = np.array(sorted(set(date))); dix = {d: i for i, d in enumerate(days)}
out = dict(o=np.array([idx[z] for z in o], np.int32), e=np.array([idx[z] for z in e], np.int32),
           day=np.array([dix[d] for d in date], np.int16), tmin=tmin.astype(np.float32), stratum=st.astype(np.int8),
           station_id=zones.astype(str), station_name=np.array([f"{lk.Borough.get(z, '')}: {lk.Zone.get(z, '')}" for z in zones]),
           days=np.array([str(d) for d in days]), lat=np.full(zones.size, np.nan), lon=np.full(zones.size, np.nan))
f = f"data/processed/tlc_{yy % 100:02d}{m:02d}_weekday_trips.npz"
np.savez_compressed(f, **out)
print(f, "trips", o.size, "zones", zones.size, "days", days.size, "by stratum", np.bincount(st))
