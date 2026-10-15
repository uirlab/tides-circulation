"""Extract SegmentID, Street, TrafDir, BikeLane, BIKE_TRAFDIR from each LION release (no geometry).
Run from the repository root (cluster recommended): python src/lion_extract.py -> data/processed/lion_attrs.csv.gz"""
import glob
import os

import pandas as pd
import pyogrio

rows = []
for z in sorted(glob.glob("data/raw/lion/nyclion_*.zip")):
    v = os.path.basename(z)[8:11]
    path = f"/vsizip/{z}/lion/lion.gdb"
    layers = [l[0] for l in pyogrio.list_layers(path)]
    lyr = "lion" if "lion" in layers else [l for l in layers if l.lower() == "lion"][0]
    cols = [c for c in pyogrio.read_info(path, layer=lyr)["fields"]]
    want = {k: next((c for c in cols if c.lower() == k.lower()), None) for k in ("SegmentID", "Street", "TrafDir", "BikeLane", "BIKE_TRAFDIR")}
    _, tbl = pyogrio.read_arrow(path, layer=lyr, columns=[c for c in want.values() if c], read_geometry=False)
    df = tbl.to_pandas()
    df = df.rename(columns={c: k for k, c in want.items() if c})
    df["release"] = v
    rows.append(df)
    print(v, len(df), {k: bool(c) for k, c in want.items()}, flush=True)
out = pd.concat(rows, ignore_index=True)
out["SegmentID"] = out["SegmentID"].astype(str).str.strip().str.zfill(7)
out.to_csv("data/processed/lion_attrs.csv.gz", index=False)
print("rows", len(out))
