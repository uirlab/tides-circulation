"""Trip volume per time bin (06-22 h, weekdays as in the analysis files) for Fig 2d. Run from the repository root (cluster recommended).
Output: results/figdata_volume.json  {city: [counts per bin]} (15-min bins; Taipei 60-min)."""
import json

import numpy as np

out = {}
for c, w in [("citibike_202605", 15), ("divvy_mj_202605", 15), ("bluebikes_202605", 15), ("g_lon", 15), ("g_seo", 15), ("g_mex", 15),
             ("g_bue", 15), ("g_tpe", 60), ("g_tpem", 60), ("g_tpex", 60), ("g_bar", 60), ("g_baw", 15), ("g_ybk", 60)]:
    d = np.load(f"data/processed/{c}_weekday_trips.npz", allow_pickle=True)
    tm = d["tmin"].astype(float); nb = int(16 * 60 / w)
    out[c.replace("_202605", "")] = np.histogram(tm, bins=np.arange(nb + 1) * w + 360)[0].tolist()
json.dump(out, open("results/figdata_volume.json", "w"))
print({k: sum(v) for k, v in out.items()})
