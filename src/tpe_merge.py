"""Post-hoc data cleaning for Taipei MRT (not pre-specified).
Interchange stations shared with the Circular Line appear under two codes (line-prefixed and plain, e.g. G大坪林 / 大坪林);
cross-operator transfers are then recorded as one-directional 'trips' between the two codes. Merge codes by plain name
and drop the resulting same-station records. Run from the repository root (cluster recommended): python src/tpe_merge.py
Output: data/processed/g_tpem_weekday_trips.npz
"""
import re

import numpy as np

d = dict(np.load("data/processed/g_tpe_weekday_trips.npz", allow_pickle=True))
names = d["station_name"].astype(str)
plain = np.array([re.sub(r"^[A-Za-z]+", "", n) for n in names])
uniq, new = np.unique(plain, return_inverse=True)
o, e = new[d["o"]], new[d["e"]]
keep = o != e
print("codes", names.size, "->", uniq.size, "| merged:", sorted(set(names) - set(uniq)), "| dropped same-station records", int((~keep).sum()), "of", keep.size)
out = {k: v[keep] if (isinstance(v, np.ndarray) and v.shape[:1] == keep.shape) else v for k, v in d.items()}
out["o"], out["e"] = o[keep], e[keep]
out["station_id"] = uniq; out["station_name"] = uniq
for k in ("lat", "lon"):
    out[k] = np.full(uniq.size, np.nan)
np.savez_compressed("data/processed/g_tpem_weekday_trips.npz", **out)
