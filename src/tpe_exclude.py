"""Post-hoc cleaning v2 for Taipei MRT: drop every trip that starts or ends at
one of the four interchange stations shared with the Circular Line (板橋, 大坪林, 景安, 頭前庄), whose records mix
cross-operator transfer legs. Input: data/processed/g_tpem_weekday_trips.npz (codes merged). Run on a cluster.
Output: data/processed/g_tpex_weekday_trips.npz (station index unchanged; excluded stations simply carry no trips)."""
import numpy as np

d = dict(np.load("data/processed/g_tpem_weekday_trips.npz", allow_pickle=True))
names = d["station_name"].astype(str)
bad = np.isin(names, ["板橋", "大坪林", "景安", "頭前庄"]); assert bad.sum() == 4
keep = ~(bad[d["o"]] | bad[d["e"]])
print("dropped", int((~keep).sum()), "of", keep.size, f"({(~keep).mean():.1%})")
out = {k: v[keep] if (isinstance(v, np.ndarray) and v.shape[:1] == keep.shape) else v for k, v in d.items()}
np.savez_compressed("data/processed/g_tpex_weekday_trips.npz", **out)
