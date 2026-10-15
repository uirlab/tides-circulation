"""R2b threshold curves from existing results. Descriptive; run on a cluster.
Writes results/rb/r2b_thresholds.json."""
import json
import os

import numpy as np
from scipy.stats import binom

SYS15 = {"New York": "results/0cb_strata_real.json", "Chicago": "results/1_replication_divvy_mj_real.json",
         "Boston": "results/1_replication_bluebikes_real.json", "London": "results/1_replication_g_lon_real.json",
         "Seoul": "results/1_replication_g_seo_real.json", "Mexico City": "results/1_replication_g_mex_real.json",
         "Buenos Aires": "results/1_replication_g_bue_real.json", "SF Bay (bike)": "results/1_replication_g_baw_real.json"}
SYS60 = {"YouBike": "results/1_replication_g_ybk_real.json", "Taipei Metro": "results/1_replication_g_tpex_real.json",
         "Bay Wheels (60 min)": "results/1_replication_g_baw_b60_real.json", "BART": "results/1_replication_g_bar_real.json"}
PAIRS = {"Taipei": ("YouBike", "Taipei Metro"), "Bay Area": ("Bay Wheels (60 min)", "BART")}


def curves(path):
    r = json.load(open(path)); d, pb = r["decision"], r["per_bin"]
    z = np.array([b["z"] for b in pb]); h = np.array([b["h"] for b in pb]); hb = np.array([b["h_big"] for b in pb])
    split_ok = d["rho"] > 0 and d["p_split"] < 0.01 and d.get("rho_big", d["rho"]) > 0 and d.get("p_split_big", d["p_split"]) < 0.01
    half = z.size / 2
    go_by_h = {f"{t:.2f}": bool((z >= 5).sum() >= half and np.median(h) >= t and np.median(hb) >= t and split_ok)
               for t in np.round(np.arange(0.05, 0.201, 0.01), 2)}
    share_z = {str(k): float((z >= k).mean()) for k in range(3, 11)}
    return dict(median_h=float(np.median(h)), go_by_h_threshold=go_by_h, share_windows_z_ge=share_z, h=h.tolist())


out = {"systems": {}, "within_city": {}}
for name, p in {**SYS15, **SYS60}.items():
    if os.path.exists(p):
        out["systems"][name] = curves(p)
for city, (b, r) in PAIRS.items():
    hb, hr = np.array(out["systems"][b]["h"]), np.array(out["systems"][r]["h"])
    k = int((hb > hr).sum()); n = hb.size
    out["within_city"][city] = dict(hours_bike_gt_rail=k, n_hours=n, median_bike=float(np.median(hb)), median_rail=float(np.median(hr)),
                                    rule_met_by_required_count={str(m): bool(k >= m and np.median(hb) > np.median(hr)) for m in range(9, 17)},
                                    one_sided_p_by_required_count={str(m): float(binom.sf(m - 1, n, 0.5)) for m in range(9, 17)})
os.makedirs("results/rb", exist_ok=True)
json.dump(out, open("results/rb/r2b_thresholds.json", "w"), indent=1)
for c, v in out["within_city"].items():
    print(c, v["hours_bike_gt_rail"], "/", v["n_hours"])
for s, v in out["systems"].items():
    print(s, round(v["median_h"], 3), "GO for h' thresholds up to", max([float(t) for t, g in v["go_by_h_threshold"].items() if g] or [0]))
