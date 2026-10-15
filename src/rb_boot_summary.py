"""R2a summary: paired day-subsampling intervals for the
within-city bike-minus-rail difference in h'. Run from the repository root (cluster recommended) after src/rb_boot.py.
Writes results/rb/r2a_boot_summary.json and prints the rule outcome per city."""
import glob
import json

import numpy as np

PAIRS = {"Taipei": ("g_ybk", "g_tpex"), "Bay Area": ("g_baw", "g_bar")}


def load(unit):
    out = {}
    for f in glob.glob(f"results/rb/boot/{unit}_*.json"):
        r = json.load(open(f)); out[r["seed"]] = (np.array([b["h"] for b in r["per_bin"]]), r["draw"])
    return out


res = {}
for city, (b, r) in PAIRS.items():
    B, R = load(b), load(r)
    seeds = sorted(set(B) & set(R))
    assert all(B[s][1] == R[s][1] for s in seeds), "bike and rail replicates must use the same days"
    hb = np.array([B[s][0] for s in seeds]); hr = np.array([R[s][0] for s in seeds])
    d_hour = hb - hr                                    # replicate x hour
    d_med = np.median(hb, axis=1) - np.median(hr, axis=1)
    lo, hi = np.percentile(d_med, [2.5, 97.5])
    hl, hh = np.percentile(d_hour, [2.5, 97.5], axis=0)
    res[city] = dict(n_replicates=len(seeds), median_bike=float(np.median(np.median(hb, axis=1))), median_rail=float(np.median(np.median(hr, axis=1))),
                     diff_median=dict(point=float(np.median(d_med)), ci95=[float(lo), float(hi)]),
                     ratio_median=float(np.median(np.median(hb, axis=1) / np.median(hr, axis=1))),
                     hours_ci_above_zero=int((hl > 0).sum()), hours_ci_below_zero=int((hh < 0).sum()), n_hours=int(hb.shape[1]),
                     per_hour=[dict(hour=6 + i, diff=float(np.median(d_hour[:, i])), ci95=[float(hl[i]), float(hh[i])]) for i in range(hb.shape[1])],
                     share_reps_rule_13of16=float(np.mean([(x > y).sum() >= 13 and np.median(x) > np.median(y) for x, y in zip(hb, hr)])),
                     rule="ROBUST" if lo > 0 else "NOT ROBUST")
    print(city, res[city]["n_replicates"], "reps | diff in medians", round(res[city]["diff_median"]["point"], 3), np.round(res[city]["diff_median"]["ci95"], 3),
          "| hours with CI > 0:", res[city]["hours_ci_above_zero"], "/", res[city]["n_hours"], "| CI < 0:", res[city]["hours_ci_below_zero"],
          "| reps meeting 13/16 rule:", round(res[city]["share_reps_rule_13of16"], 2), "|", res[city]["rule"])
json.dump(res, open("results/rb/r2a_boot_summary.json", "w"), indent=1)
