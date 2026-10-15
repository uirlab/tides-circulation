"""Uncertainty extension: delete-d jackknife standard errors and 95% intervals for the
headline h' of every system and for the within-city bike-minus-rail contrasts, from the 75% weekday subsamples written by
src/rb_boot.py. Run from the repository root (cluster recommended). Writes results/rb/uncertainty_summary.json.

With n weekdays, subsamples of m = round(0.75 n) days (d = n - m deleted) and N random subsamples, the delete-d jackknife
variance is v = m / (d N) * sum_s (theta_s - mean theta)^2 (Shao & Wu 1989). The spread of the subsamples alone understates the
sampling variability of the full-sample estimate by the factor sqrt(m / d), about 1.7. Intervals are theta_hat +/- 1.96 sqrt(v),
centred on the full-sample estimate; the raw 2.5-97.5 percentiles of the subsamples are also reported."""
import glob
import json
import os

import numpy as np

FULL = {"citibike": "results/0cb_strata_real.json", "divvy_mj": "results/1_replication_divvy_mj_real.json",
        "bluebikes": "results/1_replication_bluebikes_real.json", "g_lon": "results/1_replication_g_lon_real.json",
        "g_seo": "results/1_replication_g_seo_real.json", "g_mex": "results/1_replication_g_mex_real.json",
        "g_bue": "results/1_replication_g_bue_real.json", "g_baw": "results/1_replication_g_baw_real.json",
        "tlc_2505": "results/1_replication_tlc_2505_real.json"}  # taxi-zone application
FULL60 = {"g_ybk": "results/1_replication_g_ybk_real.json", "g_tpex": "results/1_replication_g_tpex_real.json",
          "g_baw": "results/1_replication_g_baw_b60_real.json", "g_bar": "results/1_replication_g_bar_real.json"}
PAIRS = {"Taipei": ("g_ybk", "g_tpex"), "Bay Area": ("g_baw", "g_bar")}


def trips_npz(unit):
    f = f"data/processed/{unit}_202605_weekday_trips.npz"
    d = np.load(f if os.path.exists(f) else f"data/processed/{unit}_weekday_trips.npz", allow_pickle=True)
    t = d["tmin"]; w = (t >= 6 * 60) & (t < 22 * 60)  # trips in the analysed hours (06:00-22:00), as in Fig. 2f
    return np.bincount(d["day"][w], minlength=d["days"].size), d["days"].size


def stats(per_bin, n_trips):
    h = np.array([b["h"] for b in per_bin]); g = np.array([b["g"] for b in per_bin])
    gs = np.array([b["gs"] for b in per_bin]); m0 = np.array([b["null_mean"] for b in per_bin])
    return dict(median_h=float(np.median(h)), circ_per_trip=float((g - m0).sum() / n_trips),
                tide_per_trip=float((gs - g).sum() / n_trips), h=h)


def load(folder, unit, full_path):
    per_day, n = trips_npz(unit)
    reps = {}
    for f in glob.glob(f"{folder}/{unit}_*.json"):
        r = json.load(open(f))
        reps[r["seed"]] = stats(r["per_bin"], int(per_day[r["draw"]].sum()))
    full = stats(json.load(open(full_path))["per_bin"], int(per_day.sum()))
    m = int(round(0.75 * n)); return full, reps, m / (n - m), n, m


def jk(full_val, rep_vals, factor):
    x = np.asarray(rep_vals, float); se = float(np.sqrt(factor * x.var(ddof=0)))
    return dict(estimate=float(full_val), se=se, ci95=[float(full_val - 1.96 * se), float(full_val + 1.96 * se)],
                subsample_mean=float(x.mean()), subsample_pct95=[float(v) for v in np.percentile(x, [2.5, 97.5])], n_subsamples=int(x.size))


out = {"bin15": {}, "bin60": {}, "within_city": {}, "method": __doc__}
for tag, folder, table in (("bin15", "results/rb/boot15", FULL), ("bin60", "results/rb/boot", FULL60)):
    for u, p in table.items():
        full, reps, fac, n, m = load(folder, u, p)
        if not reps:
            continue
        out[tag][u] = dict(n_days=n, m_days=m, factor=fac, **{k: jk(full[k], [r[k] for r in reps.values()], fac)
                                                                  for k in ("median_h", "circ_per_trip", "tide_per_trip")})
for city, (b, r) in PAIRS.items():
    fb, Rb, fac, n, m = load("results/rb/boot", b, FULL60[b]); fr, Rr, _, _, _ = load("results/rb/boot", r, FULL60[r])
    s = sorted(set(Rb) & set(Rr))
    out["within_city"][city] = dict(
        diff_median_h=jk(fb["median_h"] - fr["median_h"], [Rb[k]["median_h"] - Rr[k]["median_h"] for k in s], fac),
        ratio_circ_per_trip=jk(fb["circ_per_trip"] / fr["circ_per_trip"], [Rb[k]["circ_per_trip"] / Rr[k]["circ_per_trip"] for k in s], fac),
        ratio_tide_per_trip_rail_over_bike=jk(fr["tide_per_trip"] / fb["tide_per_trip"], [Rr[k]["tide_per_trip"] / Rb[k]["tide_per_trip"] for k in s], fac),
        hours_bike_gt_rail=int((fb["h"] > fr["h"]).sum()), n_hours=int(fb["h"].size))
    ph = [jk(fb["h"][i] - fr["h"][i], [Rb[k]["h"][i] - Rr[k]["h"][i] for k in s], fac) for i in range(fb["h"].size)]
    out["within_city"][city].update(per_hour=[dict(hour=6 + i, **x) for i, x in enumerate(ph)],
                                    hours_ci_above_zero=sum(x["ci95"][0] > 0 for x in ph), hours_ci_below_zero=sum(x["ci95"][1] < 0 for x in ph))
json.dump(out, open("results/rb/uncertainty_summary.json", "w"), indent=1)
for tag in ("bin15", "bin60"):
    for u, v in out[tag].items():
        x = v["median_h"]; print(tag, u, f"{x['estimate']:.3f} ± {x['se']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}] (subsample pct {np.round(x['subsample_pct95'], 3)}, N={x['n_subsamples']})")
for c, v in out["within_city"].items():
    print(c, {k: (round(x["estimate"], 3), np.round(x["ci95"], 3).tolist()) for k, x in v.items() if isinstance(x, dict)}, v["hours_bike_gt_rail"], "/", v["n_hours"],
          "| hours CI>0:", v["hours_ci_above_zero"], "CI<0:", v["hours_ci_below_zero"])
