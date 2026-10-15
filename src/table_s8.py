"""Supplementary Table S8: every main-text estimate with its uncertainty. Reads results/rb/uncertainty_summary.json
 and the local result files; writes results/tables/table_s8.md. Run from the project root: python3 src/table_s8.py"""
import json
import os
import re

import numpy as np
from scipy import stats

J = lambda p: json.load(open(p))
U = J("results/rb/uncertainty_summary.json")


def cp(k, n):
    return (stats.beta.ppf(0.025, k, n - k + 1) if k else 0.0), (stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fp(p):
    """P to two significant figures, with a power of ten below 0.001."""
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def jk(x, d=2):
    return f"{x['estimate']:.{d}f} ({x['ci95'][0]:.{d}f}–{x['ci95'][1]:.{d}f}); SE {x['se']:.{max(d, 3)}f}"


rows = []
add = lambda *r: rows.append(r)
# Results 1
z = np.arctanh(-0.855); s = 1 / np.sqrt(400 - 3)
add("1", "Morning vs evening net inflow, 400 New York stations", f"r = −0.855 (95% CI {np.tanh(z - 1.96 * s):.3f} to {np.tanh(z + 1.96 * s):.3f})", "Fisher z")
tri = np.load("results/figdata_motiv_triangles.npz")
for k, lab in (("zr", "observed"), ("zn", "tide-only model (5 draws)")):
    x = tri[k]; c = int((np.abs(x) > 3).sum()); lo, hi = cp(c, x.size)
    add("1", f"Triangle-hours with z beyond ±3, {lab}", f"{c} of {x.size:,} = {pct(c / x.size, 2)} ({pct(lo, 2)}–{pct(hi, 2)})", "Clopper–Pearson")
for lab, k, n in (("No-loop check: false GO", 0, 20), ("Power check: GO at the observed strength", 20, 20), ("Power near threshold (estimated h′ 0.10)", 4, 20)):
    lo, hi = cp(k, n); add("1", f"Calibration, New York: {lab}", f"{k}/{n} ({pct(lo, 0)}–{pct(hi, 0)})", "Clopper–Pearson")
d = J("results/0cb_strata_real.json")["decision"]
add("1", "New York split-half correlation", f"ρ = {d['rho']:.2f}; permutation P < 0.001 (1,000 permutations; floor 0.001)", "permutation")
cg = J("results/coarsegrain.json")
add("1", "New York circulation share after subtracting the tide-only artefact, 15 min → whole day",
    f"{pct(cg['15min']['frac_corrected'], 0)} → {pct(cg['16h']['frac_corrected'], 0)}; simulation SD of the artefact ≤ {max(v['loop_sim_sd'] for v in cg.values()):.0f} deviance units (< 0.3% of the circulation component)", "3 simulations")
# Results 2: h' per system
NAME = {"citibike": "New York", "divvy_mj": "Chicago", "bluebikes": "Boston", "g_lon": "London", "g_seo": "Seoul", "g_mex": "Mexico City",
        "g_bue": "Buenos Aires", "g_baw": "SF Bay Area (Bay Wheels)", "tlc_2505": "NYC taxi and ride-hail (zones)", "g_ybk": "Taipei (YouBike)", "g_tpex": "Taipei Metro", "g_bar": "BART"}
for u, v in U["bin15"].items():
    add("3" if u == "tlc_2505" else "1–2", f"Median h′, 15-min bins: {NAME[u]}", jk(v["median_h"]) + f"; {v['median_h']['n_subsamples']} subsamples of {v['m_days']} of {v['n_days']} weekdays", "delete-d jackknife")
for u, v in U["bin60"].items():
    add("2", f"Median h′, 60-min bins: {NAME[u]}", jk(v["median_h"], 3) + f"; {v['median_h']['n_subsamples']} subsamples of {v['m_days']} of {v['n_days']} weekdays", "delete-d jackknife")
for c, v in U["within_city"].items():
    add("2", f"{c}: bike minus rail, median h′", jk(v["diff_median_h"]) + f"; hourly intervals above zero in {v['hours_ci_above_zero']}/16, below in {v['hours_ci_below_zero']}/16", "delete-d jackknife, paired days")
    k = v["hours_bike_gt_rail"]; add("2", f"{c}: hours with bike > rail", f"{k}/16; one-sided sign test P = {fp(stats.binom.sf(k - 1, 16, 0.5))} (hours treated as independent)", "sign test")
    add("2", f"{c}: rail/bike tide per trip", jk(v["ratio_tide_per_trip_rail_over_bike"]), "delete-d jackknife, paired days")
    add("2", f"{c}: bike/rail circulation per trip", jk(v["ratio_circ_per_trip"], 1), "delete-d jackknife, paired days")
# Results 3
for f, nm in (("hab_citibike_jun", "New York"), ("hab_u_chi_jo25", "Chicago"), ("hab_u_bos_jo25", "Boston")):
    r = J(f"results/{f}.json")
    add("3", f"Habit, {nm} (n = {r['n_pairs']:,} pairs)", f"β = {r['beta_R']:.2f}; top-decile share {pct(r['excess_share_top_decile'], 0)}, enrichment {r['enrichment_top_decile']:.2f}; P < 0.001", "permutation (2,000; floor 0.0005)")
m = J("results/mode.json")
for k, nm in (("M1_nyc_june", "New York"), ("M1_chi", "Chicago"), ("M1_bos", "Boston")):
    r = m[k]; add("3", f"Transit-connected pairs, {nm}", f"β = {r['beta_T']:.3f}; enrichment {r['enrichment_T1']:.2f}; P < 0.001", "permutation (2,000)")
x = J("results/xfer_g_xj6.json")
add("3", "Taipei transfer share, May → June 2023 (T1)", f"β = {x['beta_tau']:.3f}; enrichment {x['enrichment_top_decile']:.2f}; P < 0.001", "permutation (2,000)")
add("3", "Taipei, transfer rides removed (T2)", "enrichment 1.43; P = 0.0035", "permutation (2,000)")
t3 = x["T3"]; k = round(t3["share_agree"] * t3["n"]); lo, hi = cp(k, t3["n"])
add("3", "Taipei, direction agreement (T3)", f"{k} of {t3['n']:,} = {pct(k / t3['n'])} ({pct(lo)}–{pct(hi)}); P = {fp(t3['p'])}", "Clopper–Pearson; binomial")
add("3", "Casual > member circulation, e-bikes, New York", f"31 of 31 months; two-sided sign test P = {fp(2 * 0.5 ** 31)}", "sign test")
h = J("results/hact.json")["primary"]
add("3", "Land-use interactions, share of loop excess explained out of sample", f"{pct(h['dES'], 2)} ({pct(h['ci95'][0], 2)}–{pct(h['ci95'][1], 2)})", "as in section 6.1")
# Results 4
for f, nm in (("1_oneway_citibike_jun", "New York"), ("1_oneway_u_chi_jo25", "Chicago"), ("1_oneway_u_bos_jo25", "Boston")):
    r = J(f"results/{f}.json"); tw = r["two_way_control"]; k2 = round(tw["f_fixed_orientation"] * tw["n"]); lo2, hi2 = cp(k2, tw["n"])
    add("4", f"One-way agreement, {nm}", f"{r['matches']} of {r['n_one_way_pairs']:,} = {pct(r['f'])} ({pct(r['ci95'][0])}–{pct(r['ci95'][1])}); one-sided P = {fp(r['p_one_sided'])}; two-way control {pct(tw['f_fixed_orientation'])} ({pct(lo2)}–{pct(hi2)}), n = {tw['n']}", "Clopper–Pearson; binomial (approximate)")
lo, hi = cp(472, 506); add("4", "Midtown one-way pairs with the traffic", f"472 of 506 = {pct(472 / 506)} ({pct(lo)}–{pct(hi)})", "Clopper–Pearson (descriptive)")
e = J("results/devES.json")["real"]
add("4", "Street direction, share of loop excess explained (all pairs)", f"{pct(e['ES_dev_june'])} ({pct(e['ci95'][0])}–{pct(e['ci95'][1])})", "day-block bootstrap (500)")
add("4", "Street direction, pairs with ≥ 30 trips", f"{pct(J('results/devES_ge30.json')['ES_dev_ge30'])} (interval not computed)", "—")
rb = J("results/4_rebalancing.json")["fits"]["test"]
for k in rb["beta"]:
    ci = rb["ci"].get(k) if isinstance(rb["ci"], dict) else None
    add("4", f"Rebalancing, June 2025: β, {dict(V='trip volume', T='tidal imbalance', Lam='circulation intensity')[k]} (per SD)", f"{rb['beta'][k]:+.3f}" + (f" ({ci[0]:+.3f} to {ci[1]:+.3f})" if ci else " (interval not stored)"), "least squares")
add("4", "Rebalancing: out-of-sample R² change from adding circulation", f"{J('results/4_rebalancing.json')['oos_r2_gain_lambda']:+.4f} (interval not computed)", "—")

for r in J("results/ampm/summary.json"):
    add("3", f"Morning–evening r*, {r['unit']}", f"{r['r_star']:.2f} ({r['ci95'][0]:.2f} to {r['ci95'][1]:.2f}); {r['verdict']}; partial given midday {r['partial_given_mid']:.2f} (exploratory)", "split-half disattenuated correlation; station bootstrap")
for u, lab in (("g_xj6", "May → June"), ("g_x04", "Mar → Apr"), ("g_x10", "Sep → Oct"), ("g_x12", "Nov → Dec")):
    r = J(f"results/ampm/xfer_{u}_real.json")
    add("3", f"Taipei transfer pairs minus other pairs, r* ({lab})", f"{r['delta']:+.2f} ({r['ci_delta'][0]:+.2f} to {r['ci_delta'][1]:+.2f}); transfer {r['r_transfer']:.2f}, other {r['r_other']:.2f}", "station bootstrap")
tab = ["| Sec. | Quantity | Estimate (95% CI) and test | Method |", "|---|---|---|---|"] + [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
P = "results/tables/table_s8.md"; os.makedirs("results/tables", exist_ok=True)
s = open(P).read() if os.path.exists(P) else "<!-- S8 table: generated by src/table_s8.py -->\n<!-- /S8 -->\n"
block = ("<!-- S8 table: generated by src/table_s8.py -->\n" + "\n".join(tab) + "\n<!-- /S8 -->").replace("-0.", "−0.").replace("+-", "−")
s = re.sub(r"<!-- S8 table: generated by src/table_s8.py -->.*?<!-- /S8 -->", lambda _: block, s, flags=re.S)
open(P, "w").write(s); print("\n".join(tab))
