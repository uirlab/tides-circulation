"""Extended Data figures ED1-ED7, ED9, ED10 (ED8 = src/ed_oneway_distance.py). Same palette as the main figures.
All inputs are existing results; nothing here is a new test. Also writes results/pertrip_deviance.json (ED6 table).
Run from the project root: python3 src/ed_figures.py  ->  results/figures/ed<k>_*.png|pdf
"""
import glob
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "results/figures"
SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = lambda p: json.load(open(p))


def letter(ax, s, x=-0.15, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig); print("written", name)


# ---------------- ED1 synthetic calibration ----------------
c0, cs = J("results/0c_irrev_synthetic.json"), J("results/0cb_strata_synthetic.json")
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.45))
ax = axs[0]; letter(ax, "a")
sc = sorted(c0["S2"], key=float); x = [float(s) for s in sc]
ax.plot(x, [c0["S2"][s]["median_h_over_reps"] for s in sc], color=TERRA, marker="o", ms=3.5, lw=1.4, label="unstratified")
ss = sorted(cs["S2"], key=float)
ax.plot([float(s) for s in ss], [cs["S2"][s]["median_h"] for s in ss], color=SLATE, marker="s", ms=3.5, lw=1.4, label="stratified")
ax.axhline(0.10, color=INK2, lw=0.6, ls=(0, (3, 2))); ax.set_xscale("log", base=2); ax.set_xticks(x); ax.set_xticklabels(sc)
ax.set_xlabel("planted loop-field scale"); ax.set_ylabel("recovered h′ (median of 20)"); ax.legend(frameon=False, fontsize=6)
ax = axs[1]; letter(ax, "b")
go = lambda v: sum(n for k, n in v["verdicts"].items() if k in ("GO", "PASS")) / 20
ax.bar(np.arange(len(sc)) - 0.18, [go(c0["S2"][s]) for s in sc], 0.36, color=TERRA, lw=0, label="unstratified")
pos = [sc.index(s) for s in ss]
ax.bar(np.array(pos) + 0.18, [go(cs["S2"][s]) for s in ss], 0.36, color=SLATE, lw=0, label="stratified")
ax.set_xticks(range(len(sc))); ax.set_xticklabels(sc); ax.set_ylim(0, 1.05); ax.set_xlabel("planted loop-field scale")
ax.set_ylabel("share of 20 replicates passing"); ax.axhline(0.8, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax = axs[2]; letter(ax, "c")
vals = [("tide\nonly", c0["S1"]["median_h_over_reps"], TERRA), ("mixed\nriders,\nunstrat.", cs["S1"]["heterogeneity_artifact_unstratified_median_h"], MUTE),
        ("mixed\nriders,\nstratified", cs["S1"]["median_h"], SLATE), ("fast\ndrift", c0["S3"]["median_h_over_reps"], INK2)]
ax.bar(range(4), [v[1] for v in vals], color=[v[2] for v in vals], lw=0, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v[1] + (0.008 if v[1] >= 0 else -0.02), f"{v[1]:+.3f}", ha="center", fontsize=5.8, color=INK, va="bottom" if v[1] >= 0 else "top")
ax.axhline(0, color=MUTE, lw=0.6); ax.axhline(0.10, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(range(4)); ax.set_xticklabels([v[0] for v in vals], fontsize=5.6); ax.set_ylabel("median h′ (no true loops)"); ax.set_ylim(-0.21, 0.14)
save(fig, "ed1_calibration")

# ---------------- ED2 all monthly units ----------------
files = [f for f in glob.glob("results/3_units/u_*_real.json") if not re.search(r"_A\d|g30|g120", f)]
fig, axs = plt.subplots(2, 1, figsize=(7.4, 3.6), sharex=True, gridspec_kw=dict(hspace=0.15))
months = [f"{y}-{m:02d}" for y in (2024, 2025, 2026) for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-08"]
col = {"nyc": "#a8552f", "chi": TERRA, "bos": "#e0a888"}; nm = {"nyc": "New York", "chi": "Chicago", "bos": "Boston"}
n_single = n_pooled = 0
OTHER = {("nyc", "2025-05"), ("bos", "2026-05")}  # same pipeline, run for the sensitivity analysis / the replication
files.append("BASE")
for f in files:
    if f == "BASE":  # NYC May 2026, the main stratified baseline (results/0cb_strata_real.json)
        d = J("results/0cb_strata_real.json")["decision"]; xs = [months.index("2026-05")]
        for ax, v in ((axs[0], d["median_h"]), (axs[1], d["rho"])):
            ax.plot(xs[0], v, "D", color=col["nyc"], ms=4, mec=INK, mew=0.6)
        continue
    m = re.search(r"u_(nyc|chi|bos)_([\d\-+]+)_real", f)
    if m is None:  # e.g. u_chi_jo25 (Jul-Oct pooled unit for H-OW), not a monthly unit
        continue
    c, span = m.group(1), m.group(2).split("+")
    d = J(f)["decision"]; xs = [months.index(s) for s in span if s in months]
    if not xs:
        continue
    pooled = len(span) > 1; n_pooled += pooled; n_single += not pooled
    for ax, v in ((axs[0], d["median_h"]), (axs[1], d["rho"])):
        if pooled:
            ax.plot([min(xs) - 0.3, max(xs) + 0.3], [v, v], color=col[c], lw=2.2, alpha=0.55, solid_capstyle="round")
        else:
            other = (c, span[0]) in OTHER
            ax.plot(xs[0], v, "D" if other else "o", color=col[c], ms=4 if other else 3.5, mec=INK if other else SURF, mew=0.6 if other else 0.4)
for c in col:
    axs[0].plot([], [], "o", color=col[c], label=nm[c])
axs[0].legend(frameon=False, fontsize=6, ncol=3, loc="upper left"); letter(axs[0], "a", x=-0.07); letter(axs[1], "b", x=-0.07)
axs[0].set_ylabel("h′ (15-min bins)"); axs[1].set_ylabel("split-half ρ")
ticks = [j for j, mm in enumerate(months) if mm.endswith("-01") or mm.endswith("-07")]
axs[1].set_xticks(ticks); axs[1].set_xticklabels([months[j][:4] if months[j].endswith("-01") else "Jul" for j in ticks])
axs[0].set_ylim(0, 0.45); axs[1].set_ylim(0, 0.8)
axs[0].text(0.99, 0.95, "dots: single months; bars: pooled months; diamonds: units run for other analyses; all GO", transform=axs[0].transAxes,
            ha="right", va="top", fontsize=5.8, color=INK2)
save(fig, "ed2_units")

# ---------------- ED3 h' vs bin width ----------------
pairs = [("New York", "results/0cb_strata_real.json", "results/0cb_strata_b60_real.json"),
         ("Chicago", "results/1_replication_divvy_mj_real.json", "results/1_replication_divvy_mj_b60_real.json"),
         ("Boston", "results/1_replication_bluebikes_real.json", "results/1_replication_bluebikes_b60_real.json"),
         ("SF Bay", "results/1_replication_g_baw_real.json", "results/1_replication_g_baw_b60_real.json"),
         ("London", "results/1_replication_g_lon_real.json", "results/1_replication_g_lon_b60_real.json"),
         ("Seoul", "results/1_replication_g_seo_real.json", "results/1_replication_g_seo_b60_real.json"),
         ("Mexico City", "results/1_replication_g_mex_real.json", "results/1_replication_g_mex_b60_real.json"),
         ("Buenos Aires", "results/1_replication_g_bue_real.json", "results/1_replication_g_bue_b60_real.json")]
fig, ax = plt.subplots(figsize=(3.6, 2.8)); letter(ax, "", x=0)
lab = []
for n_, a, b in pairs:
    h15, h60 = J(a)["decision"]["median_h"], J(b)["decision"]["median_h"]
    ax.plot([0, 1], [h15, h60], color=TERRA, lw=1.0, alpha=0.8); ax.scatter([0, 1], [h15, h60], s=10, color=TERRA, zorder=3, lw=0); lab.append((h60, n_))
for n_, f, cc in (("YouBike (Taipei)", "results/1_replication_g_ybk_real.json", TERRA), ("Taipei metro", "results/1_replication_g_tpex_real.json", SAGE),
                  ("BART", "results/1_replication_g_bar_real.json", SAGE)):
    v = J(f)["decision"]["median_h"]; ax.scatter([1], [v], s=24, fc=SURF if cc == SAGE else cc, ec=cc, lw=1.4, zorder=4); lab.append((v, n_))
lab.sort(); ys = []
for v, n_ in lab:
    y = max(v, ys[-1] + 0.028) if ys else v; ys.append(y); ax.text(1.06, y, n_, va="center", fontsize=5.8, color=SAGE if n_ in ("Taipei metro", "BART") else INK)
ax.set_xlim(-0.15, 1.7); ax.set_ylim(0, 0.62); ax.set_xticks([0, 1]); ax.set_xticklabels(["15 min", "60 min"]); ax.set_ylabel("h′")
save(fig, "ed3_binwidth")

# ---------------- ED4 Taipei versions ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.35, width_ratios=[0.8, 1.2]))
ax = axs[0]; letter(ax, "a")
vers = [("raw codes\n(pre-specified)", "g_tpe"), ("codes merged", "g_tpem"), ("interchanges\nexcluded", "g_tpex")]
hv = [J(f"results/1_replication_{k}_real.json")["decision"]["median_h"] for _, k in vers]
ax.bar(range(3), hv, color=[MUTE, MUTE, SAGE], width=0.6, lw=0)
for i, v in enumerate(hv):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=6.2)
ax.axhline(0.10, color=INK2, lw=0.6, ls=(0, (3, 2))); ax.set_xticks(range(3)); ax.set_xticklabels([v[0] for v in vers], fontsize=6)
ax.set_ylabel("Taipei MRT h′ (60-min bins)"); ax.set_ylim(0, 0.13)
ax = axs[1]; letter(ax, "b", x=-0.1)
st = np.load("results/figdata_stations_g_tpe.npz"); u = np.unique([re.sub(r"^[A-Za-z]+", "", n) for n in st["station_name"]])
for k, cc, mk, lab_ in (("g_tpem", MUTE, "o", "codes merged"), ("g_tpex", SAGE, "s", "interchanges excluded")):
    c = np.load(f"results/figdata_circ_{k}.npz"); z = c["eps"] / c["se"]; ex = z * z - 1; K = len(u)
    L = np.bincount(c["lo"], ex, K) + np.bincount(c["hi"], ex, K); V = np.bincount(c["lo"], c["t"], K) + np.bincount(c["hi"], c["t"], K)
    okk = V > 0; ax.scatter(V[okk] / V.sum(), L[okk] / L.sum(), s=10, color=cc, marker=mk, alpha=0.8, lw=0, label=lab_)
    if k == "g_tpem":
        for nm_ in ("景安", "板橋", "大坪林"):
            i = int(np.where(u == nm_)[0][0]); ax.annotate({"景安": "Jing'an", "板橋": "Banqiao", "大坪林": "Dapinglin"}[nm_],
                                                             (V[i] / V.sum(), L[i] / L.sum()), xytext=(6, 0), textcoords="offset points", fontsize=5.8, color=INK2)
m_ = 0.17; ax.plot([0, m_], [0, m_], color=INK2, lw=0.6, ls=(0, (3, 2))); ax.set_xlim(0, 0.07); ax.set_ylim(0, m_)
ax.set_xlabel("station share of trips"); ax.set_ylabel("station share of loop excess"); ax.legend(frameon=False, fontsize=6)
save(fig, "ed4_taipei")

# ---------------- ED5 transit dose-response ----------------
pts = []
for f in glob.glob("results/mode_E1r*.json") + ["results/mode.json"]:
    r = 400 if f.endswith("mode.json") else int(f.split("_E1r")[1].split(".")[0]); m = J(f)["M1_nyc_june"]
    pts.append((r, m["enrichment_T1"], m["frac_T1"], m["p"]))
pts.sort(); R_, E_, F_, P_ = map(np.array, zip(*pts))
fig, ax = plt.subplots(figsize=(3.6, 2.5))
ax.fill_between(R_, 1, E_, color=SAGE, alpha=0.22, lw=0); ax.plot(R_, E_, color=SAGE, lw=1.5)
ax.scatter(R_, E_, s=16, c=[SAGE if p < 0.01 else SURF for p in P_], edgecolors=SAGE, zorder=3)
for r, e, f_, p in pts:
    ax.text(r, e + 0.03, f"{f_:.0%}", ha="center", fontsize=5.6, color=INK2)
ax.axhline(1, color=INK2, lw=0.6, ls=(0, (3, 2))); ax.set_xlabel("radius to a shared rapid-transit route (m)"); ax.set_ylabel("loop-excess enrichment")
ax.text(0.98, 0.95, "filled: p < 0.01; labels: share of pairs within radius", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=INK2)
save(fig, "ed5_transit_radius")

# ---------------- ED6 per-trip deviance ----------------
V = J("results/figdata_volume.json")
units = [("YouBike", "Taipei", "bike", "results/1_replication_g_ybk_real.json", "g_ybk"), ("Taipei MRT", "Taipei", "rail", "results/1_replication_g_tpex_real.json", "g_tpex"),
         ("Bay Wheels", "SF Bay", "bike", "results/1_replication_g_baw_b60_real.json", "g_baw"), ("BART", "SF Bay", "rail", "results/1_replication_g_bar_real.json", "g_bar"),
         ("Citi Bike", "New York", "bike", "results/0cb_strata_b60_real.json", "citibike"), ("Ttareungyi", "Seoul", "bike", "results/1_replication_g_seo_b60_real.json", "g_seo")]
tab = []
for nm_, city, mode, f, k in units:
    pb = J(f)["per_bin"]; G_ = sum(b["g"] for b in pb); GS = sum(b["gs"] for b in pb); N0 = sum(b["null_mean"] for b in pb); n = sum(V[k])
    tab.append(dict(system=nm_, city=city, mode=mode, trips=n, tide_per_trip=(GS - G_) / n, circ_per_trip=(G_ - N0) / n, share=(G_ - N0) / (GS - N0)))
json.dump(dict(note="Exploratory: deviance per trip at 60-min bins, from per-bin G2 sums (tide = G2_S - G2_QS; circulation = G2_QS - null mean).",
               rows=tab), open("results/pertrip_deviance.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(4.4, 2.5))
x = np.arange(len(tab))
ax.bar(x - 0.19, [r["tide_per_trip"] for r in tab], 0.36, color=SLATE, lw=0, label="tide")
ax.bar(x + 0.19, [r["circ_per_trip"] for r in tab], 0.36, color=[TERRA if r["mode"] == "bike" else SAGE for r in tab], lw=0, label="circulation")
ax.set_xticks(x); ax.set_xticklabels([f"{r['system']}\n{r['city']}" for r in tab], fontsize=5.8); ax.set_ylabel("deviance per trip (60-min bins)")
ax.legend(frameon=False, fontsize=6)
save(fig, "ed6_pertrip")

# ---------------- ED7 rider type, both bike types ----------------
rows = []
for f in sorted(glob.glob("results/cas/u_nyc_*.json")):
    d = J(f)["per_stratum_15min"]; rows.append((f.split("u_nyc_")[1][:7], {k: d[k]["loop_fraction"] for k in d}, {k: d[k]["trips"] for k in d}))
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
TOP = 0.9
for ax, bt, lab_ in ((axs[0], "ebike", "a"), (axs[1], "classic", "b")):
    letter(ax, lab_, x=-0.1)
    for rt, cc in (("casual", TERRA), ("member", SLATE)):
        y = np.array([r[1][f"{bt}-{rt}"] for r in rows]); x = np.arange(len(rows)); yc = np.clip(y, -0.1, TOP)
        ax.plot(x, yc, color=cc, lw=1.5, label=rt)
        bad = (y > TOP) | (y < -0.1)
        ax.plot(x[bad], yc[bad], "^" if bt else "v", color=cc, ms=4, ls="none")
    ax.set_title({"ebike": "e-bike", "classic": "classic bike"}[bt], fontsize=7, color=INK, loc="left")
    ticks = [i for i, r in enumerate(rows) if r[0].endswith("-01")]; ax.set_xticks(ticks); ax.set_xticklabels([rows[i][0][:4] for i in ticks])
    ax.set_ylim(-0.12, TOP + 0.05)
axs[0].set_ylabel("circulation fraction"); axs[0].legend(frameon=False, fontsize=6, loc="lower right")
axs[1].text(0.98, 0.04, "triangles: values outside the axis (small winter samples)", transform=axs[1].transAxes, ha="right", fontsize=5.6, color=INK2)
save(fig, "ed7_ridertype")

# ---------------- ED9 what does not explain circulation ----------------
dv, dv30, hact, mode, ex2 = J("results/devES.json"), J("results/devES_ge30.json"), J("results/hact.json"), J("results/mode.json"), J("results/exp_loop_sources2.json")
items = [("one-way street direction\n(all pairs)", dv["real"]["ES_dev_june"], dv["real"]["ci95"]),
         ("one-way street direction\n(≥ 30 trips)", dv30["ES_dev_ge30"], None),
         ("land-use interactions", hact["primary"]["dES"], hact["primary"]["ci95"]),
         ("downhill × transit", mode["M2"]["dES"], mode["M2"]["dES_ci"])]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.75, width_ratios=[1.1, 1]))
ax = axs[0]; letter(ax, "a", x=-0.45)
for i, (lab_, v, ci) in enumerate(items):
    ax.barh(i, max(v, 0), color=SLATE if "one-way" in lab_ else MUTE, height=0.6, lw=0)
    if ci:
        ax.plot(ci, [i, i], color=INK, lw=0.9)
    ax.text(max(v, ci[1] if ci else v) + 0.002, i, f"{v:.1%}", va="center", fontsize=6)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=6); ax.invert_yaxis(); ax.set_xlim(-0.005, 0.08); ax.axvline(0, color=INK2, lw=0.6)
ax.set_xlabel("share of loop excess explained (out of sample)")
ax = axs[1]; letter(ax, "b", x=-0.45)
b = ex2["ols_std_beta"]; keys = [("habit_R", "habit"), ("log_t", "trip count"), ("absD", "street asymmetry |D|"), ("log_dist", "distance"),
                                ("river_cross", "crosses a river/bridge"), ("greenway_G", "greenway")]
for i, (k, lab_) in enumerate(keys):
    v, se = b[k]; ax.barh(i, v, color=TERRA if k == "habit_R" else MUTE, height=0.6, lw=0); ax.plot([v - 1.96 * se, v + 1.96 * se], [i, i], color=INK, lw=0.9)
ax.axvline(0, color=INK2, lw=0.6); ax.set_yticks(range(len(keys))); ax.set_yticklabels([x[1] for x in keys], fontsize=6); ax.invert_yaxis()
ax.set_xlabel("standardised association with loop excess\n(exploratory, NYC May 2026)")
save(fig, "ed9_not_geography")

# ---------------- ED10 rebalancing sensitivity + congestion-pricing null ----------------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.45))
ax = axs[0]; letter(ax, "a")
var = [("|U| ≥ 2", "results/4_rebalancing_D1u2.json"), ("|U| ≥ 3\n(baseline)", "results/4_rebalancing.json"), ("|U| ≥ 5", "results/4_rebalancing_D1u5.json"),
       ("gap 5 min", "results/4_rebalancing_D2gap5.json"), ("gap 15 min", "results/4_rebalancing_D2gap15.json")]
for i, (lab_, f) in enumerate(var):
    t = J(f)["fits"]["test"]
    for dx, k, cc in ((-0.17, "T", SLATE), (0.17, "Lam", TERRA)):
        ax.bar(i + dx, t["beta"][k], 0.32, color=cc, lw=0); ax.plot([i + dx] * 2, t["ci"][k], color=INK, lw=0.9)
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(len(var))); ax.set_xticklabels([v[0] for v in var], fontsize=5.8)
ax.set_ylabel("coefficient on operator moves\n(June 2025, held out)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=SLATE, label="tidal imbalance"), Patch(fc=TERRA, label="circulation intensity")], frameon=False, fontsize=6)
ax = axs[1]; letter(ax, "b")
cg = J("results/3_congestion_real.json"); keys = ["resid_all_early", "resid_all_late", "resid_cbd_early", "resid_cbd_late", "resid_out_early", "resid_out_late"]
for i, k in enumerate(keys):
    m = cg["main"][k]; ax.plot(m["ci"], [i, i], color=INK, lw=0.9); ax.plot(m["est"], i, "o", color=TERRA, ms=4)
    if k in cg.get("placebo", {}):
        pm = cg["placebo"][k]; ax.plot(pm["ci"], [i + 0.3] * 2, color=MUTE, lw=0.9); ax.plot(pm["est"], i + 0.3, "o", color=MUTE, ms=3)
ax.axvline(0, color=INK2, lw=0.6); ax.set_yticks(range(len(keys)))
ax.set_yticklabels([k.replace("resid_", "").replace("_", ", ").replace("all", "all zones").replace("cbd", "charge zone").replace("out", "outside") for k in keys], fontsize=6)
ax.invert_yaxis(); ax.set_xlabel("change in for-hire circulation (DiD)\norange: policy; grey: placebo year")
save(fig, "ed10_rebalancing_congestion")
