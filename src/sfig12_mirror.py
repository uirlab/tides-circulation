"""Supplementary Fig. 12: morning-evening mirror test.
Reads results/ampm/*_real.json and results/ampm/summary.json; writes results/figures/SFig12_mirror.png.
Run from the project root: python3 src/sfig12_mirror.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.edgecolor": MUTE, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = lambda p: json.load(open(p))
UNITS = [("g_lon", "London"), ("g_baw", "Bay Wheels"), ("g_bue", "Buenos Aires"), ("divvy_mj", "Chicago"), ("bluebikes", "Boston"),
         ("g_seo", "Seoul"), ("g_ybk", "YouBike"), ("g_mex", "Mexico City"), ("citibike", "New York"), (None, None),
         ("g_bar", "BART"), ("g_tpex", "Taipei Metro")]
PART = {r["unit"]: r["partial_given_mid"] for r in J("results/ampm/summary.json")}
PNAME = {"g_lon": "London", "g_baw": "Bay Wheels", "g_bue": "Buenos Aires", "divvy_mj": "Chicago", "bluebikes": "Boston", "g_seo": "Seoul",
         "g_ybk": "YouBike (60 min)", "g_mex": "Mexico City", "citibike": "New York", "g_bar": "BART (60 min)", "g_tpex": "Taipei Metro (60 min)"}

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.45))
for y, (u, nm) in enumerate(UNITS):
    if u is None:
        continue
    r = J(f"results/ampm/{u}_real.json"); x = r["AM_PM"]; rail = u in ("g_bar", "g_tpex"); col = SAGE if rail else TERRA
    ax.errorbar(x["r_star"], y, xerr=[[x["r_star"] - x["ci95"][0]], [x["ci95"][1] - x["r_star"]]], fmt="o", color=col, ms=4,
                elinewidth=0.9, capsize=1.8, capthick=0.8)
    ax.plot(PART[PNAME[u]], y, marker="|", color=INK2, ms=7, mew=1.0)
ax.set_yticks([y for y, (u, _) in enumerate(UNITS) if u]); ax.set_yticklabels([nm for u, nm in UNITS if u])
ax.set_ylim(len(UNITS) - 0.4, -0.6); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
for v, ls in ((-0.5, (0, (3, 2))), (0, "-"), (0.5, (0, (3, 2)))):
    ax.axvline(v, color=MUTE if v else INK2, lw=0.6, ls=ls)
ax.set_xlim(-1.05, 0.6); ax.set_xlabel("morning–evening correlation r*")
ax.text(-1.0, -0.95, "mirrored", fontsize=6, color=INK2); ax.text(0.03, -0.95, "one-way", fontsize=6, color=INK2)
ax.text(0.98, 0.93, "dots: r* (95% CI)\nbars: midday\nremoved\n(exploratory)", transform=ax.transAxes, ha="right", va="top", fontsize=5.6, color=INK2)
ax.text(-0.02, 1.04, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, ha="right")

months = [("g_x04", "Mar → Apr"), ("g_xj6", "May → Jun"), ("g_x10", "Sep → Oct"), ("g_x12", "Nov → Dec")]
for i, (u, lab) in enumerate(months):
    r = J(f"results/ampm/xfer_{u}_real.json")
    for dx, k, c, col in ((-0.13, "r_transfer", "ci_transfer", TERRA), (0.13, "r_other", "ci_other", MUTE)):
        v = r[k]; lo, hi = r[c]
        bx.errorbar(i + dx, v, yerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=4, elinewidth=0.9, capsize=1.8, capthick=0.8)
bx.axhline(0, color=INK2, lw=0.6); bx.set_xticks(range(4)); bx.set_xticklabels([m for _, m in months], fontsize=6.2)
bx.set_ylim(-0.75, 0.15); bx.set_ylabel("r* (YouBike, 2023)")
bx.text(0.03, 0.11, "transfer pairs (top decile)", transform=bx.transAxes, color=TERRA, fontsize=6.2, va="bottom", fontweight="bold")
bx.text(0.03, 0.03, "other pairs", transform=bx.transAxes, color=INK2, fontsize=6.2, va="bottom", fontweight="bold")
bx.text(-0.02, 1.04, "b", transform=bx.transAxes, fontsize=9, fontweight="bold", color=INK, ha="right")
fig.savefig("results/figures/SFig12_mirror.png", dpi=240, bbox_inches="tight")
print("written SFig12_mirror.png")
