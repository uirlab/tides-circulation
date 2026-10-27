"""Fig 4 v4: circulation for operations and planning. Same palette and conventions as src/fig1_v3.py.
Panels: a-b rebalancing against tidal imbalance and circulation intensity (New York, June 2025), c monthly enrichment of
circulation on Taipei's transit-linked pairs around the 28 Feb 2024 fare change, d the ratio of changes for enrichment and
for the ride share of those pairs, with the pre-specified margin. One-way streets: src/sfig13_oneway.py.
Inputs: results/figdata_rebal_june2025.csv, results/4_rebalancing.json, results/fare2024.json (src/fare.py combine).
Run from the repository root: python3 src/fig4_v4.py  ->  results/figures/fig4_v4.pdf|png
"""
import csv
import json
import re

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figdata_io import stations  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

OUT = "results/figures"
SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
TSEQ = LinearSegmentedColormap.from_list("terra_seq", ["#f6ece4", "#ecc9b3", "#dda283", "#c9744d", "#9c5232"])
SSEQ = LinearSegmentedColormap.from_list("slate_seq", ["#e4ebf2", "#9fb3c9", "#4d6e92", "#2c4a6b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = lambda p: json.load(open(p))
KX = np.cos(np.radians(40.75))


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="right" if x < 0 else "left")


fig = plt.figure(figsize=(7.4, 5.4))
G = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.42)
g2 = G[0].subgridspec(1, 2, wspace=0.28)
g3 = G[1].subgridspec(1, 2, width_ratios=[1.35, 1], wspace=0.42)

# rebalancing data
rows = list(csv.DictReader(open("results/figdata_rebal_june2025.csv")))
V = np.array([float(r["V"]) for r in rows]); T = np.array([float(r["T"]) for r in rows]); L_ = np.array([float(r["Lam"]) for r in rows])
RB = np.array([float(r["RB"]) for r in rows]); y = np.log1p(RB)
s = stations("citibike")  # cached in results/ (src/figdata_io.py)
nrm = lambda x: re.sub(r"\s+", " ", str(x).lower()).strip()
cmap_ = {nrm(n): (a, b) for n, a, b in zip(s["station_name"], s["lat"].astype(float), s["lon"].astype(float))}
ll = np.array([cmap_.get(nrm(r["st"]), (np.nan, np.nan)) for r in rows], float)

# (c, d) component-plus-residual from the full model
Zs = lambda a: (a - a.mean()) / a.std()
X = np.c_[np.ones(len(y)), Zs(V), Zs(T), Zs(L_)]; b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b
rb = J("results/4_rebalancing.json")
for k_, (j, col, nm, lab) in enumerate(((2, SLATE, "tidal imbalance", "a"), (3, TERRA, "circulation intensity", "b"))):
    ax = fig.add_subplot(g2[k_]); letter(ax, lab, x=-0.18)
    # express the predictor in May-2025 standard deviations, as in the fitted model (src/rebalancing.py standardises June by
    # May's mean and SD), so that the plotted slope equals the reported June coefficient; the fit itself is unchanged
    bj = rb["fits"]["test"]["beta"][{2: "T", 3: "Lam"}[j]]; xs = X[:, j] * b[j] / bj; pr = e + bj * xs; keep = np.abs(xs) < 3.5
    ax.hexbin(xs[keep], pr[keep], gridsize=20, cmap=LinearSegmentedColormap.from_list("m", [SURF, col]), mincnt=1, lw=0.2, edgecolors=SURF)
    xx = np.linspace(-3, 3, 10); ax.plot(xx, bj * xx, color=INK, lw=1.4)
    ax.set_xlabel(f"{nm} (May s.d.)"); ax.set_ylim(-1.3, 1.5); ax.set_box_aspect(0.78); ax.set_anchor("C"); clean(ax)
    if k_ == 0:
        ax.set_ylabel("operator moves\n(log, partial residual)")
        ax.text(0.04, 0.96, "β = +0.17 [0.13, 0.22]", transform=ax.transAxes, fontsize=6.2, va="top", color=INK)
    else:
        ax.set_yticklabels([])
        ax.text(0.04, 0.96, "β = −0.02 [−0.04, 0.01]\nΔR² = −0.0002", transform=ax.transAxes, fontsize=6.2, va="top", color=INK)

# (c) event study: monthly enrichment on transit-linked pairs, Dec-May, fare-change year against the year before
fr = J("results/fare2024.json"); t1, s2 = fr["T1_enrichment"], fr["S2_ride_share"]
mon = ["Dec", "Jan", "Feb", "Mar", "Apr", "May"]
U = {"Y0": ["g_f2212", "g_f2301", "g_f2302", "g_f2303", "g_f2304", "g_f2305"],
     "Y1": ["g_f2312", "g_f2401", "g_f2402", "g_f2403", "g_f2404", "g_f2405"]}
ax = fig.add_subplot(g3[0]); letter(ax, "c", x=-0.14)
ax.axvspan(2.5, 5.4, color=PALE, lw=0, zorder=0)
ax.axvline(2.5, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(2.58, 1.795, "first 30 min free\nfrom 28 Feb 2024", fontsize=6, color=INK, va="top")
for k, (col, lab, dx) in {"Y0": (MUTE, "Dec 2022–May 2023", -0.08), "Y1": (TERRA, "Dec 2023–May 2024", 0.08)}.items():
    lm = np.array([t1["monthly"][u] for u in U[k]]); se = np.array([t1["monthly_se"][u] for u in U[k]]); x = np.arange(6) + dx
    ax.errorbar(x, np.exp(lm), yerr=[np.exp(lm) - np.exp(lm - 1.96 * se), np.exp(lm + 1.96 * se) - np.exp(lm)], fmt="o", ms=3.4,
                color=col, ecolor=col, elinewidth=0.8, capsize=0, zorder=3)
    for a, b_ in ((0, 3), (3, 6)):
        ax.plot(x[a:b_], [np.exp(lm[a:b_].mean())] * (b_ - a), color=col, lw=1.4, alpha=0.8, zorder=2)
    ax.text(0.03, 0.17 if k == "Y1" else 0.08, lab + (" (fare change)" if k == "Y1" else " (no change)"), transform=ax.transAxes,
            fontsize=6, color=TERRA if k == "Y1" else INK2, fontweight="bold")
ax.set_xticks(range(6)); ax.set_xticklabels(mon); ax.set_xlim(-0.4, 5.4); ax.set_ylim(1.1, 1.8)
ax.set_ylabel("circulation enrichment\non transit-linked pairs"); clean(ax)

# (d) ratio of changes (post/pre in the fare-change year over post/pre in the year before)
ax = fig.add_subplot(g3[1]); letter(ax, "d", x=-0.5)
ax.axvspan(0.85, 1.18, color=PALE, lw=0, zorder=0); ax.axvline(1, color=INK, lw=0.6, ls=(0, (3, 2)))
for y_, (r, nm, col) in enumerate(((t1, "circulation\nenrichment", TERRA), (s2, "share of rides", INK2))):
    lo, hi = r["ratio_ci95"]
    ax.errorbar(r["ratio"], y_, xerr=[[r["ratio"] - lo], [hi - r["ratio"]]], fmt="o", ms=4.2, color=col, ecolor=col, elinewidth=1.1, capsize=2)
    ax.text(r["ratio"], y_ - 0.22, f"{r['ratio']:.2f} [{lo:.2f}, {hi:.2f}]", ha="center", va="bottom", fontsize=6, color=col)
ax.set_yticks([0, 1]); ax.set_yticklabels(["circulation\nenrichment", "share of rides"]); ax.set_ylim(1.6, -0.6)
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.set_xlim(0.7, 1.25); ax.set_xticks([0.8, 1.0, 1.2]); ax.set_xlabel("ratio of changes, 2024 vs 2023")
ax.text(1.015, 1.5, "±15% margin", fontsize=5.8, color=INK2, va="center")
clean(ax)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig4_v4.{ext}", dpi=240, bbox_inches="tight")
print("written fig4_v4")
