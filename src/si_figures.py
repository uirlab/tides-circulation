"""Supplementary figures that merge panels moved out of the main figures (2026-10-05 reorganisation):
  SFig 8  one-way streets: robustness tiles (old Fig 4c) + agreement vs distance (old ED8)
  SFig 9  robustness of the existence test (old Fig 2f)
  SFig 11 operations and policy: rebalancing station map (old Fig 4d) + threshold sensitivity + congestion pricing (old ED10)
Other SI figures are renamed copies of earlier outputs (src/sync_draft.sh). Same palette as the main figures.
Run from the project root: python3 src/si_figures.py -> results/figures/si_*.png
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.stats import binomtest

SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
TSEQ = LinearSegmentedColormap.from_list("terra_seq", ["#f6ece4", "#ecc9b3", "#dda283", "#c9744d", "#9c5232"])
SSEQ = LinearSegmentedColormap.from_list("slate_seq", ["#e4ebf2", "#9fb3c9", "#4d6e92", "#2c4a6b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.edgecolor": MUTE, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = lambda p: json.load(open(p))
OUT = "results/figures"


def letter(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def tiles(ax, var, key, lo, span, fmt):
    for j, (lab, f) in enumerate(var):
        d = J(f); v = d[key] if key in d else d["decision"]["median_h"]
        ax.add_patch(plt.Rectangle((j + 0.05, 0.05), 0.9, 0.9, fc=TSEQ((v - lo) / span), ec=SURF, lw=0.8))
        ax.text(j + 0.5, 0.5, fmt.format(v), ha="center", va="center", fontsize=6.4, color="white" if (v - lo) / span > 0.55 else INK, fontweight="bold")
        ax.text(j + 0.5, -0.1, lab, ha="center", va="top", fontsize=5.8, color=INK2)
    ax.set_xlim(0, len(var)); ax.set_ylim(-0.6, 1.45); ax.axis("off")


# ---------- SFig 8: one-way streets ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.3, width_ratios=[1.2, 1]))
ax = axs[0]; letter(ax, "a", x=0.0)
B = "results/1_oneway_citibike_jun"
var = [("baseline", B + ".json"), ("30 m", B + "_B1near30.json"), ("75 m", B + "_B1near75.json"), ("30°", B + "_B2ang30.json"),
       ("60°", B + "_B2ang60.json"), ("≥ 20", B + "_B3min20.json"), ("≥ 50", B + "_B3min50.json")]
tiles(ax, var, "f", 0.5, 0.45, "{:.0%}")
for x0, x1, lab in ((1, 3, "street distance"), (3, 5, "alignment"), (5, 7, "min. trips")):
    ax.plot([x0 + 0.1, x1 - 0.1], [1.12, 1.12], color=MUTE, lw=0.6); ax.text((x0 + x1) / 2, 1.2, lab, ha="center", fontsize=5.8, color=INK2)
ax = axs[1]; letter(ax, "b")
d = np.load("results/figdata_oneway_citibike_jun.npz", allow_pickle=True)
one = d["kind"] == "one"; agree = (d["legal_ab"] == d["fav_ab"])[one]; dist = d["dist_m"][one]
edges = np.quantile(dist, np.linspace(0, 1, 7)); x = np.arange(6)
for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
    m = (dist >= a) & (dist <= b); k, n = int(agree[m].sum()), int(m.sum()); ci = binomtest(k, n).proportion_ci(0.95)
    ax.bar(i, k / n, color=TERRA, width=0.7, lw=0); ax.plot([i, i], [ci.low, ci.high], color=INK, lw=0.9)
    ax.text(i, 0.04, f"{n}", ha="center", fontsize=5.4, color="white")
ax.axhline(0.5, color=INK2, lw=0.6, ls=(0, (3, 2))); ax.set_ylim(0, 1); ax.set_xticks(x)
ax.set_xticklabels([f"{a / 1000:.1f}–{b / 1000:.1f}" for a, b in zip(edges[:-1], edges[1:])], fontsize=5.8)
ax.set_xlabel("distance between stations (km)"); ax.set_ylabel("share favouring\nthe legal direction")
fig.savefig(f"{OUT}/si_oneway.png", dpi=240, bbox_inches="tight"); plt.close(fig); print("written si_oneway")

# ---------- SFig 9: robustness of the existence test ----------
fig, ax = plt.subplots(figsize=(6.4, 1.6))
U = "results/3_units/u_nyc_2025-05"
var = [("baseline", U + "_real.json"), ("30-min bins", U + "_A1bin30_real.json"), ("λ 0.03", U + "_A2lam003_real.json"),
       ("λ 0.3", U + "_A2lam03_real.json"), ("λ_δ 0.3", U + "_A3lamd03_real.json"), ("λ_δ 3", U + "_A3lamd3_real.json"),
       ("group 30 s", U + "g30_real.json"), ("group 120 s", U + "g120_real.json"), ("07–21 h", U + "_A5win0721_real.json")]
tiles(ax, var, "median_h", 0.1, 0.22, "{:.2f}")
fig.savefig(f"{OUT}/si_robustness.png", dpi=240, bbox_inches="tight"); plt.close(fig); print("written si_robustness")

# ---------- SFig 11: operations and policy ----------
fig = plt.figure(figsize=(7.4, 2.7)); g = fig.add_gridspec(1, 3, width_ratios=[0.8, 1.1, 1.1], wspace=0.45)
rows = list(csv.DictReader(open("results/figdata_rebal_june2025.csv")))
T = np.array([float(r["T"]) for r in rows]); RB = np.array([float(r["RB"]) for r in rows])
s = stations("citibike")  # cached in results/ (src/figdata_io.py)
nrm = lambda x: re.sub(r"\s+", " ", str(x).lower()).strip()
cmap_ = {nrm(n): (a, b) for n, a, b in zip(s["station_name"], s["lat"].astype(float), s["lon"].astype(float))}
ll = np.array([cmap_.get(nrm(r["st"]), (np.nan, np.nan)) for r in rows], float); ok = np.isfinite(ll[:, 0]); o = np.argsort(RB[ok])
ax = fig.add_subplot(g[0]); letter(ax, "a", x=0.0, y=1.0)
sc = ax.scatter(ll[ok, 1][o], ll[ok, 0][o], c=T[ok][o], s=2 + 22 * np.sqrt(RB[ok][o] / RB.max()), cmap=SSEQ, lw=0, alpha=0.9,
                vmin=np.percentile(T, 5), vmax=np.percentile(T, 95))
ax.set_xlim(-74.03, -73.86); ax.set_ylim(40.63, 40.86); ax.set_aspect(1 / np.cos(np.radians(40.75))); ax.axis("off")
cax = ax.inset_axes([0.6, 0.06, 0.36, 0.035]); cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.outline.set_visible(False); cb.set_ticks([])
cax.set_title("tidal imbalance", fontsize=5.6, color=INK2, pad=2)
ax.text(0.98, 0.98, "size: operator moves", transform=ax.transAxes, fontsize=5.6, color=INK2, va="top", ha="right")
ax = fig.add_subplot(g[1]); letter(ax, "b")
var = [("|U|≥2", "results/4_rebalancing_D1u2.json"), ("|U|≥3\nbase", "results/4_rebalancing.json"), ("|U|≥5", "results/4_rebalancing_D1u5.json"),
       ("gap\n5 min", "results/4_rebalancing_D2gap5.json"), ("gap\n15 min", "results/4_rebalancing_D2gap15.json")]
for i, (lab, f) in enumerate(var):
    t = J(f)["fits"]["test"]
    for dx, k, cc in ((-0.17, "T", SLATE), (0.17, "Lam", TERRA)):
        ax.bar(i + dx, t["beta"][k], 0.32, color=cc, lw=0); ax.plot([i + dx] * 2, t["ci"][k], color=INK, lw=0.9)
ax.axhline(0, color=INK2, lw=0.6); ax.set_xticks(range(len(var))); ax.set_xticklabels([v[0] for v in var], fontsize=5.6)
ax.set_ylabel("coefficient on operator moves\n(June 2025, held out)", fontsize=6.2)
ax.legend(handles=[Patch(fc=SLATE, label="tidal imbalance"), Patch(fc=TERRA, label="circulation intensity")], frameon=False, fontsize=5.8, loc="upper left")
ax = fig.add_subplot(g[2]); letter(ax, "c")
cg = J("results/3_congestion_real.json")
keys = ["resid_all_early", "resid_all_late", "resid_cbd_early", "resid_cbd_late", "resid_out_early", "resid_out_late"]
for i, k in enumerate(keys):
    m = cg["main"][k]; ax.plot(m["ci"], [i, i], color=INK, lw=0.9); ax.plot(m["est"], i, "o", color=TERRA, ms=4)
    if k in cg.get("placebo", {}):
        pm = cg["placebo"][k]; ax.plot(pm["ci"], [i + 0.3] * 2, color=MUTE, lw=0.9); ax.plot(pm["est"], i + 0.3, "o", color=MUTE, ms=3)
ax.axvline(0, color=INK2, lw=0.6); ax.set_yticks(range(len(keys)))
ax.set_yticklabels([k.replace("resid_", "").replace("_", ", ").replace("all", "all zones").replace("cbd", "charge zone").replace("out", "outside") for k in keys], fontsize=5.8)
ax.invert_yaxis(); ax.set_xlabel("change in for-hire circulation\n(policy: orange; placebo year: grey)", fontsize=6.2)
fig.savefig(f"{OUT}/si_operations.png", dpi=240, bbox_inches="tight"); plt.close(fig); print("written si_operations")
