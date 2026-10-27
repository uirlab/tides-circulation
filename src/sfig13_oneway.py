"""Supplementary Fig. 13: one-way streets. a Midtown Manhattan close-up (same-street one-way pairs over the street grid),
b held-out legal-direction agreement. Input: results/figdata_oneway_citibike_jun.npz (src/oneway.py DUMP_PAIRS=1), results/1_oneway_*.json.
Run from the repository root: python3 src/sfig13_oneway.py  ->  results/figures/sfig13_oneway.pdf|png
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


fig = plt.figure(figsize=(7.4, 3.9))
G = fig.add_gridspec(1, 1)
g1 = G[0].subgridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)

G = fig.add_gridspec(2, 1, height_ratios=[1.15, 0.82], hspace=0.26)
g1 = G[0].subgridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)
g2 = G[1].subgridspec(1, 2, wspace=0.28)

# (a) Midtown close-up
d = np.load("results/figdata_oneway_citibike_jun.npz", allow_pickle=True)
BOX = (-74.006, -73.966, 40.738, 40.772)
ax = fig.add_subplot(g1[0]); letter(ax, "a", x=0.0, y=1.0)
pts, lens, ow = d["way_pts"], d["way_lens"], d["way_oneway"]
segs_two, segs_one = [], []
for g_, o_ in zip(np.split(pts, np.cumsum(lens)[:-1]), ow):
    if (g_[:, 1].max() < BOX[0]) or (g_[:, 1].min() > BOX[1]) or (g_[:, 0].max() < BOX[2]) or (g_[:, 0].min() > BOX[3]):
        continue
    (segs_one if o_ else segs_two).append(np.c_[g_[:, 1], g_[:, 0]])
ax.add_collection(LineCollection(segs_two, colors=GRID, linewidths=0.9, zorder=0))
ax.add_collection(LineCollection(segs_one, colors=MUTE, linewidths=0.6, zorder=0))
one = d["kind"] == "one"
inbox = one & (d["lon_a"] > BOX[0]) & (d["lon_a"] < BOX[1]) & (d["lat_a"] > BOX[2]) & (d["lat_a"] < BOX[3]) \
      & (d["lon_b"] > BOX[0]) & (d["lon_b"] < BOX[1]) & (d["lat_b"] > BOX[2]) & (d["lat_b"] < BOX[3]) & (d["dist_m"] <= 1500)
order = np.argsort(d["t"][inbox]); idx = np.flatnonzero(inbox)[order]
tmax = np.log(d["t"][inbox].max())
for i in idx:
    fa = d["fav_ab"][i]
    p0 = (d["lon_a"][i], d["lat_a"][i]) if fa else (d["lon_b"][i], d["lat_b"][i])
    p1 = (d["lon_b"][i], d["lat_b"][i]) if fa else (d["lon_a"][i], d["lat_a"][i])
    legal = d["legal_ab"][i] == fa
    w = 0.4 + 1.1 * (np.log(d["t"][i]) - np.log(30)) / (tmax - np.log(30))
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=f"-|>,head_length={2.2 + w:.1f},head_width={1.4 + 0.6 * w:.1f}", lw=w,
                                 color=TERRA if legal else INK2, alpha=0.7 if legal else 0.75, shrinkA=1, shrinkB=1, zorder=2 if legal else 3))
nl = int((d["legal_ab"][inbox] == d["fav_ab"][inbox]).sum()); ni = int(inbox.sum())
ax.set_xlim(BOX[0], BOX[1]); ax.set_ylim(BOX[2], BOX[3]); ax.set_aspect(1 / KX); ax.axis("off")
from matplotlib.patches import Rectangle  # opaque backing so arrows do not run over the scale bar and legend
ax.add_patch(Rectangle((0.0, 0.0), 0.30, 0.085, transform=ax.transAxes, fc=SURF, ec="none", alpha=0.92, zorder=9))
ax.plot([BOX[0] + 0.002, BOX[0] + 0.002 + 1 / 111 / KX], [BOX[2] + 0.0015] * 2, color=INK2, lw=1.0, zorder=10, solid_capstyle="butt")
ax.text(BOX[0] + 0.002, BOX[2] + 0.0025, "1 km", fontsize=5.6, color=INK2, zorder=10)
bb = dict(fc=SURF, ec="none", alpha=0.92, pad=1.6)
ax.text(0.98, 0.04, f"with the legal direction  {nl}", transform=ax.transAxes, ha="right", fontsize=6.2, color=TERRA, fontweight="bold", bbox=bb, zorder=10)
ax.text(0.98, 0.10, f"against it  {ni - nl}", transform=ax.transAxes, ha="right", fontsize=6.2, color=INK, fontweight="bold", bbox=bb, zorder=10)

# (b) held-out agreement vs two-way controls
ax = fig.add_subplot(g1[1]); letter(ax, "b", x=-0.22)
rows = [("New York", "results/1_oneway_citibike_jun.json", "Jun 2026"), ("Chicago", "results/1_oneway_u_chi_jo25.json", "Jul–Oct 2025"),
        ("Boston", "results/1_oneway_u_bos_jo25.json", "Jul–Oct 2025")]
for i, (nm, f, per) in enumerate(rows):
    r = J(f); fo, ft = r["f"], r["two_way_control"]["f_fixed_orientation"]
    y1, y2 = 2.2 * i, 2.2 * i + 0.8
    ax.barh(y1, fo, color=TERRA, height=0.7, lw=0); ax.barh(y1, 1 - fo, left=fo, color=PALE, height=0.7, lw=0)
    ax.barh(y2, ft, color=MUTE, height=0.7, lw=0); ax.barh(y2, 1 - ft, left=ft, color=PALE, height=0.7, lw=0)
    # 95% Clopper-Pearson intervals (one-way: as stored by src/oneway.py; two-way control computed here)
    n2 = r["two_way_control"]["n"]; k2 = round(ft * n2)
    ci2 = [beta.ppf(0.025, k2, n2 - k2 + 1) if k2 else 0.0, beta.ppf(0.975, k2 + 1, n2 - k2) if k2 < n2 else 1.0]
    for yy, ff, (lo, hi) in ((y1, fo, r["ci95"]), (y2, ft, ci2)):
        ax.errorbar(ff, yy, xerr=[[ff - lo], [hi - ff]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6)
    ax.text(0.02, y1, f"{fo:.0%}", va="center", fontsize=6.6, color="white", fontweight="bold")
    ax.text(0.02, y2, f"{ft:.0%}", va="center", fontsize=6.2, color="white")
    ax.text(-0.03, y1, f"{nm} one-way\nn = {r['n_one_way_pairs']:,}", va="center", ha="right", fontsize=6.4, color=INK)  # n on the left, clear of the error bars
    ax.text(-0.03, y2, "two-way", va="center", ha="right", fontsize=5.8, color=INK2)
ax.axvline(0.5, color=INK, lw=0.6, ls=(0, (3, 2)))
ax.set_xlim(0, 1); ax.set_ylim(5.4, -0.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "50%", "100%"]); ax.set_xlabel("pairs favouring the legal direction (held out)"); clean(ax)

print("a: pairs in box", ni, "legal", nl)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/sfig13_oneway.{ext}", dpi=240, bbox_inches="tight")
print("written sfig13_oneway")
