"""Supplementary Fig. 14: taxi and ride-hail circulation between New York taxi zones (May 2025).
Run from the repository root: python3 src/sfig14_taxi.py  ->  results/figures/sfig14_taxi.pdf|png
"""
import glob
import json
import struct
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Wedge

OUT = "results/figures"
SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = lambda p: json.load(open(p))
CITY = {"nyc": ("New York", "#a8552f", "o"), "chi": ("Chicago", TERRA, "s"), "bos": ("Boston", "#e0a888", "^")}  # one hue, three tones


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="right" if x < 0 else "left")


def arrow(ax, p, q, col, lw=1.6, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=4,head_width=2.5", color=col, lw=lw, shrinkA=7, shrinkB=7,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=2))


fig = plt.figure(figsize=(3.6, 3.6))
G = fig.add_gridspec(1, 1)
# (f) the same decomposition on taxi and ride-hail trips between New York's taxi zones (May 2025)
def read_zones(shp, dbf):
    """Taxi-zone polygons keyed by LocationID (minimal shapefile and dBase reader)."""
    b = open(dbf, "rb").read(); nrec = struct.unpack("<I", b[4:8])[0]; hl, rl = struct.unpack("<HH", b[8:12]); fields, q = [], 32
    while b[q] != 0x0D:
        fields.append((b[q:q + 11].split(b"\0")[0].decode(), b[q + 16])); q += 32
    ids = []
    for i in range(nrec):
        r = b[hl + i * rl + 1: hl + (i + 1) * rl]; off = 0; rec = {}
        for nm, ln in fields:
            rec[nm] = r[off:off + ln].decode("latin1").strip(); off += ln
        ids.append(int(float(rec["LocationID"])))
    s = open(shp, "rb").read(); pos = 100; polys = []
    while pos < len(s):
        _, clen = struct.unpack(">ii", s[pos:pos + 8]); rec = s[pos + 8:pos + 8 + 2 * clen]; pos += 8 + 2 * clen
        if struct.unpack("<i", rec[:4])[0] != 5:
            polys.append([]); continue
        npart, npt = struct.unpack("<ii", rec[36:44]); parts = list(struct.unpack(f"<{npart}i", rec[44:44 + 4 * npart])) + [npt]
        pts = np.frombuffer(rec[44 + 4 * npart:44 + 4 * npart + 16 * npt], "<f8").reshape(-1, 2)
        polys.append([pts[parts[i]:parts[i + 1]] for i in range(npart)])
    out = {}
    for i, pl in zip(ids, polys):
        out.setdefault(i, []).extend(pl)
    return out


def centroid(rings):
    ring = max(rings, key=len); x, y = ring[:, 0], ring[:, 1]; c = x * np.roll(y, -1) - np.roll(x, -1) * y; a = c.sum() / 2
    return np.array([((x + np.roll(x, -1)) * c).sum() / (6 * a), ((y + np.roll(y, -1)) * c).sum() / (6 * a)]) if abs(a) > 0 else ring.mean(0)


ax = fig.add_subplot(G[0, 0]); ax.set_aspect("equal"); ax.axis("off")
ZD = "data/raw/tlc/zones/"
Z = read_zones(ZD + "taxi_zones.shp", ZD + "taxi_zones.dbf")
for rings in Z.values():
    for ring in rings:
        ax.fill(ring[:, 0], ring[:, 1], fc=PALE, ec=SURF, lw=0.25, zorder=0)
cz = {k: centroid(v) for k, v in Z.items() if v}
zid = np.load("results/figdata_stations_tlc_2505.npz", allow_pickle=True)["station_id"].astype(int)
cp = np.load("results/figdata_circ_tlc_2505.npz"); zz = cp["eps"] / cp["se"]; ok = cp["t"] >= 200
top = np.flatnonzero(ok)[np.argsort(-np.abs(zz[ok]))[:45]]; zmax = np.abs(zz[top]).max()
for k in top:
    a, b_ = zid[cp["lo"][k]], zid[cp["hi"][k]]
    if a not in cz or b_ not in cz:
        continue
    p_, q_ = (cz[a], cz[b_]) if cp["eps"][k] > 0 else (cz[b_], cz[a])
    ax.add_patch(FancyArrowPatch(p_, q_, arrowstyle="-|>,head_length=2.4,head_width=1.4", color=SLATE, alpha=0.75,
                                 lw=0.4 + 1.6 * abs(zz[k]) / zmax, connectionstyle="arc3,rad=-0.25", shrinkA=1, shrinkB=1, zorder=2))
ht = J("results/1_replication_tlc_2505_real.json")["decision"]["median_h"]; rt = J("results/ampm/tlc_2505_real.json")["AM_PM"]["r_star"]
ax.text(0.0, 0.99, "taxi and ride-hail, New York", transform=ax.transAxes, fontsize=6.4, color=INK, fontweight="bold", va="top")
ax.text(0.0, 0.91, f"h′ = {ht:.2f}   r* = {rt:.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=6.0, color=SLATE, va="top")
xs = np.concatenate([r[:, 0] for v in Z.values() for r in v]); ys = np.concatenate([r[:, 1] for v in Z.values() for r in v])
ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max() + 0.12 * (ys.max() - ys.min()))
ax.plot([xs.max() - 32808, xs.max()], [ys.min() + 2000] * 2, color=INK2, lw=1.0, solid_capstyle="butt")  # 10 km in feet
ax.text(xs.max() - 16404, ys.min() + 5000, "10 km", ha="center", fontsize=5.4, color=INK2)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/sfig14_taxi.{ext}", dpi=240, bbox_inches="tight")
print("written sfig14_taxi")
