"""Fig 3 v3: circulation marks habitual journeys on particular pairs. Same palette and conventions as src/fig1_v3.py / src/fig2_v3.py.
Panels: a schematic (round trip vs unreturned leg: both one-way within a window, told apart by the other peak), b habit
concentration (Lorenz), c enrichment (habit vs transit), d Taipei transfer pairs, e morning-evening mirror test, f taxi-zone circulation map (geography is Supplementary Fig. 10; rider type Supplementary Fig. 7).
Run from the repository root: python3 src/fig3_v3.py  ->  results/figures/fig3_v3.pdf|png
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


fig = plt.figure(figsize=(7.4, 6.6))
G = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.15, 1, 1], hspace=0.5, wspace=0.42)

# (a) two journeys that both add one-way flow within a window; the evening peak tells them apart
ax = fig.add_subplot(G[0, 0]); ax.set_xlim(0, 0.9); ax.set_ylim(-0.12, 1.12); ax.axis("off"); letter(ax, "a", x=0.0, y=0.99)
for y0, title, back_col, back_ls, back_lab, verdict, vcol in ((0.78, "round trip", TERRA, "-", "bike back, evening", "reversed: r* ≈ −1", TERRA),
                                                               (0.2, "unreturned leg", SAGE, (0, (2, 1.5)), "train back, evening", "one peak: r* ≈ 0", SAGE)):
    H, W = (0.12, y0), (0.72, y0)
    for p_, lab in ((H, "home"), (W, "work")):
        ax.add_patch(Circle(p_, 0.045, fc=SURF, ec=INK2, lw=1.0, zorder=3)); ax.text(p_[0], p_[1] - 0.1, lab, ha="center", va="top", fontsize=5.8, color=INK2)
    arrow(ax, H, W, TERRA, rad=-0.4); arrow(ax, W, H, back_col, rad=-0.4, ls=back_ls)
    ax.text(0.42, y0 + 0.17, "bike out, morning", ha="center", fontsize=5.8, color=TERRA)
    ax.text(0.42, y0 - 0.2, back_lab, ha="center", va="top", fontsize=5.8, color=back_col)
    ax.text(0.0, y0 + 0.26, title, fontsize=6.8, fontweight="bold", color=INK)
    ax.text(0.9, y0 + 0.26, verdict, fontsize=6.2, color=vcol, fontweight="bold", ha="right")
ax.text(0.45, -0.12, "within each peak, both add one-way flow on the pair", ha="center", fontsize=5.6, color=INK2, style="italic")

# (b) habit concentration
ax = fig.add_subplot(G[0, 1]); letter(ax, "b", x=-0.2)
conc = []
for c, f in (("nyc", "citibike_jun"), ("chi", "u_chi_jo25"), ("bos", "u_bos_jo25")):
    d = np.load(f"results/figdata_hab_{f}.npz"); o = np.argsort(-d["R"], kind="stable"); e = d["e"][o]  # signed excess, as in the pre-specified test
    cx = np.r_[0, np.arange(1, e.size + 1) / e.size]; cy = np.r_[0, np.cumsum(e) / e.sum()]
    nm, col, mk = CITY[c]; ax.plot(cx, cy, color=col, lw=1.6); top = d["R"] >= np.quantile(d["R"], 0.9); conc.append((nm, col, d["e"][top].sum() / d["e"].sum()))  # top decile (ties included), as src/hab.py
ax.fill_between([0, 1], [0, 1], 0, color=PALE, lw=0, zorder=0)
ax.plot([0, 1], [0, 1], color=MUTE, lw=0.8, ls=(0, (3, 2))); ax.axvline(0.1, color=GRID, lw=0.8, zorder=0)
for j, (nm, col, v) in enumerate(conc):
    ax.text(0.97, 0.30 - 0.09 * j, f"{nm} {v:.0%}", ha="right", fontsize=6.2, color=col, fontweight="bold")
ax.text(0.97, 0.39, "top 10% hold", ha="right", fontsize=5.8, color=INK2)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([0, 0.1, 0.5, 1]); ax.set_xticklabels(["", "10%", "50%", "100%"])
ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50%", "100%"])
ax.set_xlabel("station pairs, most habitual first"); ax.set_ylabel("cumulative loop excess", labelpad=2); clean(ax)

# (c) enrichment: habit vs transit
ax = fig.add_subplot(G[0, 2]); letter(ax, "c", x=-0.2)
mode = J("results/mode.json")
hab = {"nyc": J("results/hab_citibike_jun.json")["enrichment_top_decile"], "chi": J("results/hab_u_chi_jo25.json")["enrichment_top_decile"],
       "bos": J("results/hab_u_bos_jo25.json")["enrichment_top_decile"]}
tra = {"nyc": mode["M1_nyc_june"]["enrichment_T1"], "chi": mode["M1_chi"]["enrichment_T1"], "bos": mode["M1_bos"]["enrichment_T1"]}
for i, c in enumerate(("nyc", "chi", "bos")):
    nm, col, _ = CITY[c]
    ax.bar(i - 0.19, hab[c], width=0.36, color=col, lw=0); ax.bar(i + 0.19, tra[c], width=0.36, color=SAGE, alpha=0.75, lw=0)
    ax.text(i - 0.19, hab[c] + 0.06, f"{hab[c]:.1f}×", ha="center", fontsize=6); ax.text(i + 0.19, tra[c] + 0.06, f"{tra[c]:.1f}×", ha="center", fontsize=6)
ax.axhline(1, color=INK2, lw=0.7, ls=(0, (3, 2)))
ax.set_xticks(range(3)); ax.set_xticklabels([CITY[c][0] for c in ("nyc", "chi", "bos")]); ax.set_ylim(0, 4.9)
ax.set_ylabel("loop-excess enrichment"); ax.tick_params(axis="x", length=0)
ax.legend(handles=[Patch(fc=TERRA, label="habitual pairs (top 10%)"), Patch(fc=SAGE, alpha=0.75, label="rapid-transit pairs")],
          frameon=False, fontsize=5.8, loc="upper left", ncol=1, borderaxespad=0.1); clean(ax)

# (d) direct evidence: Taipei feeder pairs (H-XFER, pre-specified; tau from May 2023, circulation from June 2023)
ax = fig.add_subplot(G[1, 0]); letter(ax, "d", x=-0.2)
xa, xn = J("results/xfer_g_xj6.json"), J("results/xfer_g_xn6.json")
q = np.arange(1, 11)
ax.bar(q, xa["enrichment_by_tau_decile"], width=0.72, color=TERRA, alpha=0.75, lw=0, label="all rides")
ax.step(np.r_[q - 0.5, 10.5], np.r_[xn["enrichment_by_tau_decile"], xn["enrichment_by_tau_decile"][-1]], where="post",
        color=INK2, lw=1.1, label="non-transfer rides")
ax.axhline(1, color=INK2, lw=0.7, ls=(0, (3, 2)))
ax.text(10, xa["enrichment_by_tau_decile"][-1] + 0.06, f"{xa['enrichment_by_tau_decile'][-1]:.1f}×", ha="center", fontsize=6.2, color=INK)
t3 = xa["T3"]
ax.text(0.03, 0.97, f"direction = transfer\ndirection in {t3['share_agree']:.0%}", transform=ax.transAxes,
        fontsize=5.6, color=INK2, va="top")
ax.set_xticks([1, 5, 10]); ax.set_xticklabels([f"{xa['tau_decile_medians'][0]:.0%}", f"{xa['tau_decile_medians'][4]:.0%}",
                                               f"{xa['tau_decile_medians'][9]:.0%}"])
ax.set_xlabel("transfer share of rides\n(pair decile; Taipei)"); ax.set_ylabel("loop-excess enrichment")
ax.legend(frameon=False, fontsize=5.4, loc="center left", bbox_to_anchor=(0.0, 0.66)); ax.set_ylim(0, 2.6); clean(ax)

# (e) morning-evening mirror test (pre-specified; rail systems are positive controls)
ax = fig.add_subplot(G[1, 1]); letter(ax, "e", x=-0.42)
UN = [("g_lon", "London"), ("g_baw", "Bay Wheels"), ("g_bue", "Buenos Aires"), ("divvy_mj", "Chicago"), ("bluebikes", "Boston"),
      ("g_seo", "Seoul"), ("g_ybk", "YouBike"), ("g_mex", "Mexico City"), ("citibike", "New York"), ("tlc_2505", "NYC taxi"),
      ("g_bar", "BART"), ("g_tpex", "Taipei Metro")]
COLS = {"tlc_2505": SLATE, "g_bar": SAGE, "g_tpex": SAGE}
GAP = lambda y: y + 0.6 * (y >= 9) + 0.6 * (y >= 10)  # gaps separate bike share, taxi and rail
for y, (u, nm) in enumerate(UN):
    r = J(f"results/ampm/{u}_real.json")["AM_PM"]; col = COLS.get(u, TERRA); yy = GAP(y)
    ax.errorbar(r["r_star"], yy, xerr=[[r["r_star"] - r["ci95"][0]], [r["ci95"][1] - r["r_star"]]], fmt="o", color=col, ms=3.2,
                elinewidth=0.8, capsize=1.4, capthick=0.7)
ax.set_yticks([GAP(y) for y in range(len(UN))]); ax.set_yticklabels([nm for _, nm in UN], fontsize=5.8)
ax.set_ylim(len(UN) + 0.8, -0.7); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.axvline(0, color=INK2, lw=0.6); ax.axvline(-0.5, color=MUTE, lw=0.6, ls=(0, (3, 2)))
ax.set_xlim(-1.0, 0.25); ax.set_xticks([-1, -0.5, 0]); ax.set_xlabel("morning–evening r*")
ax.text(-0.98, -0.95, "reversed", fontsize=5.6, color=INK2); ax.text(0.22, -0.95, "unrelated", fontsize=5.6, color=INK2, ha="right")
clean(ax)

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


ax = fig.add_subplot(G[1, 2]); ax.set_aspect("equal"); ax.axis("off"); letter(ax, "f", x=0.0, y=1.0)
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
    fig.savefig(f"{OUT}/fig3_v3.{ext}", dpi=240, bbox_inches="tight")
print("written fig3_v3")
