"""Fig 1 v3: from observation to decomposition. Same muted palette as figures_v2.
Panels: a three-station example (tide only; tide plus circulation), b NYC morning vs evening net flows, c station x hour
tide heatmap, d triangle loop test vs tide-only null, e scale (world map), f the split across aggregation windows.
Explanations live in the caption, not on the figure.
Needs results/figdata_motiv_triangles.npz (src/motiv_triangles.py) and data/raw/naturalearth/ne_110m_land.shp.
Run from the project root:
  python3 src/fig1_v3.py                 -> results/figures/fig1_v3.pdf|png  (panel b style from B_STYLE, default arcs_weighted)
  VARIANTS=1 python3 src/fig1_v3.py      -> results/figures/fig1b_variants.png and fig1_v3_<style>.png for each style
"""
import json
import os
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
from scipy.cluster.vq import kmeans2

OUT = "results/figures"; os.makedirs(OUT, exist_ok=True)
SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
SEQ = LinearSegmentedColormap.from_list("slate_seq", ["#eef1f4", "#c9d4e0", "#9fb3c9", "#7290b0", "#4d6e92", "#2c4a6b"])
DIV = LinearSegmentedColormap.from_list("div", [TERRA, "#e6c3ae", SURF, "#b7c8da", SLATE])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"})
STYLES = {"arcs_fine": "curved arcs, thin and faint", "arcs_weighted": "curved arcs, width and opacity by net flow",
          "zones": "aggregated: 60 zones, arrows by net flow", "zones_nodes": "aggregated: zone arrows + zone net inflow"}


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="right" if x < 0 else "left")


def read_polygons(path):
    """Minimal ESRI shapefile reader for polygon records (type 5)."""
    b = open(path, "rb").read(); pos = 100; out = []
    while pos < len(b):
        _, clen = struct.unpack(">ii", b[pos:pos + 8]); rec = b[pos + 8:pos + 8 + 2 * clen]; pos += 8 + 2 * clen
        if struct.unpack("<i", rec[:4])[0] != 5:
            continue
        npart, npt = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{npart}i", rec[44:44 + 4 * npart])) + [npt]
        pts = np.frombuffer(rec[44 + 4 * npart:44 + 4 * npart + 16 * npt], "<f8").reshape(-1, 2)
        out += [pts[parts[i]:parts[i + 1]] for i in range(npart)]
    return out


# ---------------- data for panel b ----------------
from figdata_io import netflow_citibike, stations  # cached aggregates in results/ (no trip data needed)
_st = stations("citibike"); LAT, LON = _st["lat"].astype(float), _st["lon"].astype(float)
CX, CY = -73.984, 40.754  # Midtown
KX = np.cos(np.radians(40.75)); DIST = np.hypot((LON - CX) * KX, LAT - CY); N = len(LAT)
HOURS = ((8, "8–9 am"), (17, "5–6 pm"))
NETS = {}
_C = netflow_citibike(tuple(h for h, _ in HOURS))
for h0, _ in HOURS:
    NETS[h0] = _C[h0] - _C[h0].T
OK = np.isfinite(LAT) & np.isfinite(LON)
ZC, zl = kmeans2(np.c_[LON[OK] * KX, LAT[OK]], 60, seed=1, minit="++")  # zones for the aggregated styles (fixed seed)
ZL = np.full(N, 60); ZL[OK] = zl  # stations without coordinates go to a dropped 61st bin


def arc(p, q, bend=0.18, n=16):
    """Quadratic Bezier from p to q bending to the right of travel (so direction reads from curvature)."""
    p, q = np.asarray(p), np.asarray(q); m = (p + q) / 2; v = q - p
    c = m + bend * np.array([v[1], -v[0]])
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * p + 2 * (1 - s) * s * c + s ** 2 * q


def draw_b(fig, spec, style, label=True):
    sub = spec.subgridspec(1, 2, wspace=0.02)
    for j, (h0, nm) in enumerate(HOURS):
        ax = fig.add_subplot(sub[j]); net = NETS[h0]
        if style.startswith("arcs"):
            a, b = np.where(net >= 8); w = net[a, b]; inb = DIST[b] < DIST[a]; od = np.argsort(w)
            segs = [arc((LON[a[i]] * KX, LAT[a[i]]), (LON[b[i]] * KX, LAT[b[i]])) for i in od]
            segs = [np.c_[s_[:, 0] / KX, s_[:, 1]] for s_ in segs]
            base = np.array([to_rgba(SLATE if inb[i] else TERRA) for i in od])
            if style == "arcs_fine":
                base[:, 3] = 0.22; lw = 0.25
            else:
                wn = np.clip((w[od] - 8) / 40, 0, 1); base[:, 3] = 0.12 + 0.7 * wn; lw = 0.2 + 1.6 * wn
            ax.add_collection(LineCollection(segs, colors=base, linewidths=lw, capstyle="round"))
        else:
            Z = np.zeros((61, 61)); np.add.at(Z, (ZL[:, None].repeat(N, 1), ZL[None, :].repeat(N, 0)), np.nan_to_num(net))
            Z = (Z - Z.T)[:60, :60] / 2; a, b = np.where(Z > 0); w = Z[a, b]
            top = np.argsort(-w)[:70]; a, b, w = a[top], b[top], w[top]
            zx, zy = ZC[:, 0] / KX, ZC[:, 1]; zd = np.hypot((zx - CX) * KX, zy - CY)
            if style == "zones_nodes":
                inflow = Z.sum(0) - Z.sum(1)  # net arrivals per zone
                sc = np.abs(inflow) / np.abs(inflow).max()
                ax.scatter(zx, zy, s=4 + 90 * sc, c=[SLATE if v > 0 else TERRA for v in inflow], alpha=0.28, lw=0, zorder=1)
            wmax = w.max()
            for i in np.argsort(w):
                col = SLATE if zd[b[i]] < zd[a[i]] else TERRA; f = w[i] / wmax
                ax.add_patch(FancyArrowPatch((zx[a[i]], zy[a[i]]), (zx[b[i]], zy[b[i]]), connectionstyle="arc3,rad=-0.25",
                                             arrowstyle=f"-|>,head_length={2 + 3 * f:.1f},head_width={1.2 + 1.8 * f:.1f}",
                                             lw=0.3 + 2.4 * f, color=col, alpha=0.35 + 0.55 * f, shrinkA=1, shrinkB=1, zorder=2))
        ax.set_xlim(-74.03, -73.91); ax.set_ylim(40.68, 40.83); ax.set_aspect(1 / KX); ax.axis("off")
        ax.text(0.5, -0.01, nm, transform=ax.transAxes, fontsize=6.8, color=INK, va="top", ha="center")
        if j == 0 and label:
            letter(ax, "b", x=0.0)
    return ax


def build(style, path):
    fig = plt.figure(figsize=(7.4, 7.2))
    G = fig.add_gridspec(3, 1, height_ratios=[0.95, 1, 0.95], hspace=0.38)
    r1 = G[0].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.06)
    r2 = G[1].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.5)
    r3 = G[2].subgridspec(1, 2, width_ratios=[1.9, 1], wspace=0.22)

    # (a) the worked example of the Introduction: three stations at 8 am, tide alone and with circulation
    ax = fig.add_subplot(r1[0]); ax.set_xlim(0, 2.45); ax.set_ylim(-0.14, 1.06); ax.set_aspect("equal"); ax.axis("off")
    letter(ax, "a", x=0.0, y=0.98)
    P3 = {"R": (0.16, 0.80), "M": (0.16, 0.14), "O": (0.86, 0.47)}
    NM = {"R": "residential\nstreet", "M": "metro\nexit", "O": "offices"}
    def three(dx, circ):
        q = {k: (x + dx, y) for k, (x, y) in P3.items()}
        yy, xx = np.mgrid[-0.1:1.05:140j, 0.45:1.12:100j]
        ax.imshow(np.exp(-((xx - 0.86) ** 2 + (yy - 0.47) ** 2) / 0.04), extent=(dx + 0.45, dx + 1.12, -0.1, 1.05), origin="lower",
                  cmap=LinearSegmentedColormap.from_list("pull", [SURF, "#c9d4e0", "#7290b0"]), alpha=0.6, zorder=0)
        for a_, b_, w in (("R", "O", 2.4), ("M", "O", 1.5), ("R", "M", 0.8)):  # tide: width follows the difference in pull
            ax.add_patch(FancyArrowPatch(q[a_], q[b_], arrowstyle="-|>,head_length=4,head_width=2.5", color=SLATE, lw=w,
                                         shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.12", zorder=2))
        if circ:  # extra one-way travel from the metro exit to the offices, beyond the pull
            ax.add_patch(FancyArrowPatch(q["M"], q["O"], arrowstyle="-|>,head_length=5,head_width=3", color=TERRA, lw=2.2,
                                         shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=-0.28", zorder=3))
            ax.text(dx + 0.56, 0.17, "beyond\nthe pull", fontsize=6, color=TERRA, fontweight="bold", ha="left", va="top", linespacing=0.95)
        for k, (x, y) in q.items():
            ax.add_patch(Circle((x, y), 0.055, fc=SLATE if k == "O" else SURF, ec=SLATE, lw=1.1, zorder=4))
            ax.text(x + (0.02 if k == "O" else -0.08), y + (0.11 if k == "O" else 0), NM[k], fontsize=6, color=INK,
                    ha="center" if k == "O" else "right", va="bottom" if k == "O" else "center", linespacing=0.95, zorder=5)
        ax.text(dx + 0.5, -0.1, "circulation: tide + one-way loop" if circ else "tide only", ha="center", va="top", fontsize=7,
                color=TERRA if circ else SLATE, fontweight="bold")
    three(0.12, False); three(1.3, True)
    ax.text(1.21, 0.47, "8 am", fontsize=6.4, color=INK2, ha="center", va="center", style="italic")

    # (b) New York net flows
    draw_b(fig, r1[1], style)

    # (c) tides
    m = np.load("results/figdata_motiv_triangles.npz"); net, am, pm = m["net"], m["am"], m["pm"]
    ax = fig.add_subplot(r2[0]); letter(ax, "c", x=-0.08)
    H = net[:, 6:22]; H = H / (np.abs(H).max(1, keepdims=True) + 1e-9); H = H[np.argsort(-am)]
    ax.imshow(H, aspect="auto", cmap=DIV, vmin=-1, vmax=1, interpolation="nearest", extent=(6, 22, len(H), 0))
    ax.set_xticks([6, 9, 12, 15, 18, 21]); ax.set_yticks([]); ax.set_xlabel("hour of day"); ax.set_ylabel("stations")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1.02, 0.97, "net\ninflow", transform=ax.transAxes, fontsize=6, color=SLATE, va="top", fontweight="bold")
    ax.text(1.02, 0.03, "net\noutflow", transform=ax.transAxes, fontsize=6, color=TERRA, va="bottom", fontweight="bold")
    ax.text(1.02, 0.5, f"r = {np.corrcoef(am, pm)[0, 1]:.2f}", transform=ax.transAxes, fontsize=6.2, color=INK, va="center")

    # (d) loops
    zr, zn = m["zr"], m["zn"]
    ax = fig.add_subplot(r2[1]); letter(ax, "d", x=-0.16)
    bins = np.linspace(-7, 7, 57); c = 0.5 * (bins[1:] + bins[:-1])
    hn = np.maximum(np.histogram(zn, bins, density=True)[0], 1e-5); hr = np.maximum(np.histogram(zr, bins, density=True)[0], 1e-5)
    ax.fill_between(c, 1e-5, hn, step="mid", color=MUTE, alpha=0.55, lw=0); ax.step(c, hn, where="mid", color=INK2, lw=0.8)
    ax.fill_between(c, 1e-5, hr, step="mid", color=TERRA, alpha=0.3, lw=0); ax.step(c, hr, where="mid", color=TERRA, lw=1.1)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1); ax.set_xlim(-7, 7)
    for xx in (-3, 3):
        ax.axvline(xx, color=INK2, lw=0.6, ls=(0, (2, 2)))
    fr, fn = np.mean(np.abs(zr) > 3), np.mean(np.abs(zn) > 3)
    ax.text(6.8, 0.5, f"{fr:.1%}", ha="right", va="top", fontsize=7, color=TERRA, fontweight="bold")
    ax.text(6.8, 0.2, f"{fn:.2%}", ha="right", va="top", fontsize=7, color=INK2, fontweight="bold")
    ax.text(-6.8, 0.5, "observed", color=TERRA, fontsize=6.4, fontweight="bold", va="top")
    ax.text(-6.8, 0.2, "tide only", color=INK2, fontsize=6.4, va="top")
    ax.set_xlabel("triangle loop score z"); ax.set_ylabel("density"); clean(ax)

    # (e) scale
    ax = fig.add_subplot(r3[0]); letter(ax, "e", x=0.0, y=0.98)
    for p in read_polygons("data/raw/naturalearth/ne_110m_land.shp"):
        ax.add_patch(Polygon(p, closed=True, fc=PALE, ec=GRID, lw=0.3))
    # weekday trips between different stations, 06:00-22:00 (the analysed set, from the preprocessed counts)
    sys_ = [("New York", -74.0, 40.7, 2.90e6, "bike", (-45, 22)), ("Boston", -71.1, 42.4, 0.27e6, "bike", (-48, 60)),
            ("Chicago", -87.6, 41.9, 0.52e6, "bike", (-86, 63)), ("Mexico City", -99.1, 19.4, 1.09e6, "bike", (-92, 2)),
            ("Buenos Aires", -58.4, -34.6, 0.16e6, "bike", (-90, -32)), ("London", -0.1, 51.5, 0.50e6, "bike", (-22, 59)),
            ("Seoul", 127.0, 37.6, 1.84e6, "bike", (135, 49)), ("Taipei bike", 121.5, 25.0, 4.74e6, "bike", (140, 12)),
            ("Taipei metro", 121.5, 25.0, 39.7e6, "metro", (140, -4)), ("SF bike", -122.4, 37.8, 0.23e6, "bike", (-118, 60)),
            ("BART", -122.4, 37.8, 3.57e6, "metro", (-126, 18))]
    for nm, lon, lat, n, kind, pos in sys_:
        s = 14 + 60 * np.log10(n / 1e5)
        if kind == "bike":
            ax.scatter(lon, lat, s=s, color=TERRA, alpha=0.85, ec="white", lw=0.6, zorder=4)
        else:
            ax.scatter(lon, lat, s=s * 1.6, fc="none", ec=SAGE, lw=1.6, zorder=3)
        lab = f"{nm}\n{n / 1e6:.1f}M" if n >= 1e6 else f"{nm}\n{n / 1e3:.0f}k"
        ax.annotate(lab, (lon, lat), xytext=pos, textcoords="data", ha="center", va="center", fontsize=5.8, color=INK, linespacing=0.95,
                    zorder=4, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=1, shrinkB=3))
    ax.set_xlim(-135, 150); ax.set_ylim(-45, 68); ax.set_aspect("equal"); ax.axis("off")

    # (f) the split, from 15 min to a day (corrected for the tide-only mixing artefact)
    cg = json.load(open("results/coarsegrain.json"))  # corrected for the tide-only simulation (results/coarsegrain.md)
    win = ["15min", "1h", "4h", "16h"]; lab = ["15 min", "1 h", "4 h", "day"]
    lf = np.array([cg[w]["frac_corrected"] for w in win]); raw = np.array([cg[w]["frac_real"] for w in win]); x = np.arange(4)
    ax = fig.add_subplot(r3[1]); letter(ax, "f", x=-0.2)
    ax.fill_between(x, 0, lf, color=TERRA, alpha=0.9, lw=0); ax.fill_between(x, lf, 1, color=SLATE, alpha=0.88, lw=0)
    ax.text(0.12, 0.88, "tide", color="white", fontsize=7, fontweight="bold"); ax.text(0.12, 0.08, "circulation", color="white", fontsize=7, fontweight="bold")
    ax.plot(x, raw, color="white", lw=1.0, ls=(0, (2, 1.5))); ax.text(2.95, raw[-1] + 0.03, "uncorrected", color="white", fontsize=5.6, ha="right", va="bottom")
    for i, v in enumerate(lf):
        ax.text(min(max(i, 0.3), 2.75), v - 0.05, f"{v:.0%}", ha="center", va="top", fontsize=6.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_xlim(0, 3); ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "50%", "100%"]); ax.set_xlabel("time window"); clean(ax)

    for ext in ("pdf", "png") if path.endswith("fig1_v3") else ("png",):
        fig.savefig(f"{path}.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if os.environ.get("VARIANTS"):
    fig = plt.figure(figsize=(7.4, 7.0)); G = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08)
    for i, (st, desc) in enumerate(STYLES.items()):
        ax = draw_b(fig, G[i // 2, i % 2], st, label=False)
        ax.text(-1.0, 1.04, f"{'ABCD'[i]}  {desc}", transform=ax.transAxes, fontsize=7, color=INK, fontweight="bold")
    fig.savefig(f"{OUT}/fig1b_variants.png", dpi=240, bbox_inches="tight"); plt.close(fig)
    for st in STYLES:
        build(st, f"{OUT}/fig1_v3_{st}")
build(os.environ.get("B_STYLE", "arcs_weighted"), f"{OUT}/fig1_v3")
print("written fig1_v3")
