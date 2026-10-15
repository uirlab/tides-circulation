"""Fig 2 v3: circulation is general in bike share and small on metro. Same palette and conventions as src/fig1_v3.py
(letters only; explanations are in the figure caption).
Panels: a circulation maps (8 systems), b h' at 15 min, c 15 -> 60 min with the metro, d h' through the day,
e every month 2024-2026 (US), f robustness of the existence claim.
Inputs: results/figdata_circ_<city>.npz (src/circ_pairs.py), station coordinates (processed npz for US systems,
results/figdata_coords_<city>.npz from src/station_coords.py), replication / unit / sensitivity JSONs.
Run from the project root: python3 src/fig2_v3.py  ->  results/figures/fig2_v3.pdf|png
"""
import glob
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figdata_io import stations  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.cluster.vq import kmeans2

OUT = "results/figures"
SLATE, TERRA, SAGE = "#3b6a9b", "#c9744d", "#4f9a6e"
INK, INK2, MUTE, GRID, SURF, PALE = "#2b2a27", "#6b6862", "#b9b4aa", "#e7e3dc", "#fbfaf7", "#f1eee8"
SEQ = LinearSegmentedColormap.from_list("slate_seq", ["#eef1f4", "#c9d4e0", "#9fb3c9", "#7290b0", "#4d6e92", "#2c4a6b"])
TSEQ = LinearSegmentedColormap.from_list("terra_seq", ["#f6ece4", "#ecc9b3", "#dda283", "#c9744d", "#9c5232"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": MUTE, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = lambda p: json.load(open(p))


def clean(ax):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom", ha="right" if x < 0 else "left")


# Taipei: interchange stations excluded (g_tpex, post hoc; results/global.md) by default; TPE=raw (pre-specified) or TPE=merged
TPE = {"raw": "g_tpe", "merged": "g_tpem"}.get(os.environ.get("TPE", ""), "g_tpex")  # default: interchange stations excluded
TAG = {"g_tpe": "_tperaw", "g_tpem": "_tpemerged", "g_tpex": ""}[TPE]
VOL = J("results/figdata_volume.json")
# system key, label, 15-min result, 60-min result, coordinates source, metro?
SYS = [("g_mex", "Mexico City", "results/1_replication_g_mex_real.json", "results/1_replication_g_mex_b60_real.json", "coords", False),
       ("g_seo", "Seoul", "results/1_replication_g_seo_real.json", "results/1_replication_g_seo_b60_real.json", "coords", False),
       ("citibike", "New York", "results/0cb_strata_real.json", "results/0cb_strata_b60_real.json", "npz", False),
       ("g_bue", "Buenos Aires", "results/1_replication_g_bue_real.json", "results/1_replication_g_bue_b60_real.json", "stations", False),
       ("g_lon", "London", "results/1_replication_g_lon_real.json", "results/1_replication_g_lon_b60_real.json", "coords", False),
       ("divvy_mj", "Chicago", "results/1_replication_divvy_mj_real.json", "results/1_replication_divvy_mj_b60_real.json", "npz", False),
       ("bluebikes", "Boston", "results/1_replication_bluebikes_real.json", "results/1_replication_bluebikes_b60_real.json", "npz", False),
       ("g_baw", "SF Bay (bike)", "results/1_replication_g_baw_real.json", "results/1_replication_g_baw_b60_real.json", "stations", False),
       ("g_ybk", "Taipei (bike)", None, "results/1_replication_g_ybk_real.json", "coords", False),
       (TPE, "Taipei metro", None, f"results/1_replication_{TPE}_real.json", "coords", True),
       ("g_bar", "BART", None, "results/1_replication_g_bar_real.json", "coords", True)]
BYKEY = {x[0]: x for x in SYS}
# 95% intervals from the day-level delete-d jackknife (src/rb_uncert_summary.py ; )
UNC = J("results/rb/uncertainty_summary.json") if os.path.exists("results/rb/uncertainty_summary.json") else {"bin15": {}, "bin60": {}}
EXTENT = {"g_baw": ((-122.52, -122.38), (37.73, 37.82))}  # Bay Wheels: San Francisco (East Bay and San Jose clusters off-frame)
# map grid (3 x 4): bike systems by 15-min h', then the within-city pairs on the last row
GRID_ORDER = [["g_mex", "g_seo", "citibike", "g_bue"], ["g_lon", "divvy_mj", "bluebikes", None], ["g_ybk", TPE, "g_baw", "g_bar"]]
H15 = {k: J(p15)["decision"]["median_h"] for k, _, p15, _, _, m in SYS if p15}
PB = lambda p: np.array([b["h"] for b in J(p)["per_bin"]])
H60 = {k: J(p60)["decision"]["median_h"] for k, _, _, p60, _, _ in SYS if os.path.exists(p60)}


def coords(key, src):
    if src == "npz":
        d = stations(key); return d["lat"].astype(float), d["lon"].astype(float)  # cached in results/ (src/figdata_io.py)
    d = np.load(f"results/figdata_{'coords' if src == 'coords' else 'stations'}_{key}.npz")
    return d["lat"].astype(float), d["lon"].astype(float)


def arc(p, q, bend=0.18, n=14):
    p, q = np.asarray(p), np.asarray(q); m = (p + q) / 2; v = q - p
    c = m + bend * np.array([v[1], -v[0]]); s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * p + 2 * (1 - s) * s * c + s ** 2 * q


NZ, TOPK = 40, 45  # zones per system; zone pairs drawn per system


def circ_map(ax, key, src, metro):
    """Zone-level circulation pattern: station-pair residuals (beyond the tide) pooled by inverse variance into zone pairs;
    the TOPK zone pairs with the largest |z| are drawn, bending right of the favoured direction; width/opacity scaled
    within each system (data volume differs ~250x across systems, so magnitudes are shown in panels b-d, not here)."""
    lat, lon = coords(key, src); kx = np.cos(np.radians(np.nanmedian(lat)))
    c = np.load(f"results/figdata_circ_{key}.npz"); lo, hi, eps, se, t = c["lo"], c["hi"], c["eps"], c["se"], c["t"]
    fin = np.isfinite(lat) & np.isfinite(lon)
    zc, zl = kmeans2(np.c_[lon[fin] * kx, lat[fin]], NZ, seed=1, minit="++")
    Z = np.full(lat.size, -1); Z[fin] = zl
    ok = (Z[lo] >= 0) & (Z[hi] >= 0) & (Z[lo] != Z[hi]) & (se > 0)
    za, zb, e, w = Z[lo][ok], Z[hi][ok], eps[ok], 1 / se[ok] ** 2
    flip = za > zb; za, zb = np.where(flip, zb, za), np.where(flip, za, zb); e = np.where(flip, -e, e)  # orient lower -> higher zone
    k = za * NZ + zb
    sw = np.bincount(k, w, NZ * NZ); swe = np.bincount(k, w * e, NZ * NZ)
    has = sw > 0; pe = np.zeros(NZ * NZ); pe[has] = swe[has] / sw[has]; zz = pe * np.sqrt(sw)
    idx = np.argsort(-np.abs(zz))[:TOPK]; idx = idx[np.abs(zz[idx]) > 0]  # strongest evidence within this system
    A, B = idx // NZ, idx % NZ; sg = zz[idx]
    a = np.where(sg > 0, A, B); b = np.where(sg > 0, B, A); m = np.abs(sg)
    od = np.argsort(m); wn = (m[od] - m.min()) / max(m.max() - m.min(), 1e-9)  # scaled within system: pattern, not magnitude
    segs = [arc(zc[a[i]], zc[b[i]], bend=0.22) for i in od]
    segs = [np.c_[s_[:, 0] / kx, s_[:, 1]] for s_ in segs]
    col = np.tile(to_rgba(SAGE if metro else TERRA), (len(od), 1)); col[:, 3] = 0.25 + 0.7 * wn
    ax.add_collection(LineCollection(segs, colors=col, linewidths=0.4 + 2.4 * wn, capstyle="round", zorder=1))
    ax.scatter(zc[:, 0] / kx, zc[:, 1], s=3, color=INK2, alpha=0.35, lw=0, zorder=2)
    fin = np.isfinite(lat) & np.isfinite(lon)
    ax.scatter(lon[fin], lat[fin], s=0.6 if not metro else 3, color=MUTE, alpha=0.35, lw=0, zorder=0)
    x0, x1 = np.nanpercentile(lon[fin], [2, 98]); y0, y1 = np.nanpercentile(lat[fin], [2, 98])
    if key in EXTENT:  # multi-cluster systems: frame the core
        (x0, x1), (y0, y1) = EXTENT[key]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; half = max((x1 - x0) * kx, y1 - y0) / 2 * 1.08
    ax.set_xlim(cx - half / kx, cx + half / kx); ax.set_ylim(cy - half, cy + half); ax.set_aspect(1 / kx); ax.axis("off")
    km = 5 if half * 111 > 6 else 2  # scale bar
    bx, by = cx - half / kx * 0.92, cy - half * 0.93
    ax.plot([bx, bx + km / 111 / kx], [by, by], color=INK2, lw=1.0, solid_capstyle="butt", zorder=6)
    ax.text(bx, by + half * 0.04, f"{km} km", fontsize=5.4, color=INK2, va="bottom", zorder=6,
            bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=0.4))


fig = plt.figure(figsize=(7.4, 10.4))
G = fig.add_gridspec(3, 1, height_ratios=[1.8, 0.95, 1.1], hspace=0.26)

# (a) circulation maps; last row = within-city pairs (bike, rail)
ga = G[0].subgridspec(3, 4, wspace=0.04, hspace=0.2)
for r_, row in enumerate(GRID_ORDER):
    for c_, key in enumerate(row):
        ax = fig.add_subplot(ga[r_, c_])
        if key is None:
            ax.axis("off")
            ax.text(0.1, 0.6, "bike share", transform=ax.transAxes, fontsize=7, color=TERRA, fontweight="bold")
            ax.text(0.1, 0.45, "rail", transform=ax.transAxes, fontsize=7, color=SAGE, fontweight="bold")
            continue
        _, nm, _, _, src, metro = BYKEY[key]; circ_map(ax, key, src, metro)
        bb = dict(fc=SURF, ec="none", alpha=0.8, pad=0.6)
        ax.text(0.02, 1.0, nm, transform=ax.transAxes, fontsize=7, fontweight="bold", color=SAGE if metro else INK, va="bottom", zorder=5)  # title above the map, clear of arcs
        if r_ == 0 and c_ == 0:
            letter(ax, "a", x=-0.08, y=1.12)

# (b) h' at 15 min; (c) 15 -> 60 min with the metro
gb = G[1].subgridspec(1, 2, width_ratios=[1.1, 1], wspace=0.45)
ax = fig.add_subplot(gb[0]); letter(ax, "b", x=-0.32)
bike = [s for s in SYS if not s[5] and s[2]]
for i, (key, nm, *_ ) in enumerate(bike):
    v = H15[key]
    ax.barh(i, v, color=TERRA, alpha=0.55 + 0.45 * (v / 0.45), height=0.66, lw=0)
    hi = v
    if key in UNC["bin15"]:
        lo, hi = UNC["bin15"][key]["median_h"]["ci95"]
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.6, capthick=0.6)
    ax.text(max(v, hi) + 0.008, i, f"{v:.2f}", va="center", fontsize=6.4, color=INK)
ax.set_yticks(range(len(bike))); ax.set_yticklabels([s[1] for s in bike]); ax.tick_params(axis="y", length=0)
ax.axvline(0.10, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax.set_ylim(len(bike) - 0.4, -0.6); ax.set_xlim(0, 0.5); ax.set_xlabel("circulation share h′ (15-min bins)"); clean(ax); ax.spines["left"].set_visible(False)

# (c) within-city bike vs rail, hour by hour
gcc = gb[1].subgridspec(2, 1, hspace=0.35)
for j, (city, bk, rk, bl, rl) in enumerate([("Taipei", "g_ybk", TPE, "YouBike", "metro"), ("SF Bay Area", "g_baw", "g_bar", "Bay Wheels", "BART")]):
    ax = fig.add_subplot(gcc[j])
    if j == 0:
        letter(ax, "c", x=-0.2)
    hb, hr = PB(BYKEY[bk][3]), PB(BYKEY[rk][3]); x = np.arange(17) + 6
    ax.fill_between(x, 0, np.r_[np.clip(hb, 0, None), max(hb[-1], 0)], step="post", color=TERRA, alpha=0.5, lw=0)
    ax.step(x, np.r_[hb, hb[-1]], where="post", color=TERRA, lw=0.8)
    ax.step(x, np.r_[hr, hr[-1]], where="post", color=SAGE, lw=1.8)
    k = int((hb > hr).sum())
    ax.text(6.2, 0.98, city, fontsize=6.8, fontweight="bold", color=INK, va="top")
    ax.text(21.9, 0.98, f"{bl} {np.median(hb):.2f}", fontsize=6, color=TERRA, ha="right", va="top", fontweight="bold")
    ax.text(21.9, 0.86, f"{rl} {np.median(hr):.2f}", fontsize=6, color=SAGE, ha="right", va="top", fontweight="bold")
    ax.text(6.2, 0.86, f"bike > rail in {k}/16 h", fontsize=5.6, color=INK2, va="top")
    ax.set_xlim(6, 22); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.3, 0.6]); ax.set_ylabel("h′ (60 min)", fontsize=6.3)
    ax.set_xticks([6, 9, 12, 15, 18, 21]); ax.set_xticklabels([] if j == 0 else [6, 9, 12, 15, 18, 21])
    if j == 1:
        ax.set_xlabel("hour of day")
    clean(ax)

# (d) h' through the day (ridgeline); (e) every month; (f) robustness
gc = G[2].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.42)
ax = fig.add_subplot(gc[0]); letter(ax, "d", x=-0.25)
RID = [x for x in SYS if x[2]]  # bike systems with 15-min data (within-city units are in panel c)
for i, (key, nm, p15, p60, _, metro) in enumerate(RID):
    pb = J(p60 if metro else p15)["per_bin"]; h = np.clip([b["h"] for b in pb], 0, None); nb = len(h); w = 16 / nb
    x = 6 + np.arange(nb + 1) * w; y = np.r_[h, h[-1]]
    base = len(RID) - 1 - i; sc = 1.6
    ax.fill_between(x, base, base + sc * y, step="post", color=SAGE if metro else TERRA, alpha=0.35 if metro else 0.6, lw=0)
    ax.step(x, base + sc * y, where="post", color=SAGE if metro else TERRA, lw=0.7)
    ax.axhline(base, color=GRID, lw=0.5, zorder=0)
    v = np.asarray(VOL[key], float); v = v / v.max()  # trip volume, scaled to each system's peak
    ax.plot(6 + (np.arange(v.size) + 0.5) * 16 / v.size, base + 0.8 * v, color=INK2, lw=0.6, alpha=0.7, ls=(0, (2, 1.2)), zorder=3)
    ax.text(5.7, base + 0.12, nm, ha="right", va="bottom", fontsize=6, color=SAGE if metro else INK)
ax.set_xlim(6, 22); ax.set_ylim(-0.1, len(RID) + 0.2); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xticks([6, 9, 12, 15, 18, 21]); ax.set_xlabel("hour of day"); clean(ax)
ax.plot([22.3, 22.3], [len(RID) - 1, len(RID) - 1 + sc * 0.5], color=INK2, lw=0.8, clip_on=False)
ax.text(22.45, len(RID) - 1 + sc * 0.25, "h′ 0.5", ha="left", va="center", fontsize=5.6, color=INK2)

gr = gc[1].subgridspec(2, 1, height_ratios=[1, 1.1], hspace=0.75)
ax = fig.add_subplot(gr[0]); letter(ax, "e", x=-0.03)
months = [f"{y}-{m:02d}" for y in (2024, 2025, 2026) for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-08"]
M = np.full((3, len(months)), np.nan); S = np.zeros((3, len(months)), int)
for r, c in enumerate(("nyc", "chi", "bos")):
    for j, m in enumerate(months):
        f = f"results/3_units/u_{c}_{m}_real.json"; sfile = f"results/3_units/u_{c}_{m}_synthetic.json"
        if c == "nyc" and m == "2026-05":
            M[r, j] = J("results/0cb_strata_real.json")["decision"]["median_h"]; S[r, j] = 1
        elif os.path.exists(f):
            M[r, j] = J(f)["decision"]["median_h"]; S[r, j] = 1
        elif os.path.exists(sfile):
            S[r, j] = 2
            if glob.glob(f"results/3_units/u_{c}_*{m}*+*_real.json") + glob.glob(f"results/3_units/u_{c}_*+*{m}*_real.json"):
                S[r, j] = 3
im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=TSEQ, vmin=0.1, vmax=0.32, interpolation="nearest")
for r in range(3):
    for j in range(len(months)):
        if S[r, j] in (2, 3):
            ax.add_patch(plt.Rectangle((j - 0.5, r - 0.5), 1, 1, fc=PALE, ec=SURF, lw=0.8, hatch="////" if S[r, j] == 3 else None))
        elif S[r, j] == 0:
            ax.add_patch(plt.Rectangle((j - 0.5, r - 0.5), 1, 1, fc=SURF, ec=SURF))
        else:
            ax.add_patch(plt.Rectangle((j - 0.5, r - 0.5), 1, 1, fc="none", ec=SURF, lw=0.8))
ax.set_yticks(range(3)); ax.set_yticklabels(["New York", "Chicago", "Boston"], fontsize=6.3)
ticks = [j for j, m in enumerate(months) if m.endswith("-01")]
ax.set_xticks(ticks); ax.set_xticklabels([months[j][:4] for j in ticks], fontsize=6.3)
for s_ in ax.spines.values():
    s_.set_visible(False)
ax.tick_params(length=0)
cax = ax.inset_axes([1.03, 0.0, 0.035, 1.0])
cb = fig.colorbar(im, cax=cax, orientation="vertical", ticks=[0.1, 0.2, 0.3]); cb.ax.tick_params(labelsize=5.6, length=0); cb.outline.set_visible(False)
cax.set_title("h′", fontsize=6, color=INK2, pad=3)

# (f) per trip: rail tides are not weaker, rail circulation is (exploratory; results/pertrip_deviance.json)
PT = {r["system"]: r for r in J("results/pertrip_deviance.json")["rows"]}
gf = gr[1].subgridspec(1, 2, wspace=0.35)
groups = [("Taipei", "YouBike", "Taipei MRT"), ("SF Bay", "Bay Wheels", "BART")]
for j, (key, lab) in enumerate((("tide_per_trip", "tide per trip"), ("circ_per_trip", "circulation per trip"))):
    ax = fig.add_subplot(gf[j])
    if j == 0:
        letter(ax, "f", x=-0.55)
    for g, (city, b_, r_) in enumerate(groups):
        for k, (sysn, col) in enumerate(((b_, TERRA), (r_, SAGE))):
            v = PT[sysn][key]; x = g + (k - 0.5) * 0.38
            ax.bar(x, v, width=0.34, color=col, alpha=0.9 if key == "circ_per_trip" else 0.55, lw=0)
            u = {"YouBike": "g_ybk", "Taipei MRT": "g_tpex", "Bay Wheels": "g_baw", "BART": "g_bar"}[sysn]; hi = v
            if u in UNC["bin60"]:
                lo, hi = UNC["bin60"][u][key]["ci95"]
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5)
            ax.text(x, hi + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=5.4, color=INK)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups], fontsize=6.2); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.43); ax.set_yticks([0, 0.2, 0.4]); ax.tick_params(axis="y", labelsize=5.8)
    ax.text(0.03, 1.0, lab, transform=ax.transAxes, fontsize=6.4, color=INK, va="bottom")
    if j == 0:
        ax.set_ylabel("deviance per trip\n(60-min bins)", fontsize=6.0)
    clean(ax)
ax.text(0.98, 0.72, "bike share", transform=ax.transAxes, ha="right", fontsize=5.8, color=TERRA, fontweight="bold")
ax.text(0.98, 0.61, "rail", transform=ax.transAxes, ha="right", fontsize=5.8, color=SAGE, fontweight="bold")

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}/fig2_v3{TAG}.{ext}", dpi=240, bbox_inches="tight")
print("written fig2_v3; 60-min available for", sorted(H60))
