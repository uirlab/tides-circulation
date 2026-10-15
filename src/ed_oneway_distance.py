"""Extended Data: one-way legal-direction agreement vs pair distance (NYC June 2026, held out; descriptive).
Input: results/figdata_oneway_citibike_jun.npz (src/oneway.py DUMP_PAIRS=1). Output: results/figures/ed_oneway_distance.png|pdf"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

TERRA, INK, INK2, MUTE, SURF, PALE = "#c9744d", "#2b2a27", "#6b6862", "#b9b4aa", "#fbfaf7", "#f1eee8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTE, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
d = np.load("results/figdata_oneway_citibike_jun.npz", allow_pickle=True)
one = d["kind"] == "one"; agree = (d["legal_ab"] == d["fav_ab"])[one]; dist = d["dist_m"][one]
edges = np.quantile(dist, np.linspace(0, 1, 7)); out = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (dist >= a) & (dist <= b); k, n = int(agree[m].sum()), int(m.sum()); ci = binomtest(k, n).proportion_ci(0.95)
    out.append(((a + b) / 2, k / n, ci.low, ci.high, n, a, b))
fig, ax = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(len(out))
ax.bar(x, [o[1] for o in out], color=TERRA, width=0.7, lw=0)
ax.vlines(x, [o[2] for o in out], [o[3] for o in out], color=INK, lw=0.9)
ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (3, 2)))
for i, o in enumerate(out):
    ax.text(i, 0.04, f"n={o[4]}", ha="center", fontsize=5.4, color="white")
ax.set_xticks(x); ax.set_xticklabels([f"{o[5] / 1000:.1f}–{o[6] / 1000:.1f}" for o in out], fontsize=6)
ax.set_xlabel("distance between stations (km)"); ax.set_ylabel("one-way pairs favouring\nthe legal direction"); ax.set_ylim(0, 1)
fig.savefig("results/figures/ed_oneway_distance.png", dpi=240, bbox_inches="tight"); fig.savefig("results/figures/ed_oneway_distance.pdf", bbox_inches="tight")
for o in out: print(f"{o[5]:.0f}-{o[6]:.0f} m  f={o[1]:.3f} [{o[2]:.3f},{o[3]:.3f}] n={o[4]}")
