"""SI Note 1 toy example: a changing tide that is reversible in every window looks rotational once aggregated.
Exact expected flows (no sampling noise) on a triangle of stations A, B, C. Two cases: (1) only the tide (alpha) changes
between windows; (2) the tide and the busy pairs (gamma) change.
Each window k is tide-only: mu_ab^(k) = alpha_a^(k) * gamma_ab with gamma symmetric (quasi-symmetric, reversible).
Reports, per window and for the day total:
  (i)  cycle affinity  log[(mu_AB mu_BC mu_CA)/(mu_BA mu_CB mu_AC)]  (0 iff the cycle criterion holds),
  (ii) Hodge curl share of the net flow  A = mu - mu^T  (as in Aoki et al. 2022), on the triangle.
Run from the project root: python3 src/toy_tide.py -> results/toy_tide.json, results/figures/si_toy_tide.png
"""
import json

import numpy as np

S = ["A", "B", "C"]
E = [(0, 1), (1, 2), (2, 0)]  # oriented cycle A->B->C->A
gamma = np.array([[0, 10.0, 4.0], [10.0, 0, 6.0], [4.0, 6.0, 0]])  # symmetric pair intensities (distance, popularity)
# three windows with different tides: morning (A crowded, flows leave A), midday (flat), evening (C crowded)
alphas = {"morning": np.array([3.0, 1.0, 1.0]), "midday": np.array([1.0, 1.0, 1.0]), "evening": np.array([1.0, 1.0, 3.0])}


def flows(alpha):
    return alpha[:, None] * gamma


def cycle_affinity(mu):
    f = mu[0, 1] * mu[1, 2] * mu[2, 0]; r = mu[1, 0] * mu[2, 1] * mu[0, 2]
    return float(np.log(f / r))


def hodge_curl_share(mu):
    """Net flow on the 3 oriented edges; gradient = least-squares fit s_j - s_i; curl = remainder (on a triangle, the
    remainder is the uniform circulation around the cycle)."""
    a = np.array([mu[i, j] - mu[j, i] for i, j in E])
    G = np.zeros((3, 3))
    for e, (i, j) in enumerate(E):
        G[e, i], G[e, j] = -1, 1
    s, *_ = np.linalg.lstsq(G, a, rcond=None)
    curl = a - G @ s
    return float((curl ** 2).sum() / (a ** 2).sum()), a.tolist(), curl.tolist()


def run(gammas):
    res = {}; total = np.zeros((3, 3))
    for k, al in alphas.items():
        mu = al[:, None] * gammas[k]; total += mu
        a = np.array([mu[i, j] - mu[j, i] for i, j in E])
        res[k] = dict(cycle_affinity=cycle_affinity(mu), net_flow=a.tolist(),
                      hodge_curl_share=(hodge_curl_share(mu)[0] if np.abs(a).sum() > 0 else 0.0))
    sh, a, c = hodge_curl_share(total)
    res["day_total"] = dict(cycle_affinity=cycle_affinity(total), net_flow=a, hodge_curl_share=sh,
                            odds_ratio=float(np.exp(cycle_affinity(total))))
    return res


# Case 1: only the tide changes (alpha_k varies; pair intensities gamma fixed)
case1 = run({k: gamma for k in alphas})
# Case 2: the tide AND the busy pairs change (morning activity concentrated on A-B, evening on B-C)
g_m = np.array([[0, 16.0, 2.0], [16.0, 0, 3.0], [2.0, 3.0, 0]])
g_e = np.array([[0, 3.0, 2.0], [3.0, 0, 16.0], [2.0, 16.0, 0]])
case2 = run({"morning": g_m, "midday": gamma, "evening": g_e})
# Control for the single-window net-flow curl: uniform pair intensities
gu = np.where(gamma > 0, 6.0, 0)
mu_u = alphas["morning"][:, None] * gu
out = dict(case1_tide_only_changes=case1, case2_tide_and_busy_pairs_change=case2,
           control_uniform_gamma_morning_curl_share=hodge_curl_share(mu_u)[0])
json.dump(out, open("results/toy_tide.json", "w"), indent=1)
for cname in ("case1_tide_only_changes", "case2_tide_and_busy_pairs_change"):
    print("==", cname)
    for k, v in out[cname].items():
        print("  ", k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print("control uniform gamma, morning curl share:", out["control_uniform_gamma_morning_curl_share"])

# ---------------- SI figure ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

SLATE, TERRA, INK, INK2, MUTE, SURF = "#3b6a9b", "#c9744d", "#2b2a27", "#6b6862", "#b9b4aa", "#fbfaf7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "figure.facecolor": SURF, "savefig.facecolor": SURF})
P = {0: (0.5, 0.92), 1: (0.08, 0.18), 2: (0.92, 0.18)}
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.8), gridspec_kw=dict(wspace=0.05, hspace=0.7))
cases = [("case1_tide_only_changes", {k: gamma for k in alphas}, "1  tide changes; busy pairs fixed"),
         ("case2_tide_and_busy_pairs_change", {"morning": g_m, "midday": gamma, "evening": g_e}, "2  tide and busy pairs change")]
for r, (cname, gms, title) in enumerate(cases):
    total = sum(alphas[k][:, None] * gms[k] for k in alphas)
    for c, k in enumerate(list(alphas) + ["day_total"]):
        ax = axs[r, c]; ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.12, 1.1); ax.set_aspect("equal"); ax.axis("off")
        mu = total if k == "day_total" else alphas[k][:, None] * gms[k]
        al = alphas.get(k, total.sum(1) / total.sum(1).mean())
        for i, p in P.items():
            ax.add_patch(Circle(p, 0.07, fc=SLATE if (k != "day_total" and al[i] > 1) else SURF, ec=INK2, lw=1.0, zorder=3))
            ax.text(p[0], p[1], S[i], ha="center", va="center", fontsize=6.5, color="white" if (k != "day_total" and al[i] > 1) else INK, zorder=4)
        for i, j in E:
            net = mu[i, j] - mu[j, i]
            if abs(net) < 1e-9:
                continue
            a_, b_ = (P[i], P[j]) if net > 0 else (P[j], P[i])
            ax.add_patch(FancyArrowPatch(a_, b_, arrowstyle="-|>,head_length=4,head_width=2.5", lw=0.6 + 0.08 * abs(net),
                                         color=MUTE, shrinkA=9, shrinkB=9, zorder=2))
            mx, my = (P[i][0] + P[j][0]) / 2, (P[i][1] + P[j][1]) / 2
            ax.text(mx, my, f"{abs(net):.0f}", fontsize=5.6, color=INK2, ha="center", va="center", bbox=dict(fc=SURF, ec="none", pad=0.4))
        d = out[cname][k]
        aff = d["cycle_affinity"]; col = TERRA if abs(aff) > 1e-9 else INK2
        ax.text(0.5, -0.1, f"cycle affinity {aff:+.3f}\nnet-flow curl {d['hodge_curl_share']:.0%}", ha="center", va="top", fontsize=5.8, color=col)
        ax.set_title(k.replace("_", " "), fontsize=7, color=INK)
    axs[r, 0].text(-0.15, 1.2, title, transform=axs[r, 0].transAxes, fontsize=7.4, fontweight="bold", color=INK)
fig.savefig("results/figures/si_toy_tide.png", dpi=240, bbox_inches="tight")
print("written si_toy_tide.png")
