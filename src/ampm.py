"""Morning-evening mirror test.

Per station pair, the residual log-odds of direction beyond the fitted tide (per window x stratum Bradley-Terry fit,
Fisher-weighted pooling, as in src/loop_map.py) in the morning peak, the evening peak and midday, each on two halves
of the weekdays. r* is the disattenuated correlation between the morning and evening residual signals, built only from
cross-half covariances: about -1 for mirrored same-mode round trips, 0 for one-way legs, +1 for static asymmetry.

Run from the project root :
    CITY=citibike python src/ampm.py synthetic      # gate; must pass before any real unit
    CITY=<unit> [BIN_MIN=60] python src/ampm.py real
Outputs: results/ampm/<CITY>_synthetic.json, results/ampm/<CITY>_real.json
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import expit

sys.path.insert(0, os.path.dirname(__file__))
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402

PER = {"AM": (420, 600), "PM": (960, 1140), "MID": (660, 840)}  # minutes after midnight, [start, end)
MIN_T, MIN_PEAK, N_SPLIT, N_BOOT = 30, 5, 20, 1000  # >= 30 trips over both peaks, >= 5 in each
PLANT_FRAC, DELTA, R_SYN, N_SPLIT_SYN, N_BOOT_SYN = 0.30, 0.5, 20, 4, 200  # plant 30% of eligible pairs
OUT = "results/ampm"
D = None
G = {}  # globals shared with forked workers


def setup():
    global D
    D = S.load()
    tm, K = D["tm"], D["K"]
    D["per"] = np.full(tm.size, -1, np.int64)
    for j, (a, b) in enumerate(PER.values()):
        D["per"][(tm >= a) & (tm < b)] = j
    key = D["lo"] * K + D["hi"]
    tot = {}
    for j in (0, 1):  # eligibility: >= 30 trips in each peak, full month, all strata
        uk, c = np.unique(key[D["per"] == j], return_counts=True)
        tot[j] = dict(zip(uk.tolist(), c.tolist()))
    keys = np.array(sorted(k for k, c in tot[0].items()
                           if c >= MIN_PEAK and tot[1].get(k, 0) >= MIN_PEAK and c + tot[1][k] >= MIN_T), np.int64)
    tam = np.array([tot[0][k] for k in keys], float); tpm = np.array([tot[1][k] for k in keys], float)
    # pairs weighted by information, the harmonic combination of their morning and evening trip counts
    G.update(keys=keys, key=key, cluster=keys // K, grp=D["b"] * S.NS + D["st"], wt=tam * tpm / (tam + tpm))


def eps_pairs(y, mask):
    """Fisher-weighted residual log-odds for the eligible pairs, from trips in mask (NaN where absent)."""
    keys, K, P = G["keys"], D["K"], G["keys"].size
    num, den = np.zeros(P), np.zeros(P)
    idx = np.flatnonzero(mask)
    g = G["grp"][idx]; order = np.argsort(g, kind="stable"); idx, g = idx[order], g[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    for sel in np.split(idx, cuts):
        if sel.size < 2:
            continue
        plo, phi, n1, t, _ = I.aggregate(D["lo"][sel], D["hi"][sel], y[sel], K)
        u = I.fit_bt(plo, phi, n1, t, K, I.LAM)
        eta = u[plo] - u[phi]; p = expit(eta)
        # one-step score estimate of the pair's log-odds shift, sum(n1 - t p) / sum(t p (1 - p)),
        # instead of the pseudo-count residual, whose shrinkage bias has the sign of the tide and so reverses between peaks
        w = t * p * (1 - p)
        j = np.searchsorted(keys, plo * K + phi); ok = (j < P) & (keys[np.minimum(j, P - 1)] == plo * K + phi)
        np.add.at(num, j[ok], (n1 - t * p)[ok]); np.add.at(den, j[ok], w[ok])
    out = np.full(P, np.nan); out[den > 0] = num[den > 0] / den[den > 0]
    return out


def split_eps(y, seed, periods):
    """eps[period][half] for one random balanced split of the weekdays."""
    days = np.random.default_rng(seed).permutation(D["ndays"]); h1 = np.isin(D["day"], days[: D["ndays"] // 2])
    return {p: [eps_pairs(y, (D["per"] == j) & h) for h in (h1, ~h1)] for p, j in zip(PER, range(3)) if p in periods}


def wcov(x, z, w):
    ok = ~(np.isnan(x) | np.isnan(z)) & (w > 0)  # NaN-safe
    if ok.sum() < 10:
        return np.nan
    w, x, z = w[ok], x[ok], z[ok]
    mx, mz = np.average(x, weights=w), np.average(z, weights=w)
    return float(np.average((x - mx) * (z - mz), weights=w))


def rstar(E, w, a="AM", b="PM"):
    """Split-averaged cross-half covariances -> disattenuated correlation, plus the two reliabilities.
    Amendment 2: within each split all four covariances use the same pairs (defined in both halves of both periods)."""
    c, ra, rb = [], [], []
    for e in E:
        ok = ~np.isnan(e[a][0] + e[a][1] + e[b][0] + e[b][1]); ww = w * ok
        c.append(0.5 * (wcov(e[a][0], e[b][1], ww) + wcov(e[a][1], e[b][0], ww)))
        ra.append(wcov(e[a][0], e[a][1], ww)); rb.append(wcov(e[b][0], e[b][1], ww))
    c_ab, r_a, r_b = np.nanmean(c), np.nanmean(ra), np.nanmean(rb)
    r = c_ab / np.sqrt(r_a * r_b) if (r_a > 0 and r_b > 0) else np.nan
    return r, r_a, r_b


def boot(E, n_boot, seed, pairs=(("AM", "PM"),)):
    cl = G["cluster"]; ucl, inv = np.unique(cl, return_inverse=True); rng = np.random.default_rng(seed)
    res = {f"{a}_{b}": [] for a, b in pairs}; rel = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, ucl.size, ucl.size), minlength=ucl.size)[inv].astype(float) * G["wt"]
        for a, b in pairs:
            r, ra, rb = rstar(E, w, a, b); res[f"{a}_{b}"].append(r)
            if (a, b) == ("AM", "PM"):
                rel.append((ra, rb))
    return {k: np.array(v) for k, v in res.items()}, np.array(rel)


def summarise(E, n_boot, seed, pairs):
    w1 = G["wt"]
    bs, rel = boot(E, n_boot, seed, pairs)
    out = {}
    for a, b in pairs:
        r, ra, rb = rstar(E, w1, a, b); v = bs[f"{a}_{b}"]; v = v[~np.isnan(v)]
        out[f"{a}_{b}"] = dict(r_star=float(r), ci95=[float(x) for x in np.percentile(v, [2.5, 97.5])] if v.size else [np.nan, np.nan],
                               rel_a=float(ra), rel_b=float(rb), n_boot_defined=int(v.size))
    lo_rel = np.nanpercentile(rel, 2.5, axis=0)
    signal = bool(lo_rel[0] > 0 and lo_rel[1] > 0)
    x = out["AM_PM"]; r, (l, h) = x["r_star"], x["ci95"]
    # STATIC category added; ONE-WAY made two-sided
    verdict = ("NO SIGNAL" if not signal else "MIRRORED" if (r <= -0.5 and h < -0.2) else
               "STATIC" if (r >= 0.5 and l > 0.2) else "ONE-WAY" if (abs(r) < 0.2 and l > -0.5 and h < 0.5) else "MIXED")
    out.update(signal=signal, reliability_lower=[float(v) for v in lo_rel], verdict=verdict)
    return out


def _real_split(seed):
    return split_eps(D["y"], seed, ("AM", "PM", "MID"))


def _syn_task(args):
    scen, r = args
    rng = np.random.default_rng(1000 + r)
    keys = G["keys"]; P = keys.size
    shift = np.zeros(D["lo"].size)
    if scen != "null":
        n_plant = int(round(PLANT_FRAC * P)); pick = rng.choice(P, n_plant, replace=False); s = rng.choice([-1.0, 1.0], n_plant)
        j = np.searchsorted(keys, G["key"]); ok = (j < P) & (keys[np.minimum(j, P - 1)] == G["key"])
        sgn = np.zeros(P); sgn[pick] = s; ps = np.zeros(D["lo"].size); ps[ok] = sgn[j[ok]]
        am, pm = D["per"] == 0, D["per"] == 1
        if scen == "mirror":
            shift = DELTA * ps * (am.astype(float) - pm.astype(float))
        elif scen == "oneway":  # disjoint halves of the planted pairs, morning-only and evening-only
            half = np.zeros(P, bool); half[pick[: n_plant // 2]] = True; hp = np.zeros(D["lo"].size, bool); hp[ok] = half[j[ok]]
            shift = DELTA * ps * ((am & hp) | (pm & ~hp))
        elif scen == "static":
            shift = DELTA * ps * (am | pm)
    y = (rng.random(D["lo"].size) < expit(G["eta0"] + shift)).astype(float)
    E = [split_eps(y, sd, ("AM", "PM")) for sd in range(N_SPLIT_SYN)]
    o = summarise(E, N_BOOT_SYN, 7 + r, (("AM", "PM"),))
    return dict(scenario=scen, rep=r, r_star=o["AM_PM"]["r_star"], ci95=o["AM_PM"]["ci95"], signal=o["signal"], verdict=o["verdict"])


def tide_eta():
    """Per-trip log-odds of lo->hi under the per-window x stratum tide fit (no circulation)."""
    eta = np.zeros(D["lo"].size); K = D["K"]
    m = D["per"] >= 0
    idx = np.flatnonzero(m); g = G["grp"][idx]; o = np.argsort(g, kind="stable"); idx, g = idx[o], g[o]
    for sel in np.split(idx, np.flatnonzero(np.diff(g)) + 1):
        plo, phi, n1, t, inv = I.aggregate(D["lo"][sel], D["hi"][sel], D["y"][sel], K)
        u = I.fit_bt(plo, phi, n1, t, K, I.LAM); eta[sel] = (u[plo] - u[phi])[inv]
    return eta


def main():
    stage = sys.argv[1]
    setup(); os.makedirs(OUT, exist_ok=True)
    meta = dict(city=I.CITY, bin_min=I.W, n_pairs=int(G["keys"].size), periods=PER, min_trips=MIN_T)
    print(meta, flush=True)
    if stage == "synthetic":
        G["eta0"] = tide_eta()
        tasks = [(s, r) for s in ("null", "mirror", "oneway", "static") for r in range(R_SYN)]
        with Pool(I.NWORKERS) as pool:
            res = pool.map(_syn_task, tasks)
        crit = {"null": lambda x: (not x["signal"]) or abs(x["r_star"]) <= 0.3, "mirror": lambda x: x["signal"] and x["r_star"] <= -0.6,  # unchanged
                "oneway": lambda x: x["signal"] and abs(x["r_star"]) <= 0.2, "static": lambda x: x["signal"] and x["r_star"] >= 0.6}
        passed = {s: int(sum(crit[s](x) for x in res if x["scenario"] == s)) for s in crit}
        out = dict(meta, replicates=res, passed=passed, gate=all(v >= 16 for v in passed.values()))
        json.dump(out, open(f"{OUT}/{I.CITY}_synthetic.json", "w"), indent=1)
        print(passed, "GATE", out["gate"], flush=True)
    elif stage == "real":
        syn = json.load(open(f"{OUT}/citibike_synthetic.json"))
        if not syn["gate"]:
            print("synthetic gate failed; real test not run"); return
        with Pool(min(I.NWORKERS, N_SPLIT)) as pool:
            E = pool.map(_real_split, range(N_SPLIT))
        o = summarise(E, N_BOOT, 11, (("AM", "PM"), ("AM", "MID"), ("PM", "MID")))
        json.dump(dict(meta, **o), open(f"{OUT}/{I.CITY}_real.json", "w"), indent=1)
        print(json.dumps(dict(meta, **o), indent=1), flush=True)


if __name__ == "__main__":
    main()
