"""Are transfer pairs less mirrored?. Uses the estimator of src/ampm.py.
Run from the project root :
    python src/ampm_xfer.py g_xj6 data/processed/xfer_pairs.npz synthetic      # gate, primary unit only
    python src/ampm_xfer.py <unit> <tau file> real
Outputs: results/ampm/xfer_<unit>_{synthetic,real}.json"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import expit

unit, tau_file, stage = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["CITY"] = unit; os.environ["BIN_MIN"] = "60"
sys.path.insert(0, os.path.dirname(__file__))
import ampm as A  # noqa: E402
import irrev as I  # noqa: E402

N_SPLIT, N_BOOT, R_SYN, N_SPLIT_SYN, N_BOOT_SYN = 20, 1000, 20, 4, 200


def groups():
    """Transfer share for each eligible pair (matched by sorted station-id pair, as in src/xfer.py); top decile flag."""
    names = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
    X = np.load(tau_file, allow_pickle=True); idx = {p: i for i, p in enumerate(X["pair"])}
    K = A.D["K"]; lo, hi = A.G["keys"] // K, A.G["keys"] % K; a, b = names[lo], names[hi]
    pk = np.where(a < b, np.char.add(np.char.add(a, "\t"), b), np.char.add(np.char.add(b, "\t"), a))
    j = np.array([idx.get(p, -1) for p in pk]); has = j >= 0
    tau = np.full(lo.size, np.nan); tau[has] = X["tau"][j[has]]
    top = has & (tau >= np.nanquantile(tau, 0.9)); other = has & ~top
    return tau, top, other


def stats(E, w, top, other):
    r1 = A.rstar(E, w * top)[0]; r0 = A.rstar(E, w * other)[0]
    return r1, r0, r1 - r0


def summarise(E, top, other, n_boot, seed):
    w = A.G["wt"]; r1, r0, d = stats(E, w, top, other)
    ucl, inv = np.unique(A.G["cluster"], return_inverse=True); rng = np.random.default_rng(seed); bs = []
    for _ in range(n_boot):
        m = np.bincount(rng.integers(0, ucl.size, ucl.size), minlength=ucl.size)[inv].astype(float)
        bs.append(stats(E, w * m, top, other))
    bs = np.array(bs); ci = lambda c: [float(x) for x in np.nanpercentile(bs[:, c], [2.5, 97.5])]
    sup = bool(d > 0 and ci(2)[0] > 0)
    verdict = "STRONG" if (sup and r1 > -0.2) else "SUPPORTED" if sup else "NOT SUPPORTED"
    return dict(r_transfer=float(r1), ci_transfer=ci(0), r_other=float(r0), ci_other=ci(1), delta=float(d), ci_delta=ci(2),
                n_transfer=int(top.sum()), n_other=int(other.sum()), verdict=verdict)


def _real_split(seed):
    return A.split_eps(A.D["y"], seed, ("AM", "PM"))


def _syn(r):
    scen, rep = r
    rng = np.random.default_rng(2000 + rep); P = A.G["keys"].size; top, other = A.G["top"], A.G["other"]
    j = np.searchsorted(A.G["keys"], A.G["key"]); ok = (j < P) & (A.G["keys"][np.minimum(j, P - 1)] == A.G["key"])
    am, pm = A.D["per"] == 0, A.D["per"] == 1
    mir = np.zeros(P); one_am = np.zeros(P); one_pm = np.zeros(P)
    for grp in (top, other):
        cand = np.flatnonzero(grp); pick = rng.choice(cand, int(round(0.3 * cand.size)), replace=False)
        s = rng.choice([-1.0, 1.0], pick.size)
        if scen == "power" and grp is top:
            h = pick.size // 2; one_am[pick[:h]] = s[:h]; one_pm[pick[h:]] = s[h:]
        else:
            mir[pick] = s
    tr = lambda v: np.where(ok, v[np.minimum(j, P - 1)], 0.0)
    shift = A.DELTA * (tr(mir) * (am.astype(float) - pm.astype(float)) + tr(one_am) * am + tr(one_pm) * pm)
    y = (rng.random(A.D["lo"].size) < expit(A.G["eta0"] + shift)).astype(float)
    E = [A.split_eps(y, sd, ("AM", "PM")) for sd in range(N_SPLIT_SYN)]
    o = summarise(E, top, other, N_BOOT_SYN, 9 + rep)
    return dict(scenario=scen, rep=rep, **o)


def main():
    A.setup(); os.makedirs(A.OUT, exist_ok=True)
    tau, top, other = groups(); A.G.update(top=top, other=other)
    meta = dict(unit=unit, tau_file=tau_file, n_eligible=int(A.G["keys"].size), n_with_tau=int(np.isfinite(tau).sum()),
                tau_top_decile_min=float(np.nanquantile(tau, 0.9)))
    print(meta, flush=True)
    if stage == "synthetic":
        A.G["eta0"] = A.tide_eta()
        with Pool(I.NWORKERS) as pool:
            res = pool.map(_syn, [(s, r) for s in ("null", "power") for r in range(R_SYN)])
        fp = sum(x["verdict"] != "NOT SUPPORTED" for x in res if x["scenario"] == "null")
        pw = sum(x["verdict"] != "NOT SUPPORTED" for x in res if x["scenario"] == "power")
        out = dict(meta, replicates=res, false_supported=fp, power_supported=pw, gate=bool(fp <= 2 and pw >= 16))
        json.dump(out, open(f"{A.OUT}/xfer_{unit}_synthetic.json", "w"), indent=1); print(fp, pw, "GATE", out["gate"], flush=True)
    else:
        if not json.load(open(f"{A.OUT}/xfer_g_xj6_synthetic.json"))["gate"]:
            print("synthetic gate failed; real test not run"); return
        with Pool(min(I.NWORKERS, N_SPLIT)) as pool:
            E = pool.map(_real_split, range(N_SPLIT))
        o = dict(meta, **summarise(E, top, other, N_BOOT, 13))
        # descriptive: r* by tau quintile
        q = np.full(tau.size, -1); f = np.isfinite(tau); q[f] = np.minimum((np.argsort(np.argsort(tau[f])) * 5) // f.sum(), 4)
        o["r_by_tau_quintile"] = [float(A.rstar(E, A.G["wt"] * (q == k))[0]) for k in range(5)]
        json.dump(o, open(f"{A.OUT}/xfer_{unit}_real.json", "w"), indent=1); print(json.dumps(o, indent=1), flush=True)


if __name__ == "__main__":
    main()
