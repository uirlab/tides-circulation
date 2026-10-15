"""Deviance-based (shrinkage-free) explained share of the loop excess by street direction D. Run from the repository root (cluster recommended):  NWORKERS=32 python src/dev_es.py  -> results/devES.json
Synthetic validation runs first; the real ES_dev only if it passes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import expit, xlogy

sys.path.insert(0, "src")
NW = int(os.environ.get("NWORKERS", "8"))
GR = (np.sqrt(5) - 1) / 2


def month_cells(city):
    os.environ["CITY"] = city
    for m in ("irrev", "irrev_strat"):
        sys.modules.pop(m, None)
    import irrev as I
    import irrev_strat as S
    D = S.load(); K = D["K"]
    sid = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
    P = np.load("data/processed/legal_paths_citibike.npz")
    pos = {s: i for i, s in enumerate(P["station_id"].astype(str))}; L = P["L"]
    ix = np.array([pos.get(s, -1) for s in sid])
    nd = D["ndays"]
    cells = []
    for b in range(I.NB):
        for s in range(S.NS):
            m = (D["b"] == b) & (D["st"] == s)
            if m.sum() < 2:
                continue
            key = D["lo"][m] * K + D["hi"][m]
            uk, inv = np.unique(key, return_inverse=True)
            plo, phi = uk // K, uk % K
            T = np.bincount(D["day"][m] * uk.size + inv, minlength=nd * uk.size).reshape(nd, uk.size).astype(np.uint16)
            N1 = np.bincount(D["day"][m] * uk.size + inv, weights=D["y"][m], minlength=nd * uk.size).reshape(nd, uk.size).astype(np.uint16)
            ia, ib = ix[plo], ix[phi]
            ok = (ia >= 0) & (ib >= 0)
            lab = np.where(ok, L[np.maximum(ia, 0), np.maximum(ib, 0)], np.inf); lba = np.where(ok, L[np.maximum(ib, 0), np.maximum(ia, 0)], np.inf)
            fin = np.isfinite(lab) & np.isfinite(lba) & (lab > 0) & (lba > 0)
            Dp = np.where(fin, np.log(np.where(fin, lba, 1) / np.where(fin, lab, 1)), 0.0)
            ids = np.char.add(np.char.add(sid[plo].astype("U16"), "|"), sid[phi].astype("U16"))
            cells.append(dict(plo=plo, phi=phi, N1=N1, T=T, D=Dp, ids=ids, K=K))
    return cells, I.LAM


def g2(plo, phi, n1, t, K, lam, off):
    import irrev as I
    ok = t > 0
    plo, phi, n1, t, off = plo[ok], phi[ok], n1[ok], t[ok], off[ok]
    u = I.fit_bt(plo, phi, n1, t, K, lam, off=off)
    p = np.clip(expit(u[plo] - u[phi] + off), 1e-12, 1 - 1e-12)
    n0 = t - n1
    return float(2 * (xlogy(n1, n1) - xlogy(n1, t * p) + xlogy(n0, n0) - xlogy(n0, t * (1 - p))).sum()), u


def cell_dev(c, beta, lam, wday=None, n1_override=None):
    w = np.ones(c["T"].shape[0]) if wday is None else wday
    t = w @ c["T"].astype(float); n1 = (w @ c["N1"].astype(float)) if n1_override is None else n1_override
    return g2(c["plo"], c["phi"], n1, t, c["K"], lam, beta * c["D"])[0]


_C = {}


def _init(payload):
    _C.update(payload)


def _dev_task(args):
    which, beta, wseed = args
    cells, lam = _C[which], _C["lam"]
    w = None
    if wseed is not None:
        nd = cells[0]["T"].shape[0]
        w = np.bincount(np.random.default_rng(wseed).integers(0, nd, nd), minlength=nd).astype(float)
    return sum(cell_dev(c, beta, lam, w, c.get("n1sim")) for c in cells), sum(cell_dev(c, 0.0, lam, w, c.get("n1sim")) for c in cells)


def total_dev(pool, which, beta):
    return pool.map(_dev_task, [(which, beta, None)])[0][0]


def golden(f, a=-2.0, b=2.0, tol=1e-3):
    c, d = b - GR * (b - a), a + GR * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc; c = b - GR * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + GR * (b - a); fd = f(d)
    return (a + b) / 2


def e0(cells, lam, rng, B=20):
    tot = 0.0
    for c in cells:
        t = c["T"].sum(0).astype(float); n1 = c["N1"].sum(0).astype(float)
        _, u = g2(c["plo"], c["phi"], n1, t, c["K"], lam, np.zeros(t.size))
        ok = t > 0
        p = expit(u[c["plo"]] - u[c["phi"]])
        vals = [g2(c["plo"], c["phi"], rng.binomial(t.astype(int), p).astype(float), t, c["K"], lam, np.zeros(t.size))[0] for _ in range(B)]
        tot += float(np.mean(vals))
    return tot


def sim_month(cells, lam, beta_true, G, rng):
    out = []
    for c in cells:
        t = c["T"].sum(0).astype(float); n1 = c["N1"].sum(0).astype(float)
        _, u = g2(c["plo"], c["phi"], n1, t, c["K"], lam, np.zeros(t.size))
        g = np.array([G.get(x, 0.0) for x in c["ids"]])
        p = expit(u[c["plo"]] - u[c["phi"]] + beta_true * c["D"] + g)
        cc = dict(c); cc["n1sim"] = rng.binomial(t.astype(int), p).astype(float)
        out.append(cc)
    return out


def _sim_task(args):
    i, beta_true = args
    rng = np.random.default_rng(700 + i)
    keys = _C["keys"]
    G = dict(zip(keys, rng.standard_normal(len(keys)) * _C["sdG"]))
    lam = _C["lam"]
    may = sim_month(_C["may"], lam, beta_true, G, rng); jun = sim_month(_C["jun"], lam, beta_true, G, rng)
    f = lambda bb: sum(cell_dev(c, bb, lam, None, c["n1sim"]) for c in may)
    bh = golden(f)
    gq = sum(cell_dev(c, 0.0, lam, None, c["n1sim"]) for c in jun)
    gd = sum(cell_dev(c, bh, lam, None, c["n1sim"]) for c in jun)
    go = sum(cell_dev(c, beta_true, lam, None, c["n1sim"]) for c in jun)
    den = gq - _C["e0_jun"]
    return dict(i=i, beta_true=beta_true, beta_hat=bh, es=(gq - gd) / den, oracle=(gq - go) / den)


def main():
    t0 = time.time()
    may, lam = month_cells("citibike"); jun, _ = month_cells("citibike_jun")
    rng = np.random.default_rng(81)
    E0 = e0(jun, lam, rng)
    gq_real = sum(cell_dev(c, 0.0, lam) for c in jun)
    E_real = gq_real - E0
    wsum = sum(float((c["T"].sum(0) * 0.25).sum()) for c in jun)  # ~ sum t p(1-p) with p ~ 0.5 (scale only)
    sdG = float(np.sqrt(E_real / wsum))
    wD2 = sum(float((c["T"].sum(0) * 0.25 * c["D"] ** 2).sum()) for c in jun)
    beta_plant = float(np.sqrt(0.10 * E_real / max(wD2, 1e-9)))
    keys = sorted({x for c in may + jun for x in c["ids"]})
    out = dict(E0_jun=E0, G2QS_jun=gq_real, loop_excess_jun=E_real, sdG=sdG, beta_plant=beta_plant)
    print("prep", out, round(time.time() - t0), flush=True)
    payload = dict(may=may, jun=jun, lam=lam, keys=keys, sdG=sdG, e0_jun=E0)
    _C.update(payload)  # fork inherits copy-on-write (no per-worker pickling)
    with Pool(NW) as pool:
        sims = pool.map(_sim_task, [(i, beta_plant) for i in range(20)] + [(100 + i, 0.0) for i in range(20)])
        pl = [s for s in sims if s["beta_true"] > 0]; nu = [s for s in sims if s["beta_true"] == 0]
        bias = float(np.median([s["es"] - s["oracle"] for s in pl])); null_med = float(np.median([abs(s["es"]) for s in nu]))
        brec = float(np.median([s["beta_hat"] / s["beta_true"] for s in pl]))
        out["synthetic"] = dict(median_bias=bias, median_oracle=float(np.median([s["oracle"] for s in pl])), null_median_abs=null_med,
                                beta_recovery=brec, pass_=bool(abs(bias) <= 0.01 and null_med <= 0.01 and abs(brec - 1) <= 0.2))
        print("synthetic", out["synthetic"], round(time.time() - t0), flush=True)
        if not out["synthetic"]["pass_"]:
            out["note"] = "synthetic validation failed; real ES_dev not computed"
            json.dump(out, open("results/devES.json", "w"), indent=1); return
        # ---- real ----
        f = lambda bb: sum(pool.map(_dev_one, [("may", bb, k) for k in range(len(may))]))
        bh = golden(f)
        gd = sum(pool.map(_dev_one, [("jun", bh, k) for k in range(len(jun))]))
        es = (gq_real - gd) / E_real
        boots = pool.map(_boot, [(r, bh) for r in range(500)])
        ci = [float(np.percentile([(q - d) / E_real for q, d in boots], 2.5)), float(np.percentile([(q - d) / E_real for q, d in boots], 97.5))]
    verdict = "MAJOR" if (bh > 0 and es >= 0.30 and ci[0] > 0) else ("MATERIAL" if (bh > 0 and es >= 0.10 and ci[0] > 0) else "MINOR")
    out["real"] = dict(beta_hat_may=bh, ES_dev_june=float(es), ci95=ci, verdict=verdict)
    out["runtime_s"] = round(time.time() - t0)
    json.dump(out, open("results/devES.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


def _dev_one(args):
    which, beta, k = args
    return cell_dev(_C[which][k], beta, _C["lam"])


def _boot(args):
    r, bh = args
    nd = _C["jun"][0]["T"].shape[0]
    w = np.bincount(np.random.default_rng(9000 + r).integers(0, nd, nd), minlength=nd).astype(float)
    q = sum(cell_dev(c, 0.0, _C["lam"], w) for c in _C["jun"]); d = sum(cell_dev(c, bh, _C["lam"], w) for c in _C["jun"])
    return q, d


if __name__ == "__main__":
    main()
