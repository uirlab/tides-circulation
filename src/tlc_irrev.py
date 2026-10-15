"""Step 3: did NYC congestion pricing durably reorganise circulation?

Run from the repository root (cluster recommended), one stage per process:
    python src/tlc_irrev.py power    # synthetic injection check (must pass first)
    python src/tlc_irrev.py real     # main DiD + placebo + volume comparator (refuses to run unless power passed)
Residual field per period: per (stratum, 15-min bin) Bradley-Terry fit pooled over the period's weekdays;
residual log-odds beyond the potential, pooled over bins/strata with Fisher weights. Day-block bootstrap.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, os.path.dirname(__file__))
from irrev import fit_bt  # noqa: E402

KZ = 266
LAM = 0.1
NB, NS = 64, 2
B_REAL, B_SYN, R_SYN = 1000, 200, 20
MIN_PAIR = 200
NW = int(os.environ.get("NWORKERS", "8"))
PERIODS = [(ys, p) for ys in ("y22", "y23", "y24") for p in ("pre", "early", "late")]
CBD = set(json.load(open("data/processed/tlc_cbd_zones.json"))["cbd_zones"]) if os.path.exists("data/processed/tlc_cbd_zones.json") else set()
PIDX = {p: i for i, p in enumerate(PERIODS)}  # stable integer per period (Python's str hash is salted per process)
KEEP_F = "data/processed/tlc_keep_pairs.npy"


def keep_pairs():
    """Fixed pair universe (counts only, no outcomes): pairs with >= MIN_PAIR trips summed over all 9 periods."""
    if os.path.exists(KEEP_F):
        return np.load(KEEP_F)
    tot = np.zeros(KZ * KZ)
    for ys, per in PERIODS:
        d = np.load(f"data/processed/tlc_{ys}_{per}.npz")
        tot += np.bincount(d["lo"].astype(np.int64) * KZ + d["hi"].astype(np.int64), minlength=KZ * KZ)
    keep = np.flatnonzero(tot >= MIN_PAIR)
    np.save(KEEP_F, keep)
    return keep


KEEP = keep_pairs() if all(os.path.exists(f"data/processed/tlc_{ys}_{per}.npz") for ys, per in PERIODS) else np.arange(KZ * KZ)
NP = KEEP.size
POS = np.full(KZ * KZ, -1, np.int64); POS[KEEP] = np.arange(NP)


# ---------- per-period cell tables ----------

def cells(ys, per):
    """For each (stratum, bin): unique pair keys and per-day (n1, t) count matrices (days x pairs)."""
    d = np.load(f"data/processed/tlc_{ys}_{per}.npz")
    lo, hi, y, b, st, day = (d[k].astype(np.int64) for k in ("lo", "hi", "y", "b", "st", "day"))
    nd = int(d["days"].size)
    out = {}
    for s in range(NS):
        for k in range(NB):
            m = (st == s) & (b == k)
            if not m.any():
                continue
            key = lo[m] * KZ + hi[m]
            uk, inv = np.unique(key, return_inverse=True)
            idx = day[m] * uk.size + inv
            T = np.bincount(idx, minlength=nd * uk.size).reshape(nd, uk.size).astype(np.float32)
            N1 = np.bincount(idx, weights=y[m], minlength=nd * uk.size).reshape(nd, uk.size).astype(np.float32)
            out[(s, k)] = (uk, N1, T)
    return out, nd


def day_weights(ys, per, nd, B, seed):
    """Row 0 = point estimate (all ones); rows 1..B-1 = multinomial day-resampling counts."""
    rng = np.random.default_rng([seed, PIDX[(ys, per)]])
    Wm = np.ones((B, nd), np.float32)
    for r in range(1, B):
        Wm[r] = np.bincount(rng.integers(0, nd, nd), minlength=nd)
    return Wm


def field(cell_tab, Wm, inject=None, rng=None):
    """Residual field for each replicate row of Wm. Returns sums (B x NP): Sw*eps, Sw, St.
    inject: optional dict with 'pot' {(s,k): u}, 'F' (NP,), 'G' (NP,) -> resample directions (power check)."""
    B = Wm.shape[0]
    swe = np.zeros((B, NP)); sw = np.zeros((B, NP)); stt = np.zeros((B, NP))  # compact pair index (KEEP)
    for (s, k), (uk, N1, T) in cell_tab.items():
        plo, phi = (uk // KZ).astype(np.int64), (uk % KZ).astype(np.int64)
        cpos = POS[uk]
        if inject is not None:
            fg = np.where(cpos >= 0, inject["F"][np.maximum(cpos, 0)] + inject["G"][np.maximum(cpos, 0)], 0.0)
            eta0 = inject["pot"][(s, k)][plo] - inject["pot"][(s, k)][phi] + fg
            N1 = rng.binomial(T.astype(np.int64), expit(eta0)[None, :]).astype(np.float32)
        n1b = Wm @ N1; tb = Wm @ T
        for r in range(B):
            n1, t = n1b[r].astype(float), tb[r].astype(float)
            ok = t > 0
            u = fit_bt(plo[ok], phi[ok], n1[ok], t[ok], KZ, LAM)
            eta = u[plo[ok]] - u[phi[ok]]
            p = expit(eta)
            eps = logit((n1[ok] + 0.5) / (t[ok] + 1.0)) - eta
            w = t[ok] * p * (1 - p)
            ii = cpos[ok]; sel = ii >= 0
            swe[r, ii[sel]] += (w * eps)[sel]; sw[r, ii[sel]] += w[sel]; stt[r, ii[sel]] += t[ok][sel]
    return swe, sw, stt


def potentials(cell_tab):
    pot = {}
    for (s, k), (uk, N1, T) in cell_tab.items():
        plo, phi = (uk // KZ).astype(np.int64), (uk % KZ).astype(np.int64)
        n1, t = N1.sum(0).astype(float), T.sum(0).astype(float)
        pot[(s, k)] = fit_bt(plo, phi, n1, t, KZ, LAM)
    return pot


# ---------- stability statistics ----------

def wcorr(a, b, w):
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    return float((w * (a - ma) * (b - mb)).sum() / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()))


def pair_masks(stt0):
    """Fixed pair sets from point-estimate trip counts: >= MIN_PAIR trips in BOTH compared periods."""
    lo, hi = KEEP // KZ, KEEP % KZ
    touch = np.array([(a in CBD) or (b in CBD) for a, b in zip(lo, hi)])
    return touch


def stability(F1, F2, mask, kind="resid"):
    """F = (swe, sw, stt) for one replicate row. resid: corr of eps; vol: corr of log volume."""
    if kind == "resid":
        e1, e2 = F1[0][mask] / F1[1][mask], F2[0][mask] / F2[1][mask]
        w = 1.0 / (1.0 / F1[1][mask] + 1.0 / F2[1][mask])
        return wcorr(e1, e2, w)
    v1, v2 = np.log(F1[2][mask]), np.log(F2[2][mask])
    return wcorr(v1, v2, np.ones(mask.sum()))


def did_table(fields, treat, ctrl, touch):
    """fields[(ys, per)] = (swe, sw, stt) arrays (B x NP). Returns DiD estimates and bootstrap draws."""
    res = {}
    base = {}
    for ys in (treat, ctrl):
        t_all = [fields[(ys, p)][2][0] for p in ("pre", "early", "late")]
        base[ys] = t_all
    B = min(fields[k][0].shape[0] for k in fields)
    for kind in ("resid", "vol"):
        for sub in ("all", "cbd", "out"):
            for late in ("early", "late"):
                vals = []
                for r in range(B):
                    dd = 0.0
                    for ys, sgn in ((treat, 1), (ctrl, -1)):
                        f_pre = tuple(x[r] for x in fields[(ys, "pre")]); f_post = tuple(x[r] for x in fields[(ys, late)])
                        m = (base[ys][0] >= MIN_PAIR) & (fields[(ys, late)][2][0] >= MIN_PAIR)
                        m = m & (touch if sub == "cbd" else (~touch if sub == "out" else True))
                        dd += sgn * stability(f_pre, f_post, m, kind)
                    vals.append(dd)
                vals = np.array(vals)
                res[f"{kind}_{sub}_{late}"] = dict(est=float(vals[0]), ci=[float(np.percentile(vals[1:], 2.5)), float(np.percentile(vals[1:], 97.5))],
                                                   sd=float(vals[1:].std(ddof=1)))
    return res


def classify(R, kind):
    e, l = R[f"{kind}_all_early"], R[f"{kind}_all_late"]
    ce, co = R[f"{kind}_cbd_early"], R[f"{kind}_out_early"]
    changed = e["est"] < 0 and e["ci"][1] < 0 and ce["est"] < co["est"] and ce["ci"][1] < co["ci"][0]
    if not changed:
        return "NO EFFECT", changed
    durable = l["est"] < 0 and l["ci"][1] < 0 and abs(l["est"]) >= 0.5 * abs(e["est"])
    return ("DURABLE" if durable else "TRANSIENT"), changed


# ---------- stages ----------

def _real_task(args):
    ys, per, r0, r1 = args
    tab, nd = cells(ys, per)
    Wm = day_weights(ys, per, nd, B_REAL, seed=101)[r0:r1]
    return (ys, per, r0), field(tab, Wm)


def _gather(results, B):
    fields = {}
    for (ys, per, r0), (swe, sw, stt) in results:
        if (ys, per) not in fields:
            fields[(ys, per)] = [np.zeros((B, NP)), np.zeros((B, NP)), np.zeros((B, NP))]
        for i, a in enumerate((swe, sw, stt)):
            fields[(ys, per)][i][r0:r0 + a.shape[0]] = a
    return {k: tuple(v) for k, v in fields.items()}


def real():
    pw = json.load(open("results/3_congestion_power.json"))
    if not pw["requirement_met"]:
        print("Power requirement not met; real statistic not run (amend spec first).", flush=True); return
    t = time.time()
    chunk = 50
    tasks = [(ys, per, r0, min(r0 + chunk, B_REAL)) for ys, per in PERIODS for r0 in range(0, B_REAL, chunk)]
    with Pool(NW) as pool:
        results = pool.map(_real_task, tasks)
    fields = _gather(results, B_REAL)
    touch = pair_masks(None)
    main_ = did_table(fields, "y24", "y23", touch)
    plac = did_table(fields, "y23", "y22", touch)
    cls_r, ch_r = classify(main_, "resid"); cls_v, _ = classify(main_, "vol")
    cls_p, ch_p = classify(plac, "resid")
    final = "NO EFFECT (placebo fired)" if ch_p else cls_r
    zr = abs(main_["resid_all_early"]["est"]) / main_["resid_all_early"]["sd"]
    zv = abs(main_["vol_all_early"]["est"]) / main_["vol_all_early"]["sd"]
    h3 = (cls_r != cls_v) or (zr >= 2 * zv)
    out = dict(decision=dict(circulation=final, residual_class=cls_r, volume_class=cls_v, placebo_class=cls_p,
                             placebo_fired=bool(ch_p), H3_irreversibility_adds_info=bool(h3), z_resid_early=zr, z_vol_early=zv),
               main=main_, placebo=plac, n_cbd_zones=len(CBD), runtime_s=round(time.time() - t))
    json.dump(out, open("results/3_congestion_real.json", "w"), indent=1)
    print(json.dumps(out["decision"], indent=1), flush=True)


def _power_task(args):
    inj, ys, per, F, G = args
    tab, nd = cells(ys, per)
    pot = potentials(tab)
    rng = np.random.default_rng([inj, PIDX[(ys, per)]])
    Wm = day_weights(ys, per, nd, B_SYN, seed=900 + inj)
    return (inj, ys, per), field(tab, Wm, inject=dict(pot=pot, F=F, G=G), rng=rng)


def power():
    """Inject a cyclic-like field on CBD-touching pairs into the treatment year's early+late periods.
    Truth field F per year = that year's real pre-period residual field (point estimate), applied to all its periods."""
    t = time.time()
    touch = pair_masks(None)
    Fy = {}
    with Pool(NW) as pool:
        pre = pool.map(_real_task, [(ys, "pre", 0, 1) for ys in ("y23", "y24")])
    for (ys, per, _), (swe, sw, stt) in pre:
        Fy[ys] = np.where(sw[0] > 0, swe[0] / np.maximum(sw[0], 1e-12), 0.0)
    m = touch & (Fy["y24"] != 0)
    sdF = float(Fy["y24"][m].std())
    rng = np.random.default_rng(77)
    tasks = []
    for inj in range(R_SYN):
        G = np.zeros(NP)
        G[m] = rng.standard_normal(m.sum()) * np.sqrt(0.10) * sdF  # 10% of residual-field variance on CBD pairs
        for ys in ("y23", "y24"):
            for per in ("pre", "early", "late"):
                tasks.append((inj, ys, per, Fy[ys], G if (ys == "y24" and per != "pre") else np.zeros(NP)))
    with Pool(NW) as pool:
        results = pool.map(_power_task, tasks)
    by_inj = {}
    for (inj, ys, per), f in results:
        by_inj.setdefault(inj, {})[(ys, per)] = f
    classes = []
    for inj, fields in sorted(by_inj.items()):
        R = did_table(fields, "y24", "y23", touch)
        classes.append(classify(R, "resid")[0])
    n_dur = classes.count("DURABLE")
    out = dict(classes={c: classes.count(c) for c in set(classes)}, n_durable=n_dur, requirement_met=n_dur >= 0.8 * R_SYN,
               sd_F_cbd=sdF, injected_sd=float(np.sqrt(0.10) * sdF), B_boot=B_SYN, runtime_s=round(time.time() - t))
    json.dump(out, open("results/3_congestion_power.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    {"power": power, "real": real}[sys.argv[1]]()
