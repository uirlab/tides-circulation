"""Step 1: share of the loop field explained by street direction.
Usage (project root, after src/street_graph.py):  python src/mechanism_share.py
Writes results/1b_mechanism_share.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MIN_T = 30


def residuals(city):
    os.environ["CITY"] = city
    for m in ("irrev", "irrev_strat", "loop_map"):
        sys.modules.pop(m, None)
    import irrev_strat as S  # noqa: E402
    import loop_map as L  # noqa: E402
    D = S.load()
    lo, hi, eps, se, t, _ = L.pair_residuals(D)
    d = np.load(f"data/processed/{city}_202605_weekday_trips.npz", allow_pickle=True)
    sid = d["station_id"].astype(str)
    k = t >= MIN_T
    return sid[lo[k]], sid[hi[k]], eps[k], se[k], t[k]


def covariate(osm, a_ids, b_ids):
    P = np.load(f"data/processed/legal_paths_{osm}.npz")
    pos = {s: i for i, s in enumerate(P["station_id"].astype(str))}
    L = P["L"]
    ia = np.array([pos.get(s, -1) for s in a_ids]); ib = np.array([pos.get(s, -1) for s in b_ids])
    ok = (ia >= 0) & (ib >= 0)
    lab = np.full(a_ids.size, np.inf); lba = np.full(a_ids.size, np.inf)
    lab[ok] = L[ia[ok], ib[ok]]; lba[ok] = L[ib[ok], ia[ok]]
    ok &= np.isfinite(lab) & np.isfinite(lba) & (lab > 0) & (lba > 0)
    Dv = np.full(a_ids.size, np.nan)
    Dv[ok] = np.log(lba[ok] / lab[ok])
    und = np.where(ok, np.minimum(lab, lba), np.nan)
    return Dv, ok, und


def fit_beta(eps, se, D):
    w = 1 / se**2
    return float((w * D * eps).sum() / (w * D * D).sum())


def es(eps, se, D, beta):
    w = 1 / se**2
    n = eps.size
    den = (w * eps**2).sum() - n
    return float(1 - ((w * (eps - beta * D) ** 2).sum() - n) / den)


def verdict(beta, e, lo):
    if beta > 0 and e >= 0.30 and lo > 0:
        return "MAJOR"
    if beta > 0 and e >= 0.10 and lo > 0:
        return "MATERIAL"
    return "MINOR"


def boot_es(eps, se, D, beta, B=1000, seed=3):
    rng = np.random.default_rng(seed)
    n = eps.size
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v.append(es(eps[i], se[i], D[i], beta))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def city_set(city, osm):
    a, b, eps, se, t = residuals(city)
    D, ok, und = covariate(osm, a, b)
    return dict(a=a[ok], b=b[ok], eps=eps[ok], se=se[ok], t=t[ok], D=D[ok], und=und[ok], n_all=int(a.size))


def synthetic(may, jun, rng):
    """Spec synthetic check on the realised June pair set (and May set for fitting)."""
    def sim(S, beta):
        return beta * S["D"] + rng.standard_normal(S["D"].size) * S["se"]
    # choose beta_true so that expected ES on June = 0.10: ES ~ beta^2 sum w D^2 / (beta^2 sum w D^2 + ... ) -> solve
    wD2 = (jun["D"] ** 2 / jun["se"] ** 2).sum()
    # with eps = beta D + noise: expected excess = beta^2 * sum w D^2; explained = same -> ES ~ 1 when model true.
    # To get ES = 0.10 we add an unexplained cyclic-like component c with excess 9x the explained part.
    out = {}
    wDm = (may["D"] ** 2 / may["se"] ** 2).sum()

    def draw(es_target):
        if es_target is None:  # null: loops present, unrelated to D
            return (sim(may, 0.0) + rng.standard_normal(may["D"].size) * may["se"],
                    sim(jun, 0.0) + rng.standard_normal(jun["D"].size) * jun["se"])
        b0 = 0.1
        r = (1 - es_target) / es_target  # unexplained / explained excess
        xj = np.sqrt(r * b0**2 * wD2 / jun["D"].size) * jun["se"]
        xm = np.sqrt(r * b0**2 * wDm / may["D"].size) * may["se"]
        return (sim(may, b0) + rng.standard_normal(may["D"].size) * xm, sim(jun, b0) + rng.standard_normal(jun["D"].size) * xj)

    for label, target in (("es10", 0.10), ("es15", 0.15), ("null", None)):
        mat = det = 0
        for _ in range(200):
            em, ej = draw(target)
            beta = fit_beta(em, may["se"], may["D"])
            e = es(ej, jun["se"], jun["D"], beta)
            lo = boot_es(ej, jun["se"], jun["D"], beta, B=100, seed=int(rng.integers(1e9)))[0]
            mat += verdict(beta, e, lo) in ("MATERIAL", "MAJOR"); det += lo > 0
        out[label] = dict(ge_material=mat / 200, ci_excludes_0=det / 200)
    out["pass"] = (out["es10"]["ci_excludes_0"] >= 0.80 and out["es15"]["ge_material"] >= 0.80
                   and out["null"]["ge_material"] <= 0.05)
    return out


def main():
    rng = np.random.default_rng(2026)
    may = city_set("citibike", "citibike")
    jun = city_set("citibike_jun", "citibike")
    syn = synthetic(may, jun, rng)
    out = dict(synthetic=syn, n_pairs_may=int(may["D"].size), n_pairs_jun=int(jun["D"].size),
               n_pairs_may_before_covariate=may["n_all"], n_pairs_jun_before_covariate=jun["n_all"])
    if not syn["pass"]:
        out["note"] = "synthetic check failed; real ES not computed (amend spec first)"
        json.dump(out, open("results/1b_mechanism_share.json", "w"), indent=1); print(json.dumps(out, indent=1)); return
    beta = fit_beta(may["eps"], may["se"], may["D"])
    e = es(jun["eps"], jun["se"], jun["D"], beta)
    ci = boot_es(jun["eps"], jun["se"], jun["D"], beta)
    out["primary"] = dict(beta_may=beta, ES_out_june=e, ci95=ci, verdict=verdict(beta, e, ci[0]))
    # descriptive
    q = np.nanpercentile(jun["D"], np.linspace(0, 100, 11))
    dec = []
    for lo_, hi_ in zip(q[:-1], q[1:]):
        m = (jun["D"] >= lo_) & (jun["D"] <= hi_)
        w = 1 / jun["se"][m] ** 2
        dec.append(dict(D=[float(lo_), float(hi_)], n=int(m.sum()), mean_eps=float((w * jun["eps"][m]).sum() / w.sum())))
    out["deciles_june"] = dec
    short = jun["und"] <= 1000
    out["ES_out_june_le_1km"] = es(jun["eps"][short], jun["se"][short], jun["D"][short], beta)
    out["frac_pairs_D_nonzero_june"] = float(np.mean(np.abs(jun["D"]) > 1e-3))
    other = {}
    for city, osm in (("divvy_mj", "divvy"), ("bluebikes", "bluebikes")):
        C = city_set(city, osm)
        bc = fit_beta(C["eps"], C["se"], C["D"])
        other[city] = dict(n=int(C["D"].size), beta_own=bc, ES_in_sample=es(C["eps"], C["se"], C["D"], bc),
                           ES_transfer_nyc_beta=es(C["eps"], C["se"], C["D"], beta))
    out["other_cities"] = other
    json.dump(out, open("results/1b_mechanism_share.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
