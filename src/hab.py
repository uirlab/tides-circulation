"""H-HAB confirmatory: do habitual pairs carry the loop excess?
Usage (project root): python src/hab.py <city> <osm> [strata e.g. 0,2] [by_month]
Writes results/hab_<city>[_<strata>].json. Power check runs first; the real statistic only if it passes.
"""
import json
import os
import sys

import numpy as np
from scipy.special import expit

city, osm = sys.argv[1], sys.argv[2]
strata = tuple(int(x) for x in sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] != "all" else None
by_month = len(sys.argv) > 4 and sys.argv[4] == "by_month"
os.environ["CITY"] = city
if by_month:
    os.environ["BY_MONTH"] = "1"
sys.path.insert(0, "src")
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
import loop_map as L  # noqa: E402
import mechanism_share as M  # noqa: E402

D0 = S.load(); K = D0["K"]
sid = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
D = D0 if strata is None else {k: (v[np.isin(D0["st"], strata)] if isinstance(v, np.ndarray) and v.shape == D0["lo"].shape else v) for k, v in D0.items()}
grp = D.get("month", np.zeros(D["lo"].size, np.int64))

lo, hi, eps, se, t, _ = L.pair_residuals(D)
MIN_T = int(os.environ.get("MIN_T", "30")); SLOT = int(os.environ.get("SLOT_MIN", "30")); SENS = os.environ.get("SENS", "")
k = t >= MIN_T
lo, hi, eps, se, t = lo[k], hi[k], eps[k], se[k], t[k]
key = lo * K + hi

p_trip = np.zeros(D["lo"].size)
for g in np.unique(grp):
    for b in range(I.NB):
        for s in range(S.NS):
            m = (grp == g) & (D["b"] == b) & (D["st"] == s)
            if m.sum() < 2:
                continue
            plo, phi, n1, tt, inv = I.aggregate(D["lo"][m], D["hi"][m], D["y"][m], K)
            u = I.fit_bt(plo, phi, n1, tt, K, I.LAM)
            p_trip[np.flatnonzero(m)] = expit(u[plo] - u[phi])[inv]
acc = np.zeros(key.size); rng = np.random.default_rng(13)
for _ in range(2):
    Dn = dict(D); Dn["y"] = (rng.random(D["lo"].size) < p_trip).astype(float)
    l0, h0, e0, s0, t0, _ = L.pair_residuals(Dn)
    pos = {kk: i for i, kk in enumerate(l0 * K + h0)}
    for i, kk in enumerate(key):
        j = pos.get(kk)
        acc[i] += (1 - (e0[j] / s0[j]) ** 2) if j is not None else 0.0
e_p = (eps / se) ** 2 - 1 + acc / 2

# habit R from trip times only (directions pooled); for by_month units: per month, then averaged
pk = D["lo"] * K + D["hi"]
sel = np.isin(pk, key)
slot = (D["tm"] // SLOT).astype(np.int64)
Rs = []
for g in np.unique(grp):
    mg = sel & (grp == g)
    days_g = np.unique(D["day"][mg]).size
    code = np.unique((pk[mg] * 64 + slot[mg]) * 64 + D["day"][mg])
    ps, cnt = np.unique(code // 64, return_counts=True)
    mx = {}
    for p_, c_ in zip(ps // 64, cnt):
        if c_ > mx.get(p_, 0):
            mx[p_] = c_
    Rs.append(np.array([mx.get(kk, 0) for kk in key]) / max(days_g, 1))
R = np.mean(Rs, axis=0)

Dv, okd, und = M.covariate(osm, sid[lo], sid[hi])
ok = okd
y, R, t_, und_, aD = e_p[ok], R[ok], t[ok], und[ok], np.abs(Dv[ok])
dec = np.minimum((np.argsort(np.argsort(t_)) * 10) // len(t_), 9)
Xc = np.c_[np.eye(10)[dec], np.log(und_), aD]


def beta_R(yv, Rv):
    z = (Rv - Rv.mean()) / Rv.std()
    X = np.c_[Xc, z]
    b = np.linalg.lstsq(X, yv, rcond=None)[0]
    return b[-1] / yv.std()


def perm_within(Rv, rng):
    out = Rv.copy()
    for d_ in range(10):
        m = dec == d_
        out[m] = rng.permutation(Rv[m])
    return out


def test(yv, Rv, rng, B=2000, normal=False):
    obs = beta_R(yv, Rv)
    null = np.array([beta_R(yv, perm_within(Rv, rng)) for _ in range(B)])
    if normal:  # power-check p from a normal approximation to the permutation null
        from scipy.stats import norm
        p = float(norm.sf((obs - null.mean()) / null.std()))
    else:
        p = float((1 + (null >= obs).sum()) / (B + 1))
    top = Rv >= np.quantile(Rv, 0.9)
    enr = float((yv[top].sum() / yv.sum()) / top.mean())
    return obs, p, enr


rng = np.random.default_rng(14)
if os.environ.get("DUMP_PAIRS"):  # descriptive dump for figures only (no test, no verdict)
    np.savez_compressed(f"results/figdata_hab_{city}.npz", R=R, e=y, t=t_, und=und_)
    print("dumped", y.size); sys.exit(0)
# power check
hit = rej = 0
zR = (R - R.mean()) / R.std()
for i in range(100):
    yb = y
    Rp = perm_within(R, rng)  # destroy real association, keep within-decile distribution
    ypl = yb + 0.15 * y.std() * ((Rp - Rp.mean()) / Rp.std())
    o, p, _ = test(ypl, Rp, rng, B=200, normal=True)
    hit += (o > 0 and p < 0.001)
    o2, p2, _ = test(yb, perm_within(R, rng), rng, B=200, normal=True)
    rej += (o2 > 0 and p2 < 0.001)  # false SUPPORTED
power = dict(planted_supported=hit / 100, unplanted_false_supported=rej / 100)
power["pass"] = power["planted_supported"] >= 0.8 and power["unplanted_false_supported"] <= 0.05
out = dict(city=city, strata=strata, n_pairs=int(ok.sum()), power=power)
if power["pass"]:
    o, p, enr = test(y, R, rng)
    verdict = "SUPPORTED" if (o > 0 and p < 0.001 and enr >= 1.5) else ("REJECTED" if (o <= 0 or p >= 0.05) else "INCONCLUSIVE")
    out.update(beta_R=float(o), p_perm=p, enrichment_top_decile=enr, verdict=verdict,
               excess_share_top_decile=float(y[R >= np.quantile(R, 0.9)].sum() / y.sum()))
else:
    out["note"] = "power check failed; real statistic not computed"
tag = "" if strata is None else "_" + "".join(map(str, strata))
print(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
json.dump(out, open(f"results/hab_{city}{tag}{SENS}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
