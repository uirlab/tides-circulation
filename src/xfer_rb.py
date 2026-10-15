"""R5 replication: src/xfer.py with the covariate file, primary/secondary role,
enrichment quantile and output name as environment variables (XPAIRS, TOPQ, OUTTAG); the statistics are unchanged.
Original docstring: H-XFER test. Mirrors src/hab.py.
Usage (repository root): python src/xfer.py g_xj6   (T1 + T3)   |   python src/xfer.py g_xn6   (T2)
Power check runs first; the real statistic only if it passes. Writes results/xfer_<unit>.json."""
import json
import os
import sys

import numpy as np
from scipy.special import expit
from scipy.stats import binomtest, norm

unit = sys.argv[1]
PRIMARY = unit == "g_xj6" or unit.startswith("g_x")  # all-rides units: T1 + T3; non-transfer units g_n..: T2
os.environ["CITY"] = unit; os.environ["BIN_MIN"] = "60"
sys.path.insert(0, "src")
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
import loop_map as L  # noqa: E402

D = S.load(); K = D["K"]
names = np.load(I.DATA, allow_pickle=True)["station_id"].astype(str)
lo, hi, eps, se, t, _ = L.pair_residuals(D)
k = t >= 30; lo, hi, eps, se, t = lo[k], hi[k], eps[k], se[k], t[k]
key = lo * K + hi

# debiasing correction, exactly as H-HAB
p_trip = np.zeros(D["lo"].size)
for b in range(I.NB):
    for s in range(S.NS):
        m = (D["b"] == b) & (D["st"] == s)
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

# habit R (60-min slots, from June trip times only)
pk = D["lo"] * K + D["hi"]; sel = np.isin(pk, key); slot = (D["tm"] // 60).astype(np.int64)
days = np.unique(D["day"][sel]).size
code = np.unique((pk[sel] * 64 + slot[sel]) * 64 + D["day"][sel]); ps, cnt = np.unique(code // 64, return_counts=True)
mx = {}
for p_, c_ in zip(ps // 64, cnt):
    mx[p_] = max(mx.get(p_, 0), c_)
R = np.array([mx.get(kk, 0) for kk in key]) / max(days, 1)

# May covariates
X = np.load(os.environ.get("XPAIRS", "data/processed/xfer_pairs.npz"), allow_pickle=True)
idx = {p: i for i, p in enumerate(X["pair"])}
a_n, b_n = names[lo], names[hi]
srt = np.where(a_n < b_n, 0, 1)  # 0: lo is first in the sorted name pair
pk_s = np.where(srt == 0, np.char.add(np.char.add(a_n, "\t"), b_n), np.char.add(np.char.add(b_n, "\t"), a_n))
j = np.array([idx.get(p, -1) for p in pk_s]); ok = (j >= 0)
jj = j[ok]; tau = X["tau"][jj]; dur = X["dur"][jj]; ok2 = np.isfinite(dur) & (dur > 0)
y, tau, dur, R_, t_ = e_p[ok][ok2], tau[ok2], dur[ok2], R[ok][ok2], t[ok][ok2]
dec = np.minimum((np.argsort(np.argsort(t_)) * 10) // len(t_), 9)
zR = (R_ - R_.mean()) / R_.std()
Xc = np.c_[np.eye(10)[dec], np.log(dur), zR]


def beta(yv, tv):
    z = (tv - tv.mean()) / tv.std()
    bb = np.linalg.lstsq(np.c_[Xc, z], yv, rcond=None)[0]
    return bb[-1] / yv.std()


def perm(tv, rng):
    out = tv.copy()
    for d_ in range(10):
        m = dec == d_
        out[m] = rng.permutation(tv[m])
    return out


def test(yv, tv, rng, B=2000, normal=False):
    obs = beta(yv, tv); null = np.array([beta(yv, perm(tv, rng)) for _ in range(B)])
    p = float(norm.sf((obs - null.mean()) / null.std())) if normal else float((1 + (null >= obs).sum()) / (B + 1))
    top = tv >= np.quantile(tv, float(os.environ.get("TOPQ", "0.9")))
    return obs, p, float((yv[top].sum() / yv.sum()) / top.mean())


rng = np.random.default_rng(14)
hit = rej = 0
for _ in range(100):
    tp = perm(tau, rng)
    ypl = y + 0.15 * y.std() * ((tp - tp.mean()) / tp.std())
    o, p, _ = test(ypl, tp, rng, B=200, normal=True); hit += (o > 0 and p < 0.001)
    o2, p2, _ = test(y, perm(tau, rng), rng, B=200, normal=True); rej += (o2 > 0 and p2 < 0.001)
power = dict(planted_supported=hit / 100, unplanted_false_supported=rej / 100)
power["pass"] = power["planted_supported"] >= 0.8 and power["unplanted_false_supported"] <= 0.05
out = dict(unit=unit, n_pairs=int(y.size), median_tau=float(np.median(tau)), power=power)
if power["pass"]:
    o, p, enr = test(y, tau, rng)
    if PRIMARY:
        verdict = "SUPPORTED" if (o > 0 and p < 0.001 and enr >= 1.2) else ("REJECTED" if (o <= 0 or p >= 0.05) else "INCONCLUSIVE")
    else:
        verdict = "SUPPORTED" if (o > 0 and p < 0.01) else "NOT SUPPORTED"
    out.update(beta_tau=float(o), p_perm=p, enrichment_top_decile=enr, verdict=verdict,
               excess_share_top_decile=float(y[tau >= np.quantile(tau, float(os.environ.get("TOPQ", "0.9")))].sum() / y.sum()))
    # descriptive curve for the figure: enrichment by tau decile
    q = np.minimum((np.argsort(np.argsort(tau)) * 10) // len(tau), 9)
    out["enrichment_by_tau_decile"] = [float((y[q == d_].sum() / y.sum()) / (q == d_).mean()) for d_ in range(10)]
    out["tau_decile_medians"] = [float(np.median(tau[q == d_])) for d_ in range(10)]
else:
    out["note"] = "power check failed; real statistic not computed"

if PRIMARY:  # T3 direction
    xab, xba = X["x_ab"][jj][ok2], X["x_ba"][jj][ok2]
    zz = (eps / se)[ok][ok2]; ep = eps[ok][ok2]; sr = srt[ok][ok2]
    fav_first = np.where(sr == 0, ep > 0, ep < 0)  # June favoured direction is sorted-first -> sorted-second
    pb = np.array([binomtest(int(a), int(a + b), 0.5).pvalue if a + b > 0 else 1.0 for a, b in zip(xab, xba)])
    m3 = (pb < 0.05) & (np.abs(zz) >= 2)
    agree = fav_first[m3] == (xab[m3] > xba[m3]); n3 = int(m3.sum())
    t3 = dict(n=n3, share_agree=float(agree.mean()) if n3 else None)
    if n3 >= 30:
        t3["p"] = float(binomtest(int(agree.sum()), n3, 0.5, alternative="greater").pvalue)
        t3["verdict"] = "SUPPORTED" if (t3["share_agree"] > 0.5 and t3["p"] < 0.01) else "NOT SUPPORTED"
    else:
        t3["verdict"] = "UNDERPOWERED"
    out["T3"] = t3
conv = lambda o: o.item() if hasattr(o, "item") else str(o)
print(json.dumps(out, indent=1, default=conv))
os.makedirs("results/rb", exist_ok=True)
json.dump(out, open(f"results/rb/xfer_{unit}{os.environ.get('OUTTAG', '')}.json", "w"), indent=1, default=conv)
