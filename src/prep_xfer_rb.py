"""R5 replication of H-XFER on other 2023 month pairs.
Parameterised copy of src/prep_xfer.py: identical logic, with the covariate month (XM), outcome month (XO), holidays and
minimum May rides per pair (XMIN) as arguments. Run from the repository root (cluster recommended):
    python src/prep_xfer_rb.py <XM> <XO> [XMIN] [pairs_only]
Outputs: data/processed/g_x<XO>_weekday_trips.npz (all rides), g_n<XO>_weekday_trips.npz (non-transfer rides),
data/processed/xfer_pairs_rb_<XM><XO>[_min<XMIN>].npz"""
import sys

import numpy as np
import pandas as pd

XM, XO = int(sys.argv[1]), int(sys.argv[2])
XMIN = int(sys.argv[3]) if len(sys.argv) > 3 else 30
PAIRS_ONLY = len(sys.argv) > 4 and sys.argv[4] == "pairs_only"
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, "src")
import prep_global as P  # noqa: E402
import prep_xfer_lib as X  # noqa: E402  (full(), canon(), xfer(), cells() from src/prep_xfer.py, unchanged)

HOL23 = {3: [], 4: ["2023-04-03", "2023-04-04", "2023-04-05"], 5: ["2023-05-01"], 6: ["2023-06-22", "2023-06-23"],
         9: ["2023-09-29"], 10: ["2023-10-09", "2023-10-10"], 11: [], 12: []}  # fixed in the spec before data were opened

if not PAIRS_ONLY:
    fj = X.full(XO); xj = X.canon(X.xfer(XO), set(fj.s) | set(fj.e))
    cf, cx = X.cells(fj), X.cells(xj)
    cn = (cf - cx.reindex(cf.index).fillna(0)).clip(lower=0)
    print("outcome month", XO, "rides", int(cf.sum()), "transfer", int(cx.sum()))
    for pre, c in (("x", cf), ("n", cn)):
        key = f"{pre}{XO:02d}"; name = f"g_{key}"
        P.HOL[key], P.YEAR[key], P.MONTH[key] = HOL23[XO], 2023, XO
        c = c[c > 0].reset_index(); t = c.h + pd.Timedelta(minutes=30)
        P.finish(name, t, t, c.s, c.e, np.zeros(len(c), int), weight=c.n.values, group=False)
else:
    fj = X.full(XO)

fm = X.full(XM); xm = X.canon(X.xfer(XM), set(fm.s) | set(fm.e))
hol = {pd.Timestamp(h).date() for h in HOL23[XM]}
tm = pd.to_datetime(fm.t); keep = (tm.dt.weekday < 5) & (~tm.dt.date.isin(hol)) & (fm.s != fm.e)
fm = fm[keep]; xt = pd.to_datetime(xm.t); xm = xm[(xt.dt.weekday < 5) & (~xt.dt.date.isin(hol)) & (xm.s != xm.e)]
und = lambda d: pd.Series([tuple(sorted(p)) for p in zip(d.s, d.e)])
nm = und(fm).value_counts(); nx = und(xm).value_counts()
tau = (nx.reindex(nm.index).fillna(0) / nm)[nm >= XMIN]
dx = pd.Series(list(zip(xm.s, xm.e))).value_counts()
sec = pd.to_timedelta(fj.dur, errors="coerce").dt.total_seconds()
dur = pd.Series(sec.values, index=und(fj).values).groupby(level=0).median()
keys = np.array(["\t".join(k) for k in tau.index])
tag = f"{XM:02d}{XO:02d}" + (f"_min{XMIN}" if XMIN != 30 else "")
np.savez_compressed(f"data/processed/xfer_pairs_rb_{tag}.npz", pair=keys, tau=tau.values, n_may=nm.reindex(tau.index).values,
                    x_ab=np.array([dx.get(k, 0) for k in tau.index]), x_ba=np.array([dx.get((k[1], k[0]), 0) for k in tau.index]),
                    dur=dur.reindex(tau.index).values)
print("covariate month", XM, "pairs with >=", XMIN, "rides", len(tau), "median tau", float(tau.median()),
      "share of rides that are transfers", float(nx.sum() / nm.sum()))
