"""Descriptive per-pair circulation field for figure maps (Fig 2a). No decision rule.
For one CITY: per (bin x stratum) Bradley-Terry fit, residual log-odds beyond the tide for each station pair,
Fisher-weighted over bins and strata (same estimator as src/loop_map.py; directions only, magnitudes are
shrinkage-biased; see ).
Run from the project root:  CITY=g_lon python src/circ_pairs.py   (Taipei: CITY=g_tpe BIN_MIN=60)
Output: results/figdata_circ_<CITY>.npz  (lo, hi, eps, se, t; eps > 0 means lo -> hi is favoured beyond the tide)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import irrev as I  # noqa: E402
import irrev_strat as S  # noqa: E402
from loop_map import pair_residuals  # noqa: E402

D = S.load()
lo, hi, eps, se, t, _ = pair_residuals(D)
np.savez_compressed(f"results/figdata_circ_{I.CITY}.npz", lo=lo, hi=hi, eps=eps, se=se, t=t)
print(I.CITY, "pairs", lo.size, "pairs with t>=30", int((t >= 30).sum()))
