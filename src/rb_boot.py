"""R2a day-block subsampling (75% of weekdays without replacement). Run from the repository root (cluster recommended):
    python src/rb_boot.py <unit> <seed_from> <seed_to>
For each seed: draw 75% of the unit's weekdays without replacement (rng seed = seed, so both modes of a city get the same days),
write data/processed/rb/bs_<unit>_<seed>_weekday_trips.npz, run potentials + real at 60-minute bins with the gate of the
original unit (SKIP_SYN_GATE=1), keep only the per-bin h, z, g, gs and null mean in results/rb/boot/<unit>_<seed>.json,
then delete the replicate's trip and potentials files.
Environment (uncertainty extension, ): BIN_MIN (default 60) sets the bin width; with BIN_MIN other
than 60 the outputs go to results/rb/boot<BIN_MIN>/ and the replicate files carry the bin width in their name."""
import glob
import json
import os
import subprocess
import sys

import numpy as np

unit, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
BIN = os.environ.get("BIN_MIN", "60")
BOOT = "results/rb/boot" + ("" if BIN == "60" else BIN)
src = f"data/processed/{unit}_202605_weekday_trips.npz"
src = src if os.path.exists(src) else f"data/processed/{unit}_weekday_trips.npz"
d = dict(np.load(src, allow_pickle=True))
n = d["days"].size
by_day = [np.flatnonzero(d["day"] == k) for k in range(n)]
os.makedirs("data/processed/rb", exist_ok=True); os.makedirs(BOOT, exist_ok=True); os.makedirs("results/1_replication_rb", exist_ok=True)
for seed in range(s0, s1 + 1):
    out_json = f"{BOOT}/{unit}_{seed}.json"
    if os.path.exists(out_json):
        continue
    # subsample 75% of weekdays without replacement (resampling with replacement duplicates days, and the
    # duplicated trips add noise to the observed deviance that the null expectation does not remove, inflating h')
    draw = np.sort(np.random.default_rng(seed).choice(n, size=int(round(0.75 * n)), replace=False))
    idx = np.concatenate([by_day[k] for k in draw]); newday = np.concatenate([np.full(by_day[k].size, j) for j, k in enumerate(draw)])
    rep = {k: (v[idx] if (isinstance(v, np.ndarray) and v.shape[:1] == d["o"].shape) else v) for k, v in d.items()}
    rep["day"] = newday.astype(np.int16); rep["days"] = np.array([f"{d['days'][k]}#{j}" for j, k in enumerate(draw)])
    name = f"rb/bs_{unit}_{seed}" + ("" if BIN == "60" else f"_b{BIN}")
    np.savez_compressed(f"data/processed/{name}_weekday_trips.npz", **rep)
    env = dict(os.environ, CITY=name, BIN_MIN=BIN, SKIP_SYN_GATE="1")
    for stage in ("potentials", "real"):
        subprocess.run([sys.executable, "src/irrev_strat.py", stage], env=env, check=True, stdout=subprocess.DEVNULL)
    res = json.load(open(f"results/1_replication_{name}_real.json"))
    keep = dict(unit=unit, seed=seed, bin_min=int(BIN), draw=draw.tolist(), n_trips=int(idx.size), median_h=res["decision"]["median_h"],
                per_bin=[{k: b[k] for k in ("bin", "h", "z", "g", "gs", "null_mean")} for b in res["per_bin"]])
    json.dump(keep, open(out_json, "w"))
    for f in glob.glob(f"data/processed/{name}_*") + glob.glob(f"results/1_replication_{name}_*"):
        os.remove(f)
    print(unit, seed, round(keep["median_h"], 4), flush=True)
