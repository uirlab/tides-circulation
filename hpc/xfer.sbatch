#!/bin/bash
# H-XFER: prep, pipeline gates, then T1+T3 (all rides) and T2 (non-transfer rides).
#SBATCH --job-name=xfer
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16
PY=${PYTHON:-python}
[ -f data/processed/xfer_pairs.npz ] || $PY src/prep_xfer.py
for u in g_xj6 g_xn6; do
  CITY=$u BIN_MIN=60 $PY src/irrev_strat.py all
  $PY src/xfer.py $u
done
echo "end=$(date -Is)"
