#!/bin/bash
#SBATCH --job-name=xfer2
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
$PY src/xfer.py g_xj6
CITY=g_xn6 BIN_MIN=60 $PY src/irrev_strat.py all
$PY src/xfer.py g_xn6
echo "end=$(date -Is)"
