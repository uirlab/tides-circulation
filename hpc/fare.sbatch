#!/bin/bash
# Taipei fare-change analysis. STEP=prep | power | month (array 1-12) | combine
#   sbatch --export=STEP=prep hpc/fare.sbatch; then STEP=power, STEP=month with --array=1-12, then STEP=combine
#SBATCH --job-name=fare
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j-%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=${SLURM_CPUS_PER_TASK}
PY=${PYTHON:-python}
UNITS=(g_f2212 g_f2301 g_f2302 g_f2303 g_f2304 g_f2305 g_f2312 g_f2401 g_f2402 g_f2403 g_f2404 g_f2405)
echo "step=$STEP start=$(date -Is)"
case $STEP in
  prep) $PY src/prep_fare.py ;;
  power) $PY src/fare.py power ;;
  month) $PY src/fare.py month ${UNITS[$((SLURM_ARRAY_TASK_ID-1))]} ;;
  combine) $PY src/fare.py combine ;;
esac
echo "end=$(date -Is)"
