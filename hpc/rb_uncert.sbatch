#!/bin/bash
# Uncertainty extension: 75% weekday subsamples at 15-minute bins for the eight bike-share
# headline units, 100 seeds each, 10 seeds per array task (tasks 0-79).
#SBATCH --job-name=rbunc
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16 BIN_MIN=15
U=(citibike divvy_mj bluebikes g_lon g_seo g_mex g_bue g_baw); i=$SLURM_ARRAY_TASK_ID
u=${U[$((i / 10))]}; s0=$(( (i % 10) * 10 + 1 )); s1=$((s0 + 9))
echo "unit=$u seeds=$s0-$s1 start=$(date -Is)"
${PYTHON:-python} src/rb_boot.py $u $s0 $s1
echo "end=$(date -Is)"
