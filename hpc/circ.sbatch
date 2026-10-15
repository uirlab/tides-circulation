#!/bin/bash
# Fig 2a circulation fields, one city per array task:  sbatch --array=0-7 src/aicr_circ.sbatch
#SBATCH --job-name=circ
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
CITIES=(citibike divvy_mj bluebikes g_lon g_seo g_mex g_bue g_tpe)
C=${CITIES[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CITY=$C
[ "$C" = g_tpe ] && export BIN_MIN=60
[ "$C" = divvy_mj ] && export BY_MONTH=1
echo "city=$C start=$(date -Is)"
${PYTHON:-python} src/circ_pairs.py
echo "end=$(date -Is)"
