#!/bin/bash
# : bike systems at 60-min bins.  sbatch --array=0-6 src/aicr_res60.sbatch
#SBATCH --job-name=res60
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
CITIES=(citibike divvy_mj bluebikes g_lon g_seo g_mex g_bue)
C=${CITIES[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=${SLURM_CPUS_PER_TASK:-8}
export CITY=$C BIN_MIN=60 SENS=_b60
[ "$C" = divvy_mj ] && export BY_MONTH=1
echo "city=$C start=$(date -Is)"
${PYTHON:-python} src/irrev_strat.py all
echo "end=$(date -Is)"
