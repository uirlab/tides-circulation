#!/bin/bash
# : tide-only simulation through the windowed decomposition. sbatch --array=101-103 src/aicr_coarse.sbatch
#SBATCH --job-name=coarse
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=20 CITY=citibike SIM_TIDE=$SLURM_ARRAY_TASK_ID
${PYTHON:-python} src/tidal_share.py > /dev/null
echo "end=$(date -Is)"
