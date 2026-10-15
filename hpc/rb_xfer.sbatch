#!/bin/bash
# R5 transfer-test replications: one array task per line of src/rb_xfer_tasks.txt (1-indexed).
#SBATCH --job-name=rbxfer
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" src/rb_xfer_tasks.txt); eval "export $LINE"
echo "task: $LINE  start=$(date -Is)"
${PYTHON:-python} src/xfer_rb.py $UNIT
echo "end=$(date -Is)"
