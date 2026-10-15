#!/bin/bash
# Review-stage robustness pipeline runs: one array task per line of src/rb_run_tasks.txt (environment for irrev_strat.py all).
#SBATCH --job-name=rbrun
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
mkdir -p results/1_replication_rb data/processed/rb
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" src/rb_run_tasks.txt)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=32
eval "export $LINE"
echo "task: $LINE  start=$(date -Is)"
${PYTHON:-python} src/irrev_strat.py all
echo "end=$(date -Is)"
