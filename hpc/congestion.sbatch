#!/bin/bash
# Step 3 on a Slurm cluster: power check, then the real DiD (which self-gates on the power result). One process per stage.
#SBATCH --job-name=congestion
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=${SLURM_CPUS_PER_TASK:-64}
echo "host=$(hostname) start=$(date -Is)"
${PYTHON:-python} src/tlc_irrev.py power
${PYTHON:-python} src/tlc_irrev.py real
echo "end=$(date -Is)"
