#!/bin/bash
# Run one stage of src/irrev.py on a Slurm cluster's cpu partition.
# Usage (on a Slurm cluster, from the repository root):
#     sbatch src/aicr_cpu.sbatch synthetic      # or: potentials | real
#SBATCH --job-name=tides
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
STAGE=${1:?stage required: potentials|synthetic|real|strata}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NWORKERS=${SLURM_CPUS_PER_TASK:-64}
echo "host=$(hostname) stage=$STAGE workers=$NWORKERS start=$(date -Is)"
if [ "$STAGE" = strata ]; then
  ${PYTHON:-python} src/irrev_strat.py all
else
  ${PYTHON:-python} src/irrev.py "$STAGE"
fi
echo "end=$(date -Is)"
