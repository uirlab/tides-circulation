#!/bin/bash
# Sensitivity analyses: one array task per line of src/sens_tasks.txt.
#SBATCH --job-name=sens
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NWORKERS=32
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" src/sens_tasks.txt)
eval "export $(echo "$LINE" | sed -E 's/ ARGS=.*//')"
ARGS=$(echo "$LINE" | sed -nE 's/.* ARGS=(.*)$/\1/p' | tr -d '"')
echo "task: $LINE"
if [ -n "${PREP:-}" ] && [ ! -f data/processed/${PREP}_weekday_trips.npz ]; then
  ${PYTHON:-python} src/prep_unit.py "$PREP" citibike 2025-05
fi
${PYTHON:-python} src/$PY $ARGS
echo "end $(date -Is)"
