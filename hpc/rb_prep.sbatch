#!/bin/bash
# Review-stage robustness prep: one array task per line of src/rb_prep_tasks.txt (1-indexed).
#SBATCH --job-name=rbprep
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" src/rb_prep_tasks.txt)
echo "task: $CMD  start=$(date -Is)"
eval "${CMD/python/${PYTHON:-python}}"
echo "end=$(date -Is)"
