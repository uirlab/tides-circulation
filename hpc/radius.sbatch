#!/bin/bash
# Exploratory transit dose-response for Fig 3 (descriptive; no decision rule): H-MODE M1 at extra radii.
#   sbatch --array=0-3 src/aicr_radius.sbatch      (250 m and 600 m exist from the sensitivity run)
#SBATCH --job-name=radius
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -eo pipefail
R=(150 800 1200 2000); r=${R[$SLURM_ARRAY_TASK_ID]}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NWORKERS=16 RADIUS=$r SENS=_E1r$r M1_ONLY=1
${PYTHON:-python} src/modesub.py
echo "end $(date -Is)"
