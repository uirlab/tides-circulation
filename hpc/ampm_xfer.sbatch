#!/bin/bash
# Transfer pairs vs other pairs in the mirror test.
# Task 0: synthetic gate on g_xj6; tasks 1-4: real units (submitted with a dependency on task 0).
#SBATCH --job-name=ampmx
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16
U=(g_xj6 g_xj6 g_x04 g_x10 g_x12)
T=(xfer_pairs xfer_pairs xfer_pairs_rb_0304 xfer_pairs_rb_0910 xfer_pairs_rb_1112)
i=$SLURM_ARRAY_TASK_ID; stage=$([ "$i" = 0 ] && echo synthetic || echo real)
echo "unit=${U[$i]} tau=${T[$i]} stage=$stage start=$(date -Is)"
${PYTHON:-python} -X faulthandler src/ampm_xfer.py ${U[$i]} data/processed/${T[$i]}.npz $stage
echo "end=$(date -Is)"
