#!/bin/bash
# Morning-evening mirror test. Task 0: synthetic gate (New York);
# tasks 1-11: real units, submitted with a dependency on task 0 (the script refuses to run if the gate failed).
#SBATCH --job-name=ampm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16
U=(citibike citibike divvy_mj bluebikes g_lon g_seo g_mex g_bue g_baw g_ybk g_tpex g_bar)
B=(15 15 15 15 15 15 15 15 15 60 60 60)
i=$SLURM_ARRAY_TASK_ID; stage=$([ "$i" = 0 ] && echo synthetic || echo real)
echo "unit=${U[$i]} bin=${B[$i]} stage=$stage start=$(date -Is)"
CITY=${U[$i]} BIN_MIN=${B[$i]} ${PYTHON:-python} -X faulthandler src/ampm.py $stage
echo "end=$(date -Is)"
