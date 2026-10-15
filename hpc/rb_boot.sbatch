#!/bin/bash
# R2a day-block bootstrap: 4 units x 200 seeds, 10 seeds per array task (tasks 0-79).
#SBATCH --job-name=rbboot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16
U=(g_ybk g_tpex g_baw g_bar); i=$SLURM_ARRAY_TASK_ID
u=${U[$((i / 20))]}; s0=$(( (i % 20) * 10 + 1 )); s1=$((s0 + 9))
echo "unit=$u seeds=$s0-$s1 start=$(date -Is)"
${PYTHON:-python} src/rb_boot.py $u $s0 $s1
echo "end=$(date -Is)"
