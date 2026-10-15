#!/bin/bash
# Taipei MRT with interchange stations excluded (post-hoc cleaning): sbatch src/aicr_tpem.sbatch
#SBATCH --job-name=tpex
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16 CITY=g_tpex BIN_MIN=60
${PYTHON:-python} src/tpe_exclude.py
${PYTHON:-python} src/irrev_strat.py all
${PYTHON:-python} src/circ_pairs.py
