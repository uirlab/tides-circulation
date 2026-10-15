#!/bin/bash
# : prep + pipeline for BART (g_bar), Bay Wheels (g_baw), YouBike (g_ybk).
#   sbatch --array=0-2 src/aicr_withincity.sbatch
#SBATCH --job-name=wcity
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=16
U=(bar baw ybk); u=${U[$SLURM_ARRAY_TASK_ID]}; C=g_$u
PY=${PYTHON:-python}
echo "unit=$C start=$(date -Is)"
$PY src/prep_global.py $u
if [ "$u" = baw ]; then
  CITY=$C $PY src/irrev_strat.py all                       # 15-min reference
  CITY=$C BIN_MIN=60 SENS=_b60 $PY src/irrev_strat.py all  # decision resolution
  CITY=$C $PY src/circ_pairs.py
else
  CITY=$C BIN_MIN=60 $PY src/irrev_strat.py all            # hourly data: 60-min only
  CITY=$C BIN_MIN=60 $PY src/circ_pairs.py
fi
echo "end=$(date -Is)"
