#!/bin/bash
# Taxi-zone application: prep, existence test (synthetic gate then real), mirror test.
#SBATCH --job-name=taxi
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=32
PY=${PYTHON:-python}
[ -f data/processed/tlc_2505_weekday_trips.npz ] || $PY src/prep_tlc_unit.py 2025 5
CITY=tlc_2505 $PY -X faulthandler src/irrev_strat.py all
CITY=tlc_2505 BIN_MIN=15 $PY -X faulthandler src/ampm.py real
echo done $(date -Is)
