#!/bin/bash
# Step 3: one array task per unit in src/units_step3_fallback.txt -> prep_unit, then irrev_strat (potentials, synthetic, real-if-gated).
#SBATCH --job-name=units-fb
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NWORKERS=${SLURM_CPUS_PER_TASK:-32}
read -r UNIT CITYRAW MONTHS < <(sed -n "${SLURM_ARRAY_TASK_ID}p" src/units_step3_fallback.txt)
echo "unit=$UNIT city=$CITYRAW months=$MONTHS host=$(hostname) start=$(date -Is)"
mkdir -p results/3_units
[ -f data/processed/${UNIT}_weekday_trips.npz ] || ${PYTHON:-python} src/prep_unit.py "$UNIT" "$CITYRAW" "$MONTHS"
CITY=$UNIT ${PYTHON:-python} src/irrev_strat.py all
echo "end=$(date -Is)"
