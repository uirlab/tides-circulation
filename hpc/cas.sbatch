#!/bin/bash
#SBATCH --job-name=cas
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
export OMP_NUM_THREADS=1 NWORKERS=32 ONLY_15=1
U=$(sed -n "${SLURM_ARRAY_TASK_ID}p" src/units_cas.txt)
CITY=$U ${PYTHON:-python} src/tidal_share.py > /dev/null
echo done $U
