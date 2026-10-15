#!/bin/bash
# Review-stage robustness downloads.
#SBATCH --job-name=rbfetch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
bash src/fetch_rb.sh
echo "end=$(date -Is)"
