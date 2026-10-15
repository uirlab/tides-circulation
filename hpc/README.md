# Slurm job scripts

The job scripts used for the heavy steps (calibrated nulls, subsampling, monthly units, covariate tests). They were
written for one Slurm cluster; adapt the partition, memory and CPU settings to yours, and set `PYTHON` to your
interpreter (for example `export PYTHON=$CONDA_PREFIX/bin/python`). Each script is run from the repository root, for
example `sbatch --array=0-79 hpc/rb_uncert.sbatch`. The `*_tasks.txt` and `units_*.txt` files list the array tasks.
Download jobs (`steps24_download.sbatch`, `tlc_download.sbatch`) fetch the monthly and taxi data described in
`data/README.md`.
