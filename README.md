# Tides and circulation in urban mobility

Code and derived results for the article
**"Tides and circulation in urban mobility"** (Y. Liu, Q. R. Wang, X. Qu; submitted).

The article decomposes the directional asymmetry of origin–destination flows in each short time window into a
**tide**, explained by how crowded places are, and **circulation**, the part that runs around loops of places
preferentially one way. A tide is exactly quasi-symmetric within a window, so circulation is measured as the lack of
fit of a Bradley–Terry model, against a calibrated null in which the tide may drift. The analysis covers nine
bike-share systems and two metros (New York, Chicago, Boston, San Francisco Bay Area, London, Seoul, Mexico City,
Buenos Aires, Taipei).

## Quick start: rebuild the figures and tables without any data

All figures and Supplementary Table S8 can be rebuilt from the derived results in `results/`; no trip data are needed.

```bash
conda env create -f environment.yml && conda activate tides-circulation   # or: pip install -r requirements.txt
make test      # synthetic checks of the estimator (seconds)
make figures   # Figs. 1-4 and Supplementary Figs. 1-12 -> results/figures/
make tables    # Supplementary Table S8 -> results/tables/table_s8.md
```

The rebuilt figures are identical to those in the article.

## Repository layout

| Folder | Contents |
|---|---|
| `src/` | All analysis code: data preparation, the decomposition and test, covariate tests, robustness, uncertainty, figures |
| `scripts/` | Download scripts for the public data sources |
| `hpc/` | Slurm job scripts used for the heavy steps (set `PYTHON` to your interpreter) |
| `data/` | Download instructions, licences and checksums; raw and processed data are not committed |
| `results/` | Derived results: test outputs (JSON), pair- and station-level aggregates, figure data (see `results/README.md`) |
| `tests/` | Quick synthetic tests of the core estimator |

## Full pipeline from raw data

The full pipeline needs the raw data (about 50 GB including the taxi records, see `data/README.md`) and, for the
calibrated nulls, a multi-core machine or cluster. Scripts are run from the repository root; the city or unit is
selected with the environment variable `CITY`, the window length with `BIN_MIN` (default 15) and the number of worker
processes with `NWORKERS`. Random seeds are fixed in the code.

| Step | Scripts | Output |
|---|---|---|
| 1. Download | `scripts/download_core.sh`, `scripts/fetch_global.sh`, `scripts/fetch_rb.sh`, `scripts/fetch_landuse.sh`, `src/fetch_osm.py`, `src/fetch_mode_data.py`, `src/fetch_overture.py`, `hpc/steps24_download.sbatch`, `hpc/tlc_download.sbatch` | `data/raw/` |
| 2. Prepare trip arrays | `src/prep_citibike.py`, `src/prep_lyft.py`, `src/prep_global.py`, `src/prep_unit.py`, `src/prep_xfer.py`, `src/prep_xfer_rb.py`, `src/prep_rb.py`, `src/tpe_merge.py`, `src/tpe_exclude.py`, `src/tlc_prep.py`, `src/station_table.py`, `src/station_coords.py` | `data/processed/*_weekday_trips.npz` |
| 3. Existence test (synthetic gate, then data) | `src/irrev.py` (unstratified), `src/irrev_strat.py all` (stratified; `CITY=…`, `BIN_MIN=60` for the rail comparison) | `results/0cb_strata_real.json`, `results/1_replication_*`, `results/3_units/` |
| 4. Resolution and coarse-graining | `src/tidal_share.py`, `src/toy_tide.py` | `results/coarsegrain.json`, `results/5_tidal_share_*` |
| 5. Pair-level fields | `src/loop_map.py`, `src/circ_pairs.py`, `src/bin_volume.py`, `src/motiv_triangles.py` | `results/figdata_*` |
| 6. Covariate tests | `src/hab.py` (habit), `src/modesub.py` (transit proximity), `src/xfer.py`, `src/xfer_rb.py` (Taipei transfers), `src/hact.py` (land use), `src/explore_loop_sources*.py` (exploratory) | `results/hab_*`, `mode*.json`, `xfer_*`, `hact.json`, `cas*` |
| 7. Morning–evening comparison | `src/ampm.py`, `src/ampm_xfer.py` (`synthetic` then `real`) | `results/ampm/` |
| 7b. Taxi-zone application | `src/prep_tlc_unit.py 2025 5`, then steps 3 and 7 with `CITY=tlc_2505` (`hpc/taxi.sbatch`) | `results/1_replication_tlc_2505_*`, `results/ampm/tlc_2505_real.json` |
| 8. One-way streets and explained share | `src/street_graph.py`, `src/lion_extract.py`, `src/oneway.py`, `src/dev_es.py`, `src/dev_es30.py`, `src/mechanism_share.py`, `src/shrinkage_calibration.py` | `results/1_oneway_*`, `devES*.json` |
| 9. Rebalancing; congestion pricing | `src/rebalancing.py`; `src/tlc_irrev.py` | `results/4_rebalancing*.json`; `results/3_congestion_*` |
| 10. Robustness and uncertainty | `src/rb_spatial.py`, `src/rb_boot.py`, `src/rb_boot_summary.py`, `src/rb_thresholds.py`, `src/rb_ridge_check.py`, `src/rb_uncert_summary.py` | `results/rb/`, `results/1_replication_rb/` |
| 11. Figures and tables | see below | `results/figures/`, `results/tables/` |

The `hpc/` folder holds the job scripts used for each step, with the cluster-specific interpreter path replaced by
`${PYTHON:-python}`.

## Figures and tables

| Article item | Script | Output file |
|---|---|---|
| Fig. 1 | `src/fig1_v3.py` | `fig1_v3.png` |
| Fig. 2 | `src/fig2_v3.py` | `fig2_v3.png` |
| Fig. 3 | `src/fig3_v3.py` | `fig3_v3.png` |
| Fig. 4 | `src/fig4_v3.py` | `fig4_v3.png` |
| Supplementary Fig. 1 | `src/toy_tide.py` | `si_toy_tide.png` |
| Supplementary Figs. 2–7, 10 | `src/ed_figures.py` | `ed1_calibration`, `ed2_units`, `ed3_binwidth`, `ed4_taipei`, `ed5_transit_radius`, `ed7_ridertype`, `ed9_not_geography` |
| Supplementary Figs. 8, 9, 11 | `src/si_figures.py` (with `src/ed_oneway_distance.py`) | `si_oneway`, `si_robustness`, `si_operations` |
| Supplementary Fig. 12 | `src/sfig12_mirror.py` | `SFig12_mirror.png` |
| Supplementary Table S8 | `src/table_s8.py` | `results/tables/table_s8.md` |
| Supplementary Tables S1–S7 | assembled from the result files named in each table's source line | — |

## Data and licences

No trip records are redistributed; see `data/README.md` for every source, its licence and how to download it. The code
is released under the MIT licence (`LICENSE`). The derived results are released under CC BY 4.0, except that files
derived from OpenStreetMap are under ODbL (`LICENSE-data.md`, which also lists the required source credits). This work
is not affiliated with or endorsed by any data provider.

## Citation

See `CITATION.cff`. A Zenodo DOI will be added on publication.

## Contact

Qi R. Wang, Department of Civil and Environmental Engineering, Northeastern University (q.wang@northeastern.edu).
