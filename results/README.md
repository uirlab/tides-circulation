# Derived results

Released under CC BY 4.0, except OpenStreetMap-derived files (ODbL); see `../LICENSE-data.md`. No trip records are
included. All files are outputs of the scripts in `src/` (see the main README for the step that writes each).

| Pattern | Contents |
|---|---|
| `0c_irrev_*`, `0cb_strata_*` | Existence test for New York (unstratified; stratified), per window: deviances, null mean and SD, z, h′ |
| `1_replication_<unit>_{real,synthetic}.json` | The same test for every other system and resolution (`_b60` = 60-minute bins), with the synthetic gate |
| `3_units/`, `cas/` | Monthly units in New York, Chicago and Boston; rider-type comparison by month |
| `1_replication_rb/`, `1_replication_*_rb*` | Review-stage robustness units (spatial units, second month, estimator settings) |
| `rb/` | Day subsamples (`boot/`, `boot15/`), their summaries, the delete-*d* jackknife intervals (`uncertainty_summary.json`), threshold curves, transfer replications |
| `ampm/` | Morning–evening comparison and transfer-pair comparison |
| `hab_*`, `mode*.json`, `xfer_*`, `hact.json`, `exp_loop_sources*.json` | Covariate tests |
| `1_oneway_*`, `devES*.json`, `1b_*` | One-way streets and explained shares (`streets_top` lists derive from OpenStreetMap) |
| `4_rebalancing*.json`, `3_congestion_*` | Rebalancing; congestion pricing |
| `coarsegrain.json`, `5_tidal_share_*`, `toy_tide.json`, `pertrip_deviance.json` | Resolution, coarse-graining and per-trip deviances |
| `figdata_*` | Pair-level circulation fields, station coordinates and other aggregates used by the figures (`figdata_oneway_citibike_jun.npz` contains OpenStreetMap street data; ODbL) |
| `figures/`, `tables/` | Created by `make figures` and `make tables` |
