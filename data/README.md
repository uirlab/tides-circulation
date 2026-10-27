# Data

No trip records are included in this repository. The bike-share data of Citi Bike, Divvy, Bluebikes and Bay Wheels may
be used as source material in published non-commercial studies but not redistributed as a stand-alone dataset, and some
other sources carry no stated licence (see the table). The scripts below download every source from its publisher into
`data/raw/<source>/`. `data/checksums_sha256.txt` lists the SHA-256 checksums of the files used in the paper. Some
publishers revise files after release; the checksums identify the versions we analysed.

By downloading, you accept each provider's terms. This work is not affiliated with or endorsed by any provider.

| Data (period used) | Folder | How to obtain | Licence |
|---|---|---|---|
| Citi Bike, New York (May–June 2026; monthly from Jan 2024 to Aug 2026) | `citibike/` | `scripts/download_core.sh`; monthly units: `hpc/steps24_download.sbatch` | Lyft Bike Data License Agreement (no stand-alone redistribution) |
| Divvy, Chicago (May–June 2026; monthly from Jan 2025) | `divvy/` | as above | Lyft Bike Data License Agreement |
| Bluebikes, Boston (May 2026; monthly from Jan 2025) | `bluebikes/` | as above | Lyft Bike Data License Agreement |
| Bay Wheels, San Francisco Bay Area (May and Sep 2025) | `baywheels/` | `scripts/download_core.sh` | Lyft Bike Data License Agreement |
| Santander Cycles, London (Apr–May 2026) | `tfl/` | `scripts/fetch_global.sh`, `scripts/fetch_rb.sh` | TfL open data (OGL-based); "Powered by TfL Open Data" |
| Ttareungyi, Seoul (Apr–May 2026; dataset OA-15182) | `seoul/` | `scripts/fetch_global.sh`, `scripts/fetch_rb.sh` | KOGL Type 1 (attribution) |
| Ecobici, Mexico City (Apr–May 2026) | `ecobici_mx/` | `scripts/fetch_global.sh`, `scripts/fetch_rb.sh` | none stated by the provider |
| Ecobici, Buenos Aires (2026) | `ecobici_ba/` | `scripts/fetch_global.sh` | CC BY 2.5 AR |
| YouBike 2.0, Taipei (Apr–May 2026; 2023 months; Dec 2022–Feb 2023 and Jan–May 2024) | `youbike/` | `scripts/download_core.sh`, `scripts/fetch_rb.sh`, `scripts/fetch_fare.sh` | Open Government Data License v1 (data.taipei) |
| YouBike transfer rides, Taipei (2023) | `youbike/xfer_2023MM.csv` | data.taipei dataset d51e8336-fceb-4275-a81a-2a246f4fca1b; `scripts/fetch_rb.sh` | Open Government Data License v1 (data.taipei) |
| Taipei Metro hourly OD (Apr–May 2026) | `taipei_mrt/` | `scripts/fetch_global.sh`, `scripts/fetch_rb.sh` | Open Government Data License v1 (data.taipei) |
| BART hourly OD (2025) and GTFS | `bart/` | `scripts/download_core.sh` | CC BY (BART ridership data); developer terms (GTFS) |
| NYC TLC yellow and high-volume for-hire trips (May 2025 for the taxi-zone application; Oct–Jun, 2022–2025, for congestion pricing) | `tlc/` | `hpc/tlc_download.sbatch` (about 15 GB); the taxi-zone boundaries and lookup table (`tlc/zones/`, `tlc/taxi_zone_lookup.csv`) are included | none stated (NYC open data) |
| Citi Bike station status, every 2 minutes (May–June 2025) | `station_status/` | Kaggle dataset `rosenthal/citi-bike-stations` (requires a Kaggle account) | CC BY 4.0 (archive); Lyft licence for the underlying feed |
| OpenStreetMap street network | `osm/` | `src/fetch_osm.py` (Overpass API) | ODbL 1.0 |
| NYC DCP LION street centrelines (22c–26c) | `lion/` | `hpc/steps24_download.sbatch` | NYC open data terms |
| GTFS: MTA subway, CTA, MBTA | `gtfs/` | `src/fetch_mode_data.py` | operator developer terms |
| USGS 3DEP elevations (EPQS) | (cached in `data/processed/`) | `src/fetch_mode_data.py` | public domain |
| LODES 2023 (residents and jobs) | `lodes/` | `scripts/fetch_landuse.sh` | US Census, public domain |
| Overture Maps places (release 2026-09-23.1) | `overture/` | `src/fetch_overture.py` (DuckDB query of the public S3 bucket) | CDLA-Permissive-2.0, Apache-2.0, CC0 by source |
| Natural Earth 1:110m land (Fig. 1a) | `naturalearth/` (included) | included in the repository | public domain |

The OSM, Overture and GTFS extracts are queries of live services. Re-running them today gives current data, which can
differ slightly from the versions we used; the derived inputs we used are in `results/`.
