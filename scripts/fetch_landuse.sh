#!/bin/bash
# Downloads: LODES8 2023 RAC/WAC/xwalk (NY, IL, MA), NYC PLUTO 25v2 CSV, Overture Places bbox extracts.
set -uo pipefail
mkdir -p data/raw/lodes data/raw/pluto data/raw/overture
for st in ny il ma; do
  curl -sSfL -o data/raw/lodes/${st}_rac_S000_JT00_2023.csv.gz https://lehd.ces.census.gov/data/lodes/LODES8/${st}/rac/${st}_rac_S000_JT00_2023.csv.gz || echo FAIL rac $st
  curl -sSfL -o data/raw/lodes/${st}_wac_S000_JT00_2023.csv.gz https://lehd.ces.census.gov/data/lodes/LODES8/${st}/wac/${st}_wac_S000_JT00_2023.csv.gz || echo FAIL wac $st
  curl -sSfL -o data/raw/lodes/${st}_xwalk.csv.gz https://lehd.ces.census.gov/data/lodes/LODES8/${st}/${st}_xwalk.csv.gz || echo FAIL xwalk $st
done
curl -sSfL -o data/raw/pluto/nyc_pluto_25v2_csv.zip https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/pluto/nyc_pluto_25v2_csv.zip || echo FAIL pluto
${PYTHON:-python} - <<'PY'
import duckdb, json
bb = json.load(open("data/processed/landuse_bboxes.json"))
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
src = "s3://overturemaps-us-west-2/release/2026-09-23.1/theme=places/type=place/*"
for c, (w, s, e, n) in bb.items():
    q = f"""COPY (SELECT id, names.primary AS name, categories.primary AS cat, confidence,
                (bbox.xmin + bbox.xmax)/2 AS lon, (bbox.ymin + bbox.ymax)/2 AS lat
         FROM read_parquet('{src}', hive_partitioning=1)
         WHERE bbox.xmin >= {w} AND bbox.xmax <= {e} AND bbox.ymin >= {s} AND bbox.ymax <= {n})
         TO 'data/raw/overture/places_{c}.parquet' (FORMAT PARQUET)"""
    con.execute(q); print(c, con.execute(f"SELECT count(*) FROM 'data/raw/overture/places_{c}.parquet'").fetchone()[0], flush=True)
PY
ls -la data/raw/lodes data/raw/pluto data/raw/overture; du -sh data/raw/lodes data/raw/pluto data/raw/overture
