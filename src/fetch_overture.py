"""Downloads: Overture Places (release 2026-09-23.1, CDLA-Permissive-2.0) for the three
station bounding boxes in data/processed/landuse_bboxes.json -> data/raw/overture/places_<city>.parquet (run on a cluster)."""
import json

import duckdb

bb = json.load(open("data/processed/landuse_bboxes.json"))
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
src = "s3://overturemaps-us-west-2/release/2026-09-23.1/theme=places/type=place/*"
for c, (w, s, e, n) in bb.items():
    con.execute(f"""COPY (SELECT id, names.primary AS name, basic_category AS cat, taxonomy.primary AS tax_primary,
                taxonomy.hierarchy[1] AS tax_top, operating_status, confidence, sources[1].license AS src_license,
                (bbox.xmin + bbox.xmax)/2 AS lon, (bbox.ymin + bbox.ymax)/2 AS lat
                FROM read_parquet('{src}', hive_partitioning=1)
                WHERE bbox.xmin >= {w} AND bbox.xmax <= {e} AND bbox.ymin >= {s} AND bbox.ymax <= {n})
                TO 'data/raw/overture/places_{c}.parquet' (FORMAT PARQUET)""")
    print(c, con.execute(f"SELECT count(*) FROM 'data/raw/overture/places_{c}.parquet'").fetchone()[0], flush=True)
