#!/bin/bash
# Download NYC TLC Yellow + HVFHV parquet (Oct-Jun of 2022-23, 2023-24, 2024-25) + taxi zones to data/raw/tlc/.
# Usage (from the repository root): sbatch hpc/tlc_download.sbatch
#SBATCH --job-name=tlc-dl
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
D=data/raw/tlc; mkdir -p $D
B=https://d37ci6vzurychx.cloudfront.net/trip-data
get() {  # url dest ; verifies size against Content-Length
  local u=$1 f=$2 want got
  want=$(curl -sI "$u" | awk 'tolower($1)=="content-length:"{print $2}' | tr -d '\r')
  for a in 1 2 3 4 5; do
    [ -s "$f" ] && [ "$(stat -c %s "$f")" = "$want" ] && { echo "ok $f $want"; return 0; }
    curl -sS --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f"
  done
  echo "FAILED $u (want $want, got $(stat -c %s "$f" 2>/dev/null))"; return 1
}
fail=0
for y in 2022 2023 2024; do for m in 10 11 12 01 02 03 04 05 06; do
  yy=$y; [ $((10#$m)) -lt 10 ] && yy=$((y+1))
  for t in yellow fhvhv; do get "$B/${t}_tripdata_${yy}-${m}.parquet" "$D/${t}_tripdata_${yy}-${m}.parquet" || fail=1; done
done; done
get https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip $D/taxi_zones.zip || fail=1
get https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv $D/taxi_zone_lookup.csv || fail=1
du -sh $D; ls $D | wc -l; echo "fail=$fail"
