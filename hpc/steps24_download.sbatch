#!/bin/bash
# Downloads for steps 2-4: Citi Bike 2023-2026,
# Divvy + Bluebikes Jan 2025-Aug 2026, NYC DCP LION 22c-26c. Each file verified against Content-Length.
#SBATCH --job-name=dl-steps24
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -uo pipefail
get() {
  local u=$1 f=$2 want
  want=$(curl -sIL "$u" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' | tr -d '\r')
  [ -z "$want" ] && { echo "MISSING $u"; return 1; }
  for a in 1 2 3 4 5; do
    [ -s "$f" ] && [ "$(stat -c %s "$f")" = "$want" ] && { echo "ok $f $want"; return 0; }
    curl -sSL --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f"
  done
  echo "FAILED $u"; return 1
}
fail=0
mkdir -p data/raw/citibike data/raw/divvy data/raw/bluebikes data/raw/lion
get https://s3.amazonaws.com/tripdata/2023-citibike-tripdata.zip data/raw/citibike/2023-citibike-tripdata.zip || fail=1
for ym in 202401 202402 202403 202404 202405 202406 202407 202408 202409 202410 202411 202412 \
          202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512 202601 202602 202603 202604 202607 202608; do
  get https://s3.amazonaws.com/tripdata/${ym}-citibike-tripdata.zip data/raw/citibike/${ym}-citibike-tripdata.zip || fail=1
done
for ym in 202501 202502 202503 202504 202505 202506 202507 202508 202509 202510 202511 202512 202601 202602 202603 202604 202605 202606 202607 202608; do
  get https://divvy-tripdata.s3.amazonaws.com/${ym}-divvy-tripdata.zip data/raw/divvy/${ym}-divvy-tripdata.zip || fail=1
  b=https://s3.amazonaws.com/hubway-data/${ym}-bluebikes-tripdata.zip
  [ "$ym" = 202511 ] && b=https://s3.amazonaws.com/hubway-data/${ym}-bluebikes-tripdata.csv.zip
  get $b data/raw/bluebikes/$(basename $b) || fail=1
done
for v in 22a 22c 23a 23b 23c 23d 24a 24b 24c 24d 25a 25b 25c 25d 26a 26b 26c; do
  get https://s-media.nyc.gov/agencies/dcp/assets/files/zip/data-tools/bytes/lion/nyclion_${v}.zip data/raw/lion/nyclion_${v}.zip || fail=1
done
du -sh data/raw/citibike data/raw/divvy data/raw/bluebikes data/raw/lion; echo "fail=$fail"
