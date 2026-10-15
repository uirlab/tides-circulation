#!/bin/bash
# Downloads the trip data for the cross-system comparison that are not covered by the other download scripts:
# Citi Bike, Divvy and Bluebikes (May and June 2026), Bay Wheels (May and September 2025), BART hourly
# origin-destination data (2025) and GTFS, and YouBike 2.0 rides (May 2026; May and June 2023).
# Run from the repository root:  bash scripts/download_core.sh
# By downloading, you accept each provider's licence (see data/README.md). Files are checked against
# data/checksums_sha256.txt where a checksum is listed.
set -uo pipefail
UA="Mozilla/5.0 (research; tides-circulation)"
get() { local u=$1 f=$2; mkdir -p "$(dirname "$f")"; [ -s "$f" ] && { echo "exists $f"; return; }
  curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f" && echo "ok $f" || echo "FAIL $u"; }
for m in 202605 202606; do
  get "https://s3.amazonaws.com/tripdata/${m}-citibike-tripdata.zip"        "data/raw/citibike/${m}-citibike-tripdata.zip"
  get "https://divvy-tripdata.s3.amazonaws.com/${m}-divvy-tripdata.zip"     "data/raw/divvy/${m}-divvy-tripdata.zip"
  get "https://s3.amazonaws.com/hubway-data/${m}-bluebikes-tripdata.zip"    "data/raw/bluebikes/${m}-bluebikes-tripdata.zip"
done
for m in 202505 202509; do
  get "https://s3.amazonaws.com/baywheels-data/${m}-baywheels-tripdata.csv.zip" "data/raw/baywheels/${m}-baywheels-tripdata.csv.zip"
done
get "https://afcweb.bart.gov/ridership/origin-destination/date-hour-soo-dest-2025.csv.gz" "data/raw/bart/date-hour-soo-dest-2025.csv.gz"
get "https://www.bart.gov/dev/schedules/google_transit.zip"                           "data/raw/bart/google_transit.zip"
ZP=%E7%A5%A8%E8%AD%89%E5%88%B7%E5%8D%A1%E8%B3%87%E6%96%99   # 票證刷卡資料
for ym in 2026-05 2023-05 2023-06; do y=${ym%-*}; m=${ym#*-}
  get "https://tcgbusfs.blob.core.windows.net/dotapp/youbike_second_ticket_opendata/${y}/${ym}/${y}${m}_YouBike2.0${ZP}.zip" "data/raw/youbike/${y}${m}_youbike.zip"
done
echo "Transfer rides (YouBike after MRT/bus, 2023): download the monthly CSVs of data.taipei dataset"
echo "  d51e8336-fceb-4275-a81a-2a246f4fca1b (臺北市轉乘YouBike租借資料) to data/raw/youbike/xfer_2023MM.csv;"
echo "  scripts/fetch_rb.sh fetches March, April and September-December 2023 by resource id."
if command -v shasum >/dev/null; then (shasum -a 256 -c data/checksums_sha256.txt --ignore-missing 2>/dev/null | grep -v ': OK$') || true; fi
