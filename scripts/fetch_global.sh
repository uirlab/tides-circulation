#!/bin/bash
# Downloads: May 2026 for London, Seoul, Mexico City, Buenos Aires (2026 zip), Taipei MRT OD.
set -uo pipefail
UA="Mozilla/5.0 (research; mobility-irreversibility)"
mkdir -p data/raw/{tfl,seoul,ecobici_mx,ecobici_ba,taipei_mrt}
get() { local u=$1 f=$2; [ -s "$f" ] && { echo "exists $f"; return; }
  curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f" && echo "ok $f $(stat -c %s "$f")" || echo "FAIL $u"; }
get "https://cycling.data.tfl.gov.uk/usage-stats/443JourneyDataExtract01May2026-16May2026.csv" data/raw/tfl/443JourneyDataExtract01May2026-16May2026.csv
get "https://cycling.data.tfl.gov.uk/usage-stats/444JourneyDataExtract17May2026-31May2026.csv" data/raw/tfl/444JourneyDataExtract17May2026-31May2026.csv
f=data/raw/seoul/seoul_bike_rental_2605.csv
[ -s $f ] || { curl -sSf -A "$UA" --retry 3 -X POST "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false" --data "infId=OA-15182&seq=149&infSeq=1" -o $f.part && mv $f.part $f && echo "ok $f $(stat -c %s $f)" || echo "FAIL seoul"; }
get "https://ecobici.cdmx.gob.mx/wp-content/uploads/2026/06/2026-05.csv" data/raw/ecobici_mx/2026-05.csv
get "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/transporte-y-obras-publicas/bicicletas-publicas/recorridos-realizados-2026.zip" data/raw/ecobici_ba/recorridos-realizados-2026.zip
get "http://tcgmetro.blob.core.windows.net/stationod/%E8%87%BA%E5%8C%97%E6%8D%B7%E9%81%8B%E6%AF%8F%E6%97%A5%E5%88%86%E6%99%82%E5%90%84%E7%AB%99OD%E6%B5%81%E9%87%8F%E7%B5%B1%E8%A8%88%E8%B3%87%E6%96%99_202605.csv" data/raw/taipei_mrt/taipei_mrt_od_202605.csv
du -sh data/raw/{tfl,seoul,ecobici_mx,ecobici_ba,taipei_mrt}
for f in data/raw/tfl/443*.csv data/raw/seoul/*.csv data/raw/ecobici_mx/*.csv data/raw/taipei_mrt/*.csv; do echo "== $f"; head -c 600 "$f" | iconv -f $( [[ $f == *seoul* ]] && echo cp949 || echo utf-8) -t utf-8 -c | head -3; done
unzip -l data/raw/ecobici_ba/recorridos-realizados-2026.zip | tail -4
