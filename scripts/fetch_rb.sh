#!/bin/bash
# Downloads for the robustness checks; every file is under 1 GB.
set -uo pipefail
UA="Mozilla/5.0 (research; mobility-irreversibility)"
mkdir -p data/raw/{tfl,seoul,ecobici_mx,taipei_mrt,youbike,baywheels}
get() { local u=$1 f=$2; [ -s "$f" ] && { echo "exists $f"; return; }
  curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f" && echo "ok $f $(stat -c %s "$f")" || echo "FAIL $u"; }
ZP=%E7%A5%A8%E8%AD%89%E5%88%B7%E5%8D%A1%E8%B3%87%E6%96%99   # 票證刷卡資料
# R3: April 2026 and September 2025
get "https://cycling.data.tfl.gov.uk/usage-stats/441JourneyDataExtract01Apr2026-15Apr2026.csv" data/raw/tfl/441JourneyDataExtract01Apr2026-15Apr2026.csv
get "https://cycling.data.tfl.gov.uk/usage-stats/442JourneyDataExtract16Apr2026-30Apr2026.csv" data/raw/tfl/442JourneyDataExtract16Apr2026-30Apr2026.csv
f=data/raw/seoul/seoul_bike_rental_2604.csv
[ -s $f ] || { curl -sSf -A "$UA" --retry 3 -X POST "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?&useCache=false" --data "infId=OA-15182&seq=148&infSeq=1" -o $f.part && mv $f.part $f && echo "ok $f $(stat -c %s $f)" || echo "FAIL seoul"; }
get "https://ecobici.cdmx.gob.mx/wp-content/uploads/2026/05/2026-04.csv" data/raw/ecobici_mx/2026-04.csv
get "http://tcgmetro.blob.core.windows.net/stationod/%E8%87%BA%E5%8C%97%E6%8D%B7%E9%81%8B%E6%AF%8F%E6%97%A5%E5%88%86%E6%99%82%E5%90%84%E7%AB%99OD%E6%B5%81%E9%87%8F%E7%B5%B1%E8%A8%88%E8%B3%87%E6%96%99_202604.csv" data/raw/taipei_mrt/taipei_mrt_od_202604.csv
get "https://tcgbusfs.blob.core.windows.net/dotapp/youbike_second_ticket_opendata/2026/2026-04/202604_YouBike2.0$ZP.zip" data/raw/youbike/202604_youbike.zip
get "https://s3.amazonaws.com/baywheels-data/202509-baywheels-tripdata.csv.zip" data/raw/baywheels/202509-baywheels-tripdata.csv.zip
# R5: YouBike 2023 rides and transfer rides (data.taipei d51e8336 resources)
declare -A RID=([3]=65ee5383-cf71-4dc3-aeff-655b6f9c8356 [4]=acf1e719-d5e0-471e-a261-5989d34c8ddb [9]=f678018a-339f-40d0-a0f8-c544a8911b55
                [10]=c0d56ec4-d176-47c1-897f-b747be1db0c3 [11]=be19e3ae-6c6e-480a-ace7-529bd84c1c3e [12]=8ddb45ba-a0f5-41d6-85a0-1691096fdcf8)
for m in 3 4 9 10 11 12; do mm=$(printf %02d $m)
  get "https://tcgbusfs.blob.core.windows.net/dotapp/youbike_second_ticket_opendata/2023/2023-$mm/2023${mm}_YouBike2.0$ZP.zip" data/raw/youbike/2023${mm}_youbike.zip
  get "https://data.taipei/api/dataset/d51e8336-fceb-4275-a81a-2a246f4fca1b/resource/${RID[$m]}/download" data/raw/youbike/xfer_2023${mm}.csv
done
echo "== headers (first lines only)"
for f in data/raw/tfl/441*.csv data/raw/ecobici_mx/2026-04.csv data/raw/taipei_mrt/taipei_mrt_od_202604.csv data/raw/youbike/xfer_202303.csv; do echo "-- $f"; head -c 300 "$f" | head -2; echo; done
head -c 300 data/raw/seoul/seoul_bike_rental_2604.csv | iconv -f cp949 -t utf-8 -c | head -2
for z in data/raw/youbike/202604_youbike.zip data/raw/youbike/202303_youbike.zip data/raw/baywheels/202509-baywheels-tripdata.csv.zip; do unzip -l "$z" | sed -n 4p; done
