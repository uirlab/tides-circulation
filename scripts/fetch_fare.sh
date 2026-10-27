#!/bin/bash
# Downloads for the Taipei fare-change analysis (YouBike 2.0 rides, Dec 2022-Feb 2023 and Jan-May 2024);
# every file is under 60 MB. The other months used here (Mar-May, Sep-Dec 2023, and the Sep-Nov 2023 transfer rides)
# come from scripts/download_core.sh and scripts/fetch_rb.sh.
set -uo pipefail
UA="Mozilla/5.0 (research; mobility-irreversibility)"
mkdir -p data/raw/youbike
get() { local u=$1 f=$2; [ -s "$f" ] && { echo "exists $f"; return; }
  curl -sSfL -A "$UA" --retry 3 -o "$f.part" "$u" && mv "$f.part" "$f" && echo "ok $f $(stat -c %s "$f")" || echo "FAIL $u"; }
ZP=%E7%A5%A8%E8%AD%89%E5%88%B7%E5%8D%A1%E8%B3%87%E6%96%99   # 票證刷卡資料
for ym in 2022-12 2023-01 2023-02 2024-01 2024-02 2024-03 2024-04 2024-05; do y=${ym%-*}; m=${ym#*-}
  get "https://tcgbusfs.blob.core.windows.net/dotapp/youbike_second_ticket_opendata/${y}/${ym}/${y}${m}_YouBike2.0${ZP}.zip" "data/raw/youbike/${y}${m}_youbike.zip"
done
