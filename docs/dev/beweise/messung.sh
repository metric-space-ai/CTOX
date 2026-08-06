#!/bin/zsh
# Dieselben Messungen vor und nach der Installation. Nichts erzaehlt, alles gemessen.
# Aufruf: messung.sh <marke>   z.B. messung.sh vorher
MARKE=${1:-unbenannt}
PORT=8765
OUT=/Volumes/tmp/ctox-pipeline/messung-$MARKE.txt
: > "$OUT"

echo "=== Messung '$MARKE' — $(date '+%d.%m. %H:%M:%S') ===" | tee -a "$OUT"

PID=$(pgrep -x ctox-real | head -1)
if [ -n "$PID" ]; then
  ps -o etime=,rss= -p "$PID" | awk '{printf "  Daemon: pid '"$PID"', Laufzeit %s, RSS %.2f GB\n", $1, $2/1048576}' | tee -a "$OUT"
  echo "  Verbindungen auf business-os-rxdb: $(lsof -p $PID 2>/dev/null | grep -c 'business-os-rxdb.sqlite3$')" | tee -a "$OUT"
  echo "  FDs gesamt: $(lsof -p $PID 2>/dev/null | wc -l | tr -d ' ')" | tee -a "$OUT"
else
  echo "  Daemon: NICHT GEFUNDEN" | tee -a "$OUT"
fi

echo "--- Shell, fuenf Aufrufe nacheinander (kalt nach warm) ---" | tee -a "$OUT"
for i in 1 2 3 4 5; do
  curl -s -o /dev/null -w "  #$i  %{http_code}  ttfb=%{time_starttransfer}s  gesamt=%{time_total}s  %{size_download} Bytes\n" \
       --max-time 60 -H 'Accept-Encoding: gzip, br' "http://127.0.0.1:$PORT/" | tee -a "$OUT"
done

echo "--- Kompression: was kommt wirklich ueber die Leitung ---" | tee -a "$OUT"
curl -s -D - -o /dev/null --max-time 60 -H 'Accept-Encoding: gzip, br' "http://127.0.0.1:$PORT/" \
  | grep -iE "content-encoding|content-length|vary|cache-control|connection" | sed 's/^/  /' | tee -a "$OUT"

echo "--- Sechs gleichzeitige Anfragen (Worker-Pool) ---" | tee -a "$OUT"
for i in 1 2 3 4 5 6; do
  ( curl -s -o /dev/null -w "  parallel #$i  %{time_total}s\n" --max-time 120 "http://127.0.0.1:$PORT/" ) &
done
wait 2>/dev/null
sleep 1

echo "--- Sellify: was der Peer gebunden hat ---" | tee -a "$OUT"
S=$HOME/.local/state/ctox/business-os-rxdb.sqlite3
for t in sellify_people sellify_companies sellify_campaigns sellify_activities; do
  v0=$(sqlite3 "file:$S?mode=ro" "SELECT count(*) FROM ctox_business_os__${t}__v0;" 2>/dev/null)
  v1=$(sqlite3 "file:$S?mode=ro" "SELECT count(*) FROM ctox_business_os__${t}__v1;" 2>/dev/null)
  echo "  $t: v0=${v0:-?}  v1=${v1:-?}" | tee -a "$OUT"
done
curl -s --max-time 30 "http://127.0.0.1:$PORT/installed-modules/sellify/schema.js" 2>/dev/null \
  | grep -c '"version": 1' | awk '{print "  ausgelieferte Deklarationen version:1 = "$1"  (1 = alter Stand, 5 = repariert)"}' | tee -a "$OUT"

echo "--- Platte ---" | tee -a "$OUT"
df -H /System/Volumes/Data | tail -1 | awk '{print "  frei: "$4"  ("$5" belegt)"}' | tee -a "$OUT"

echo "  → $OUT"
