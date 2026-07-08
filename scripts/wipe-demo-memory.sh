#!/usr/bin/env bash
# Wipe the demo memory EVERYWHERE — the only correct way to "почистить базу".
#
# The object memory lives in TWO places that must be cleared together:
#   1. Glasses: Qdrant Edge shards + thumbnail JPEGs (files/objects_shard_*, object_thumbs)
#   2. Mac relay: HUD rail + pushed thumbs, held in embed_server process RAM
# Wiping only one side leaves ghost cards on the dashboard (bitten twice at rehearsals).
#
# Usage: ./scripts/wipe-demo-memory.sh
set -euo pipefail

SERIAL="${GLASSES_SERIAL:-YOUR_GLASSES_SERIAL}"
PORT=9000
EMBED_DIR="$HOME/Work/edge-mission-control"

# 1) Host first: restart the relay (its rail/thumbs live in process memory).
OLD_PID=$(lsof -tnP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1)
if [ -n "$OLD_PID" ]; then kill "$OLD_PID"; sleep 2; fi
(cd "$EMBED_DIR" && HF_HUB_OFFLINE=1 nohup uv run uvicorn embed_server:app --host 0.0.0.0 --port $PORT \
    > /tmp/embed_server.log 2>&1 &)
printf "relay restarting"
until curl -s -m 2 -o /dev/null "http://localhost:$PORT/poll"; do sleep 2; printf .; done
echo " — FRESH"

# 2) Glasses: stop the app, remove shards + thumbs.
adb -s "$SERIAL" get-state >/dev/null 2>&1 || { echo "ERROR: glasses not reachable over adb"; exit 1; }
adb -s "$SERIAL" shell am force-stop tech.qdrant.glasses
adb -s "$SERIAL" shell run-as tech.qdrant.glasses sh -c "'rm -rf files/objects_shard_mac files/objects_shard_ondevice files/object_thumbs'" 2>/dev/null || true
echo "glasses memory wiped"

# 3) Bring the app back and prove both sides are empty.
adb -s "$SERIAL" shell am start -n tech.qdrant.glasses/.MainActivity >/dev/null
RAIL=$(curl -s "http://localhost:$PORT/poll?since=-1")
echo "relay rail after wipe: $RAIL"
echo "DONE — refresh the dashboard tab (it may still RENDER old cards until reloaded)."
