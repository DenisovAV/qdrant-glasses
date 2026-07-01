#!/usr/bin/env bash
# Keep the browser→glasses MJPEG tunnel alive.
#
# The RayNeo firmware blocks inbound TCP to app servers on wlan0 (WiFi), so the ONLY reliable
# path from the desktop browser to the on-glasses server is USB `adb forward`. That tunnel drops
# whenever the app restarts or USB hiccups, which looks like "the stream froze / went blank".
# This watchdog re-establishes it within ~2s of any drop.
#
# Usage:  ./scripts/stream-tunnel.sh
# Then open:  http://localhost:8081   (HUD)  or  http://localhost:8081/stream (raw MJPEG)
#
# Also (re)creates the embed tunnel used by the Mac SigLIP2 crop encoder.

set -u
HOST_HUD_PORT=8081       # desktop port → device 8080 (the MJPEG/HUD server)
DEV_HUD_PORT=8080
EMBED_PORT=9000          # device → desktop (Mac SigLIP2 endpoint), reverse
CHECK_EVERY=2

echo "stream-tunnel watchdog: keeping localhost:${HOST_HUD_PORT} → glasses:${DEV_HUD_PORT} alive"
echo "open http://localhost:${HOST_HUD_PORT}  (Ctrl-C to stop)"

last_state=""
while true; do
  if ! adb get-state >/dev/null 2>&1; then
    [ "$last_state" != "nodev" ] && echo "$(date +%H:%M:%S) waiting for device (USB)..."
    last_state="nodev"
    sleep "$CHECK_EVERY"; continue
  fi

  # forward: browser → glasses HUD/stream
  if ! adb forward --list 2>/dev/null | grep -q "tcp:${HOST_HUD_PORT} tcp:${DEV_HUD_PORT}"; then
    adb forward "tcp:${HOST_HUD_PORT}" "tcp:${DEV_HUD_PORT}" >/dev/null 2>&1 \
      && echo "$(date +%H:%M:%S) (re)forwarded ${HOST_HUD_PORT}→${DEV_HUD_PORT}"
  fi

  # reverse: glasses → Mac embed endpoint (only matters when the SigLIP2 server is running)
  if ! adb reverse --list 2>/dev/null | grep -q "tcp:${EMBED_PORT} tcp:${EMBED_PORT}"; then
    adb reverse "tcp:${EMBED_PORT}" "tcp:${EMBED_PORT}" >/dev/null 2>&1 \
      && echo "$(date +%H:%M:%S) (re)reversed embed ${EMBED_PORT}"
  fi

  if [ "$last_state" != "up" ]; then echo "$(date +%H:%M:%S) tunnels up"; last_state="up"; fi
  sleep "$CHECK_EVERY"
done
