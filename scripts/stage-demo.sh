#!/usr/bin/env bash
# One-command stage prep: point the glasses at THIS Mac's current IP and bring the whole
# wireless demo up. Run it after joining the venue/hotspot network (glasses on USB once,
# to receive the setprop; unplug after it prints READY).
#
#   ./scripts/stage-demo.sh          # auto-detect IP (first active interface)
#   ./scripts/stage-demo.sh 1.2.3.4  # force an IP
set -euo pipefail

# Device serial. Defaults to the only attached device, so a fresh clone works with no config.
# With several devices attached (an emulator counts!), set it explicitly:
#   GLASSES_SERIAL=XXXXXXXX ./scripts/<script>.sh
SERIAL="${GLASSES_SERIAL:-$(adb devices | awk 'NR>1 && $2=="device" {print $1}' | grep -v '^emulator' | head -1)}"
[ -n "$SERIAL" ] || { echo "ERROR: no glasses found over adb. Plug them in, or set GLASSES_SERIAL."; exit 1; }
PORT=9000
# Host-side relay + SigLIP2 embed server (separate repo:
# https://github.com/qdrant-labs/edge-mission-control). Only needed for the MAC_ENDPOINT backend
# and the HUD dashboard; an ON_DEVICE demo needs no host at all. Override, or point it at nothing
# to skip the host half entirely:
#   EMBED_DIR=/nonexistent ./scripts/stage-demo.sh
EMBED_DIR="${EMBED_DIR:-$HOME/Work/edge-mission-control}"

# ── Mac IP on the current network ────────────────────────────────────────────
IP="${1:-}"
if [ -z "$IP" ]; then
  # Order matters: en0 is WiFi on MacBooks; fall back to the default-route interface
  # (hotspots sometimes come up as en1/bridge).
  IP=$(ipconfig getifaddr en0 2>/dev/null || true)
  if [ -z "$IP" ]; then
    IFACE=$(route -n get default 2>/dev/null | awk '/interface:/{print $2}')
    [ -n "${IFACE:-}" ] && IP=$(ipconfig getifaddr "$IFACE" 2>/dev/null || true)
  fi
fi
[ -z "$IP" ] && { echo "ERROR: can't detect Mac IP — join a network first (or pass IP as arg)"; exit 1; }
RELAY="http://$IP:$PORT"
echo "Mac relay: $RELAY"

# ── embed server (SigLIP2 + relay) ───────────────────────────────────────────
# ALWAYS restart: the relay keeps the HUD rail (object cards) in memory, so a
# leftover server shows yesterday's objects on the dashboard. Fresh start = clean rail.
RELAY_OK=0
if [ -d "$EMBED_DIR" ]; then
  # `|| true`: under `set -o pipefail`, an lsof that finds nothing listening exits 1 — and `set -e`
  # then killed this script HERE, SILENTLY, before the glasses were touched at all. "Relay not
  # running" is the normal case, not an error.
  OLD_PID=$(lsof -tnP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1 || true)
  [ -n "$OLD_PID" ] && { echo "restarting embed_server (clearing relay rail)..."; kill "$OLD_PID"; sleep 2; }
  # HF_HUB_OFFLINE: models are already cached; without it startup stalls on HuggingFace
  # HEAD checks over slow venue WiFi.
  (cd "$EMBED_DIR" && HF_HUB_OFFLINE=1 nohup uv run uvicorn embed_server:app --host 0.0.0.0 --port $PORT \
      > /tmp/embed_server.log 2>&1 &)
  # Bounded at 40s: an unbounded `until curl` hung forever whenever the relay could not come up,
  # stalling venue prep before it had done anything useful.
  printf "embed_server starting"
  for _ in $(seq 1 20); do
    if curl -s -m 2 -o /dev/null "http://localhost:$PORT/poll"; then RELAY_OK=1; break; fi
    sleep 2; printf .
  done
  [ "$RELAY_OK" = 1 ] && echo " UP (fresh)" || echo " TIMED OUT"
else
  echo "relay: $EMBED_DIR not found — skipping the host half"
fi

if [ "$RELAY_OK" != 1 ]; then
  echo "WARNING: no Mac relay. Fine for an ON_DEVICE demo (it needs no host); a MAC_ENDPOINT build"
  echo "         cannot embed, and the HUD dashboard will not render. Continuing with the"
  echo "         glasses-side prep, which is the half that matters."
fi

# ── glasses: point at this Mac, restart app ──────────────────────────────────
adb -s "$SERIAL" get-state >/dev/null 2>&1 || { echo "ERROR: glasses not on USB (needed once for setprop)"; exit 1; }

# Fresh memory on EVERY run: wipe the object store + thumbs on the glasses. The host side
# (relay rail + thumbs, kept in RAM) is already cleared by the embed_server restart above —
# both sides must be wiped together or the dashboard shows ghost cards.
adb -s "$SERIAL" shell am force-stop tech.qdrant.glasses
# Glob, not a hand-written list: CropEncoderFactory.namespace mints a new shard dir per backend,
# and a list silently misses any new one — leaving a stale shard that keeps answering dedup queries.
# moments_shard_*/moment_thumbs are the episodic-memory plan's opt-in moment path (Task 1.5,
# QdrantEdgeMomentStore) — same per-namespace glob rule, or a fresh venue run leaves a ghost
# moment shard behind exactly the way a stale objects_shard_* used to (Spec §7 / CLAUDE.md).
adb -s "$SERIAL" shell run-as tech.qdrant.glasses sh -c "'rm -rf files/objects_shard_* files/object_thumbs files/moments_shard_* files/moment_thumbs'" 2>/dev/null || true
echo "glasses memory wiped"

adb -s "$SERIAL" shell setprop persist.qdrant.relay "$RELAY"
adb -s "$SERIAL" shell setprop debug.qdrant.relay "$RELAY"
adb -s "$SERIAL" shell svc wifi enable
# UVLO guard: Battery Saver auto-clears while CHARGING and does not come back on unplug,
# so setting low_power here (on USB) is useless. Instead set the AUTO trigger to 100% —
# saver then engages by itself the moment the glasses leave the cable, at any battery level.
adb -s "$SERIAL" shell settings put global low_power_trigger_level 100
LP_TRIG=$(adb -s "$SERIAL" shell settings get global low_power_trigger_level | tr -d '\r')
[ "$LP_TRIG" = "100" ] || echo "WARNING: battery-saver auto-trigger not set (got: $LP_TRIG) — set Battery Saver manually before unplugging!"
adb -s "$SERIAL" shell am force-stop tech.qdrant.glasses
adb -s "$SERIAL" shell input keyevent KEYCODE_WAKEUP
adb -s "$SERIAL" shell am start -n tech.qdrant.glasses/.MainActivity >/dev/null

# ── verify the rail end-to-end ───────────────────────────────────────────────
# A pushing glasses client = an ESTABLISHED TCP connection into :9000 from a non-local
# address (survives log-file rotation/deletion, unlike grepping the server log).
echo -n "waiting for frames from glasses"
for i in $(seq 1 20); do
  sleep 2; printf .
  if lsof -nP -iTCP:$PORT -sTCP:ESTABLISHED 2>/dev/null | grep -vE "127.0.0.1|\[::1\]" | grep -qF -- "->"; then
    echo; echo "READY — glasses are pushing to $RELAY"
    echo "Dashboard on this Mac:  http://localhost:$PORT/"
    echo "Now: unplug USB, put the glasses on, demo away."
    exit 0
  fi
done
echo; echo "WARNING: no connection after 40s — check glasses WiFi (adb shell cmd wifi status) and the pusher log (adb logcat -s MjpegPusher)"
exit 1
