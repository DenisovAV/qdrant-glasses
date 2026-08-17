#!/usr/bin/env bash
# Wipe the demo memory EVERYWHERE — the only correct way to "почистить базу".
#
# The object memory lives in TWO places that must be cleared together:
#   1. Glasses: Qdrant Edge shards + thumbnail JPEGs — the crop-store path
#      (files/objects_shard_*, object_thumbs) AND the opt-in episodic-memory moment path
#      (files/moments_shard_*, moment_thumbs — episodic-memory plan Task 1.7, Spec §7)
#   2. Mac relay: HUD rail + pushed thumbs, held in embed_server process RAM
# Wiping only one side leaves ghost cards on the dashboard (bitten twice at rehearsals).
#
# Usage: ./scripts/wipe-demo-memory.sh
set -euo pipefail

# Device serial. Defaults to the only attached device, so a fresh clone works with no config.
# With several devices attached (an emulator counts!), set it explicitly:
#   GLASSES_SERIAL=XXXXXXXX ./scripts/<script>.sh
SERIAL="${GLASSES_SERIAL:-$(adb devices | awk 'NR>1 && $2=="device" {print $1}' | grep -v '^emulator' | head -1)}"
[ -n "$SERIAL" ] || { echo "ERROR: no glasses found over adb. Plug them in, or set GLASSES_SERIAL."; exit 1; }
PORT=9000
# Overridable: `EMBED_DIR=/nonexistent ./scripts/wipe-demo-memory.sh` skips the relay entirely.
# Useful for a no-HUD run (nothing is pushed, so the relay has no rail to clear) and when the relay
# checkout simply isn't on this machine.
EMBED_DIR="${EMBED_DIR:-$HOME/Work/edge-mission-control}"

# 1) Host first: restart the relay (its rail/thumbs live in process memory).
#
# The relay holds the HUD rail even in the ON_DEVICE crop-encoder mode, where nothing is embedded
# on the Mac at all — the rail is fed by MjpegPusher/Config.WIRELESS, which is independent of
# CropEncoderFactory.backend. So we still restart it. But we must NOT let it block the glasses
# wipe: this step used to be `until curl ...; do sleep 2; done` with no timeout, and it hung
# FOREVER whenever the relay could not come up (no embed_server checkout, no `uv`, or simply an
# on-device demo where nobody had started it) — hanging BEFORE the glasses were wiped, i.e.
# failing at the one job the script is run for. Bound the wait, warn loudly, carry on.
RELAY_OK=0
if [ -d "$EMBED_DIR" ]; then
    # `|| true`: with `set -o pipefail`, an lsof that finds nothing listening exits 1, which under
    # `set -e` killed this script on its second line — SILENTLY, before wiping anything. "Relay not
    # running" is the normal case for an ON_DEVICE demo, not an error.
    OLD_PID=$(lsof -tnP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | head -1 || true)
    if [ -n "$OLD_PID" ]; then kill "$OLD_PID"; sleep 2; fi
    (cd "$EMBED_DIR" && HF_HUB_OFFLINE=1 nohup uv run uvicorn embed_server:app --host 0.0.0.0 --port $PORT \
        > /tmp/embed_server.log 2>&1 &)
    printf "relay restarting"
    for _ in $(seq 1 20); do          # 40s ceiling — a healthy relay answers in ~5s
        if curl -s -m 2 -o /dev/null "http://localhost:$PORT/poll"; then RELAY_OK=1; break; fi
        sleep 2; printf .
    done
    [ "$RELAY_OK" = 1 ] && echo " — FRESH" || echo " — TIMED OUT"
else
    echo "relay: $EMBED_DIR not found — skipping"
fi

if [ "$RELAY_OK" != 1 ]; then
    echo "WARNING: the Mac relay is NOT running, so its in-RAM rail was not cleared."
    echo "         Harmless if you are not using the HUD dashboard (ON_DEVICE demos need no relay)."
    echo "         If the dashboard IS up, it will keep showing ghost cards: restart the relay"
    echo "         (see $EMBED_DIR) and reload the tab. Wiping the glasses anyway —"
    echo "         that is the half that actually matters."
fi

# 2) Glasses: stop the app, remove shards + thumbs.
adb -s "$SERIAL" get-state >/dev/null 2>&1 || { echo "ERROR: glasses not reachable over adb"; exit 1; }
adb -s "$SERIAL" shell am force-stop tech.qdrant.glasses
# Glob, not a hand-written list of namespaces: CropEncoderFactory.namespace grows a new shard dir
# whenever a backend is added, and a list silently misses it — leaving a stale shard that keeps
# answering dedup queries. (A bench build with an extra `objects_shard_ondevice_mclip` did exactly
# this: "wiped" store, 35 objects still in it, every crop deduped away.)
# moments_shard_*/moment_thumbs are the episodic-memory plan's opt-in moment path (Task 1.5,
# QdrantEdgeMomentStore) — same per-namespace glob rule, or a ghost moment shard survives a wipe
# exactly the way a stale objects_shard_* used to (Spec §7 / CLAUDE.md).
adb -s "$SERIAL" shell run-as tech.qdrant.glasses sh -c "'rm -rf files/objects_shard_* files/object_thumbs files/moments_shard_* files/moment_thumbs'" 2>/dev/null || true
echo "glasses memory wiped"

# 3) Re-point the relay property (setprop does NOT survive a reboot — a glasses
# reboot silently reverts the app to the compiled-in home IP) and bring the app back.
IP=$(ipconfig getifaddr en0 2>/dev/null || true)
[ -n "$IP" ] && adb -s "$SERIAL" shell setprop persist.qdrant.relay "http://$IP:$PORT" && adb -s "$SERIAL" shell setprop debug.qdrant.relay "http://$IP:$PORT"
adb -s "$SERIAL" shell am start -n tech.qdrant.glasses/.MainActivity >/dev/null

# Print the (empty) rail as proof — but only if there is a relay to ask. Under `set -e` an
# unreachable relay made this curl abort the script AFTER the wipe had already happened, so the
# run looked like a failure when it had in fact succeeded.
if [ "$RELAY_OK" = 1 ]; then
    RAIL=$(curl -s "http://localhost:$PORT/poll?since=-1" || echo "<relay unreachable>")
    echo "relay rail after wipe: $RAIL"
    echo "DONE — refresh the dashboard tab (it may still RENDER old cards until reloaded)."
else
    echo "DONE — glasses wiped. (No relay was running; nothing to prove on the host side.)"
fi
