#!/usr/bin/env bash
# RayNeo drops wlan0 on its power policy. This re-enables WiFi within ~3s of any drop so the
# wireless demo (glasses push → Mac) stays connected. Run alongside the demo.
#
# Pinned to the USB serial: with WiFi adb also connected (adb connect <ip>:5555), a bare
# `adb shell` dies with "more than one device" and the watchdog silently stops working.
set -u
SERIAL="${GLASSES_SERIAL:-YOUR_GLASSES_SERIAL}"
echo "wifi-keepalive: keeping glasses WiFi up via $SERIAL (Ctrl-C to stop)"
while true; do
  if adb -s "$SERIAL" get-state >/dev/null 2>&1; then
    if ! adb -s "$SERIAL" shell ip addr show wlan0 2>/dev/null | grep -q "state UP"; then
      adb -s "$SERIAL" shell svc wifi enable 2>/dev/null
      echo "$(date +%H:%M:%S) WiFi was down → re-enabled"
    fi
  fi
  sleep 3
done
