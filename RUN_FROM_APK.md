# Running the demo from a pre-built APK (no source needed)

This is the **operator guide**: given a ready `app-debug.apk`, install it and run the demo on the
RayNeo X3 Pro — **no Android Studio, no Gradle, no source checkout, no model downloads**. Everything
the app needs (models, native libraries) is already bundled inside the APK.

> Building the APK from source? See [`README.md`](README.md) instead.

---

## Can it run with only the APK?

**Yes.** For an **on-device (`ON_DEVICE` / TinyCLIP)** build the whole pipeline — detection,
embedding, the Qdrant Edge vector store, and voice search — runs **on the glasses**. No Mac, no
server, no network. You only need the APK, a way to install it (adb), and the glasses.

A **`MAC_ENDPOINT` / SigLIP2** build is also just an APK, but its embedding step runs on a Mac
server, so it additionally needs that server running and reachable (see the last section).

**How to tell which build you have:** after launching (below), `adb logcat` prints
`object mode ready (backend=ON_DEVICE, dim=512, …)` or `backend=MAC_ENDPOINT, dim=768`. `ON_DEVICE`
needs nothing else.

---

## What you need

- The **`app-debug.apk`** file (~730 MB — it bundles the models).
- **RayNeo X3 Pro** glasses + a USB-C **data** cable.
- Any computer with **`adb`** — the standalone
  [Android platform-tools](https://developer.android.com/tools/releases/platform-tools) are enough
  (~15 MB; no full SDK, no Android Studio).

---

## 1. Connect the glasses

Plug in over USB, then on the **glasses' display** tap **Allow** when the "Allow USB debugging?"
dialog appears (tick "always allow from this computer").

```bash
adb devices        # should list  your serial + device, product:RayNeoX3Pro
```

If more than one device is listed, set the serial for every command below:
```bash
export S=<serial>      # your glasses' serial from `adb devices`
```
(If you have only the glasses connected, you can drop the `-s "$S"` from the commands.)

---

## 2. Install

```bash
adb -s "$S" install -r app-debug.apk

# grant camera + microphone once
adb -s "$S" shell pm grant tech.qdrant.glasses android.permission.CAMERA
adb -s "$S" shell pm grant tech.qdrant.glasses android.permission.RECORD_AUDIO
```

Launch it (or just tap the app on the glasses):
```bash
adb -s "$S" shell am start -n tech.qdrant.glasses/.MainActivity
```

You're done — for an `ON_DEVICE` build the demo now works standalone.

---

## 3. Use it

Interaction is the RayNeo **action button** + short taps:

| Do this | Result |
|---|---|
| **Long-press** the action button (≥1.2 s) | **Start recording / indexing** — objects in view get detected, embedded, and saved to memory. |
| **Press** again while recording | **Stop.** |
| **Short tap** (when idle, after you've indexed something) | **Voice query** — say *"where is my keys"*; results appear on the lens. |
| **Tap** on a results screen | next result; tap past the last → back to idle. |

**Indexing:** point at **distinct** objects and hold the recording; each object is saved once
(seen across a few frames, then de-duplicated). Pan slowly across a desk/room.

**Searching:** say the **object's name**. On a TinyCLIP (on-device) build, use the **exact object
word** it detects ("chair", "cup", "keyboard", "laptop") rather than synonyms — its on-device search
matches best on the exact label. You can only find things you actually indexed.

---

## 4. Reset the memory (start fresh)

No scripts needed — wipe the on-glasses store directly (works because the debug APK is debuggable):

```bash
adb -s "$S" shell am force-stop tech.qdrant.glasses
adb -s "$S" shell run-as tech.qdrant.glasses sh -c 'rm -rf files/objects_shard_* files/object_thumbs'
adb -s "$S" shell am start -n tech.qdrant.glasses/.MainActivity
```

---

## 5. (Optional) See the memory in a browser — HUD dashboard

The app can mirror the live feed + the object rail to a laptop browser over USB:

```bash
adb -s "$S" forward tcp:8081 tcp:8080
# then open http://localhost:8081  in a browser
```
Keep the USB cable connected; the dashboard shows the camera feed and the saved-object rail with
thumbnails.

---

## Troubleshooting

**`adb devices` doesn't list the glasses** — re-plug and tap **Allow** on the glasses (a reboot
revokes USB-debugging authorization). Make sure it's a **data** cable, not charge-only.

**App shows an error screen on the lens** — a startup failure (e.g. missing accelerator library on
an unexpected device). The reason is on the screen and in `adb logcat`. The detector auto-falls-back
HTP → GPU → CPU, so most devices still run.

**Nothing gets saved when indexing** —
- On an **`ON_DEVICE`** build this shouldn't happen (no network needed). Check `adb logcat` for
  `object stored: …` while recording; if you see none, make sure the glasses are **awake** and the
  camera is pointed at real objects.
- On a **`MAC_ENDPOINT`** build, embedding needs the Mac server — see below.

**`MAC_ENDPOINT` build only — the Mac server & relay.** This build embeds crops on a Mac server
(`embed_server`, port 9000). Start that server on a Mac on the same WiFi, then point the glasses at
the Mac's IP and relaunch:
```bash
MAC_IP=<the Mac's LAN IP>     # e.g. 192.168.1.42
adb -s "$S" shell setprop persist.qdrant.relay "http://$MAC_IP:9000"
adb -s "$S" shell am force-stop tech.qdrant.glasses
adb -s "$S" shell am start -n tech.qdrant.glasses/.MainActivity
# verify reachability (want 200):
adb -s "$S" shell "curl -s -m5 -o /dev/null -w '%{http_code}\n' http://$MAC_IP:9000/poll"
```
A **stale relay IP** (from a previous network) is the #1 reason a `MAC_ENDPOINT` build silently
stores nothing — every embed times out. Re-point it at the Mac's *current* IP.

---

*For building the APK from source, the architecture, and the full script set, see
[`README.md`](README.md).*
