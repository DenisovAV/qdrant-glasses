# Qdrant Glasses — on-device object memory for AR glasses

An **on-device multimodal-RAG "object memory"** demo for the **RayNeo X3 Pro** AR glasses
(Snapdragon AR1 Gen 1 + Hexagon HTP). A lost-and-found for the real world:

```
camera → detect objects (YOLOv8n on HTP) → track/dedup → embed each crop (CLIP-family)
       → store the vector on the glasses (Qdrant Edge, Rust FFI) → ask by voice → results on the lens
```

Everything that matters runs **on the glasses**: detection on the NPU, a Qdrant Edge vector shard
on local storage, and (in on-device mode) CLIP embedding too. A browser **HUD dashboard** mirrors the
live feed + object memory to any laptop for the audience.

> **Just have the pre-built APK and want to run the demo?** You don't need this repo or a toolchain —
> see **[`RUN_FROM_APK.md`](RUN_FROM_APK.md)**. An on-device (TinyCLIP) build is fully self-contained
> and runs on the glasses with no Mac, no server, and no network. This README is the **developer**
> guide: building from source, the architecture, and the demo-ops scripts.

---

## Table of contents
- [What you need](#what-you-need)
- [First-time setup](#first-time-setup)
- [Build & install](#build--install)
- [The two embedding backends](#the-two-embedding-backends)
- [Running a demo](#running-a-demo)
- [How to use it (on the glasses)](#how-to-use-it-on-the-glasses)
- [The HUD dashboard](#the-hud-dashboard)
- [Demo operations (wipe / stage)](#demo-operations)
- [Scripts reference](#scripts-reference)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Repo layout](#repo-layout)

---

## What you need

**Hardware**
- **RayNeo X3 Pro** glasses (model `ARGF20`, **Snapdragon AR1 Gen 1** — 4 cores, Hexagon V73
  HTP, Adreno GPU). USB-C data cable.
- A **Mac/Linux laptop** for building + the HUD dashboard (and the Mac embedding server, if you use
  the `MAC_ENDPOINT` backend).

**Toolchain**
- **JDK 17–21**, **Android SDK** (compileSdk 36, minSdk 26), Android Gradle Plugin 9.1.1 (wired via
  the Gradle wrapper — just use `./gradlew`).
- **adb** on your `PATH`.

> If your `JAVA_HOME` is a JDK the Android Gradle Plugin rejects (a too-new one fails at `jlink`),
> pin an acceptable one in your **user-level** Gradle config — not in the project's committed
> `gradle.properties`, which would break everyone else's clone:
> ```properties
> # ~/.gradle/gradle.properties
> org.gradle.java.home=/Applications/Android Studio.app/Contents/jbr/Contents/Home
> ```

### Binaries you have to supply

**The Qdrant Edge AARs are committed** (Apache-2.0 — see [`NOTICE`](NOTICE)); Qdrant has no Maven
artifact for Android, and nothing builds without them. **Model weights are not**: they are large, and
YOLOv8's licence makes vendoring it a problem (below). So a clone **compiles** as-is, but fails at
startup until the models are on disk.

| Path | What | Where to get it | Needed for |
|---|---|---|---|
| `app/libs/*.aar` | Qdrant Edge — embedded vector DB (Rust, via JNA) | ***committed*** — nothing to do | **everything** (the vector store) |
| `app/src/main/assets/clip-tokenizer.json` | CLIP BPE tokenizer | ***committed*** | ON_DEVICE text queries |
| `libs/sherpa-onnx-static-link-onnxruntime-*.aar` | sherpa-onnx Android runtime, ~36 MB | [releases](https://github.com/k2-fsa/sherpa-onnx/releases) → `libs/` | **compile** (ASR runtime) |
| `app/src/main/assets/tinyclip-int8.onnx` | TinyCLIP-40M int8, ~86 MB | quantize [`wkcn/TinyCLIP-ViT-40M-32-Text-19M`](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M) | **ON_DEVICE** embedding |
| `app/src/main/assets/detect/qnn/yolov8_det.onnx` + `.data` | int8 YOLOv8n, QDQ ONNX for the QNN EP | export it yourself — see below | **default HTP detector** |
| `app/src/main/assets/detect/yolov8n.tflite`, `efficientdet_lite0.tflite` | GPU/CPU detector fallbacks | Ultralytics / MediaPipe releases | detector fallback chain |
| `app/src/main/assets/vad/`, `moonshine/`, `sherpa/` | Silero VAD + Moonshine offline ASR | sherpa-onnx model zoo | ambient transcription (LEGACY) |
| `app/src/main/assets/bge/` | bge-small text encoder | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | LEGACY "heard" channel |

> Only the **sherpa AAR** gates compilation (without it Kotlin fails on `Unresolved reference
> 'k2fsa'`). Everything else in the table is a runtime asset.

**Exporting the HTP detector.** The int8 YOLOv8n is produced with
[qai-hub](https://aihub.qualcomm.com/); the export must be a **static QDQ ONNX**, not a precompiled
QNN context binary (that binary is SoC-locked and fails to load on the AR1):

```bash
pip install qai-hub-models
python -m qai_hub_models.models.yolov8_det.export \
    --target-runtime onnx --quantize w8a8 --device "Samsung Galaxy S22 5G"
# → yolov8_det.onnx + yolov8_det.data  →  app/src/main/assets/detect/qnn/
```

ORT compiles the HTP context on-device on first run (~9 s, then cached). Input is uint8 **NCHW**
`[1,3,640,640]` — NHWC throws `ORT_INVALID_ARGUMENT`. See `YoloQnnDetector.kt`.

> **A licensing note.** This repo is Apache-2.0 ([`LICENSE`](LICENSE)), and so is the Qdrant Edge
> AAR it bundles ([`NOTICE`](NOTICE)). **YOLOv8 is Ultralytics AGPL-3.0** — a copyleft licence — so
> its weights are deliberately *not* vendored here: committing them would impose AGPL on this whole
> repository. Export your own, and mind the licence your use requires.

> **Why nothing heavy is in git.** The whole-frame CLIP weights alone are ~945 MB; bundling
> everything pushed the APK to ~1.8 GB and made USB installs flaky. Only what a given mode needs is
> kept in the APK (see `androidResources.ignoreAssetsPattern` in `app/build.gradle.kts`).

---

## First-time setup

1. **`local.properties`** (gitignored) — point Gradle at your SDK, and optionally add the Google STT
   key (only needed for the cloud-STT fallback; the on-device Android recognizer works without it):

   ```properties
   sdk.dir=/Users/you/Library/Android/sdk
   GOOGLE_STT_API_KEY=your-key-or-leave-blank
   ```

2. **Place the model assets & sherpa AAR** (see the table above).

3. **Plug in the glasses**, enable USB debugging, and tap **Allow** on the on-glasses prompt:

   ```bash
   adb devices          # should list your serial + device, product:RayNeoX3Pro
   ```

   The scripts auto-detect the attached device (an emulator is skipped). Override with
   `export GLASSES_SERIAL=<your-serial>`.

---

## Build & install

```bash
./gradlew :app:assembleDebug
adb -s "$GLASSES_SERIAL" install -r app/build/outputs/apk/debug/app-debug.apk

# grant runtime permissions once (camera + mic)
adb -s "$GLASSES_SERIAL" shell pm grant tech.qdrant.glasses android.permission.CAMERA
adb -s "$GLASSES_SERIAL" shell pm grant tech.qdrant.glasses android.permission.RECORD_AUDIO
```

Run the unit tests with `./gradlew :app:testDebugUnitTest`.

---

## The two embedding backends

Object crops are embedded into vectors by one of two backends, selected by a single constant in
`app/src/main/java/tech/qdrant/glasses/embedding/CropEncoder.kt`:

```kotlin
object CropEncoderFactory {
    val backend = Backend.ON_DEVICE   // or Backend.MAC_ENDPOINT
}
```

| | `ON_DEVICE` | `MAC_ENDPOINT` |
|---|---|---|
| Model | TinyCLIP-40M, **512-dim**, on the glasses | SigLIP2-base on the Mac, **768-dim** |
| Network | **none** — fully autonomous | needs a reachable Mac relay |
| Search quality | weaker (small/cluttered objects, synonyms) | stronger semantic separation |
| Speed | **~200 ms** / crop on CPU¹ | ~1 s / crop incl. WiFi round-trip |
| Storage namespace | `objects_shard_ondevice` | `objects_shard_mac` |

¹ Measured with the HUD dashboard off; **~870 ms with it on** — the JPEG encode competes for the
same four cores (see Architecture). CPU is not a fallback here, it is the fastest route this SoC has
for a ViT: every accelerator was measured and lost. The startup log says so out loud —
`ClipEncoder: no NNAPI EP in this ORT build (expected) — running CLIP on CPU`. The full reasoning,
with the graph-partition counts, is in the KDoc on `createAcceleratedSession`.

> **The two backends use different dims and different vector spaces.** They index into **separate**
> on-disk shards, so switching is safe — but each backend has its own memory. Objects indexed under
> one backend are invisible under the other. **After changing the backend, rebuild & reinstall.**

**Recommendation:** for a network-free, self-contained demo use `ON_DEVICE`. For the best search
relevance (and if you can run the Mac server on the same LAN) use `MAC_ENDPOINT`.

---

## Running a demo

### Option A — On-device (`ON_DEVICE` / TinyCLIP), no relay
Nothing to set up. Build with `backend = Backend.ON_DEVICE`, install, and go — embedding, storage,
and search all run on the glasses.

### Option B — With the Mac server (`MAC_ENDPOINT` / SigLIP2)
Crop embedding runs on a **separate repo**, the Mac `embed_server` (SigLIP2 + the HUD relay), which
lives in [qdrant-labs/edge-mission-control](https://github.com/qdrant-labs/edge-mission-control) and
serves on **port 9000**. Clone it wherever you like; the scripts look in `~/Work/edge-mission-control`
by default and take an `EMBED_DIR=` override. The glasses reach it either over
WiFi (same LAN) or over USB (`adb reverse`).

**One-command venue prep** (recommended — points the glasses at *this* Mac's current IP, restarts the
server, wipes memory for a fresh run, enables the UVLO Battery-Saver guard, verifies frame flow):

```bash
./scripts/stage-demo.sh            # auto-detect the Mac's IP
./scripts/stage-demo.sh 1.2.3.4    # or force an IP
```

**Manual** (if you don't want a memory wipe): point the relay at the Mac's current IP and restart the
app (the relay is read once at startup):

```bash
MAC_IP=$(ipconfig getifaddr en0)
adb -s "$GLASSES_SERIAL" shell setprop persist.qdrant.relay "http://$MAC_IP:9000"
adb -s "$GLASSES_SERIAL" shell am force-stop tech.qdrant.glasses
adb -s "$GLASSES_SERIAL" shell am start -n tech.qdrant.glasses/.MainActivity
```

`persist.qdrant.relay` survives reboots; `debug.qdrant.relay` is a volatile override that wins over
it. **The relay endpoint is the #1 demo gotcha** — see [Troubleshooting](#troubleshooting).

---

## How to use it (on the glasses)

Interaction is the RayNeo **action button** (a system broadcast, keycode 289) plus taps:

| Action | What it does |
|---|---|
| **Long-press** the action button (in Idle, ≥1.2 s) | **Start recording / indexing** — the app detects objects, embeds their crops, and stores them in memory. |
| **Release** (within ~1 s of start) | keeps recording (the start-press release is ignored). |
| **Press** while recording | **Stop** recording. |
| **Tap** (short press) in Idle | **Start a voice query** → speak *"where is my keys"* → results appear on the lens. |
| **Tap** on a results screen | advance to the next result card; tap past the last → back to Idle. |

**Indexing tips**
- An object is committed to memory only after it's seen with confidence across **~3 frames**, then
  **de-duplicated** (cosine ≥ 0.90) — so panning across a scene stores each object once.
- Point at **distinct** objects; the same object won't be stored twice.

**Voice-search tips**
- Say the **object name** ("where is my cup"). Question words are stripped automatically.
- Search matches on **cosine OR exact detector-label word**. On `ON_DEVICE`/TinyCLIP the cosine gate
  is high relative to its score scale, so it behaves mostly as **exact-label** search (say the COCO
  label — "chair", "cup", "keyboard" — not synonyms like "seat"/"sofa"). On `MAC_ENDPOINT`/SigLIP2
  semantic search works better.
- Searching for something you never indexed correctly returns **"nothing found"**.

---

## The HUD dashboard

The glasses push the live feed + the object-memory rail to a browser dashboard.

- **`MAC_ENDPOINT` (relay):** open **`http://<mac-ip>:9000/`** in a browser — the object rail with
  thumbnails, live feed, and search results. The relay keeps rail state in RAM.
- **Wired / USB (on-glasses server):** the app also serves an MJPEG + SSE HUD on the device at
  `:8080`; reach it over USB with `./scripts/stream-tunnel.sh` (RayNeo firmware blocks inbound TCP on
  WiFi, so browser→glasses only works via `adb forward`), then open `http://localhost:8081`.

---

## Demo operations

**Object memory lives in TWO places** — the glasses (Qdrant Edge shard + thumbnail JPEGs) **and** the
Mac relay's process RAM (the HUD rail + pushed thumbs). Wiping only one side leaves ghost cards.

**Always wipe with the script** (restarts the relay, wipes the glasses, relaunches, prints the empty
rail as proof):

```bash
./scripts/wipe-demo-memory.sh
```

A fresh `./scripts/stage-demo.sh` also wipes both sides — a fresh run is a fresh demo.

---

## Scripts reference

| Script | Purpose |
|---|---|
| `scripts/stage-demo.sh` | **One-command venue prep**: point the glasses at this Mac's IP, restart the embed server, wipe memory, enable WiFi + Battery-Saver (UVLO guard), relaunch, verify the glasses are pushing frames. |
| `scripts/wipe-demo-memory.sh` | Wipe the demo memory on **both** sides (glasses shard + relay RAM). |
| `scripts/stream-tunnel.sh` | USB watchdog for the wired HUD: keeps `adb forward tcp:8081→8080` (browser→glasses) and `adb reverse tcp:9000` (glasses→Mac embed) alive. |
| `scripts/wifi-keepalive.sh` | Re-enables `wlan0` within ~3 s whenever RayNeo's power policy drops WiFi, so the wireless push stays connected. |

---

## Architecture

Single-Activity Android app, package `tech.qdrant.glasses`. The pipeline is the fault-line:

```
FrameCaptureManager (CameraX, ≤640×480, KEEP_ONLY_LATEST)
   └─ GlassesViewModel  (thin orchestrator + state machine)
        ├─ PerceptionPipeline   detect → track → stream-overlay → crop → embed → dedup → store → HUD
        │     ObjectDetector (YoloQnnDetector on HTP, GPU/MediaPipe fallback)
        │     ObjectTracker  (IoU tracking + confirm/dedup)
        │     CropEncoder    (TinyCLIP on-device | SigLIP2 on the Mac)
        │     ObjectStore    (Qdrant Edge shard, Rust FFI via JNA)
        ├─ ObjectSearcher    voice query → embed → vector search → cosine-OR-label gate → results
        ├─ AppStateHolder    the single owner of AppState (Loading/Idle/Recording/Listening/…)
        ├─ HudPublisher      the FrameSink (MjpegServer wired | MjpegPusher wireless)
        └─ GlassesComponents boot sequence + teardown
```

**Load-bearing details**
- **Concurrency:** three single-thread lanes — `inferLane` (the *only* legal thread for the
  non-thread-safe tracker + TFLite), `cropLane` (network/embed, off the detect path), `streamLane`
  (JPEG encode) — with `AtomicBoolean` drop-oldest backpressure gates.
- **Accelerators:** the int8 **CNN detector runs on the Hexagon HTP at ~8 ms**. The **CLIP/ViT
  embedder runs on the CPU (~200 ms)** — and that is the fastest this SoC can do it. Every
  accelerator route was implemented and measured, and every one lost: the HTP takes only **44 of a
  ViT's 490 graph nodes** (shredding the rest into 50 partitions), the Adreno GPU delegate 198/490,
  NNAPI 72/890. Each partition boundary is a round trip back to the CPU that costs more than the
  acceleration saves. The decisive variable is int8-vs-fp32, not CPU-vs-accelerator: an int8 graph
  on the CPU beats an fp32 graph on every accelerator here. (ORT's NNAPI path cannot work at all in
  this build — see the KDoc on `createAcceleratedSession`.)
- **Latency is load-dependent:** the AR1 is a **4-core** part, so the embedder shares cores with the
  camera's YUV conversion, the detector and the HUD's JPEG encode. Closing the HUD dashboard alone
  takes a crop embed from ~870 ms to ~200 ms (`setprop debug.qdrant.hud 0`). Benchmark with it off.
- **Power:** the voice-search current spike can brown-out the glasses (UVLO); Battery Saver mitigates
  it (set by `stage-demo.sh`). The camera is never power-cycled around search.

---

## Troubleshooting

**"Indexing doesn't store anything" (`MAC_ENDPOINT`)** — the relay is unreachable. Embedding runs on
the Mac; if `persist.qdrant.relay` points at a **stale IP** (a different network) every embed times
out and nothing stores. Fix: re-point the relay at the Mac's *current* IP (`stage-demo.sh`, or the
manual `setprop` above), or switch to `ON_DEVICE`. Confirm reachability:
```bash
adb -s "$GLASSES_SERIAL" shell 'curl -s -m5 -o /dev/null -w "%{http_code}\n" http://<mac-ip>:9000/poll'  # want 200
```

**"Voice search finds nothing"** — first, you can only find what you **indexed** (check the rail on
the dashboard). Second, on `ON_DEVICE`/TinyCLIP say the **exact COCO label** ("chair", not "seat") —
its semantic gate is conservative. For better semantic search use `MAC_ENDPOINT`.

**`adb` doesn't see the glasses after a reboot** — a reboot revokes USB-debugging authorization.
Re-plug and tap **Allow** on the glasses. WiFi-adb to a stale IP will time out; USB is the reliable
path.

**App stuck on "Loading" / crashes at start** — an init failure now surfaces as an on-lens **error
screen** (with a reason) instead of hanging. A missing HTP `.so` or a missing detector asset is the
usual cause; the detector falls back HTP → GPU → CPU automatically (logged at ERROR).

**TinyCLIP is slow / on CPU** — CPU is correct here, and it is the *fastest* route this SoC has for
a ViT; do not "fix" it by moving to an accelerator. Every route was measured and every one lost (see
Architecture). If it feels slower than ~200 ms, the cause is almost certainly **pipeline load, not
the encoder**: close the HUD dashboard (`setprop debug.qdrant.hud 0`) and it drops from ~870 ms to
~200 ms on the same four cores.

**Driving the app over adb (for testing)** — start/stop recording via the action-button broadcast:
```bash
BC=com.rayneo.key_pass_to_user
adb -s "$GLASSES_SERIAL" shell "am broadcast -a $BC --es data '{\"type\":\"ACTION_DOWN\",\"keyCode\":289}'"  # then sleep >1.2s (LONG_PRESS_MS)
adb -s "$GLASSES_SERIAL" shell "am broadcast -a $BC --es data '{\"type\":\"ACTION_UP\",\"keyCode\":289}'"    # release / stop
```
Watch logcat: `GlassesVM` (state/store/embed), `YoloQnnDetector` (`YOLO-QNN inference=Nms`),
`ObjectStore`, `ClipEncoder`. A successful store logs
`object stored: <label> (track N), total=M (embed=…ms qsearch=…ms upsert=…ms)`.

---

## Repo layout

```
app/src/main/java/tech/qdrant/glasses/
  Config.kt, MainActivity.kt, GlassesViewModel.kt, AppState.kt, AppStateHolder.kt, GlassesComponents.kt
  camera/     FrameCaptureManager
  detect/     ObjectDetector + YoloQnnDetector/YoloDetector/MediaPipeDetector, YoloDecoder, ObjectTracker
  embedding/  CropEncoder + OnDeviceCropEncoder/MacEndpointEncoder, TinyCLIP/CLIP encoders, tokenizers
  pipeline/   PerceptionPipeline, CropGeometry
  search/     ObjectSearcher, QueryText, VoiceSearchManager, speech recognizers, AmbientTranscriber
  storage/    ObjectStore (Qdrant Edge), VisionMemoryStore (legacy)
  stream/     HudPublisher, MjpegServer, MjpegPusher, FrameSink, HudEvents
  legacy/     LegacyMomentPipeline (dormant whole-frame path)
  ui/         hand-built dual-eye views
app/src/main/assets/   models + web/ HUD dashboard
scripts/               stage-demo, wipe-demo-memory, stream-tunnel, wifi-keepalive
app/libs/              Qdrant Edge AARs (committed — Apache-2.0, see NOTICE)
libs/                  sherpa-onnx AAR (fetched — see "Binaries you have to supply")
```

---

*Companion (separate repo):* the Mac `embed_server` (SigLIP2 + HUD relay) —
[qdrant-labs/edge-mission-control](https://github.com/qdrant-labs/edge-mission-control), port 9000.
Only needed for the `MAC_ENDPOINT` backend; an `ON_DEVICE` build needs no host at all.
