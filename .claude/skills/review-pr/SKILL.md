---
name: review-pr
description: Comprehensive PR / branch review for qdrant_glasses — the on-device multimodal-RAG "object memory" Android app for RayNeo X3 Pro AR glasses. Runs 10 specialized reviewers in parallel (5 pipeline-stage + 5 general) plus an optional Copilot second opinion. Use this whenever reviewing a PR, reviewing a branch before merge, checking a diff, or when the user says "review", "проверь", "review the PR", "review before merge", or "ревью" — even if they don't name a PR number.
user_invocable: true
---

# qdrant_glasses PR Review

Run a comprehensive review of a diff with parallel agents — 5 pipeline-stage reviewers + 5
general reviewers — each carrying this project's hard-won constraints (HTP runs YOLO but not
CLIP; per-encoder cosine thresholds; UVLO brownout; WiFi suspend). Findings are collected,
deduplicated, and written to a report.

## Context: what this app is

`qdrant_glasses` (package `tech.qdrant.glasses`) is a **single-platform Android app** for
RayNeo X3 Pro AR glasses (Snapdragon AR1 Gen 1 + Hexagon HTP). It is an on-device multimodal-RAG
**object-memory / lost-and-found demo**: the camera detects objects, crops are embedded
(CLIP-family), stored in **Qdrant Edge** (Rust FFI via JNA) on the glasses, and retrieved by
**voice search**. A HUD dashboard mirrors the state to a Mac.

Unlike a multi-platform plugin, its natural fault-line is the **pipeline**, not the OS:

```
camera → detect → track/dedup → embed (crop) → store (Qdrant Edge) → voice-search → retrieve → HUD
```

The reviewers below are organized by that pipeline. Facts that shape how the review runs:

- **Base branch is `main`.** The review works off the **local diff** by default; pass a PR number
  only if a GitHub PR actually exists.
- The **Mac relay / `embed_server`** (SigLIP2 cloud embeddings + HUD rail) lives in a
  **separate repo** ([qdrant-labs/edge-mission-control](https://github.com/qdrant-labs/edge-mission-control),
  cloned to `~/Work/edge-mission-control` by convention). It is out of scope here; this review
  covers the Android app and `scripts/`.
- **There are TWO pipelines behind `GlassesViewModel.appMode`.** `OBJECTS` (active demo):
  `CropEncoder` → `ObjectStore` → direct top-k search. `LEGACY` "moment" path (dormant in the
  shipped build): whole-frame CLIP + BGE → `VisionMemoryStore` → `MomentRetriever`. When a
  finding lands in LEGACY-only code (`MomentRetriever`, `VisionMemoryStore`, `CropEncoder.visionMinScore`),
  say so — it is lower-severity because that path is not active.

## Usage

```
/review-pr            # review current branch vs main (the default path)
/review-pr 42         # review GitHub PR #42 (only if a remote/PR exists)
/review-pr HEAD~3     # review the last 3 commits
```

## Process

### Step 1: Get the diff

Detect the base branch robustly (this repo's default is `main`; it was `master` historically, and
a clone may have neither checked out):

```bash
# Base branch: prefer main (this repo's default), fall back to master.
BASE=main
git rev-parse --verify --quiet "$BASE" >/dev/null 2>&1 || BASE=master

ARG="{arg}"   # the /review-pr argument, if any
if [ -z "$ARG" ]; then
  # No arg → review the working branch against its merge-base with BASE.
  git diff "$BASE"...HEAD > /tmp/qg-review.diff
  git diff "$BASE"...HEAD --stat
elif echo "$ARG" | grep -qE '^[0-9]+$' && git remote | grep -q .; then
  # Numeric arg AND a remote exists → treat as a GitHub PR number.
  gh pr view "$ARG" --json title,body,files --jq '.title'
  gh pr diff "$ARG" > /tmp/qg-review.diff
else
  # Anything else (a ref/range like HEAD~3, a branch name) → diff it.
  git diff "$ARG"...HEAD > /tmp/qg-review.diff 2>/dev/null || git diff "$ARG" > /tmp/qg-review.diff
fi
wc -l /tmp/qg-review.diff
```

Also capture the changed-file list — the agents route off it:

```bash
git diff "$BASE"...HEAD --name-only    # (or `gh pr diff {n} --name-only` on the PR path)
```

If the diff is empty, stop and tell the user there is nothing to review against `$BASE`.

### Step 2: Identify changed pipeline stages

All app code lives under `app/src/main/java/tech/qdrant/glasses/`. Map changed files to
pipeline stages so each agent knows whether its area is affected (an agent whose area is
untouched should say so briefly and skip deep work):

| Stage | Paths | Owns |
|-------|-------|------|
| **Perception** | `detect/`, `camera/`, `pipeline/`, `GlassesViewModel.kt` | `ObjectDetector`/`YoloDetector`/`YoloQnnDetector`/`MediaPipeDetector`, `YoloDecoder`, `ObjectTracker`, `Geometry`, `CocoLabels`, `FrameCaptureManager`, `PerceptionPipeline` (the detect→crop→embed→store hot path, and `DEDUP_COSINE`), `CropGeometry` |
| **Embedding** | `embedding/` | `CropEncoder` + impls (`OnDeviceCropEncoder`, `MacEndpointEncoder`, TinyCLIP LiteRT/ONNX), text encoders, tokenizers (`*BpeTokenizer`, `WordPieceTokenizer`), `LiteRtSession`, `BgeTextEncoder` |
| **Storage / Retrieval** | `storage/`, `search/ObjectSearcher.kt`, `search/QueryText.kt`, `search/MomentRetriever.kt` | `ObjectStore`, `VisionMemoryStore` (Qdrant Edge JNA FFI), `ObjectSearcher` + `QueryText` (the ACTIVE voice-search path: cosine-gate-OR-label-match), `MomentRetriever` (dormant) |
| **Voice / ASR** | `search/` (speech only — object search belongs to Storage/Retrieval above) | `SpeechRecognizer` + impls (`Android`/`Google`/`Vosk`/`SherpaVadAsr`), `AmbientTranscriber`, `VoiceSearchManager` |
| **Streaming / HUD** | `stream/`, `ui/`, `assets/web/` | `MjpegServer`, `MjpegPusher`, `FrameSink`, `HudEvents`, `BoxOverlay`, view classes, web dashboard |
| **Legacy (dormant)** | `legacy/` | `LegacyMomentPipeline` — whole-frame path, not in the shipped build; findings here are lower-severity, say so |
| **Build / infra** | `app/build.gradle.kts`, `gradle/libs.versions.toml`, `settings.gradle.kts`, `scripts/`, `app/src/main/assets/**` (non-code) | native packaging, deps, demo/stage scripts |

Non-app areas: root gradle, `CLAUDE.md`.

### Step 3: Launch ALL agents in parallel

**CRITICAL: launch every agent in a single message with multiple Agent tool calls so they run
concurrently.** Give each agent the diff (or changed-file list) and its checklist below. Tell
each agent to report findings as **CRITICAL / IMPORTANT / MINOR** with `file:line` references,
and to state up front if its stage was not touched by the diff.

---

## Agent Specifications

### Agent 1 — Perception: Detection & Camera Pipeline

**subagent_type:** `general-purpose`

```
You review the perception pipeline of qdrant_glasses (on-device object-memory app for RayNeo
X3 Pro AR glasses, Snapdragon AR1 Gen 1 + Hexagon HTP). Review changed files under:
  app/src/main/java/tech/qdrant/glasses/detect/ and /camera/, and GlassesViewModel.kt

CHECKLIST:
1. DETECTOR ABSTRACTION: ObjectDetector interface cleanly implemented by YoloQnnDetector (ORT
   QNN EP / Hexagon HTP — the default), YoloDetector (LiteRT on the Adreno GPU, ~113ms — the
   fallback), MediaPipeDetector (EfficientDet, CPU — the last resort). One detector is chosen in
   DetectorFactory.backend; no leaking of impl details into callers.
2. ACCELERATOR SELECTION (KEY PROJECT RULE): int8 YOLOv8n runs on Hexagon HTP via ORT QNN EP
   (~8ms) with htp burst perf mode and NCHW layout. This is the ONE model that works on HTP.
   Check: correct EP/provider options, NCHW vs NHWC, quantization assumptions. If HTP init
   fails, does it fall back *loudly* (logged) to GPU/CPU — or silently? Silent accelerator
   downgrade that makes the demo slow without a trace is a bug.
3. YOLO DECODE: YoloDecoder — anchor/grid math, xywh→xyxy, confidence * class-prob, NMS/IoU
   threshold, letterbox coordinate un-mapping (Geometry). Off-by-one/stride errors here
   silently drop or mis-place boxes.
4. TRACKING & DEDUP: ObjectTracker assigns IoU tracking IDs so the same physical object isn't
   embedded/stored repeatedly. Verify the SAVE RULE — an object is committed to memory only
   after it is seen with confidence >= ~0.4 across ~3 sightings. Check the constants and the
   dedup key.
5. FRAME LIFECYCLE & THREADING: CameraX capture → detect. Stream vs infer decoupled onto
   separate lanes/dispatchers with backpressure gates (no unbounded queue). Bitmaps recycled
   exactly once (no use-after-recycle on async frames); crops copied before the frame is
   released.
6. ORCHESTRATION: GlassesViewModel coordinates detect→embed→store→HUD and the state machine
   (Idle/Recording/Listening/Processing). No blocking calls on the main thread; volatile /
   synchronized where detector state is shared.
7. COCO LABELS: CocoLabels index→name mapping matches the model's class order.

Report CRITICAL / IMPORTANT / MINOR with file:line. State if this stage is untouched.
```

### Agent 2 — Embedding & Tokenizers

**subagent_type:** `general-purpose`

```
You review the embedding layer of qdrant_glasses (CLIP-family crop embeddings for on-device
vector search on AR glasses). Review changed files under:
  app/src/main/java/tech/qdrant/glasses/embedding/

CHECKLIST:
1. ENCODER ABSTRACTION: CropEncoder interface implemented by OnDeviceCropEncoder (TinyCLIP-512
   int8 via ORT, on the CPU) and MacEndpointEncoder (SigLIP2 on a Mac, over HTTP). The backend is
   chosen in CropEncoderFactory.backend. Swapping backends must not leak.
2. CLIP RUNS ON THE CPU, AND THAT IS THE ANSWER (KEY PROJECT RULE — measured, not assumed):
   every accelerator on this SoC was implemented and benchmarked against a CLIP-class ViT, and
   every one LOST to the CPU. The Hexagon HTP accepts 44 of the graph's 490 nodes (the rest shred
   into 50 partitions), the Adreno GPU delegate 198/490, NNAPI 72/890 — and each partition
   boundary costs a round trip back to the CPU worth more than the acceleration saves. The shipped
   onnxruntime-android-qnn AAR has NO NNAPI EP at all (zero ANeuralNetworks* imports), so
   addNnapi() always throws; see the KDoc on createAcceleratedSession. CPU ~200ms is the fast
   path. FLAG any change that moves the vision encoder onto HTP/GPU/NNAPI "for speed" — that is
   a regression, not an optimization. (The CNN detector is the opposite: it belongs on the HTP.)
3. EMBEDDING-SPACE INTEGRITY: vectors MUST be L2-normalized before storage/search; cosine is
   the metric. Vision-crop vectors and text-query vectors must come from the SAME paired
   encoder (CLIP modality gap) — never mix TinyCLIP vision with a different text tower, and
   never mix on-device vectors with Mac-endpoint (SigLIP2) vectors in one collection.
4. DIMENSIONS: on-device TinyCLIP = **512-d**, Mac SigLIP2 (`MacEndpointEncoder`) = **768-d**,
   BGE = 384-d. The vector dim MUST match the Qdrant Edge collection dim exactly (the store is
   opened with `cropEncoder.dim`, and each backend gets its own on-disk namespace). A dim
   mismatch is a CRITICAL silent corruptor. `MacEndpointEncoder` DOES validate this today (it
   throws on a wrong length and on non-finite values) — keep that guard; flag its removal, and
   flag any NEW encoder that lacks the equivalent.
5. PER-ENCODER THRESHOLDS: gates differ per encoder AND there are multiple gate systems — the
   *effective* object-search gate is `CropEncoderFactory.searchGate` (mac 0.08 / ondevice 0.25
   / cloud 0.12); `CropEncoder.visionMinScore` (mac 0.12 / ondevice 0.20) flows only into the
   dormant `MomentRetriever`. A threshold hardcoded for one encoder but applied to another —
   or edited on the wrong gate — silently wrecks recall/precision. `DEDUP_COSINE=0.90` is
   applied uniformly to both 512-d and 768-d spaces (a per-backend smell).
6. ORT ENV LIFETIME: `OrtEnvironment.getEnvironment()` is a process-wide singleton, and every
   ONNX encoder here calls `env.close()` in its own `close()`. That reads like a double-free
   hazard and was long flagged as one — but in the PINNED version (onnxruntime 1.26.0)
   `OrtEnvironment.close()` is decompiled to a literal no-op (`return;`); teardown happens only
   via a JVM shutdown hook. So it is harmless TODAY and is baseline behaviour, not something a
   diff introduces. Don't spend review budget on it — but it is a real trip-wire if the ORT pin
   ever moves, so flag an ORT version bump as needing a re-check here.
7. TOKENIZERS: the text encoder's tokenizer must match the model it was trained with. Active
   CLIP tokenizer is `RankedBpeTokenizer` (open_clip-correct); `NaiveBpeTokenizer` is present
   but INCORRECT (applies `</w>` to every piece) — flag any switch to it. Check BPE
   merges/vocab loading, byte-level pre-tokenization, special/BOS/EOS tokens (SOT 49406 / EOT
   49407), and truncation/padding to context length (77 for CLIP). A CLIP text mask must cover
   only pad positions.
8. INT8 QUANT: dequant buffers pre-allocated/reused; input normalization (mean/std) matches
   the exported model; NCHW/NHWC correct. Note the vector DB itself stores full float
   (`quantizationConfig = null`) — "int8" describes model weights, not stored vectors.

Report CRITICAL / IMPORTANT / MINOR with file:line. State if this stage is untouched.
```

### Agent 3 — Vector Storage & Retrieval (Qdrant Edge)

**subagent_type:** `general-purpose`

```
You review the vector storage + retrieval layer of qdrant_glasses. Qdrant Edge runs ON the
glasses via Rust FFI (JNA). Review changed files under:
  app/src/main/java/tech/qdrant/glasses/storage/ and search/MomentRetriever.kt

CHECKLIST:
1. FFI SAFETY (JNA → Rust qdrant-edge AAR): pointer/handle lifetime, native memory freed
   (no leaks, no double-free), buffers sized correctly, thread-confinement of native handles.
   FFI errors surfaced as exceptions, not swallowed.
2. COLLECTION CONFIG: dimension matches the active encoder (512-d on-device), distance =
   cosine, on-disk persistence path on the glasses. Collection created once / idempotently.
3. UPSERT: point ids are random UUIDs and the store is INSERT-ONLY — there is no id-based
   update. De-duplication happens entirely upstream, before upsert is reached: IoU tracking
   (`ObjectTracker.markEmbedded`) plus a semantic cosine check (`DEDUP_COSINE`, in
   `pipeline/PerceptionPipeline`). So a dedup bug shows up as DUPLICATE POINTS, never as a
   clobbered one. Payload (label, bbox, thumb path, timestamp, track id) stored and round-trips.
   Thumbnail written to disk, not into the vector.
4. SEARCH: top-k query, score threshold/gate applied consistently with the encoder in use,
   results mapped back to payload. Empty-result path handled.
5. THREAD SAFETY: store accessed from camera/infer and voice-search threads — is access
   serialized? On indexing: the shipped collections are plain brute-force over full float
   (`quantizationConfig = null`) and that is deliberate — benchmarked on this device, HNSW loses
   at demo scale (and even at 1M: a 1h46m build and a worse p95 than a binary scan). Binary
   quantization is the win at large scale but is NOT configured today, so don't describe it as
   active; flag an HNSW/quantization change that arrives without a benchmark.
6. TWO-PLACE MEMORY: object memory lives in TWO places — the glasses shard AND the Mac relay's
   RAM. Code that "clears memory" must not assume one side. (Wipes go through
   scripts/wipe-demo-memory.sh — do not hand-roll a one-sided wipe.)
7. RETRIEVAL (MomentRetriever): text→embed→search→gate→map. Query embedding uses the matching
   text encoder for the active backend.

Report CRITICAL / IMPORTANT / MINOR with file:line. State if this stage is untouched.
```

### Agent 4 — Voice / ASR

**subagent_type:** `general-purpose`

```
You review the voice / ASR pipeline of qdrant_glasses (voice-driven search on AR glasses).
Review changed files under:
  app/src/main/java/tech/qdrant/glasses/search/ (SpeechRecognizer, AndroidSpeechRecognizer,
  GoogleSpeechRecognizer, VoskSpeechRecognizer, SherpaVadAsr, AmbientTranscriber,
  VoiceSearchManager)

CHECKLIST:
1. ASR ABSTRACTION: SpeechRecognizer interface implemented by Android (one-shot
   SpeechRecognizer — beeps, whitelisted on RayNeo), Google Cloud STT (needs
   BuildConfig.GOOGLE_STT_API_KEY), Vosk (offline), SherpaVadAsr (VAD + offline Moonshine).
   Which backend serves voice-search vs ambient transcription, and is the choice explicit?
2. VAD / STREAMING vs ONE-SHOT: one-shot SpeechRecognizer emits a recording beep and can't do
   continuous ambient capture — that's why SherpaVadAsr (VAD-gated offline) exists. Check VAD
   energy/threshold tuning and multi-segment transcript concatenation.
3. AUDIO PIPELINE: AudioRecord source (VOICE_RECOGNITION / MIC), sample rate, read/process
   split to avoid buffer overflow, audio focus. Buffers not shared across threads unsafely.
   trimToSpeech bounds correct (no out-of-range slice).
4. VOICE-SEARCH ROBUSTNESS (VoiceSearchManager): the gate that rejects noise/short utterances,
   text normalization, and COCO-label matching that maps a spoken phrase to a stored object.
   Over-aggressive gating silently drops valid queries; under-gating searches on garbage.
5. MIC / POWER: mic held only while listening and released after. Voice-search current spike
   is the UVLO brownout trigger (see cross-cutting) — no needless mic warm-up loops.
6. LIFECYCLE: recognizer/AudioRecord always released in finally; no leak across search
   sessions; cancellation stops capture promptly.

Report CRITICAL / IMPORTANT / MINOR with file:line. State if this stage is untouched.
```

### Agent 5 — Streaming / HUD / Relay + UI

**subagent_type:** `general-purpose`

```
You review the streaming/HUD layer and Android UI of qdrant_glasses. The glasses run an MJPEG
+ event server (NanoHTTPD) and also push frames/events to a Mac relay dashboard. Review
changed files under:
  app/src/main/java/tech/qdrant/glasses/stream/ and /ui/, and app/src/main/assets/web/

CHECKLIST:
1. HTTP SERVER (MjpegServer / NanoHTTPD): thread-per-connection — one slow client must not
   block others; dead-client eviction so a closed browser tab doesn't leak a parked thread/FD.
   NOTE a known weakness: `Client.alive` is never set false, so eviction relies only on the
   `stalled` flag — and `pushEvent` (SSE) checks only `alive`, so a silently-dead /events
   reader is never evicted. Any change here should tighten, not loosen, that. gzip was
   force-disabled (`useGzipWhenAccepted=false`) because it buffered SSE/stream delivery to zero
   — do NOT re-enable it for /events or /stream. Routes (/stream, /events, /thumb, /static)
   return promptly.
2. FRAME PUSH (MjpegPusher, OkHttp → Mac relay): per-frame POST with backpressure (drop, don't
   queue unboundedly); relay endpoint read from system property persist.qdrant.relay /
   debug.qdrant.relay. Network failures logged, not crashing the capture loop.
3. FRAMESINK ABSTRACTION: MjpegServer and MjpegPusher decoupled behind FrameSink so local
   serving and relay push share one frame source without coupling.
4. HUD EVENTS: HudEvents model (boxes/stored/tick/mode/results) — builders correct, thumbnails
   referenced by path/URL not inlined as huge base64, event ordering sane.
5. UI / VIEW LIFECYCLE (hwui crash risk): views should be REUSED and updated, not recreated on
   every state change — view recreation caused an hwui SIGSEGV (fixed e27939c). State
   (Idle/Recording/Listening/Processing/SearchResults) drives which view is visible without
   re-inflating.
6. RESOURCE LEAKS: input/output streams, sockets, bitmaps closed in finally.
7. WEB DASHBOARD (assets/web): TWO different browser contracts exist and must stay aligned by
   hand — the on-glasses `MjpegServer` serves true SSE (`/events`, `/stream`, `/thumb`), while
   `app.js` actually polls the **Mac relay's** `/poll` + `/browser_log` API (~0.5s; the relay
   lives in the separate repo). A change to one contract's event shape must be mirrored in the
   other. No hardcoded Mac IPs baked into committed JS.

Report CRITICAL / IMPORTANT / MINOR with file:line. State if this stage is untouched.
```

### Agent 6 — Android RAG Architect

**subagent_type:** `android-rag-architect`

**Prompt:** Review this qdrant_glasses diff for architecture. It is an on-device multimodal-RAG object-memory app for RayNeo X3 Pro glasses. Focus on: the swappable abstractions (`ObjectDetector`, `CropEncoder`, `SpeechRecognizer`, `FrameSink`) and whether new code respects them; the on-device-detection + Qdrant-Edge-on-glasses + (optional) cloud-crop-embedding split; mode/backend selection (NOTE: `Config` holds only WIRELESS/MAC_BASE_URL/HUD_STREAM — `appMode` is a private val in `GlassesViewModel`, and each factory owns its own `Backend` enum: `DetectorFactory`, `CropEncoderFactory`, `EncoderFactory`, `Tokenizer`; flag drift BETWEEN them, not against `Config`); and separation between the perception, embedding, storage, and voice stages. **Also review build/native config if `app/build.gradle.kts` or `gradle/libs.versions.toml` changed:** `useLegacyPackaging=true` (QNN/FastRPC needs the `.so` on disk), `pickFirsts` scoped to `lib/x86/libonnxruntime.so` ONLY (a real arm64 ORT clash must still fail loudly), `noCompress` for `onnx/tflite/bin/data/txt`, `ignoreAssetsPattern` keeping the ~945MB whole-frame CLIP weights OUT while keeping `tinyclip-int8.onnx` IN, `abiFilters = arm64-v8a` only, and version pins (`onnxruntime-android-qnn` and `qnn` >= the qai-hub context-binary QAIRT version). Read CLAUDE.md and flag anything that breaks the two-place-memory or stage-prep contracts.

### Agent 7 — Android RAG Reviewer (domain code review)

**subagent_type:** `android-rag-reviewer`

**Prompt:** Review the changed inference/storage/audio code in this qdrant_glasses diff for domain correctness. Priorities: (1) the CLIP **modality gap** — vision and text embeddings must be the paired encoder and L2-normalized before cosine search; (2) **accelerator fail-fast** — CLIP must not run on Hexagon HTP (TCM overflow), only the CNN detector does; accelerator init failures must surface, not silently downgrade; (3) **embedding-space integrity** — no mixing vectors from different encoders/backends in one collection, dims must match; (4) thread safety across the camera/infer/voice/HTTP threads; (5) on-device verifiability. Read CLAUDE.md.

### Agent 8 — Silent Failure Hunter

**subagent_type:** `pr-review-toolkit:silent-failure-hunter`

**Prompt:** Hunt for silent failures and inappropriate fallbacks in this qdrant_glasses diff. Highest-value targets: accelerator init (QNN/HTP → GPU → CPU) that downgrades *silently* so the demo is slow with no log; FFI/JNA errors from Qdrant Edge swallowed in catch blocks; embedding failures returning zero/empty vectors that then get stored; ASR errors that leave the user stuck in Listening; MjpegPusher network failures that should be logged-and-dropped vs ones that hide a real bug. NOTE: the CLIP encoder runs on the CPU BY DESIGN (every accelerator was measured and lost) — its 'no NNAPI EP in this ORT build (expected)' log line is correct behaviour, not a silent downgrade; the bug is the *silent* ones and any fallback that corrupts data (wrong dim, unnormalized vector).

### Agent 9 — Type Design Analyzer

**subagent_type:** `pr-review-toolkit:type-design-analyzer`

**Prompt:** Analyze new/modified types and interfaces in this qdrant_glasses diff for encapsulation and invariant expression. Focus on the strategy interfaces (`ObjectDetector`, `CropEncoder`, `SpeechRecognizer`, `FrameSink`), the factory `Backend` enums / `AppMode`, `HudEvents`, detection/geometry value types (boxes, tracks), and the stored-object/payload model. Are invariants (normalized vectors, matching dims, valid box coords) expressed in the types or left implicit?

### Agent 10 — Code Reviewer (general + CLAUDE.md)

**subagent_type:** `pr-review-toolkit:code-reviewer`

**Prompt:** Review this qdrant_glasses diff for general correctness, bugs, and adherence to CLAUDE.md. Kotlin specifics: coroutine scope/cancellation, null safety, resources closed in finally, no blocking on main thread, no hardcoded IPs/serials in committed code (`stage-demo.sh` has a default serial — that's expected in scripts, not in app code). Also assess test coverage: this repo has JUnit/Robolectric unit tests (`YoloDecoderTest`, `ObjectTrackerTest`, `GeometryTest`, `CropGeometryTest`, `CocoLabelsTest`, `HudEventsTest`, `QueryTextTest`, `ObjectPayloadTest`, `AppStateHolderTest`, `SmokeTest`) — if the diff changes decode/tracking/geometry/label/HUD logic, is there a matching test? Flag untested new branches in that logic.

### Agent 11 (optional) — Copilot second opinion

Run **only if** the `copilot` CLI is installed (`command -v copilot`). Run directly via Bash
(timeout 300000ms), not as a subagent:

```bash
copilot -p "Review the current branch of qdrant_glasses — an on-device multimodal-RAG object-memory Android app (Kotlin) for RayNeo X3 Pro AR glasses (Snapdragon AR1 Gen 1 + Hexagon HTP). Pipeline: camera → YOLOv8n detect (int8 on HTP) → IoU track/dedup → CLIP crop embed (TinyCLIP-512 on CPU — measured fastest here; every accelerator lost — or Mac SigLIP2) → Qdrant Edge vector store (Rust FFI/JNA on the glasses) → voice search (VAD+offline ASR) → HUD. Base branch is main. Key rules: CLIP must NOT run on HTP (TCM overflow); vectors L2-normalized, cosine, dims must match the collection; never mix encoders/backends in one collection; accelerator fallback must be LOGGED not silent; views reused not recreated (hwui crash). Steps: run 'git diff main...HEAD', read CLAUDE.md, then report bugs / logic errors / race conditions / silent error-swallowing / memory or FFI leaks with file:line. Skip style nits. Categorize CRITICAL / IMPORTANT / MINOR." \
  --allow-all-tools --allow-all-paths --no-auto-update --output-format text 2>&1
```

Parse the output; extract findings with severities. If `copilot` is absent, skip silently and
note "Copilot: not run (CLI unavailable)" in the report.

---

## Cross-cutting constraints (every agent should keep these in mind)

These are the project's hardware/firmware realities — the equivalent of a platform's "gotchas".
A change that ignores one of these is a finding even if the code "looks" correct:

- **HTP is for the CNN detector only.** int8 YOLOv8n → Hexagon HTP (~8ms). A CLIP/ViT does not
  fit any accelerator here — measured, the HTP takes 44 of 490 graph nodes, the GPU 198, NNAPI
  72/890 — so CLIP runs on the **CPU (~200ms), by design and by measurement**. Moving it to an
  accelerator is a regression.
- **UVLO battery brownout.** The voice-search current spike can brown-out the glasses (no BCL
  in firmware). Battery Saver mitigates it — don't add needless peak-current work at search.
- **WiFi suspends on idle** (RayneoSuspendManager) and firmware **blocks inbound TCP on
  wlan0** — browser→glasses streaming only works over USB `adb forward` (see
  `scripts/stream-tunnel.sh`). Don't assume the glasses are reachable inbound over WiFi.
- **Two-place memory.** Object memory = glasses shard + Mac relay RAM. Wipe via
  `scripts/wipe-demo-memory.sh`; stage via `scripts/stage-demo.sh`. Never hand-roll a
  one-sided wipe.
- **APK size discipline.** The whole-frame CLIP weights (~945MB) stay OUT via
  `ignoreAssetsPattern`; `tinyclip-int8.onnx` (~86MB) stays IN for on-device.

## Step 4: Collect and deduplicate

After all agents return:

1. Collect every finding from every agent (including Copilot, if run).
2. Deduplicate — if two agents report the same `file:line` + concern, keep the clearest
   description and note it was multiply-reported (raises confidence).
3. Sort by severity: CRITICAL > IMPORTANT > MINOR.
4. Group by pipeline stage, then file, within each severity.

## Step 5: Generate the report

Write to `docs/pr-reviews/pr-{number-or-branch}-review.md` (gitignored — review findings are working
notes, not published artifacts):

```markdown
# PR Review: {number or branch} — {title}

**Base:** {base}  **Head:** {branch}  **Date:** {date}
**Reviewers:** 10 agents (5 pipeline-stage + 5 general){ + Copilot if run}
**Stages touched:** {perception, embedding, storage, voice, streaming, build}

## Critical Issues
{blocking — must fix before merge, with file:line and why}

## Important Issues
{should fix — not blocking}

## Minor Issues
{nice-to-have}

## Passed Checks
{stage-by-stage: what was reviewed and found clean}

## Summary
- Critical: X   Important: Y   Minor: Z
- Recommendation: APPROVE / REQUEST CHANGES / NEEDS DISCUSSION
```

## Step 6: Output summary to the user

Print a concise summary:
- Total findings by severity.
- The top 3 issues (CRITICAL first).
- Which pipeline stages were touched and which came back clean.
- Recommendation (approve / request changes).
- Path to the full report.
