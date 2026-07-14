# qdrant_glasses — project rules

## Demo memory wipe (IMPORTANT)

The demo's object memory can live in TWO places: the glasses (Qdrant Edge shards +
thumb JPEGs) **and** — if the HUD dashboard is up — the Mac relay's process RAM
(the rail + pushed thumbs). Wiping only one side leaves ghost cards on the
dashboard. When asked to "почистить базу" / wipe / clear the demo memory —
**never do it by hand**, always run:

    ./scripts/wipe-demo-memory.sh

It restarts the relay (bounded — 40s, then it warns and carries on), wipes the
glasses' shards and thumbs, relaunches the app, and prints the empty rail as
proof *if* a relay was running. The glasses half always happens: it is the half
that matters, and a missing relay must never block it.

The default backend is `ON_DEVICE`, which needs no relay at all. On a machine
without the relay checkout, skip that half explicitly:

    EMBED_DIR=/nonexistent ./scripts/wipe-demo-memory.sh    # ~1s, glasses only

## Stage / demo prep

Full venue prep (relay repoint to this Mac's current IP, both-sides wipe,
battery-saver UVLO guard, frame-flow verification) is one command:

    ./scripts/stage-demo.sh

A fresh run means a fresh demo. Same relay rules as above.

## Facts that bite

- **The embedder runs on the CPU, and that is correct.** ~200 ms/crop. Every
  accelerator on this SoC was implemented and measured against a CLIP-class ViT,
  and every one lost — the Hexagon HTP accepts 44 of the graph's 490 nodes, the
  Adreno GPU delegate 198, NNAPI 72 of 890, and each partition boundary costs a
  round trip back to the CPU. Do not "optimize" this onto an accelerator. The
  detector is the opposite: int8 YOLOv8n runs whole on the HTP at ~8 ms.
- **Latency is load-dependent, not fixed.** The AR1 Gen 1 is a **4-core** part,
  so the embedder shares cores with the camera's YUV conversion, the detector and
  the HUD's JPEG encode. Closing the HUD alone takes a crop embed from ~870 ms to
  ~200 ms (`setprop debug.qdrant.hud 0`). Benchmark with it off, and compare only
  like-for-like runs.
- **Doc rot is the norm here.** Claims in comments and docs age badly and read as
  true long after they stop being so ("CLIP runs on NNAPI", "a fresh clone
  compiles", "cropFrom copies pixels synchronously" — all were false, all cost
  real time, and the last one hid a live bug). Verify against the code before
  repeating a claim, and prefer a grep over a memory.
