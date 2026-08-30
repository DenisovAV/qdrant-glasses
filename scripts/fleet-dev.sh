#!/usr/bin/env bash
# Fleet-sync dev harness (Sovereign Fleet Memory PoC — glasses Workstream A).
# A private Qdrant "fleet hub" for local dev/demo. NOT a product component: the transport for the
# glasses is `adb reverse tcp:6333` and the server is a local container. See
# docs/superpowers/specs/2026-08-24-fleet-sync-design.md.
#
# Usage:
#   scripts/fleet-dev.sh up                 # ensure the podman Qdrant (v1.19.x) is running
#   scripts/fleet-dev.sh init               # create fleet_inbox + fleet_curated (clip 768 + text 384)
#   scripts/fleet-dev.sh seed-shard <dir>   # load an Edge shard dir (e.g. adb-pulled from the glasses)
#                                           #   and upsert its points into fleet_curated (relabel FLEET:)
#   scripts/fleet-dev.sh count              # print collection point counts
#   scripts/fleet-dev.sh pull-device        # adb-pull the glasses' local moment shard to /tmp and seed
#   scripts/fleet-dev.sh approve <id>       # flip curated=true on a fleet_inbox point (simulate approval)
#   scripts/fleet-dev.sh curate             # copy curated=true inbox points into fleet_curated (pillar 3)
set -euo pipefail

QDRANT_IMAGE="qdrant/qdrant:v1.19.0"     # matches qdrant-core 1.19.1 under Edge 0.8 (snapshot-compat, spec §2)
CONTAINER="qdrant-spike"
URL="http://localhost:6333"
VENV="${TMPDIR:-/tmp}/fleet-dev-venv"
DEVICE_SHARD="files/moments_shard_siglipnpu"   # the glasses' local Edge moment shard (run-as private)

venv_py() {
  # All progress goes to STDERR; the ONLY stdout is the python path (callers do PY=$(venv_py)).
  if [ ! -x "$VENV/bin/python" ]; then
    echo ">> creating venv $VENV" >&2
    python3.12 -m venv "$VENV" >&2
    "$VENV/bin/pip" -q install --upgrade pip >/dev/null 2>&1
    echo ">> installing qdrant-edge-py + qdrant-client (first run only)" >&2
    "$VENV/bin/pip" -q install "qdrant-edge-py==0.8.0" "qdrant-client>=1.11" >&2
  fi
  echo "$VENV/bin/python"
}

cmd="${1:-}"; shift || true
case "$cmd" in
  up)
    if podman ps --filter "name=$CONTAINER" --format '{{.Names}}' | grep -q "$CONTAINER"; then
      echo "already up"
    elif podman ps -a --filter "name=$CONTAINER" --format '{{.Names}}' | grep -q "$CONTAINER"; then
      podman start "$CONTAINER"
    else
      podman run -d --name "$CONTAINER" -p 6333:6333 -p 6334:6334 "$QDRANT_IMAGE"
    fi
    for i in $(seq 1 10); do curl -sf "$URL/" >/dev/null 2>&1 && { echo "qdrant up: $(curl -s $URL/ | head -c 120)"; exit 0; }; sleep 1; done
    echo "qdrant did not come up" >&2; exit 1
    ;;
  init)
    PY=$(venv_py)
    "$PY" - "$URL" <<'PYEOF'
import sys
from qdrant_client import QdrantClient, models
c = QdrantClient(url=sys.argv[1])
VECS = {"clip": models.VectorParams(size=768, distance=models.Distance.COSINE),
        "text": models.VectorParams(size=384, distance=models.Distance.COSINE)}
for name in ("fleet_inbox", "fleet_curated"):
    if not c.collection_exists(name):
        c.create_collection(name, vectors_config=VECS)
        print("created", name)
    else:
        print("exists", name)
PYEOF
    ;;
  seed-shard)
    dir="${1:?usage: seed-shard <edge-shard-dir>}"
    PY=$(venv_py)
    "$PY" - "$URL" "$dir" <<'PYEOF'
import sys, qdrant_edge as E
from qdrant_client import QdrantClient, models
url, shard_dir = sys.argv[1], sys.argv[2]
shard = E.EdgeShard.load(shard_dir)                     # config=None: read schema from the shard itself
sr = shard.scroll(E.ScrollRequest(limit=500, with_payload=True, with_vector=True))
recs = sr.points if hasattr(sr, "points") else (sr[0] if isinstance(sr, (list, tuple)) else sr)
c = QdrantClient(url=url)
pts, n = [], 0
for r in recs:
    pl = dict(getattr(r, "payload", None) or {})
    if pl.get("type") not in (None, "frame"):
        continue                                        # seed frames only (searchable moments)
    vecmap = getattr(r, "vector", None) or {}
    clip = vecmap.get("clip") if hasattr(vecmap, "get") else None
    if clip is None:
        continue
    pl["label"] = "FLEET: " + str(pl.get("label", "") or pl.get("moment_id", ""))
    pl["curated"] = True
    pts.append(models.PointStruct(id=n + 1, vector={"clip": list(clip)}, payload=pl)); n += 1
if pts:
    c.upsert("fleet_curated", points=pts, wait=True)
print(f"seeded fleet_curated with {len(pts)} frame points; total={c.count('fleet_curated').count}")
PYEOF
    ;;
  pull-device)
    tmp="${TMPDIR:-/tmp}/fleet-device-shard"; rm -rf "$tmp"; mkdir -p "$tmp"
    echo ">> adb-pulling the glasses' local moment shard ($DEVICE_SHARD)"
    adb exec-out run-as tech.qdrant.glasses tar c "$DEVICE_SHARD" 2>/dev/null | tar x -C "$tmp"
    sd="$tmp/$DEVICE_SHARD"
    [ -d "$sd" ] || { echo "no shard pulled (record moments first?)" >&2; exit 1; }
    echo ">> shard at $sd"; "$0" seed-shard "$sd"
    ;;
  count)
    PY=$(venv_py)
    "$PY" - "$URL" <<'PYEOF'
import sys
from qdrant_client import QdrantClient
c = QdrantClient(url=sys.argv[1])
for n in ("fleet_inbox", "fleet_curated"):
    try: print(f"{n}: {c.count(n).count}")
    except Exception as e: print(f"{n}: (missing)")
PYEOF
    ;;
  approve)
    # Simulate a human curator approving ONE device contribution (Spec §8): flip curated=true on a
    # fleet_inbox point. A real deployment would have an approval UI/workflow; the PoC shows the CONCEPT.
    id="${1:?usage: approve <point-id>}"
    PY=$(venv_py)
    "$PY" - "$URL" "$id" <<'PYEOF'
import sys
from qdrant_client import QdrantClient
c = QdrantClient(url=sys.argv[1]); pid = sys.argv[2]
# Inbox ids are UUID strings (device UP path) or ints (seed-shard). Coerce all-digit args to int so a
# numeric id targets the right point instead of a never-matching string.
pid = int(pid) if pid.isdigit() else pid
# Qdrant's set_payload with an explicit `points` selector does NOT error on a missing id — it returns
# status=completed having touched nothing. For a harness whose whole job is proving "invisible until
# approved", a silent no-op that still prints "approved" is the exact false-success this must avoid.
# So: verify the point exists BEFORE, and confirm the flag actually landed AFTER — fail loudly otherwise.
if not c.retrieve("fleet_inbox", ids=[pid]):
    sys.exit(f"approve FAILED: no point {pid!r} in fleet_inbox — nothing approved")
c.set_payload("fleet_inbox", payload={"curated": True}, points=[pid], wait=True)
back = c.retrieve("fleet_inbox", ids=[pid], with_payload=True)
if not back or back[0].payload.get("curated") is not True:
    sys.exit(f"approve FAILED: curated flag did not take on {pid!r}")
print(f"approved (curated=true): {pid}")
PYEOF
    ;;
  curate)
    # Pillar 3 (Spec §8): copy the APPROVED subset (curated=true) from fleet_inbox into fleet_curated,
    # preserving id/vectors/payload so a re-run is idempotent (upsert by id). Devices snapshot-pull ONLY
    # fleet_curated, so a contribution stays invisible to the fleet until it is approved here — then it
    # propagates on the next pull. curated=true points are the only knowledge that ever reaches a device.
    # Safe to re-run: upsert-by-id overwrites, so a partial run (e.g. an error mid-pagination, which
    # `set -e` surfaces loudly) is recovered simply by running `curate` again.
    PY=$(venv_py)
    "$PY" - "$URL" <<'PYEOF'
import sys
from qdrant_client import QdrantClient, models
c = QdrantClient(url=sys.argv[1])
flt = models.Filter(must=[models.FieldCondition(key="curated", match=models.MatchValue(value=True))])
moved, skipped, offset = 0, 0, None
while True:
    recs, offset = c.scroll("fleet_inbox", scroll_filter=flt, with_payload=True, with_vectors=True,
                            limit=256, offset=offset)
    pts = []
    for r in recs:
        vecmap = r.vector or {}
        # Preserve EVERY named vector the point carries (clip today; text too once OCR/speech fills it),
        # not just clip — the point travels whole.
        vecs = {k: list(v) for k, v in vecmap.items()} if hasattr(vecmap, "items") else {}
        if "clip" not in vecs:
            skipped += 1   # an approved point with no clip vector can't be a searchable frame — surface it, never drop silently
            continue
        pts.append(models.PointStruct(id=r.id, vector=vecs, payload=dict(r.payload or {})))
    if pts:
        c.upsert("fleet_curated", points=pts, wait=True); moved += len(pts)
    if offset is None:
        break
if skipped:
    # A curator approved these but they can't be curated — loud, so lost approvals are never invisible.
    print(f"WARNING: skipped {skipped} approved point(s) with no clip vector — NOT curated (approval lost)", file=sys.stderr)
print(f"curated {moved} approved point(s) into fleet_curated; total={c.count('fleet_curated').count}")
PYEOF
    ;;
  *)
    echo "usage: $0 {up|init|seed-shard <dir>|pull-device|count|approve <id>|curate}" >&2; exit 1
    ;;
esac
