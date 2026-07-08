# qdrant_glasses — project rules

## Demo memory wipe (IMPORTANT)

The demo's object memory lives in TWO places: the glasses (Qdrant Edge shards +
thumb JPEGs) **and** the Mac relay's process RAM (HUD rail + pushed thumbs).
When asked to "почистить базу" / wipe / clear the demo memory — NEVER do it by
hand. Always run:

    ./scripts/wipe-demo-memory.sh

It restarts the relay first, then wipes the glasses, then restarts the app and
prints the (empty) rail as proof. Wiping only one side leaves ghost cards on
the dashboard.

## Stage / demo prep

Full venue prep (new network, relay repoint, battery-saver trigger, frame-flow
verification) is one command:

    ./scripts/stage-demo.sh

It also wipes memory on both sides — a fresh run means a fresh demo.
