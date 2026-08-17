const $ = (id) => document.getElementById(id);
const feed = $("feed"), overlay = $("overlay"), octx = overlay.getContext("2d");
const RAIL_CAP = 40;
let frameDim = [640, 480];
let tracks = new Map(); // id -> {cur:{x,y,w,h}, target:{x,y,w,h}, label}

console.log("[hud] app.js loaded; feed=", !!feed, "overlay=", !!overlay, "ctx=", !!octx);
// DEBUG: mirror browser steps to the Mac so they're inspectable in /tmp/relay_debug.log
// (the browser console isn't readable from the shell). Fire-and-forget.
function blog(msg) { try { fetch("/browser_log", { method: "POST", body: msg }); } catch (e) {} }
blog("app.js loaded, feed=" + !!feed);

// Consume events by POLLING short plain fetches (the reliable path). A long-lived streaming
// fetch/EventSource to the uvicorn relay never resolves in some browsers (the promise hangs with
// bytes flowing), but SHORT fetches resolve fine — same reason the video works as a plain <img>,
// not a JS stream. So we GET /poll?since=<cursor> every 500ms: it returns all events after the
// cursor (and the full rail snapshot on the first poll), then we advance the cursor.
let _evCount = 0;
let _pollCursor = -1;   // -1 = first poll → server also replays the rail snapshot
async function pollEvents() {
  try {
    const r = await fetch("/poll?since=" + _pollCursor + "&t=" + Date.now());
    if (r.ok) {
      const data = await r.json();
      if (_pollCursor === -1) blog("poll: first poll ok, " + data.events.length + " events");
      _pollCursor = data.cursor;
      for (const line of data.events) {
        try {
          const ev = JSON.parse(line);
          if (_evCount++ < 5 || ev.t !== "boxes") console.log("[hud] event", ev.t, ev.count != null ? "count=" + ev.count : "");
          if (ev.t !== "boxes") blog("event t=" + ev.t + (ev.label ? " label=" + ev.label : "") + (ev.count != null ? " count=" + ev.count : ""));
          handle(ev);
        } catch (err) { console.error("[hud] parse failed:", err, "raw:", line); }
      }
    }
  } catch (err) { console.warn("[hud] poll error", err); }
  setTimeout(pollEvents, 500);
}
blog("pollEvents starting");
pollEvents();

function handle(ev) {
  if (ev.t === "boxes") {
    frameDim = ev.frame;
    const seen = new Set();
    for (const it of ev.items) {
      seen.add(it.id);
      const tgt = { x: it.x, y: it.y, w: it.w, h: it.h, label: it.label, score: it.score };
      const t = tracks.get(it.id);
      if (t) { t.target = tgt; t.label = it.label; t.score = it.score; }
      else tracks.set(it.id, { cur: { ...tgt }, target: tgt, label: it.label, score: it.score });
    }
    for (const id of [...tracks.keys()]) if (!seen.has(id)) tracks.delete(id);
  } else if (ev.t === "stored") {
    addToRail(ev.id, ev.label);
    $("t-count").textContent = ev.count;
    $("mem-count").textContent = ev.count + (ev.count == 1 ? " object" : " objects");
  } else if (ev.t === "tick") {
    $("t-detect").textContent = ev.detect + "ms";
    $("t-embed").textContent = ev.embed + "ms";
    $("t-store").textContent = ev.store + "ms";
    $("t-count").textContent = ev.count;
  } else if (ev.t === "mode") {
    setMode(ev.value, ev.query);
  } else if (ev.t === "results") {
    renderResults(ev.items);
  } else if (ev.t === "moment") {
    addToTimeline(ev.id, ev.ts);
    $("moment-count").textContent = ev.count + (ev.count == 1 ? " moment" : " moments");
  }
}

function addToRail(key, label) {
  const rail = $("rail");
  console.log("[hud] addToRail key=", key, "label=", label, "rail?=", !!rail, "children=", rail ? rail.children.length : "N/A");
  if (!rail) { console.error("[hud] #rail element MISSING"); return; }
  const cell = document.createElement("div");
  cell.className = "cell";
  const img = document.createElement("img");
  img.src = "/thumb/" + key;
  img.onerror = () => console.warn("[hud] thumb failed to load:", img.src);
  img.onload = () => console.log("[hud] thumb loaded:", img.src);
  const span = document.createElement("span");
  span.textContent = label;
  cell.appendChild(img); cell.appendChild(span);
  rail.prepend(cell);
  console.log("[hud] rail now has", rail.children.length, "cells");
  while (rail.children.length > RAIL_CAP) rail.removeChild(rail.lastChild);
}

const TIMELINE_CAP = 60;
// Moment timeline (Task 1.6, Codex P1 fix): mirrors addToRail's fetch/thumb/DOM approach
// (same /thumb/<key> mechanism, same .cell markup) but APPENDS instead of prepending — a
// timeline reads oldest-to-newest left-to-right, the opposite of the rail's newest-first grid.
function addToTimeline(key, tsMs) {
  const strip = $("timeline");
  if (!strip) { console.error("[hud] #timeline element MISSING"); return; }
  const cell = document.createElement("div");
  cell.className = "cell";
  const img = document.createElement("img");
  img.src = "/thumb/" + key;
  img.onerror = () => console.warn("[hud] moment thumb failed to load:", img.src);
  const span = document.createElement("span");
  span.textContent = formatTime(tsMs);
  cell.appendChild(img); cell.appendChild(span);
  strip.appendChild(cell);
  while (strip.children.length > TIMELINE_CAP) strip.removeChild(strip.firstChild);
  strip.scrollLeft = strip.scrollWidth;   // keep the just-appended (newest) moment in view
}

function formatTime(ms) {
  const d = new Date(ms);
  const pad = (n) => n.toString().padStart(2, "0");
  return pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
}

function setMode(value, query) {
  $("mode-badge").textContent = value.toUpperCase();
  $("rec").classList.toggle("hidden", value !== "recording");
  // Search panel stays visible in the side column (reference layout); only its content changes.
  if (value === "search") {
    $("search-query").textContent = query || "";
    $("results").innerHTML = "<div class='searching'>searching…</div>";
  }
}

function renderResults(items) {
  const wrap = $("results");
  wrap.innerHTML = "";
  if (!items.length) { wrap.innerHTML = "<div class='searching'>no matches</div>"; return; }
  for (const r of items) {
    const d = document.createElement("div");
    d.className = "cell";
    d.innerHTML = `<img src="/thumb/${r.id}" onerror="this.classList.add('broken')"><span>${r.label} ${(r.score*100|0)}%</span>`;
    wrap.appendChild(d);
  }
}

// render loop: scale frame coords to the displayed <img>, draw + interpolate boxes
function draw() {
  // Use the canvas's own rendered size (CSS now stretches it over the feed). Before the video
  // has loaded, feed.clientWidth can be 0 — fall back to the wrap width so boxes still scale.
  const rw = overlay.clientWidth || feed.clientWidth, rh = overlay.clientHeight || feed.clientHeight;
  if (!rw || !rh) { requestAnimationFrame(draw); return; }
  if (overlay.width !== rw || overlay.height !== rh) { overlay.width = rw; overlay.height = rh; }
  const sx = rw / frameDim[0], sy = rh / frameDim[1];
  octx.clearRect(0, 0, rw, rh);
  octx.lineWidth = 2; octx.strokeStyle = "#15c46a"; octx.fillStyle = "#15c46a";
  octx.font = "13px sans-serif";
  for (const t of tracks.values()) {
    const c = t.cur, g = t.target;
    c.x += (g.x - c.x) * 0.3; c.y += (g.y - c.y) * 0.3;
    c.w += (g.w - c.w) * 0.3; c.h += (g.h - c.h) * 0.3;
    const x = c.x * sx, y = c.y * sy, w = c.w * sx, h = c.h * sy;
    octx.strokeRect(x, y, w, h);
    const tag = t.score != null ? `${t.label} ${(t.score*100|0)}%` : t.label;
    octx.fillText(tag, x, Math.max(12, y - 4));
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
