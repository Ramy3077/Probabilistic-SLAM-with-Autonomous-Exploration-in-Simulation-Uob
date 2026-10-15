"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
const overlay = $("overlay");
const statusEl = $("status");

const css = getComputedStyle(document.documentElement);
const C = {
  truth: css.getPropertyValue("--truth").trim(),
  est: css.getPropertyValue("--accent").trim(),
  part: css.getPropertyValue("--part").trim(),
  front: css.getPropertyValue("--front").trim(),
  plan: css.getPropertyValue("--plan").trim(),
  wall: css.getPropertyValue("--wall").trim(),
};

const ROBOT_RADIUS = 0.225;
const LIDAR_BEAMS = 180;

let map = null;          // map spec (from rooms.json)
let truthLayer = null;   // offscreen canvas of the real walls
let gridLayer = null;    // offscreen canvas of the estimated occupancy grid
let gridImg = null;
let frame = null;
let trailTrue = [];
let trailEst = [];
let running = false;
let ready = false;
let dirty = true;

const worker = new Worker("worker.js");
worker.onmessage = (e) => {
  const m = e.data;
  if (m.type === "status") statusEl.textContent = m.text;
  else if (m.type === "ready") onReady(m);
  else if (m.type === "frame") onFrame(m);
  else if (m.type === "error") showError(m.text);
};
worker.onerror = (e) => showError(e.message || "Worker failed to start");
worker.postMessage({ type: "boot" });

function showError(text) {
  console.error(text);
  overlay.classList.remove("hidden");
  overlay.classList.add("error");
  statusEl.textContent = "Something went wrong: " + text;
  setRunning(false);
}

function onReady(m) {
  map = m.map;
  $("files").textContent = "Loaded from the repo: " + m.files.filter((f) => !f.endsWith("__init__.py")).join("  ");
  buildTruthLayer();
  gridLayer = document.createElement("canvas");
  gridLayer.width = map.width;
  gridLayer.height = map.height;
  gridImg = gridLayer.getContext("2d").createImageData(map.width, map.height);
  ready = true;
  $("play").disabled = false;
  $("reset").disabled = false;
  reset();
  overlay.classList.add("hidden");
  setRunning(true);
}

function buildTruthLayer() {
  truthLayer = document.createElement("canvas");
  truthLayer.width = map.width;
  truthLayer.height = map.height;
  const g = truthLayer.getContext("2d");
  const img = g.createImageData(map.width, map.height);
  const [r, gg, b] = hex(C.wall);
  for (let i = 0; i < map.height; i++) {
    const row = map.rows[i];
    const y = map.height - 1 - i; // row 0 is the bottom of the world
    for (let j = 0; j < map.width; j++) {
      if (row.charCodeAt(j) === 49) {
        const k = (y * map.width + j) * 4;
        img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b; img.data[k + 3] = 255;
      }
    }
  }
  g.putImageData(img, 0, 0);
}

function hex(h) {
  const n = parseInt(h.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function reset() {
  trailTrue = [];
  trailEst = [];
  frame = null;
  worker.postMessage({ type: "reset", particles: +$("particles").value, noise: +$("noise").value });
}

function setRunning(on) {
  running = on && ready;
  $("play").textContent = running ? "Pause" : "Play";
  worker.postMessage({ type: "run", on: running });
}

function onFrame(f) {
  frame = f;
  const last = trailTrue[trailTrue.length - 1];
  if (!last || Math.hypot(last[0] - f.true[0], last[1] - f.true[1]) > 0.01) {
    trailTrue.push([f.true[0], f.true[1]]);
    trailEst.push([f.est[0], f.est[1]]);
  }
  // occupancy probabilities (0..255) to pixels; unknown = transparent
  const d = gridImg.data;
  const W = map.width, H = map.height;
  for (let i = 0; i < H; i++) {
    const src = i * W;
    const dst = (H - 1 - i) * W;
    for (let j = 0; j < W; j++) {
      const p = f.map[src + j];
      const k = (dst + j) * 4;
      if (p > 140) {          // occupied
        const a = Math.min(255, (p - 128) * 2.2);
        d[k] = 236; d[k + 1] = 241; d[k + 2] = 247; d[k + 3] = a;
      } else if (p < 116) {   // free
        const a = Math.min(255, (128 - p) * 2.6);
        d[k] = 38; d[k + 1] = 50; d[k + 2] = 68; d[k + 3] = a;
      } else {
        d[k + 3] = 0;
      }
    }
  }
  gridLayer.getContext("2d").putImageData(gridImg, 0, 0);
  updateStats(f);
  const complete = f.done || f.explored >= 0.995;
  if (complete && running) {
    setRunning(false);
    $("play").textContent = "Run again";
  }
  frame.complete = complete;
  dirty = true;
}

function updateStats(f) {
  $("sTime").textContent = f.t.toFixed(1) + " s";
  $("sErr").textContent = f.err.toFixed(2) + " m";
  $("sHerr").textContent = ((f.herr * 180) / Math.PI).toFixed(1) + "°";
  $("sSpread").textContent = Math.round(f.spread * 100) + " cm";
  $("sExpl").textContent = Math.round(f.explored * 100) + "%";
  $("sMs").textContent = f.stepMs ? f.stepMs.toFixed(1) + " ms" : "-";
}

// ---------- drawing ----------

let view = { s: 1, ox: 0, oy: 0 };

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  dirty = true;
}
new ResizeObserver(resize).observe(canvas);

function layout() {
  const pad = 10 * (window.devicePixelRatio || 1);
  const s = Math.min((canvas.width - 2 * pad) / map.width, (canvas.height - 2 * pad) / map.height);
  view = {
    s,
    ox: (canvas.width - map.width * s) / 2,
    oy: (canvas.height - map.height * s) / 2,
  };
}

// world metres to canvas pixels
const px = (x) => view.ox + ((x - map.origin_x) / map.resolution) * view.s;
const py = (y) => view.oy + (map.height - (y - map.origin_y) / map.resolution) * view.s;
const m2px = (m) => (m / map.resolution) * view.s;

function draw() {
  requestAnimationFrame(draw);
  if (!dirty || !map) return;
  dirty = false;
  layout();
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.imageSmoothingEnabled = false;

  // faint cell grid, 1 m
  ctx.strokeStyle = "rgba(255,255,255,0.035)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const x0 = Math.ceil(map.origin_x), x1 = map.origin_x + map.width * map.resolution;
  const y0 = Math.ceil(map.origin_y), y1 = map.origin_y + map.height * map.resolution;
  for (let x = x0; x <= x1; x++) { ctx.moveTo(px(x), py(map.origin_y)); ctx.lineTo(px(x), py(y1)); }
  for (let y = y0; y <= y1; y++) { ctx.moveTo(px(map.origin_x), py(y)); ctx.lineTo(px(x1), py(y)); }
  ctx.stroke();

  const gw = map.width * view.s, gh = map.height * view.s;
  ctx.drawImage(gridLayer, view.ox, view.oy, gw, gh);

  if ($("showTruth").checked) {
    ctx.globalAlpha = 0.32;
    ctx.drawImage(truthLayer, view.ox, view.oy, gw, gh);
    ctx.globalAlpha = 1;
  }
  if (!frame) return;
  const f = frame;
  const cell = Math.max(1.5, view.s);

  // frontier cells
  ctx.fillStyle = C.front;
  for (let k = 0; k < f.frontiers.length; k += 2) {
    const i = f.frontiers[k], j = f.frontiers[k + 1];
    ctx.fillRect(view.ox + j * view.s, view.oy + (map.height - 1 - i) * view.s, cell, cell);
  }

  // LiDAR hits, placed from the estimated pose
  if ($("showScan").checked) {
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    const [ex, ey, et] = f.est;
    const inc = (2 * Math.PI) / LIDAR_BEAMS;
    for (let k = 0; k < f.ranges.length; k++) {
      const r = f.ranges[k];
      if (r <= 0) continue;
      const a = et - Math.PI + k * inc;
      ctx.fillRect(px(ex + r * Math.cos(a)) - 1, py(ey + r * Math.sin(a)) - 1, 2.2, 2.2);
    }
  }

  // planned path
  if (f.path.length >= 2) {
    ctx.strokeStyle = C.plan;
    ctx.lineWidth = Math.max(1.5, view.s * 0.5);
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(px(f.est[0]), py(f.est[1]));
    for (let k = 0; k < f.path.length; k += 2) ctx.lineTo(px(f.path[k]), py(f.path[k + 1]));
    ctx.stroke();
    ctx.setLineDash([]);
    const gx = f.path[f.path.length - 2], gy = f.path[f.path.length - 1];
    ctx.beginPath();
    ctx.arc(px(gx), py(gy), m2px(0.12), 0, Math.PI * 2);
    ctx.stroke();
  }

  trail(trailTrue, C.truth, 0.9);
  trail(trailEst, C.est, 0.9);

  if ($("showParticles").checked) {
    ctx.fillStyle = C.part;
    ctx.globalAlpha = 0.85;
    const r = Math.max(1.2, view.s * 0.35);
    for (let k = 0; k < f.particles.length; k += 3) {
      ctx.beginPath();
      ctx.arc(px(f.particles[k]), py(f.particles[k + 1]), r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  robot(f.true, C.truth, false);
  robot(f.est, C.est, true);
}

function trail(pts, color, alpha) {
  if (pts.length < 2) return;
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = Math.max(1.2, view.s * 0.4);
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(px(pts[0][0]), py(pts[0][1]));
  for (let k = 1; k < pts.length; k++) ctx.lineTo(px(pts[k][0]), py(pts[k][1]));
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function robot(p, color, filled) {
  const [x, y, th] = p;
  const r = m2px(ROBOT_RADIUS);
  ctx.lineWidth = Math.max(1.5, view.s * 0.5);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(px(x), py(y), r, 0, Math.PI * 2);
  if (filled) {
    ctx.globalAlpha = 0.25;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(px(x), py(y));
  ctx.lineTo(px(x + 1.4 * ROBOT_RADIUS * Math.cos(th)), py(y + 1.4 * ROBOT_RADIUS * Math.sin(th)));
  ctx.stroke();
}

requestAnimationFrame(draw);

// ---------- controls ----------

$("play").onclick = () => {
  if (frame && frame.complete) reset();
  setRunning(!running);
};
$("reset").onclick = () => reset();

$("speed").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  for (const o of $("speed").children) o.setAttribute("aria-checked", String(o === b));
  worker.postMessage({ type: "speed", value: +b.dataset.v });
});

$("particles").addEventListener("input", (e) => ($("pOut").textContent = e.target.value));
$("particles").addEventListener("change", () => ready && reset());
$("noise").addEventListener("input", (e) => {
  $("nOut").textContent = (+e.target.value).toFixed(2).replace(/0$/, "") + "x";
  worker.postMessage({ type: "noise", value: +e.target.value });
});
for (const id of ["showTruth", "showScan", "showParticles"]) $(id).addEventListener("change", () => (dirty = true));

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "checkbox") return;
  if (e.code === "Space" && ready) { e.preventDefault(); setRunning(!running); }
  if (e.key === "r" && ready) reset();
});
