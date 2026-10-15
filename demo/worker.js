// Runs the project's Python code in Pyodide, off the main thread.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
importScripts(PYODIDE + "pyodide.js");

const DT = 0.064;
let py = null;
let demo = null;
let running = false;
let speed = 1; // sim seconds per wall second; 0 = as fast as possible
let budget = 0;
let last = 0;
let timer = null;

const status = (text) => postMessage({ type: "status", text });

async function fetchText(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.text();
}

async function boot() {
  status("Starting Python (Pyodide 0.27.7)");
  py = await loadPyodide({ indexURL: PYODIDE, stdout: () => {}, stderr: (s) => console.warn(s) });
  status("Loading NumPy");
  await py.loadPackage("numpy");

  status("Fetching the project's modules");
  const files = JSON.parse(await fetchText("core/manifest.json"));
  const sources = await Promise.all(files.map((f) => fetchText("core/" + f)));
  py.FS.mkdirTree("/app/core");
  files.forEach((f, i) => {
    const dir = "/app/core/" + f.split("/")[0];
    py.FS.mkdirTree(dir);
    py.FS.writeFile("/app/core/" + f, sources[i]);
  });
  py.FS.mkdirTree("/app/demo");
  for (const f of ["sim.py", "runner.py"]) {
    py.FS.writeFile("/app/demo/" + f, await fetchText("py/" + f));
  }
  const mapText = await fetchText("maps/rooms.json");

  status("Importing slam.fastslam, explore.frontiers, control.path_planner");
  py.globals.set("MAP_JSON", mapText);
  py.runPython(`
import sys, json
sys.path[:0] = ["/app/core", "/app/demo"]
import runner
runner.silence_prints()
MAP = json.loads(MAP_JSON)
`);
  const map = JSON.parse(mapText);
  postMessage({ type: "ready", files, map });
}

function reset({ particles, noise }) {
  if (demo) demo.destroy();
  py.globals.set("N_PARTICLES", particles);
  py.globals.set("NOISE", noise);
  demo = py.runPython("runner.Demo(MAP, num_particles=N_PARTICLES, noise=NOISE)");
  budget = 0;
  sendFrame();
}

function buf(f, key) {
  const p = f.get(key);
  const b = p.getBuffer();
  const out = b.data.slice();
  b.release();
  p.destroy();
  return out;
}

function sendFrame(stepMs = 0) {
  const f = demo.frame();
  const msg = {
    type: "frame",
    step: f.get("step"),
    t: f.get("t"),
    err: f.get("err"),
    herr: f.get("herr"),
    spread: f.get("spread"),
    n: f.get("n"),
    explored: f.get("explored"),
    done: f.get("done"),
    doneReason: f.get("done_reason"),
    stepMs,
    true: buf(f, "true"),
    est: buf(f, "est"),
    particles: buf(f, "particles"),
    map: buf(f, "map"),
    frontiers: buf(f, "frontiers"),
    path: buf(f, "path"),
    ranges: buf(f, "ranges"),
  };
  f.destroy();
  postMessage(msg, [msg.map.buffer, msg.particles.buffer, msg.frontiers.buffer, msg.path.buffer, msg.ranges.buffer]);
}

function tick() {
  timer = null;
  if (!running || !demo) return;
  const now = performance.now();
  const elapsed = Math.min(0.25, (now - last) / 1000);
  last = now;
  budget += speed === 0 ? Infinity : (elapsed * speed) / DT;
  const t0 = performance.now();
  let steps = 0;
  // Step until caught up with the requested speed or 45 ms of work, then draw.
  try {
    while (budget >= 1 && performance.now() - t0 < 45 && !demo.done) {
      demo.step();
      budget -= 1;
      steps++;
    }
  } catch (err) {
    running = false;
    postMessage({ type: "error", text: String(err.message || err) });
    return;
  }
  if (speed === 0 || budget > 8) budget = Math.min(budget, 1);
  if (steps > 0) sendFrame((performance.now() - t0) / steps);
  if (demo.done) {
    running = false;
    return;
  }
  timer = setTimeout(tick, steps > 0 ? 0 : 8);
}

function setRunning(on) {
  running = on;
  if (on && !timer) {
    last = performance.now();
    tick();
  }
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "boot") await boot();
    else if (m.type === "reset") reset(m);
    else if (m.type === "run") setRunning(m.on);
    else if (m.type === "speed") speed = m.value;
    else if (m.type === "noise" && demo) demo.set_noise(m.value);
  } catch (err) {
    running = false;
    postMessage({ type: "error", text: String(err && err.message ? err.message : err) });
  }
};
