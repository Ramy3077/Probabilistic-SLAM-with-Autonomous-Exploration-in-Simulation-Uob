# Probabilistic SLAM with Autonomous Exploration

A particle-filter SLAM system in Python that lets a simulated robot map a building it has never seen and choose where to explore next.

Group project of three (Intelligent Robotics, University of Birmingham). Ramy Mekhzer wrote the SLAM core: motion model, particle weighting, resampling and occupancy mapping. Bassel Kenaan built the exploration and Sahib Singh the control. Details under [Team and roles](#team-and-roles).

[![tests](https://github.com/Ramy3077/Probabilistic-SLAM-with-Autonomous-Exploration-in-Simulation-Uob/actions/workflows/tests.yml/badge.svg)](https://github.com/Ramy3077/Probabilistic-SLAM-with-Autonomous-Exploration-in-Simulation-Uob/actions/workflows/tests.yml)

**Live demo: [fastslam-demo-production.up.railway.app](https://fastslam-demo-production.up.railway.app)** (runs in the browser, no install)

[![The browser demo after a run: estimated map, true and estimated trails, and the Done summary with coverage, time and final pose error](docs/demo.png)](https://fastslam-demo-production.up.railway.app)

## How the SLAM works

The robot only knows two things: how far its wheels turned and what its 360 degree LiDAR sees. A cloud of particles, each one a guess of the robot's pose, is moved forward with a differential-drive motion model whose noise grows with speed and turn rate, and which keeps a few particles on the hypothesis that the wheels slipped. Each particle is then scored with a likelihood field: every LiDAR endpoint is compared with its distance to the nearest wall in the occupancy grid built so far. The weighted mean of the particles is the pose estimate and is used to write the scan into the map. The map stores log-odds per 5 cm cell: cells along a beam are marked free and the cell at its end is marked occupied. When the weights collapse onto a few particles, the set is resampled with low-variance resampling.

This is a simplified FastSLAM: every particle shares one global map instead of carrying its own.

## Live demo

`demo/` runs the project's own Python modules in the browser through [Pyodide](https://pyodide.org): `slam/`, `models/`, `explore/` and `control/` are copied in at build time and imported unchanged. What is specific to the demo:

- **World:** the walls of `webots_worlds/SLAM_Project.wbt`, rasterised to a 5 cm grid by `demo/tools/wbt_to_grid.py`. Furniture is left out.
- **Simulator:** `demo/py/sim.py` stands in for Webots. Same robot radius, wheel base and 4 m LiDAR range, with wheel slip and range noise you can turn up.
- **Loop:** `demo/py/runner.py` follows the per-tick loop of `scripts/live_slam_frontier.py`. The A* safety-margin check is answered from a precomputed obstacle dilation so it runs fast in WebAssembly; the result is the same. The demo adds recovery the Webots loop does not have: it re-plans every 10 ticks rather than every tick, gives up on goals it stops making progress towards, drives towards frontiers that `explore/planner.py` cannot score (thin LiDAR fans seen through doorways), and stops with a summary once no reachable frontier is left.

Run it locally:

```bash
sh demo/build.sh
python3 -m http.server 8000 --directory demo/dist
# open http://localhost:8000
```

## Results (2D demo simulator)

These numbers come from the browser demo's 2D simulator (`demo/py/sim.py`) run headless, not from Webots. Each row is 10 runs (seeds 0 to 9) of the full loop until it reaches its terminal state, on the `SLAM_Project.wbt` wall layout. Noise scales the simulator's wheel slip and range noise; the filter's own noise model is unchanged. Pose error is the distance between the filter's estimate and the true position at the end of the run. Free space mapped is the share of the world's free cells that the estimated map has marked as known.

| Particles | Noise | Final pose error (m), mean ± sd | Median pose error (m) | Free space mapped (%), mean ± sd | Time to Done (s), mean ± sd |
|---|---|---|---|---|---|
| 10 | 0.5x | 0.66 ± 1.91 | 0.05 | 98.4 ± 4.9 | 76.0 ± 31.6 |
| 10 | 1.0x | 0.04 ± 0.04 | 0.03 | 99.9 ± 0.1 | 73.1 ± 12.2 |
| 10 | 2.0x | 1.95 ± 3.22 | 0.14 | 94.7 ± 8.0 | 108.7 ± 55.4 |
| 30 | 0.5x | 0.07 ± 0.05 | 0.05 | 99.7 ± 0.6 | 76.3 ± 18.7 |
| 30 | 1.0x | 0.08 ± 0.11 | 0.05 | 99.4 ± 1.8 | 81.7 ± 24.6 |
| 30 | 2.0x | 0.08 ± 0.08 | 0.06 | 99.7 ± 0.3 | 83.2 ± 32.7 |
| 100 | 0.5x | 0.07 ± 0.03 | 0.08 | 96.7 ± 6.5 | 102.6 ± 24.0 |
| 100 | 1.0x | 0.10 ± 0.20 | 0.04 | 99.5 ± 1.2 | 100.0 ± 54.0 |
| 100 | 2.0x | 0.07 ± 0.12 | 0.03 | 95.3 ± 14.5 | 72.1 ± 17.2 |

All 90 runs stopped on their own. 5 of 90 ended more than 0.5 m off, 4 of them with 10 particles; before the fixes in [docs/ACCURACY_INVESTIGATION.md](docs/ACCURACY_INVESTIGATION.md) it was 50 of 90, and the old table is kept in `results/before/`.

Reproduce (runs in parallel on all cores):

```bash
pip install numpy
python scripts/bench_demo.py
```

Per-run results are in `results/bench_demo_runs.csv` and the table above is `results/bench_demo_summary.csv`.

## Design decisions and trade-offs

- **One shared map instead of a map per particle.** Full FastSLAM gives each particle its own map, so a particle that guessed the pose wrong also carries its own wrong map and can be dropped with it. Here the particles only estimate the pose, and the best one writes into a single grid (`slam/fastslam.py`). Memory and map updates stay constant as the particle count grows, which matters in Python and in the browser. Cost: a wrong pose written into the map is only cleared slowly by later scans, and there is no loop closure. Measured against per-particle maps on 60 s runs, accuracy was the same and per-particle maps cost about 25 times the time per step; see the investigation report.
- **Pick the mapping pose before resampling.** After resampling the weights are uniform, so the pose that writes the map is taken while the weights still reflect the scan match. It is the weighted mean of the particles.
- **Particle count.** Each particle scores every LiDAR beam in the demo (180) and every second beam in Webots (256 of 512); scoring is vectorised, so cost grows slowly with the particle count. With 10 particles the filter can still lose track after a collision (see the table); the Webots loop uses 50 and the demo 40.
- **Recovery logic in the demo loop, not in the planner.** The demo needed to stop parking when frontiers remained and to end runs with a result. `explore/planner.py` and the control code belong to teammates and are what the Webots loop runs, so the fixes (re-plan every 10 ticks, drop goals that stop making progress, drive towards frontiers the planner cannot score, stop when nothing reachable is left) live in `demo/py/runner.py`. Cost: the demo and the Webots loop now behave differently, and the Webots loop does not get these fixes.
- **A 2D stand-in simulator for the demo and benchmark.** Webots cannot run in a browser or in CI, so `demo/py/sim.py` uses the same robot radius, wheel base and LiDAR range on the real wall layout. Cost: no furniture, simplified physics, and the results table is for this simulator, not Webots.

## Known limitations

- A long stall against a wall, with the wheels still turning, can still pull the estimate off and leave a smeared wall in the shared map.
- No loop closure, so error builds up over long runs.
- The benchmark numbers come from the 2D demo simulator, not Webots. The Webots worlds have not been re-run since the accuracy fixes.
- The motion model's noise parameters are set by hand, not calibrated on the Webots robot.

## Run the tests

```bash
pip install numpy pytest
pytest
```

The suite covers particle initialisation, the motion update, resampling, a short end-to-end SLAM loop, frontier detection, the evaluation metrics, and a headless run of the demo loop. CI runs it on every push.

## Run it in Webots

1. Install [Webots R2025a](https://cyberbotics.com/) (the worlds use R2025a protos) and the Python packages: `pip install -r requirements.txt`.
2. Open one of the worlds in `webots_worlds/`: `SLAM_Project.wbt`, `Small_Room.wbt`, `Maze.wbt` or `Large_Appartment.wbt`.
3. The robot's controller is `robot_controller`. By default it runs frontier exploration (`scripts/live_slam_frontier.py`).
4. To run the random-walk baseline instead, change the import at the top of `controllers/robot_controller/robot_controller.py` to `scripts.live_slam_random`.
5. Map snapshots and a CSV log are written to `controllers/robot_controller/eval_logs/` as the run goes.

## Project structure

```
controllers/robot_controller/
  robot_controller.py      Webots entry point
  slam/
    fastslam.py            particle filter: predict, weight, resample, map update
    particles.py           particle set, effective sample size, low-variance resampling
    occupancy.py           log-odds occupancy grid and inverse sensor model
    visualize.py           map snapshots (matplotlib)
  models/motion.py         differential-drive motion models
  explore/
    frontiers.py           frontier detection and clustering
    planner.py             frontier scoring and selection
  control/
    path_planner.py        A* on the occupancy grid, path simplification
    waypoint_follow.py     go-to-goal controller
    trap_recovery.py       stuck detection for the random baseline
  robots/robot.py          Webots motors, encoders and LiDAR
  eval/                    coverage and entropy metrics, CSV logger
  scripts/                 run loops (live_slam_frontier.py, live_slam_random.py,
                           benchmark_exploration.py) and plotting helpers
  tests/                   frontier and metric tests
tests/                     particle filter and SLAM loop tests
demo/                      browser demo (Pyodide) and its build script
scripts/bench_demo.py      headless benchmark of the demo simulator
results/                   benchmark CSVs
webots_worlds/             the four Webots worlds
```

`controllers/robot_controller/eval_logs/` holds two CSVs from mock runs used while developing the logger. They are not experimental results.

## Team and roles

Group project for Intelligent Robotics (06-30227), University of Birmingham.

- **Ramy Mekhzer:** SLAM core. Particle filter, motion and measurement models and the step-window inverse sensor model.
- **Bassel Kenaan:** exploration. Frontier detection and clustering, frontier selection, evaluation metrics and logging.
- **Sahib Singh:** robot control. Webots robot interface, A* path planning and waypoint following.

## Attribution

The motion model equations and the overall particle filter structure follow [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (Atsushi Sakai). The Bresenham line tracing in `slam/occupancy.py` follows [toolbuddy/2D-Grid-SLAM](https://github.com/toolbuddy/2D-Grid-SLAM). Webots provides the simulator and hardware API; NumPy and Matplotlib are used throughout.
