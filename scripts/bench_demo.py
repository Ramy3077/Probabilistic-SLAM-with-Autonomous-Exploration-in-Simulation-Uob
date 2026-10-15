"""Benchmark the browser demo's simulator headless.

Runs demo/py/runner.Demo (the project's SLAM, exploration and control modules
driven by the 2D stand-in simulator, not Webots) to its terminal state for
each particle count, noise level and seed, and writes:

    results/bench_demo_runs.csv     one row per run
    results/bench_demo_summary.csv  mean and sample sd per configuration

Usage:
    python scripts/bench_demo.py                 # full grid, 10 seeds
    python scripts/bench_demo.py --seeds 3 --particles 10 30 --noise 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "controllers" / "robot_controller"), str(ROOT / "demo" / "py")]

MAX_SIM_SECONDS = 600.0


def run_one(args):
    particles, noise, seed = args
    import numpy as np
    import runner

    runner.silence_prints()
    np.random.seed(seed)  # resampling jitter uses NumPy's global RNG
    spec = json.loads((ROOT / "demo" / "maps" / "rooms.json").read_text())
    demo = runner.Demo(spec, num_particles=particles, noise=noise, seed=seed)
    limit = int(MAX_SIM_SECONDS / runner.DT)
    while not demo.done and demo.step_count < limit:
        demo.step()
    f = demo.frame()
    return {
        "particles": particles,
        "noise": noise,
        "seed": seed,
        "sim_seconds": round(f["t"], 2),
        "coverage": round(f["explored"], 4),
        "pose_error_m": round(f["err"], 4),
        "heading_error_deg": round(math.degrees(f["herr"]), 2),
        "terminated": bool(f["done"]),
        "reason": f["done_reason"] or "time limit",
    }


def summarise(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["particles"], r["noise"]), []).append(r)
    out = []
    for (p, n), rs in sorted(groups.items()):
        def ms(key, scale=1.0):
            v = [r[key] * scale for r in rs]
            return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
        err_m, err_s = ms("pose_error_m")
        err_med = statistics.median(r["pose_error_m"] for r in rs)
        cov_m, cov_s = ms("coverage", 100.0)
        t_m, t_s = ms("sim_seconds")
        out.append({
            "particles": p, "noise": n, "runs": len(rs),
            "terminated": sum(r["terminated"] for r in rs),
            "pose_error_m_mean": round(err_m, 2), "pose_error_m_sd": round(err_s, 2),
            "pose_error_m_median": round(err_med, 2),
            "coverage_pct_mean": round(cov_m, 1), "coverage_pct_sd": round(cov_s, 1),
            "sim_seconds_mean": round(t_m, 1), "sim_seconds_sd": round(t_s, 1),
        })
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--particles", type=int, nargs="+", default=[10, 30, 100])
    ap.add_argument("--noise", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=10, help="seeds 0..N-1 per configuration")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary from an existing bench_demo_runs.csv without re-running")
    a = ap.parse_args()

    if a.summarise_only:
        with (a.out / "bench_demo_runs.csv").open() as fh:
            rows = [{**r, "particles": int(r["particles"]), "noise": float(r["noise"]),
                     "sim_seconds": float(r["sim_seconds"]), "coverage": float(r["coverage"]),
                     "pose_error_m": float(r["pose_error_m"]), "terminated": r["terminated"] == "True"}
                    for r in csv.DictReader(fh)]
    else:
        jobs = [(p, n, s) for p in a.particles for n in a.noise for s in range(a.seeds)]
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            rows = []
            for r in pool.map(run_one, jobs):
                rows.append(r)
                print(f"particles={r['particles']:>3} noise={r['noise']:<4} seed={r['seed']:<2} "
                      f"coverage={r['coverage']:.3f} err={r['pose_error_m']:.2f} m t={r['sim_seconds']:.0f} s ({r['reason']})",
                      flush=True)
        write_csv(a.out / "bench_demo_runs.csv", rows)
    summary = summarise(rows)
    write_csv(a.out / "bench_demo_summary.csv", summary)

    print("\n| Particles | Noise | Final pose error (m) | Median pose error (m) | Free space mapped (%) | Time to Done (s) |")
    print("|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['particles']} | {s['noise']}x | {s['pose_error_m_mean']:.2f} ± {s['pose_error_m_sd']:.2f} "
              f"| {s['pose_error_m_median']:.2f} "
              f"| {s['coverage_pct_mean']:.1f} ± {s['coverage_pct_sd']:.1f} "
              f"| {s['sim_seconds_mean']:.1f} ± {s['sim_seconds_sd']:.1f} |")


if __name__ == "__main__":
    main()
