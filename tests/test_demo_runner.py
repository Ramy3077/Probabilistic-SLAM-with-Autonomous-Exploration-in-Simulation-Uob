"""Headless run of the browser demo loop (demo/py) against the real modules."""
import json
from pathlib import Path

import numpy as np

import runner

MAP = Path(__file__).resolve().parent.parent / "demo" / "maps" / "rooms.json"


def test_demo_builds_a_map_and_tracks_pose():
    runner.silence_prints()
    np.random.seed(0)
    demo = runner.Demo(json.loads(MAP.read_text()), num_particles=20, noise=1.0, seed=0)
    for _ in range(150):
        demo.step()
    frame = demo.frame()

    assert frame["step"] == 150
    assert len(frame["particles"]) == 20 * 3
    prob = frame["map"]
    assert (prob > 200).sum() > 50, "no walls mapped"
    assert (prob < 60).sum() > 500, "no free space mapped"
    assert frame["err"] < 1.0
    # the robot actually drove somewhere
    assert np.hypot(*(np.array(frame["true"][:2]) - demo.world.start[:2])) > 0.3


def test_demo_run_terminates():
    """The demo stops on its own with a reason instead of parking forever."""
    runner.silence_prints()
    np.random.seed(1)
    demo = runner.Demo(json.loads(MAP.read_text()), num_particles=20, noise=1.0, seed=1)
    limit = int(300 / runner.DT)  # 300 s of simulated time
    while not demo.done and demo.step_count < limit:
        demo.step()
    frame = demo.frame()
    assert frame["done"], "run did not reach a terminal state"
    assert frame["done_reason"]
    assert frame["explored"] > 0.25
