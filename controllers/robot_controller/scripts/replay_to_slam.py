# scripts/replay_to_slam.py
import time
from io.recorder import read_jsonl
# Ramy: import your SLAM
# from slam.fastslam import FastSLAM

def main():
    # slam = FastSLAM(...)
    prev_t = None
    for pkt in read_jsonl("eval_logs/run_live.jsonl"):
        # Ramy: adapt to your API
        # slam.predict(pkt["odom"])      # motion update
        # slam.update(pkt["scan"])       # measurement update
        # grid, pose = slam.get_map_and_pose()

        # sleep to approximatae original timing
        if prev_t is not None:
            time.sleep(max(0.0, pkt["t"] - prev_t))
        prev_t = pkt["t"]

    print("Replay finished.")

if __name__ == "__main__":
    main()
