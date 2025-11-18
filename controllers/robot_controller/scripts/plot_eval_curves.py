# scripts/plot_eval_curves.py

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(path):
    ts = []
    cov = []
    ent = []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        t0 = None
        for row in reader:
            t = float(row["t"])
            if t0 is None:
                t0 = t
            ts.append(t - t0)  # relative time (seconds)
            cov.append(float(row["coverage_pct"]))
            ent.append(float(row["entropy_proxy"]))
    return ts, cov, ent


def main():
    frontier_path = Path("eval_logs/week3_frontier_mock.csv")
    random_path = Path("eval_logs/week3_random_mock.csv")

    if not frontier_path.exists() or not random_path.exists():
        print("Missing logs. Run live_exploration_loop and live_random_loop first.")
        return

    t_f, cov_f, ent_f = load_log(frontier_path)
    t_r, cov_r, ent_r = load_log(random_path)

    # --- Coverage plot ---
    plt.figure()
    plt.plot(t_f, cov_f, label="frontier")
    plt.plot(t_r, cov_r, label="random")
    plt.xlabel("time (s)")
    plt.ylabel("coverage (%)")
    plt.title("Coverage vs time (mock)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_logs/week3_coverage.png")
    print("Saved eval_logs/week3_coverage.png")

    # --- Entropy plot ---
    plt.figure()
    plt.plot(t_f, ent_f, label="frontier")
    plt.plot(t_r, ent_r, label="random")
    plt.xlabel("time (s)")
    plt.ylabel("entropy (proxy)")
    plt.title("Entropy vs time (mock)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eval_logs/week3_entropy.png")
    print("Saved eval_logs/week3_entropy.png")


if __name__ == "__main__":
    main()
