import numpy as np
from explore.frontiers import detect_frontiers, FREE, UNKNOWN, OCCUPIED
from explore.planner import choose_frontier
from eval.metrics import coverage_percent, entropy_proxy

# build a simple synthetic map
g = np.full((30, 30), UNKNOWN, dtype=int)
g[5:25, 5:25] = FREE
g[10:12, 10:20] = OCCUPIED  # obstacle band

F = detect_frontiers(g)
goal = choose_frontier(F, pose_ij=(15, 15))

print(
    f"frontiers: {len(F)}  goal: {goal}  "
    f"coverage%: {coverage_percent(g):.2f}  entropy: {entropy_proxy(g):.2f}"
)
