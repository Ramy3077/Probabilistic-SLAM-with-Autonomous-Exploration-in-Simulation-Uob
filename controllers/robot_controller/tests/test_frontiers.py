import numpy as np
from explore.frontiers import detect_frontiers, UNKNOWN, FREE, OCCUPIED

def test_frontier_detection_simple():
    g = np.array([
      [OCCUPIED, OCCUPIED, OCCUPIED, OCCUPIED],
      [OCCUPIED, FREE    , UNKNOWN , OCCUPIED],
      [OCCUPIED, FREE    , FREE    , OCCUPIED],
      [OCCUPIED, OCCUPIED, OCCUPIED, OCCUPIED],
    ])
    F = set(detect_frontiers(g))
    assert (1,1) in F  # FREE next to UNKNOWN
    assert (2,1) not in F  # FREE but no UNKNOWN 4-neighbor
