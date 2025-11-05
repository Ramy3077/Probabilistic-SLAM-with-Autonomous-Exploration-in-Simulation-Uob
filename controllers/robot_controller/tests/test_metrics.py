import numpy as np
from eval.metrics import coverage_percent, entropy_proxy

def test_metrics_sanity():
    g = np.array([[-1,-1,0,1],[0,1,1,-1]])
    assert 0 < coverage_percent(g) < 100
    assert 0 <= entropy_proxy(g) <= 1
