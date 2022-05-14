import numpy as np
from bgspy.classic import B_segment_lazy
from bgspy.theory import bgs_segment


def test_segment():
    "Test the lazy segment model against bgs_segment"
    n = 100
    rbp = 10**np.random.uniform(-8, -7, n)
    rf = 10**np.random.uniform(-8, -1, n)
    s = 10**np.random.uniform(-4, -1, n)
    mu = 1e-8
    L = np.random.randint(1, 1000, n)
    a, b, c, d = B_segment_lazy(rbp, L, s)
    lazy_B = np.exp(mu*a/(b*rf**2 + c*rf + d))
    seg_B = bgs_segment(mu, s, L, rbp, rf)
    np.testing.assert_allclose(lazy_B, seg_B)
