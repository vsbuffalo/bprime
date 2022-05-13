import numpy as np
from bgspy.theory import bgs_segment


def test_bgs_segment():
    mu, s, L, rbp, rf = 1e-8, 0.1, 1000, 1e-8, 1e-9
    np.testing.assert_allclose(bgs_segment(mu, s, L, rbp, rf), 0.99980006)



