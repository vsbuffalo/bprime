import numpy as np
from bgspy.utils import bin_chrom

def test_bin_chrom():
    # 0, 1, 2, 3, |  4, 5, 6, 7, | 8, 9, 10, 11, | 12, 13
    # remember, not right inclusive...
    np.testing.assert_equal(bin_chrom(13, 4), np.array([0, 4, 8, 12, 13]))
    np.testing.assert_equal(bin_chrom(12, 4), np.array([0, 4, 8, 12]))
