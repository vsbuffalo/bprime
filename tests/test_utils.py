import numpy as np
from bgspy.utils import bin_chrom
from bgspy.utils import masks_to_ranges, combine_features


def test_combine_features():
    """
    """
    sl = {'chr1': 13}
    feature_ranges = {'introns': {'chr1':[(1, 3), (5, 8)]}, 'cds': {'chr1':[(2, 4), (8, 9)]}}
    #expected = {'chr1': array([0., 2., 1., 1., 0., 2., 2., 2., 1., 0., 0., 0., 0.])}
    priority = ('cds', 'introns')
    a = combine_features(feature_ranges, priority, sl)
    b = {'chr1': np.array([0., 2., 1., 1., 0., 2., 2., 2., 1., 0., 0., 0., 0.])}
    assert all(a['chr1'] == b['chr1'])
    c = {'chr1': [(1, 2, 'introns'),
                  (2, 4, 'cds'),
                  (5, 8, 'introns'),
                  (8, 9, 'cds')]}
    assert c['chr1'] == masks_to_ranges(a, priority)['chr1']

def test_bin_chrom():
    # 0, 1, 2, 3, |  4, 5, 6, 7, | 8, 9, 10, 11, | 12, 13
    # remember, not right inclusive...
    np.testing.assert_equal(bin_chrom(13, 4), np.array([0, 4, 8, 12, 13]))
    np.testing.assert_equal(bin_chrom(12, 4), np.array([0, 4, 8, 12]))
