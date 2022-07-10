import numpy as np
from bgspy.likelihood import num_nonpoly, calc_loglik_components


def test_num_nonpoly():
    #
    neut_pos = {'chr1': np.array([ 0,  3,  5, 11])}
    bins = {'chr1': np.array([ 0,  4,  8, 12])}
    #                          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    #                                     |           |
    # poly sites:              x        x     x                  x
    masks = {'chr1': np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 1,  0, 1])}
    # number of poly sites            2   |     1     |      1
    # overlapping the neutral region masks
    # --
    # nfixed:                         1   |     1      |     2
    # (not poly and overlapping neutral)

    #res = {'chr1': np.array([1, 1, 2])}
    res = np.array([1, 1, 2])
    np.testing.assert_allclose(num_nonpoly(neut_pos, bins, masks)['chr1'], res)

#def test_calc_loglik_components():
