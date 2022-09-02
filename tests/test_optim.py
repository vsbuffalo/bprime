import numpy as np
from bgspy.optim import constraint_matrix
from bgspy.optim import inequality_constraint_functions


def test_constrain_matrix():
    nt, nf = 4, 2
    A = constraint_matrix(nt, nf)
    x = np.array([1e-3, 1e-8,
                  1, 2,
                  0, 1,
                  3, 4,
                  -1, 3])
    np.testing.assert_array_equal(A.dot(x), [3, 10])

