import numpy as np
import pickle
from tabulate import tabulate
from functools import partial
import ctypes
import matplotlib.pyplot as plt
import nlopt
from bgspy.likelihood import negll_c, negll, negll_numba, negll_fixmu_c, negll_fixmu_numba
from bgspy.likelihood import bounds, random_start, interp_logBw_c, access

dat = pickle.load(open('../../tests/likelihood_test_data.pkl', 'rb'))
B, Y, w = dat['B'], dat['Y'], dat['w']

nx, nw, nt, nf = B.shape
nparams = nt * nf + 2

test_theta = np.array([1.33128476e-04, 1.99781458e-09,
                       8.62855908e-03, 2.01632113e-01, 5.62221421e-01,
                       9.32838631e-03, 4.62330613e-01, 1.32512993e-01,
                       9.18201833e-02, 3.32180143e-02, 3.02959135e-03,
                       6.38781967e-01, 8.20186567e-02, 2.59546792e-01,
                       2.51440905e-01, 2.20800603e-01, 4.26892028e-02])
# test_theta = np.array([4.501e-02, 3.879e-08,
#                       0.3369, 0.0996, 0.1454,
#                       0.4887, 0.0482, 0.1766,
#                       0.0168, 0.699 , 0.2347,
#                       0.1154, 0.026 , 0.252 ,
#                       0.0423, 0.1273, 0.1914])



# numba_results = negll_numba(test_theta, Y, B, w)
# c_results = negll_c(test_theta, Y, B, w)
# np.testing.assert_almost_equal(c_results, numba_results)


x = 1.3e-8

# print("B[0]", B.flat[0])

np.random.seed(21)
# compare array accesss
for _ in range(1000):
    i = np.random.randint(0, nx)
    l = np.random.randint(0, nw)
    j = np.random.randint(0, nt)
    k = np.random.randint(0, nf)
    #print(B[i, l, j, k], access(B, i, l, j, k))
    assert B[i, l, j, k] == access(B, i, l, j, k)

print("PASSED ACCESSS")

print(f"test: ", access(B, 1061, 3, 4, 0))
for _ in range(100):
    i = int(np.random.randint(0, nx))
    l = int(np.random.randint(0, nw))
    j = int(np.random.randint(0, nt))
    k = int(np.random.randint(0, nf))
    #print(B[i, l, j, k], access(B, i, l, j, k))
    assert B[i, l, j, k] == access(B, i, l, j, k)

    print(f"{i,l,j,k} | c_interp: ", interp_logBw_c(x, w, B, i, j, k), " np: ",
          np.interp(x, w, B[i, :, j, k]))
    print(B[i, :, j, k])
    for ll in range(nw):
        print(access(B, i, ll, j, k))

    assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])
    # print("successs!")

