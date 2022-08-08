import pickle
import numpy as np
from bgspy.likelihood import negll_numba, negll_c

def reparam_theta(theta, nt, nf):
    # the theta for the negll_numba func excludes
    # mutation rate but doesn't use the simplex, so we
    # adjust for this reparameterization
    mu = theta[1]
    W = theta[2:].reshape((nt-1, nf))
    W_first_row = 1 - W.sum(axis=0)
    # adjust theta to incorporate mu
    W = mu * np.vstack((W_first_row, W))
    alt_theta = np.empty((1 + nt*nf))
    alt_theta[0] = theta[0] # set pi
    alt_theta[1:] = W.flat # set the rest
    return alt_theta

def test_compare_C_to_numba():
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']

    nx, nw, nt, nf = B.shape
    theta = np.array([4.501e-02, 3.879e-08,
                      0.3369, 0.0996, 0.1454,
                      0.4887, 0.0482, 0.1766,
                      0.0168, 0.699 , 0.2347,
                      0.1154, 0.026 , 0.252 ,
                      0.0423, 0.1273, 0.1914])

    assert(theta.size == 2 + nf*nt)

    numba_results = negll_numba(theta, Y, B, w)
    c_results = negll_c(theta, Y, B, w)
    np.testing.assert_almost_equal(c_results, numba_results)

    # now let's crank through a few more noisy ones for additional checks
    for _ in range(20):
        theta_jitter = np.random.normal(0, 0.001) + theta
        theta_jitter[:2] = theta[:2]
        numba_results = negll_numba(theta_jitter, Y, B, w)
        c_results = negll_c(theta_jitter, Y, B, w)
        np.testing.assert_almost_equal(c_results, numba_results)



# def test_compare_C_to_numba_random():
#     with open('likelihood_test_data.pkl', 'rb') as f:
#         dat = pickle.load(f)
#         B, Y, w = dat['B'], dat['Y'], dat['w']

#     nx, nw, nt, nf = B.shape
#     np.random.seed(1)

#     for _ in range(20):
#         theta = random_start(nt, nf)
#         numba_results = negll_numba(theta, Y, B, w)
#         c_results = negll_c(theta, Y, B, w)
#         np.testing.assert_almost_equal(c_results, numba_results)



