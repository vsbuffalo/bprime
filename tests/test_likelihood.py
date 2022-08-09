import pickle
import numpy as np
from bgspy.likelihood import negll_numba, negll_c, access, interp_logBw_c
from bgspy.likelihood import negll_mutation

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


def test_access_C():
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']
    nx, nw, nt, nf = B.shape
    for _ in range(10000):
        i = np.random.randint(0, nx)
        l = np.random.randint(0, nw)
        j = np.random.randint(0, nt)
        k = np.random.randint(0, nf)
        #print(B[i, l, j, k], access(B, i, l, j, k))
        assert B[i, l, j, k] == access(B, i, l, j, k)

def test_interpol_C():
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']
    nx, nw, nt, nf = B.shape
    for _ in range(100):
        i = int(np.random.randint(0, nx))
        l = int(np.random.randint(0, nw))
        j = int(np.random.randint(0, nt))
        k = int(np.random.randint(0, nf))
        x = 10**np.random.uniform(-11, -7)
        assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])

def test_interpol_C_bounds():
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']
    nx, nw, nt, nf = B.shape
    i, j, k = 100, 2, 0
    x = 1e-7
    assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])
    x = 1e-11
    assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])



def test_compare_C_to_numba():
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']

    nx, nw, nt, nf = B.shape
    theta = np.array([4.5010e-02, 3.7059e-09, 1.0956e-09, 1.5994e-09, 5.3757e-09,
                      5.3020e-10, 1.9426e-09, 1.8480e-10, 7.6890e-09, 2.5817e-09,
                      1.2694e-09, 2.8600e-10, 2.7720e-09, 4.6530e-10, 1.4003e-09,
                      2.1054e-09])
    assert(theta.size == 1 + nf*nt)

    numba_results = negll_numba(theta, Y, B, w)
    py_results = negll_mutation(theta, Y, B, w)
    c_results = negll_c(theta, Y, B, w)
    np.testing.assert_almost_equal(py_results, numba_results, decimal=1)
    # the sum in numba is a little unstable, but fine for quantities this large
    np.testing.assert_almost_equal(c_results, numba_results, decimal=0)

    # now let's crank through a few more noisy ones for additional checks
    for _ in range(20):
        theta_jitter = np.random.normal(0, 1e-10) + theta
        theta_jitter[:1] = theta[:1]
        numba_results = negll_numba(theta_jitter, Y, B, w)
        c_results = negll_c(theta_jitter, Y, B, w)
        np.testing.assert_almost_equal(c_results, numba_results)

