import pickle
import numpy as np
from bgspy.likelihood import negll, negll_c, access, interp_logBw_c
from bgspy.likelihood import random_start_mutation

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
    np.random.seed(0)
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
    np.random.seed(0)
    for _ in range(10000):
        i = int(np.random.randint(0, nx))
        j = int(np.random.randint(0, nt))
        k = int(np.random.randint(0, nf))
        x = 10**np.random.uniform(-11, -7, 1)
        assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])
        # now test bounds -- first lower
        i, j, k = 0, 0, 0
        assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])
        # now upper
        i, j, k = nx-1, nt-1, nf-1
        assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])
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
    # NOTE: we changed the behavior, so the lower bound so if it runs
    # past the lower bound, log(B) = 0 is returned.
    #assert interp_logBw_c(x, w, B, i, j, k) == np.interp(x, w, B[i, :, j, k])
    assert interp_logBw_c(x, w, B, i, j, k) == 0



def test_compare_C():
    """
    Compare the python/numba loglikelihood implementations to the C.
    """
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']

    nx, nw, nt, nf = B.shape
    theta = np.array([4.5010e-02, 1.0000e-08, 3.7059e-01, 1.0956e-01, 1.5994e-01,
       5.3757e-01, 5.3020e-02, 1.9426e-01, 1.8480e-02, 7.6890e-01,
       2.5817e-01, 1.2694e-01, 2.8600e-02, 2.7720e-01, 4.6530e-02,
       1.4003e-01, 2.1054e-01])
    assert(theta.size == 2 + nf*nt)

    py_results = negll(theta, Y, B, w)
    c_results = negll_c(theta, Y, B, w)
    # the sum in numba is a little unstable, but fine for quantities this large
    np.testing.assert_almost_equal(c_results, py_results, decimal=0)

    # now let's crank through a few more noisy ones for additional checks
    for _ in range(20):
        theta_jitter = np.random.normal(0, 1e-10) + theta
        theta_jitter[:1] = theta[:1]
        py_results = negll(theta_jitter, Y, B, w)
        c_results = negll_c(theta_jitter, Y, B, w)
        np.testing.assert_almost_equal(c_results, py_results, decimal=1)


def test_compare_C_random():
    """
    Compare the python loglikelihood implementations to the C.
    """
    with open('likelihood_test_data.pkl', 'rb') as f:
        dat = pickle.load(f)
        B, Y, w = dat['B'], dat['Y'], dat['w']

    nx, nw, nt, nf = B.shape

    for _ in range(100):
        theta = random_start_mutation(nt, nf)
        new_theta = np.zeros(nt*nf + 2)
        new_theta[0] = theta[0]
        new_theta[1] = 1.
        new_theta[2:] = theta[1:]
        py_results = negll(new_theta, Y, B, w)
        c_results = negll_c(new_theta, Y, B, w)
        np.testing.assert_almost_equal(c_results, py_results, decimal=1)


def test_new_likelihood_versus_old():
    """
    Compare the various C version of the *deprecated* linearly
    interpolated likelihood.
    """
    with open('likelihood_test_data.pkl', 'rb') as f:
       dat = pickle.load(f)
       B, Y, w = dat['B'], dat['Y'], dat['w']

    nx, nw, nt, nf = B.shape
    theta = np.array([4.5010e-02, 1.0000e-08, 3.7059e-01, 1.0956e-01, 1.5994e-01,
       5.3757e-01, 5.3020e-02, 1.9426e-01, 1.8480e-02, 7.6890e-01,
       2.5817e-01, 1.2694e-01, 2.8600e-02, 2.7720e-01, 4.6530e-02,
       1.4003e-01, 2.1054e-01])
    assert(theta.size == 2 + nf*nt)

    for _ in range(20):
        theta_jitter = np.random.normal(0, 1e-6) + theta
        theta_jitter[:1] = theta[:1]
        old_results = negll_c(theta_jitter, Y, B, w, version=1)
        new_results = negll_c(theta_jitter, Y, B, w, version=2)
        if not np.isnan(old_results):
            # we ignore different NAN behavior
            assert(old_results == new_results), theta_jitter

