import numpy as np
import pickle
import ctypes
from ctypes import POINTER, c_double

lib = np.ctypeslib.load_library('lik', '../src/')
dat = pickle.load(open('../data/test_data.pkl', 'rb'))

B, Y, w = dat['B'], dat['Y'], dat['w']

nx, nw, nt, nf = B.shape

theta = np.array([1e-3, 1e-8, #0.8, 0.8, 0.1,
                         0.2,  0.1,  0.1,
                         0.01, 0.01, 0.1,
                         0.01, 0.04, 0.03,
                         0.3,  0.01, 0.03], dtype=float)

assert(theta.size == 2 + (nt-1)*nf)

print("W=\n", theta[2:].reshape((nt-1, nf)))
print(theta[2:].reshape((nt-1, nf)).sum(axis=0))

nS, nD = np.array(Y[:, 0]), np.array(Y[:, 1])

def negloglik(theta, Y, logB, w):
    nS = np.require(Y[:, 0], int, ['ALIGNED'])
    nD = np.require(Y[:, 1], int, ['ALIGNED'])
    theta = np.require(theta, float, ['ALIGNED'])
    nS_ptr = nS.ctypes.data_as(POINTER(c_double))
    nD_ptr = nD.ctypes.data_as(POINTER(c_double))
    theta_ptr = theta.ctypes.data_as(POINTER(c_double))
    logB_ptr = logB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    w_ptr = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.negloglik.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double),
                             POINTER(c_double), POINTER(c_double), ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp))
    return lib.negloglik(theta_ptr, nS_ptr, nD_ptr, logB_ptr, w_ptr, logB.ctypes.shape, logB.ctypes.strides)

negloglik(theta, Y, B, w)
