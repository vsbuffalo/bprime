import numpy as np
import pickle
import ctypes
from ctypes import POINTER, c_double
from bgspy.likelihood import negll_numba

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

print("Is contiguous? ", B.flags['C_CONTIGUOUS'])
print("B dtype: ", B.dtype)
print("W=\n", theta[2:].reshape((nt-1, nf)))
print(theta[2:].reshape((nt-1, nf)).sum(axis=0))

nS, nD = np.array(Y[:, 0]), np.array(Y[:, 1])

lib.access.restype = c_double

B = np.require(B, float, ['ALIGNED'])

i, l, j, k = 0, 0, 0, 0
print(f"python: {i},{l},{j},{k}: ", B[i, l, j, k], end='\t//\t')
B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
shape, strides = B.ctypes.shape, B.ctypes.strides
print(f"C: {i},{l},{j},{k}: ", lib.access(B_ptr, i, l, j, k, shape, strides))

print('------')
print(w.strides)

# import sys
# sys.exit(0)
#print(B.flatten()[:50])

for _ in range(10):
    i = np.random.randint(0, nx)
    l = np.random.randint(0, nw)
    j = np.random.randint(0, nt)
    k = np.random.randint(0, nf)
    print(f"python: {i},{l},{j},{k}: ", B[i, l, j, k], end='\t//\t')
    s = B.strides
    print(f"python manual: {i},{l},{j},{k}: ",
          B.flat[(s[0]*i + s[1]*l + s[2]*j + s[3]*k) // B.dtype.itemsize], end='\t//\t')
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    shape, strides = B.ctypes.shape, B.ctypes.strides
    lib.access.argtypes = (POINTER(c_double), ctypes.c_ssize_t,  ctypes.c_ssize_t,
                           ctypes.c_ssize_t, ctypes.c_ssize_t, POINTER(ctypes.c_ssize_t),
                           POINTER(ctypes.c_ssize_t))
    print(f"C: {i},{l},{j},{k}: ", lib.access(B_ptr, i, l, j, k, shape, strides))

print("Y DTYPE", Y.dtype)

def cnegloglik(theta, Y, logB, w):
    nS = np.require(Y[:, 0].flat, np.float64, ['ALIGNED'])
    nD = np.require(Y[:, 1].flat, np.float64, ['ALIGNED'])
    theta = np.require(theta, np.float64, ['ALIGNED'])
    nS_ptr = nS.ctypes.data_as(POINTER(c_double))
    nD_ptr = nD.ctypes.data_as(POINTER(c_double))
    theta_ptr = theta.ctypes.data_as(POINTER(c_double))
    logB_ptr = logB.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    w_ptr = w.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    lib.negloglik.argtypes = (POINTER(c_double), POINTER(c_double), POINTER(c_double),
                             POINTER(c_double), POINTER(c_double), ctypes.POINTER(np.ctypeslib.c_intp), ctypes.POINTER(np.ctypeslib.c_intp))
    lib.negloglik.restype = c_double
    return lib.negloglik(theta_ptr, nS_ptr, nD_ptr, logB_ptr, w_ptr, logB.ctypes.shape, logB.ctypes.strides)

# compare
# adjust theta to incorporate mu
mu = theta[1]
W = theta[2:].reshape((nt-1, nf))
W_first_row = 1 - W.sum(axis=0)
W = mu * np.vstack((W_first_row, W))
print(W)

alt_theta = np.empty((1 + nt*nf))
alt_theta[0] = theta[0]
alt_theta[1:] = W.flat
print(alt_theta)
print("negll_numba: ", negll_numba(alt_theta, Y, B, w))
print("cnegloglik: ", cnegloglik(theta, Y, B, w))



