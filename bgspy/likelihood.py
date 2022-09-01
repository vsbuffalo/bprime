## likelihood.py -- functions for likelihood stuff
import os
import warnings
import json
import multiprocessing
import tqdm
from tabulate import tabulate
from functools import partial
from collections import Counter, defaultdict
import numpy as np
from ctypes import POINTER, c_double, c_ssize_t

# no longer needed
# HAS_JAX = False
# try:
#     import jax.numpy as jnp
#     HAS_JAX = True
# except ImportError:
#     pass

from scipy.special import xlogy, xlog1py
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy.optimize import minimize
from numba import jit
from bgspy.utils import signif
from bgspy.data import pi_from_pairwise_summaries
from bgspy.optim import run_optims, nlopt_mutation_worker

# load the library (relative to this file in src/)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
likclib = np.ctypeslib.load_library("likclib", LIBRARY_PATH)

#likclib = np.ctypeslib.load_library('lik', 'bgspy.src.__file__')

def access(B, i, l, j, k):
    """
    This is a function that tests uses the C function access()
    to grab elements of the multidimensional array B using a macro.
    This is primarily used in unit tests to ensure this is working properly.
    """
    nx, nw, nt, nf = B.shape
    B = np.require(B, np.float64, ['ALIGNED'])
    likclib.access.argtypes = (POINTER(c_double),
                                     c_ssize_t,
                                     c_ssize_t,
                                     c_ssize_t,
                                     c_ssize_t,
                                     POINTER(np.ctypeslib.c_intp))

    likclib.access.restype = c_double

    logB_ptr = B.ctypes.data_as(POINTER(c_double))
    return likclib.access(logB_ptr, i, l, j, k, B.ctypes.strides)


def bounds_mutation(nt, nf, log10_pi0_bounds=(-4, -2),
                    log10_mu_bounds=(-11, -7), paired=False):
    """
    Return the bounds on for optimization under the free mutation
    model. If paired=True, the bounds are zipped together for each
    parameter.
    """
    l = [10**log10_pi0_bounds[0]]
    u = [10**log10_pi0_bounds[1]]
    for i in range(nt):
        for j in range(nf):
            l += [10**log10_mu_bounds[0]]
            u += [10**log10_mu_bounds[1]]
    lb = np.array(l)
    ub = np.array(u)
    assert np.all(lb < ub)
    if paired:
        return list(zip(lb, ub))
    return lb, ub


def bounds_simplex(nt, nf, log10_pi0_bounds=(-4, -2),
           log10_mu_bounds=(-11, -7),
           paired=False):
    """
    Return the bounds on for optimization under the simplex model
    model. If paired=True, the bounds are zipped together for each
    parameter.
    """
    l = [10**log10_pi0_bounds[0]]
    u = [10**log10_pi0_bounds[1]]
    l += [10**log10_mu_bounds[0]]
    u += [10**log10_mu_bounds[1]]
    l += [0.]*nf*nt
    u += [1.]*nf*nt
    lb = np.array(l)
    ub = np.array(u)
    assert np.all(lb < ub)
    if paired:
        return list(zip(lb, ub))
    return lb, ub

def random_start_mutation(nt, nf,
                          log10_pi0_bounds=(-4, -3),
                          log10_mu_bounds=(-11, -7)):
    """
    Create a random start position log10 uniform over the bounds for π0
    and all the mutation parameters under the free mutation model.
    """
    pi0 = 10**np.random.uniform(log10_pi0_bounds[0], log10_pi0_bounds[1], 1)
    W = np.empty((nt, nf))
    for i in range(nt):
        for j in range(nf):
            W[i, j] = 10**np.random.uniform(log10_mu_bounds[0], log10_mu_bounds[1])
    theta = np.empty(nt*nf + 1)
    theta[0] = pi0
    theta[1:] = W.flat
    return theta

def random_start_simplex(nt, nf, log10_pi0_bounds=(-4, -3),
                         log10_mu_bounds=(-11, -7)):
    """
    Create a random start position log10 uniform over the bounds for π0
    and μ, and uniform under the DFE weights for W, under the simplex model.
    """
    pi0 = 10**np.random.uniform(log10_pi0_bounds[0], log10_pi0_bounds[1], 1)
    mu = 10**np.random.uniform(log10_mu_bounds[0], log10_mu_bounds[1], 1)
    W = np.empty((nt, nf))
    for i in range(nf):
        W[:, i] = np.random.dirichlet([1.] * nt)
        assert np.abs(W[:, i].sum() - 1.) < 1e-5
    theta = np.empty(nt*nf + 2)
    theta[0] = pi0
    theta[1] = mu
    theta[2:] = W.flat
    check_bounds(theta, *bounds_simplex(nt, nf, log10_pi0_bounds, log10_mu_bounds))
    return theta

def interp_logBw_c(x, w, B, i, j, k):
    """
    Linearly interpolate log(B) over the mutation parameter w using the C
    function. This is to test against the Python implementation.
    """
    nx, nw, nt, nf = B.shape
    B = np.require(B, np.float64, ['ALIGNED'])
    likclib.interp_logBw.argtypes = (c_double,           # x
                                     POINTER(c_double),  # *w
                                     POINTER(c_double),  # *logB
                                     c_ssize_t,          # nw
                                     c_ssize_t,          # i
                                     c_ssize_t,          # j
                                     c_ssize_t,          # k
                                     POINTER(np.ctypeslib.c_intp)) # *strides

    likclib.interp_logBw.restype = c_double

    return likclib.interp_logBw(x,
                                w.ctypes.data_as(POINTER(c_double)),
                                B.ctypes.data_as(POINTER(c_double)),
                                nw, i, j, k, B.ctypes.strides)

def R2(x, y):
    """
    Based on scipy.stats.linregress
    https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_mstats_common.py#L22-L209
    """
    complete_idx = ~(np.isnan(x) | np.isnan(y))
    x = x[complete_idx]
    y = y[complete_idx]
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=True).flat
    return ssxym / np.sqrt(ssxm * ssym)


def penalized_negll_c(theta, Y, logB, w, mu0, r):
    """
    A thin wrapper over negll_c() that imposes a penalty of the form:

     l*(θ) = l(θ | x) - r (μ - μ0)^2 / 2

    where r can be thought of as the precision (1/variance) -- this is
    essentially a Gaussian prior.
    """
    nll = negll_c(theta, Y, logB, w)
    mu = theta[1]
    return nll + r/2 * (mu - mu0)**2

def log1mexp(x):
    """
    log(1-exp(-|x|)) computed using the methods described in
    this paper: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    I've also added another condition to prevent underflow.

    NOTE: note in use currently outside negll() -- critical code in C.
    """
    assert isinstance(x, np.ndarray), "x must be a float-like np.ndarray"
    assert np.all(x < 0), "x must be < 0"
    min_negexp = np.finfo(x.dtype).minexp / np.log2(np.exp(1))
    a0 = np.log(2)
    out = np.zeros_like(x)
    #out = np.full_file(x, np.e)
    not_underflow = x > min_negexp
    abs_x = np.abs(x[not_underflow])
    out[not_underflow] = np.where(abs_x < a0, np.log(-np.expm1(-abs_x)), np.log1p(-np.exp(-abs_x)))
    return out

def negll(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, mu, W = theta[0], theta[1], theta[2:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(mu*W[j, k], w, logB[i, :, j, k])
    #pibar = pi0 * np.exp(logBw)
    log_pibar = np.log(pi0) + logBw
    #llm = nD*log_pibar + xlog1py(nS, -np.exp(log_pibar))
    llm = nD*log_pibar + nS*log1mexp(log_pibar)
    return -np.sum(llm)

## there's an imp DeprecationWarning this trigger
# if HAS_JAX:
#     def negll_jax(theta, Y, logB, w):
#         nS = Y[:, 0]
#         nD = Y[:, 1]
#         nx, nw, nt, nf = logB.shape
#         # mut weight params
#         pi0, W = theta[0], theta[1:]
#         W = W.reshape((nt, nf))
#         # interpolate B(w)'s
#         logBw = jnp.zeros(nx, dtype=jnp.float32)
#         for i in range(nx):
#             for j in range(nt):
#                 for k in range(nf):
#                     logBw = logBw.at[i].add(jnp.interp(W[j, k], w, logB[i, :, j, k]))
#         log_pibar = jnp.log(pi0) + logBw
#         llm = nD*log_pibar + nS*jnp.log1p(-jnp.exp(log_pibar))
#         return -jnp.sum(llm)

def negll_mutation(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    mu = 1.0
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    ll = 0.
    for i in range(nx):
        logBw_i = 0.
        for j in range(nt):
            for k in range(nf):
                logBw_i += np.interp(mu*W[j, k], w, logB[i, :, j, k])
        log_pibar = np.log(pi0) + logBw_i
        ll += nD[i]*log_pibar + nS[i]*np.log1p(-np.exp(log_pibar))
    return -ll

@jit(nopython=True)
def negll_numba(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, mu, W = theta[0], theta[1], theta[2:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    ll = 0.
    for i in range(nx):
        logBw_i = 0.
        for j in range(nt):
            for k in range(nf):
                logBw_i += np.interp(mu*W[j, k], w, logB[i, :, j, k])
        log_pibar = np.log(pi0) + logBw_i
        ll += nD[i]*log_pibar + nS[i]*np.log1p(-np.exp(log_pibar))
    return -ll

@jit(nopython=True)
def negll_mutation_numba(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    mu = 1.
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    ll = 0.
    for i in range(nx):
        logBw_i = 0.
        for j in range(nt):
            for k in range(nf):
                logBw_i += np.interp(mu*W[j, k], w, logB[i, :, j, k])
        log_pibar = np.log(pi0) + logBw_i
        ll += nD[i]*log_pibar + nS[i]*np.log1p(-np.exp(log_pibar))
    return -ll

def check_bounds(x, lb, ub):
    assert np.all((x >= lb) & (x <= ub))

def negll_c(theta, Y, logB, w):
    """
    θ is [π0, μ, w11, w12, ...] and should
    have dimension (nt x nf) + 2
    """
    nx, nw, nt, nf = logB.shape
    nS = np.require(Y[:, 0].flat, np.float64, ['ALIGNED'])
    nD = np.require(Y[:, 1].flat, np.float64, ['ALIGNED'])
    assert nS.shape[0] == nx
    assert theta.size == (nt * nf) + 2
    theta = np.require(theta, np.float64, ['ALIGNED'])
    logB = np.require(logB, np.float64, ['ALIGNED'])
    nS_ptr = nS.ctypes.data_as(POINTER(c_double))
    nD_ptr = nD.ctypes.data_as(POINTER(c_double))
    theta_ptr = theta.ctypes.data_as(POINTER(c_double))
    logB_ptr = logB.ctypes.data_as(POINTER(c_double))
    w_ptr = w.ctypes.data_as(POINTER(c_double))
    likclib.negloglik.argtypes = (POINTER(c_double), POINTER(c_double),
                              POINTER(c_double), POINTER(c_double),
                              POINTER(c_double),
                              # weird type for dims/strides
                              POINTER(np.ctypeslib.c_intp),
                              POINTER(np.ctypeslib.c_intp))
    likclib.negloglik.restype = c_double
    return likclib.negloglik(theta_ptr, nS_ptr, nD_ptr, logB_ptr, w_ptr,
                             logB.ctypes.shape, logB.ctypes.strides)

def normal_ll_c(theta, Y, logB, w):
    nS = np.require(Y[:, 0].flat, np.float64, ['ALIGNED'])
    nD = np.require(Y[:, 1].flat, np.float64, ['ALIGNED'])
    theta = np.require(theta, np.float64, ['ALIGNED'])
    logB = np.require(logB, np.float64, ['ALIGNED'])
    nS_ptr = nS.ctypes.data_as(POINTER(c_double))
    nD_ptr = nD.ctypes.data_as(POINTER(c_double))
    theta_ptr = theta.ctypes.data_as(POINTER(c_double))
    logB_ptr = logB.ctypes.data_as(POINTER(c_double))
    w_ptr = w.ctypes.data_as(POINTER(c_double))
    likclib.negloglik.argtypes = (POINTER(c_double), POINTER(c_double),
                              POINTER(c_double), POINTER(c_double),
                              POINTER(c_double),
                              # weird type for dims/strides
                              POINTER(np.ctypeslib.c_intp),
                              POINTER(np.ctypeslib.c_intp))
    likclib.normal_loglik.restype = c_double
    return likclib.normal_loglik(theta_ptr, nS_ptr, nD_ptr, logB_ptr, w_ptr,
                                 logB.ctypes.shape, logB.ctypes.strides)

def constraint_matrix(nt, nf):
    nparams = nt*nf + 2
    A = np.zeros((nf, nparams))
    for i in range(nf):
        W = A[i, 2:].reshape((nt, nf))
        W[:, i] = 1.
    return A

def inequality_constraint_functions(nt, nf, log10_mu_bounds=(-11, -7)):
    """
    Inequality constraints for nlopt.
     l < μW < u
     l - μW < 0
     μW - u < 0
    """
    A = constraint_matrix(nt, nf)
    lower, upper = 10**log10_mu_bounds[0], 10**log10_mu_bounds[1]
    def func_l(result, x, grad):
        mu = x[1]
        M = lower - (mu *  A.dot(x))
        for i in range(nf):
            result[i] = M[i]
    def func_u(result, x, grad):
        mu = x[1]
        M = (mu *  A.dot(x)) - upper
        for i in range(nf):
            result[i] = M[i]
    return func_l, func_u

def equality_constraint_function(nt, nf):
    """
    Equality constraints for nlopt.
    """
    A = constraint_matrix(nt, nf)
    def func(result, x, grad):
        M = A.dot(x)
        for i in range(nf):
            result[i] = M[i] - 1.
    return func

def predict_simplex(theta, logB, w):
    """
    """
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, mu, W = theta[0], theta[1], theta[2:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(mu*W[j, k], w, logB[i, :, j, k])
    return pi0*np.exp(logBw)

def predict_freemutation(theta, logB, w):
    """
    """
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(W[j, k], w, logB[i, :, j, k])
    return pi0*np.exp(logBw)

class BGSLikelihood:
    """
    Y ~ π0 B(w)

    (note Bs are stored in log-space.)
    """
    def __init__(self, w, t, logB, Y=None,
                 log10_pi0_bounds=(-5, -1),
                 seed=None):
        self.w = w
        self.t = t
        self.log10_pi0_bounds = log10_pi0_bounds
        self.log10_mu_bounds = np.log10(w[0]), np.log10(w[-1])
        try:
            assert logB.ndim == 4
            assert logB.shape[1] == w.size
            assert logB.shape[2] == t.size
        except AssertionError:
            raise AssertionError("logB has incorrection shape, should be nx x nw x nt x nf")

        if Y is not None:
            try:
                logB.shape[0] == Y.shape[0]
                Y.shape[1] == 2
            except AssertionError:
                raise AssertionError("Y shape is incorrect")
        self.Y = Y

        self.logB = logB
        self.theta_ = None
        self.dim()
        self.rng = np.random.default_rng(seed)


    def dim(self):
        """
        Dimensions are nx x nw x nt x nf.
        """
        return self.logB.shape[1:]

    @property
    def nw(self):
        return self.dim()[0]

    @property
    def nt(self):
        return self.dim()[1]

    @property
    def nf(self):
        return self.dim()[2]

    def _load_optim(self, optim_res):
        """
        Taken an OptimResult() object.
        """
        self.optim = optim_res
        # load in the best values
        self.theta_ = optim_res.theta
        self.nll_ = optim_res.nll

    def save_runs(self, filename):
        success = np.array([x.success for x in self.res_], dtype=bool)
        np.savez(filename, starts=self.starts_, nlls=self.nlls_,
                 thetas=self.thetas_, success=success)

    def profile_likelihood(self, Y, theta_fixed, nmesh, bounds=None):
        """
        """
        if bounds is None:
            bounds = self.bounds()
        is_free = np.isnan(theta_fixed)
        assert 1 <= sum(is_free) <= 2, "1 ≤ x ≤ 2 parameters must be set to np.nan (free)"
        grid = []
        free_idx = []
        fixed_idx = []
        if isinstance(nmesh, int):
            nmesh = [nmesh if free else None for free in is_free]
        for i, bounds in enumerate(bounds):
            n = nmesh[i]
            if is_free[i]:
                grid.append(10**np.linspace(*np.log10(bounds), n))
                free_idx.append(i)
            else:
                assert n is None
                grid.append([theta_fixed[i]])
                fixed_idx.append(i)
        mgrid = np.meshgrid(*grid)
        thetas = np.stack([x.flatten() for x in mgrid]).T
        n = thetas.shape[0]
        nlls = np.empty(n, dtype=float)
        for i in range(n):
            nlls[i] = negll(thetas[i, ...], Y, self.logB, self.w)
        return grid, thetas, nlls

    def predict(self, optim=None, theta=None):
        """
        Predicted π from the best fit (if optim = None). If optim is 'random', a
        random MLE optimization is chosen (e.g. to get a senes of how much
        variation there is across optimization). If optim is an integer,
        this rank of optimization results is given (e.g. optim = 0 is the
        best MLE).
        """
        if theta is not None:
            return predict_freemutation(theta, self.logB, self.w)
        if optim is None:
            theta = self.theta_
        else:
            thetas = self.optim.thetas_
            if optim == 'random':
                theta = thetas[np.random.randint(0, thetas.shape[0]), :]
            else:
                theta = thetas[optim]
        return predict_freemutation(theta, self.logB, self.w)

    def R2(self):
        """
        The R² value of the predictions against actual results.
        """
        pred_pi = self.predict()
        pi = pi_from_pairwise_summaries(self.Y)
        return R2(pred_pi, pi)

    def resid(self):
        pred_pi = self.predict()
        pi = pi_from_pairwise_summaries(self.Y)
        return (pred_pi - pi)**2

    @property
    def mle_pi0(self):
        return self.theta_[0]

    def to_npz(self, filename):
        np.savez(filename, logB=self.logB, w=self.w, t=self.t, Y=self.Y_,
                 bounds=self.bounds())

    @classmethod
    def from_npz(self, filename):
        d = np.load(filename)
        bounds = d['bounds']
        obj = BGSLikelihood(d['w'], d['t'], d['logB'],
                           (np.log10(bounds[0, 0]), np.log10(bounds[0, 1])))
        return obj

    def __repr__(self):
        rows = [f"MLE (interpolated w): {self.nw} x {self.nt} x {self.nf}"]
        rows.append(f"  w grid: {signif(self.w)} (before interpolation)")
        rows.append(f"  t grid: {signif(self.t)}")
        return "\n".join(rows)

def negll_freemut(Y, B, w):
    """
    This is a closure around data; returns a negative log- likelihood
    function around the data and a few fixed parameters (B and w).

    The core part is negll_c().
    """
    def func(theta):
        new_theta = np.full(theta.size + 1, np.nan)
        theta = np.copy(theta)
        new_theta[0] = theta[0]
        # fix mutation rate to one and let W represent mutation rates to various classes
        new_theta[1] = 1.
        new_theta[2:] = theta[1:] # times mutation rates
        #print("-->", theta, new_theta)
        return negll_c(new_theta, Y, B, w)
    return func

def negll_freemut_full(theta, grad, Y, B, w):
    """
    Like negll_freemut() but not a closure.

    grad is for nlopt and should be set to None if SciPy is being used.
    """
    new_theta = np.full(theta.size + 1, np.nan)
    new_theta[0] = theta[0]
    # fix mutation rate to one and let W represent mutation rates to various classes
    new_theta[1] = 1.
    new_theta[2:] = theta[1:] # times mutation rates
    #print("-->", theta, new_theta)
    return negll_c(new_theta, Y, B, w)

def negll_simplex_full(theta, grad, Y, B, w):
    """
    Simplex model wrapper for negll_c().

    grad is required for nlopt.
    """
    return negll_c(theta, Y, B, w)

class FreeMutationModel(BGSLikelihood):
    def __init__(self, w, t, logB, Y=None, log10_pi0_bounds=(-5, -1), seed=None):
        super().__init__(w=w, t=t, logB=logB, Y=Y,
                         log10_pi0_bounds=log10_pi0_bounds,
                         seed=seed)

    def random_start(self):
        """
        Random starts
        """
        return random_start_mutation(self.nt, self.nf,
                                     self.log10_pi0_bounds,
                                     self.log10_mu_bounds)

    def bounds(self, paired=False):
        return bounds_mutation(self.nt, self.nf,
                               self.log10_pi0_bounds,
                               self.log10_mu_bounds, paired=paired)

    def fit(self, starts=1, ncores=None, algo='ISRES'):
        """
        Fit likelihood models with mumeric optimization (either scipy or nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        engine: either 'scipy' or 'nlopt'.
        """
        algo = algo.upper()
        algos = {'L-BFGS-B':'scipy', 'ISRES':'nlopt',
                 'NELDERMEAD':'nlopt', 'NEWUOA':'nlopt'}
        assert algo in algos, f"algo must be in {algos}"
        engine = algos[algo]

        if isinstance(starts, int):
            starts = [self.random_start() for _ in range(starts)]
        ncores = min(len(starts), ncores) # don't request more cores than we need

        if engine == 'scipy':
            nll = partial(negll_freemut_full, grad=None, Y=self.Y,
                          B=self.logB, w=self.w)
            worker = partial(minimize, nll, bounds=self.bounds(paired=True),
                             method=algo, options={'eps':1e-9})
        elif engine == 'nlopt':
            nll = partial(negll_freemut_full, Y=self.Y,
                          B=self.logB, w=self.w)
            worker = partial(nlopt_mutation_worker,
                             func=nll, nt=self.nt, nf=self.nf,
                             bounds=self.bounds(), algo=algo)
        else:
            raise ValueError("engine must be 'scipy' or 'nlopt'")
        res = run_optims(worker, starts, ncores=ncores)
        self._load_optim(res)

    @property
    def mle_W(self):
        """
        Extract out the W matrix.
        """
        return self.theta_[1:].reshape((self.nt, self.nf))

    @property
    def nll(self):
        return self.nll_

    def __repr__(self):
        base_rows = super().__repr__()
        if self.theta_ is not None:
            base_rows += "\n\nFree-mutation model ML estimates:\n"
            base_rows += f"negative log-likelihood: {self.nll_}\n"
            base_rows += f"π0 = {self.mle_pi0}\n"
            W = self.mle_W.reshape((self.nt, self.nf))
            base_rows += "W = \n" + tabulate(W / W.sum(axis=0)) + "\n"
            base_rows += "μ = \n" + tabulate(W.sum(axis=0)[None, :])
        return base_rows

class SimplexModel(BGSLikelihood):
    def __init__(self, w, t, logB, Y=None, log10_pi0_bounds=(-5, -1), seed=None):
        super().__init__(w=w, t=t, logB=logB, Y=Y,
                         log10_pi0_bounds=log10_pi0_bounds,
                         seed=seed)

    def random_start(self):
        """
        Random starts
        """
        return random_start_simplex(self.nt, self.nf,
                                    self.log10_pi0_bounds,
                                    self.log10_mu_bounds)

    def bounds(self):
        return bounds_simplex(self.nt, self.nf,
                              self.log10_pi0_bounds,
                              self.log10_mu_bounds, paired=False)

    def fit(self, starts=1, ncores=None, algo='ISRES'):
        """
        Fit likelihood models with mumeric optimization (using nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        algo: the optimization algorithm to use.
        """
        algo = algo.upper()
        algos = {'L-BFGS-B':'scipy', 'ISRES':'nlopt',
                 'NELDERMEAD':'nlopt', 'NEWUOA':'nlopt'}
        assert algo in algos, f"algo must be in {algos}"
        engine = algos[algo]

        if isinstance(starts, int):
            starts = [self.random_start() for _ in range(starts)]
        # don't request more cores than we need
        ncores = min(len(starts), ncores)

        nll = partial(negll_simplex_full, Y=self.Y,
                      B=self.logB, w=self.w)
        worker = partial(nlopt_mutation_worker,
                         func=nll, nt=self.nt, nf=self.nf,
                         bounds=self.bounds(), algo=algo)

        res = run_optims(worker, starts, ncores=ncores)
        self._load_optim(res)

    @property
    def mle_W(self):
        """
        Extract out the W matrix.
        """
        return self.theta_[1:]

    def __repr__(self):
        base_rows = super().__repr__()
        if self.theta_ is not None:
            base_rows += "\n\nFree-mutation model ML estimates:\n"
            base_rows += f"π0 = {self.mle_pi0}\n"
            base_rows += "W = \n" + tabulate(self.mle_W.reshape((self.nt, self.nf)))
        return base_rows


    def __init__(self, *args, **kwargs):
        super(LikelihoodSimplex).__init__(*args, **kwargs)

    def fit(self):
        pass


