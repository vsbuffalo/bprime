## likelihood.py -- functions for likelihood stuff
import multiprocessing
import tqdm
from functools import partial
from collections import Counter
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy.optimize import minimize


def negloglik(Y, logB, w, log_params=False):
    """
    Returns negative log-likelihood function closure around
    data, Bs (stored in log-space), and the mutation weights
    w.

    The enclosed function is a function of the parameters
    θ = {π0, w_{0, 0}, ..., w_{nt, nf}} where nt is the
    selection grid size and nf is the number of features.
    """
    nS = Y[:, 0]
    nD = Y[:, 1]

    nx, nw, nt, nf = logB.shape

    def negll_logparams(log_theta):
        log_pi0, log_w_theta = log_theta[0], log_theta[1:]
        # interpolate B(w)'s
        logBw = np.zeros(nx, dtype=float)
        for i in range(nx):
            for j in range(nt):
                for k in range(nf):
                    logBw[i] += np.interp(np.exp(log_w_theta[j, k]), w, logB[i, :, j, k])
        log_pibar = np.exp(log_pi0 + logBw)
        llm = nD * log_pibar + nS * np.log1p(-np.exp(log_pibar))
        return -llm.sum()

    def negll(theta):
        pi0, w_theta = theta[0], theta[1:]
        # interpolate B(w)'s
        logBw = np.zeros(nx, dtype=float)
        for i in range(nx):
            for j in range(nt):
                for k in range(nf):
                    logBw[i] += np.interp(w_theta[j, k], w, logB[i, :, j, k])
        pibar = pi0 * np.exp(logBw)
        llm = nD * np.log(pibar) + nS * np.log1p(-pibar)
        return -llm.sum()

    if log_params:
        return negll_logparams
    return negll


def minimize_worker(start, bounds, func):
    res = minimize(func, start, bounds=bounds, options={'disp': False})
    return res

class InterpolatedMLE:
    """
    Y ~ π0 B(w)

    (note Bs are stored in log-space.)
    """
    def __init__(self, w, t, logB, log10_pi0_bounds):
        self.w = w
        self.t = t
        self.log10_pi0_bounds = log10_pi0_bounds
        try:
            assert logB.ndim == 4
            assert logB.shape[1] == w.size
            assert logB.shape[2] == t.size
        except AssertionError:
            raise AssertionError("logB has incorrection shape, should be nx x nw x nt x nf")

        self.logB = logB
        self.theta = None
        self.dim()

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

    def bounds(self, log=False):
        pi0_bounds = 10**self.log10_pi0_bounds[0], 10**self.log10_pi0_bounds[1]
        if log:
            # must be base e
            pi0_bounds = np.log(pi0_bounds[0]), np.log(pi0_bounds[1])
        bounds = [pi0_bounds]
        w = self.w
        for f in range(self.nf):
            w1, w2 = np.min(w), np.max(w)
            if log:
                w1, w1 = np.log(w1), np.log(w2)
            bounds.append((w1, w2))
        return bounds

    def random_start(self, log=False):
        start = []
        for bounds in self.bounds(log):
            if log:
                start.append(np.exp(np.random.uniform(*bounds, size=1)))
            else:
                start.append(np.random.uniform(*bounds, size=1))
        return np.array(start)

    def fit(self, Y, log_params=False, nruns=50, ncores=None,
            method='BFGS'):
        self.Y_ = Y
        nll = negloglik(Y, self.logB, self.w, log_params)
        bounds = self.bounds(log_params)

        starts = [self.random_start(log_params) for _ in range(nruns)]
        worker = partial(minimize_worker, bounds=bounds, func=nll)
        if ncores == 1 or ncores is None:
            res = list(tqdm.tqdm(p.imap(worker, starts), total=nruns))
        else:
            with multiprocessing.Pool(ncores) as p:
                res = list(tqdm.tqdm(p.imap(worker, starts), total=nruns))

        converged = [x.success for x in res]
        nconv = sum(converged)
        if all(converged):
            print(f"all {nconv}/{nruns} converged")
        else:
            nfailed = nruns-nconv
            print(f"WARNING: {nfailed}/{nruns} ({100*np.round(nfailed/nruns, 2)}%)")
        self.starts_ = starts
        self.nlls_ = [x.fun for x in res]
        self.thetas_ = [x.x for x in res]
        self.bounds_ = bounds
        self.res_ = res

        return self

    def __repr__(self):
        pass

