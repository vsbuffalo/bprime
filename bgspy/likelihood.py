## likelihood.py -- functions for likelihood stuff
import multiprocessing
import tqdm
from functools import partial
from collections import Counter
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy.optimize import minimize
from bgspy.utils import signif

def negll_logparams(log_theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    log_pi0, log_W = log_theta[0], log_theta[1:]
    log_W = log_W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(np.exp(log_W[j, k]), w, logB[i, :, j, k])
    log_pibar = np.exp(log_pi0 + logBw)
    llm = nD * log_pibar + nS * np.log1p(-np.exp(log_pibar))
    return -llm.sum()


def negll(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
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
    pibar = pi0 * np.exp(logBw)
    llm = nD * np.log(pibar) + nS * np.log1p(-pibar)
    return -llm.sum()


def minimize_worker(args):
    start, bounds, func, Y, logB, w = args
    func = partial(func, Y=Y, logB=logB, w=w)
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
        self.theta_ = None
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
        for t in range(self.nt):
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
        bounds = self.bounds(log_params)
        logB = self.logB
        w = self.w
        func = {True: negll_logparams, False: negll}[log_params]
        args = []
        starts = []
        for _ in range(nruns):
            start = self.random_start(log_params)
            starts.append(start)
            args.append((start, bounds, func, Y, logB, w))
        if ncores == 1 or ncores is None:
            res = list(tqdm.tqdm(p.imap(minimize_worker, args), total=nruns))
        else:
            with multiprocessing.Pool(ncores) as p:
                res = list(tqdm.tqdm(p.imap(minimize_worker, args), total=nruns))

        converged = [x.success for x in res]
        nconv = sum(converged)
        if all(converged):
            print(f"all {nconv}/{nruns} converged")
        else:
            nfailed = nruns-nconv
            print(f"WARNING: {nfailed}/{nruns} ({100*np.round(nfailed/nruns, 2)}%)")
        self.starts_ = starts
        self.nlls_ = np.array([x.fun for x in res])
        self.thetas_ = np.array([x.x for x in res])
        self.bounds_ = bounds
        self.res_ = res
        self.theta_ = self.thetas_[np.argmin(self.nlls_), :]
        self.nll_ = self.nlls_[np.argmin(self.nlls_), :]

        return self

    @property
    def mle_pi0(self):
        return self.theta_[0]

    @property
    def mle_W(self):
        return self.theta_[1:].reshape((self.nt, self.nf))

    def __repr__(self):
        rows = [f"MLE (interpolated w): {self.nw} x {self.nt} x {self.nf}"]
        rows.append(f"  w grid: {signif(self.w)} (before interpolation)")
        rows.append(f"  t grid: {signif(self.w)}")
        if self.theta_ is not None:
            rows.append(f"MLEs:\n  pi0 = {signif(self.mle_pi0)}\n  W = " +
                        str(signif(self.mle_W)).replace('\n', '\n   '))
            rows.append(f" negative log-likelihood: {signif(self.nll_)}")
        return "\n".join(rows)



