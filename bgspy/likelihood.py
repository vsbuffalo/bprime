## likelihood.py -- functions for likelihood stuff
import multiprocessing
import tqdm
from functools import partial
from collections import Counter, defaultdict
import numpy as np
from scipy.special import xlogy, xlog1py
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy.optimize import minimize, dual_annealing
from bgspy.utils import signif

def penalized_negll(theta, Y, logB, w, penalty=2):
    """
    Experimental
    """
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
    return -(llm.sum() - penalty*np.log(W.sum()**2))

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
    llm = xlogy(nD, pibar) + xlog1py(nS, -pibar)
    return -llm.sum()

def predict(theta, logB, w):
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


def minimize_worker(args):
    start, bounds, func, Y, logB, w = args
    func = partial(func, Y=Y, logB=logB, w=w)
    res = minimize(func, start, bounds=bounds,
                   #method='L-BFGS-B',
                   method='BFGS',
                   options={'disp': False})
    return res

class InterpolatedMLE:
    """
    Y ~ π0 B(w)

    (note Bs are stored in log-space.)
    """
    def __init__(self, w, t, logB, log10_pi0_bounds=(-5, -1)):
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

    def bounds(self):
        pi0_bounds = 10**self.log10_pi0_bounds[0], 10**self.log10_pi0_bounds[1]
        bounds = [pi0_bounds]
        w = self.w
        for t in range(self.nt):
            for f in range(self.nf):
                w1, w2 = np.min(w), np.max(w)
                bounds.append((w1, w2))
        return bounds

    def random_start(self):
        start = []
        for bounds in self.bounds():
            start.append(np.random.uniform(*bounds, size=1))
        return np.array(start)

    def fit(self, Y, nruns=50, ncores=None, annealing=False):
        self.Y_ = Y
        bounds = self.bounds()
        logB = self.logB
        w = self.w

        if annealing:
            if nruns > 1:
                warnings.warn("only one run supported for annealing!")
            func = partial(negll, Y=Y, logB=logB, w=w)
            res = dual_annealing(func, bounds)
        else:
            args = []
            starts = []
            for _ in range(nruns):
                start = self.random_start()
                starts.append(start)
                args.append((start, bounds, negll, Y, logB, w))

            if ncores == 1 or ncores is None:
                res = [minimize_worker(a) for a in tqdm.tqdm(args, total=nruns)]
            else:
                with multiprocessing.Pool(ncores) as p:
                    res = list(tqdm.tqdm(p.imap(minimize_worker, args), total=nruns))

        converged = [x.success for x in res]
        nconv = sum(converged)
        if all(converged):
            print(f"all {nconv}/{nruns} converged")
        else:
            nfailed = nruns-nconv
            print(f"WARNING: {nfailed}/{nruns} ({100*np.round(nfailed/nruns, 2)}%) optimizations failed!")
        self.starts_ = starts
        self.nlls_ = np.array([x.fun for x in res])
        self.thetas_ = np.array([x.x for x in res])
        self.bounds_ = bounds
        self.res_ = res
        self.theta_ = self.thetas_[np.argmin(self.nlls_), :]
        self.nll_ = self.nlls_[np.argmin(self.nlls_)]

        return self

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


    @property
    def mle_pi0(self):
        return self.theta_[0]

    @property
    def mle_W(self):
        return self.theta_[1:].reshape((self.nt, self.nf))

    def summary(self):
        wgrid = defaultdict(dict)
        for i in range(self.nf):
            for j in range(self.nt):
                wgrid[i][j] = float(signif(self.mle_W[j, i]))
        print(f"π0 = {self.mle_pi0}")
        for i in wgrid:
            print(f"feature {i}:")
            for j in wgrid[i]:
                sel = signif(self.t[j])
                print(f"  w({sel}) = {wgrid[i][j]}")

    def predict(self, logB=None, all=False):
        assert self.theta_ is not None, "InterpolatedMLE.theta_ is not set, call fit() first"
        logB = self.logB if logB is None else logB
        if not all:
            return predict(self.theta_, logB, self.w)
        n = self.thetas_.shape[0]
        pis = [predict(self.thetas_[i, :], logB, self.w) for i in range(n)]
        return np.stack(pis)

    def __repr__(self):
        rows = [f"MLE (interpolated w): {self.nw} x {self.nt} x {self.nf}"]
        rows.append(f"  w grid: {signif(self.w)} (before interpolation)")
        rows.append(f"  t grid: {signif(self.t)}")
        if self.theta_ is not None:
            rows.append(f"MLEs:\n  pi0 = {signif(self.mle_pi0)}\n  W = " +
                        str(signif(self.mle_W)).replace('\n', '\n   '))
            rows.append(f" negative log-likelihood: {self.nll_}")
        return "\n".join(rows)



