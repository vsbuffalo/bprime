## likelihood.py -- functions for likelihood stuff
import json
import multiprocessing
import tqdm
from functools import partial
from collections import Counter, defaultdict
import numpy as np
HAS_JAX = False
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    pass
from scipy.special import xlogy, xlog1py
from scipy.stats import binned_statistic
from scipy import interpolate
from scipy.optimize import minimize, dual_annealing
from numba import jit
from bgspy.utils import signif

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

def log1mexp(x):
    """
    log(1-exp(-|x|)) computed using the methods described in
    this paper: https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    I've also added another condition to prevent underflow.
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
    pi0, W = theta[0], theta[1:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(W[j, k], w, logB[i, :, j, k])
    #pibar = pi0 * np.exp(logBw)
    log_pibar = np.log(pi0) + logBw
    #llm = nD*log_pibar + xlog1py(nS, -np.exp(log_pibar))
    llm = nD*log_pibar + nS*log1mexp(log_pibar)
    return -np.sum(llm)

if HAS_JAX:
    def negll_jax(theta, Y, logB, w):
        nS = Y[:, 0]
        nD = Y[:, 1]
        nx, nw, nt, nf = logB.shape
        # mut weight params
        pi0, W = theta[0], theta[1:]
        W = W.reshape((nt, nf))
        # interpolate B(w)'s
        logBw = jnp.zeros(nx, dtype=jnp.float32)
        for i in range(nx):
            for j in range(nt):
                for k in range(nf):
                    logBw = logBw.at[i].add(jnp.interp(W[j, k], w, logB[i, :, j, k]))
        log_pibar = jnp.log(pi0) + logBw
        llm = nD*log_pibar + nS*jnp.log1p(-jnp.exp(log_pibar))
        return -jnp.sum(llm)

@jit(nopython=True)
def inverse_logit(x):
    return np.exp(x) / (1+np.exp(1))

def logit(x):
    return np.log(x / (1-x))

@jit(nopython=True)
def negll_numba_reparam(theta_reparam, Y, logB, w):
    theta = np.exp(theta_reparam) / (1 + np.exp(theta_reparam))
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=np.float64)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(W[j, k], w, logB[i, :, j, k])
    log_pibar = np.log(pi0) + logBw
    llm = nD*log_pibar + nS*np.log1p(-np.exp(log_pibar))
    return -np.sum(llm)


@jit(nopython=True)
def negll_numba(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=np.float64)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(W[j, k], w, logB[i, :, j, k])
    log_pibar = np.log(pi0) + logBw
    llm = nD*log_pibar + nS*np.log1p(-np.exp(log_pibar))
    return -np.sum(llm)


@jit(nopython=True)
def negll_numba_fixmu(theta, Y, logB, w, mu):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    W = W.reshape((nt-1, nf)) # simplex; one fewer parameter per column
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=np.float64)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                # mimix the simplex
                if j == 0:
                    # fixed class
                    Wjk = 1 - W[:, k].sum()
                else:
                    Wjk = W[j-1, k]
                logBw[i] += np.interp(mu*Wjk, w, logB[i, :, j, k])
    log_pibar = np.log(pi0) + logBw
    llm = nD*log_pibar + nS*np.log1p(-np.exp(log_pibar))
    return -np.sum(llm)

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
    if len(args) == 6:
        start, bounds, func, Y, logB, w = args
        func = partial(func, Y=Y, logB=logB, w=w)
    else:
        start, bounds, func, Y, logB, w, mu = args
        func = partial(func, Y=Y, logB=logB, w=w, mu=mu)
    res = minimize(func, start, bounds=bounds,
                   method='L-BFGS-B',
                   options={'disp': False})
    return res


def expand_W_simplex(w, nt, nf):
    W = w.reshape((nt-1, nf))
    return np.concatenate((1-W.sum(axis=0)[None, :], W), axis=0)

class BGSEstimator:
    """
    Y ~ π0 B(w)

    (note Bs are stored in log-space.)
    """
    def __init__(self, w, t, logB, log10_pi0_bounds=(-5, -1), seed=None):
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

    def bounds(self, paired=True, fix_mu=False):
        pi0_bounds = 10**self.log10_pi0_bounds[0], 10**self.log10_pi0_bounds[1]
        bounds = [pi0_bounds]
        w = self.w
        nt = self.nt
        if fix_mu:
            nt -= 1
        for t in range(nt):
            for f in range(self.nf):
                if fix_mu:
                    w1, w2 = (0, 1)
                else:
                    w1, w2 = np.min(w), np.max(w)
                bounds.append((w1, w2))
        if paired:
            return bounds
        return tuple(zip(*bounds))

    def random_start(self, fix_mu=False, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            assert self.rng is not None, "rng not set!"
            rng = self.rng
        start = []
        for bounds in self.bounds(fix_mu=fix_mu):
            start.append(rng.uniform(*bounds, size=1))
        return np.array(start)

    def fit(self, Y, nruns=50, seed=None, mu=None, use_numba=True,
            ncores=None, annealing=False):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.Y_ = Y
        fix_mu = mu is not None
        bounds = self.bounds(fix_mu=fix_mu)
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
                start = self.random_start(fix_mu=fix_mu, seed=seed)
                starts.append(start)
                if mu is not None:
                    assert use_numba
                    func = negll_numba_fixmu
                    args.append((start, bounds, func, Y, logB, w, mu))
                else:
                    func = negll if not use_numba else negll_numba
                    args.append((start, bounds, func, Y, logB, w))

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
        self.fix_mu_ = fix_mu

        return self

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


    @property
    def mle_pi0(self):
        return self.theta_[0]

    @property
    def mle_W(self):
        if not self.fix_mu_:
            return self.theta_[1:].reshape((self.nt, self.nf))
        return expand_W_simplex(self.theta_[1:], self.nt, self.nf)

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

    def to_npz(self, filename):
        np.savez(filename, logB=self.logB, w=self.w, t=self.t, Y=self.Y_,
                 bounds=self.bounds())

    def to_json(self, filename):
        Y = self.Y_.astype(int)
        N = Y.sum(axis=1)
        Nd = Y[:, 1].squeeze()
        # get rid of 0s
        idx = N > 0
        N = N[idx]
        Nd = Nd[idx]
        # simpler model:
        #logB = self.logB[idx, ...][..., 0][..., None]
        logB = self.logB[idx, ...]
        nx, nw, nt, nf = logB.shape
        with open(filename, 'w') as f:
            json.dump(dict(logBs=logB.tolist(),
                           w=self.w.tolist(),
                           nx=nx, nw=nw, nt=nt, nf=nf,
                           #Nd=Nd.tolist(), N=N.tolist(),
                           Y = Y[idx, ...].tolist()
                           ), f, indent=2)


    @classmethod
    def from_npz(self, filename):
        d = np.load(filename)
        bounds = d['bounds']
        obj = BGSEstimator(d['w'], d['t'], d['logB'],
                           (np.log10(bounds[0, 0]), np.log10(bounds[0, 1])))
        return obj

    def __repr__(self):
        rows = [f"MLE (interpolated w): {self.nw} x {self.nt} x {self.nf}"]
        rows.append(f"  w grid: {signif(self.w)} (before interpolation)")
        rows.append(f"  t grid: {signif(self.t)}")
        if self.theta_ is not None:
            rows.append(f"MLEs:\n  pi0 = {signif(self.mle_pi0)}\n  W = " +
                        str(signif(self.mle_W)).replace('\n', '\n   '))
            rows.append(f" negative log-likelihood: {self.nll_}")
        return "\n".join(rows)



