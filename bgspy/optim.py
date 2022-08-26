import multiprocessing
from collections import Counter
import numpy as np
import tqdm
from tabulate import tabulate
from scipy.optimize import minimize

# nlopt has different error messages; we use this for scipy to
# but just force to 'success' (1) or 'failure' (-1)
NL_OPT_CODES = {1:'success', 2:'stopval reached', 3:'ftol reached',
                4:'xtol reached', 5:'max eval', 6:'maxtime reached', -1:'failure',
                -2:'invalid args', -3:'out of memory', -4:'forced stop'}


def extract_opt_info(x):
    is_scipy = hasattr(x, 'fun')
    if is_scipy:
        nll, res, success = x.fun, x.x, x.success
        success = -1 if not success else 1
    else:
        nll, res, success = x
        assert success in NL_OPT_CODES.keys()
    return nll, res, success

def array_all(x):
    return tuple([np.array(a) for a in x])

def run_optims(workerfunc, starts, ncores=50):
    """
    """
    nstarts = len(starts)
    ncores = ncores if ncores is not None else 1
    if ncores > 1:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(workerfunc, starts), total=nstarts))
    else:
        res = list(tqdm.tqdm(map(workerfunc, starts), total=nstarts))
    nlls, thetas, success = array_all(zip(*map(extract_opt_info, res)))
    return OptimResult(nlls, thetas, success)

def nlopt_isres_worker(start, func, nt, nf):
    "TODO LEFT HERE"
    opt = nlopt.opt(nlopt.GN_ISRES, nparams)
    #opt = nlopt.opt(nlopt.AUGLAG, nparams)
    #opt.set_local_optimizer(nlopt.LN_COBYLA)
    nll = negll_nlopt(Y, Bp, w)
    opt.set_min_objective(nll)
    hl, hu = inequality_constraint_functions(nt, nf)
    tols = np.repeat(1e-11, nf)
    opt.add_inequality_mconstraint(hl, tols)
    opt.add_inequality_mconstraint(hu, tols)
    ce = equality_constraint_function(nt, nf)
    opt.add_equality_mconstraint(ce, tols)
    lb, ub = bounds_simplex(nt, nf)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_xtol_rel(1e-3)
    #opt.set_xtol_abs(1e-6)
    #opt.set_stopval(923543002497)
    opt.set_maxeval(1000000)
    assert x.size == nparams
    mle = opt.optimize(x)
    nll = opt.last_optimum_value()
    success = opt.last_optimize_result()
    return nll, mle, success

def nlopt_worker():
    pass


class OptimResult:
    def __init__(self, nlls, thetas, success, starts=None):
        self.nlls = nlls
        self.thetas = thetas
        self.success = success
        self.starts = starts

    @property
    def stats(self):
        succ = Counter(self.success)
        return {NL_OPT_CODES[k]: n for k, n in succ.items()}

    @property
    def rank(self):
        assert self.nlls is not None
        assert self.thetas is not None
        assert self.success is not None
        idx = np.argsort(self.nlls)
        # remove non-successful terminations
        idx = idx[self.success[idx] >= 1]
        return idx

    @property
    def theta(self):
        return self.thetas[self.rank[0]]

    @property
    def nll(self):
        assert self.nlls is not None
        return self.nlls[self.rank[0]]

    @property
    def valid(self):
        return self.success[self.rank[0]] >= 1

    @property
    def frac_success(self):
        x = np.mean([v >= 1 for v in self.success])
        return x

    def __repr__(self):
        code = NL_OPT_CODES[self.success[self.rank[0]]]
        return ("OptimResult\n"
               f"  success: {self.valid} (termination code: {code})\n"
               f"  stats: {self.stats} (prop success: {np.round(self.frac_success, 2)*100}%)\n"
               f"  negative log-likelihood = {self.nll}\n"
               f"  theta = {self.theta}")

