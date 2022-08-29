import multiprocessing
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
from tabulate import tabulate
import nlopt
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
    ncores = min(nstarts, ncores)
    if ncores > 1:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(workerfunc, starts), total=nstarts))
    else:
        res = list(tqdm.tqdm(map(workerfunc, starts), total=nstarts))
    nlls, thetas, success = array_all(zip(*map(extract_opt_info, res)))
    return OptimResult(nlls, thetas, success, np.array(starts))


def nlopt_mutation_worker(start, func, nt, nf, bounds,
                          xtol_rel=1e-3, maxeval=1000000, algo='ISRES'):
    """
    Use nlopt to do bounded optimization for the free-mutation
    model.
    """
    nparams = nt * nf + 1
    if algo == 'ISRES':
        nlopt_algo = nlopt.GN_ISRES
    elif algo == 'NEWUOA':
        nlopt_algo = nlopt.LN_NEWUOA_BOUND
    elif algo == 'NELDERMEAD':
        nlopt_algo = nlopt.LN_NELDERMEAD
    else:
        raise ValueError("algo must be 'isres' or 'cobyla'")

    opt = nlopt.opt(nlopt_algo, nparams)
    opt.set_min_objective(func)
    lb, ub = bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_xtol_rel(xtol_rel)
    opt.set_maxeval(maxeval)
    assert start.size == nparams
    mle = opt.optimize(start)
    nll = opt.last_optimum_value()
    success = opt.last_optimize_result()
    return nll, mle, success


def nlopt_simplex_isres_worker(start, func, nt, nf, bounds,
                               constraint_tol=1e-11, xtol_rel=1e-3,
                               maxeval=1000000):
    """
    Use nlopt to do constrained (inequality to bound DFE weights
    and equality to enforce the simplex) and bounded optimization
    for the simplex model.
    """
    opt = nlopt.opt(nlopt.GN_ISRES, nparams)
    #opt = nlopt.opt(nlopt.LN_COBYLA, nparams)
    opt.set_min_objective(func)
    hl, hu = inequality_constraint_functions(nt, nf)
    tols = np.repeat(constraint_tol, nf)
    opt.add_inequality_mconstraint(hl, tols)
    opt.add_inequality_mconstraint(hu, tols)
    ce = equality_constraint_function(nt, nf)
    opt.add_equality_mconstraint(ce, tols)
    lb, ub = bounds_simplex(nt, nf)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_xtol_rel(xtol_rel)
    opt.set_maxeval(maxeval)
    assert start.size == nparams
    mle = opt.optimize(start)
    nll = opt.last_optimum_value()
    success = opt.last_optimize_result()
    return nll, mle, success

def optim_plot(only_success=True, tail=0.5, x_percent=False, downsample=None, **runs):
    """
    Make a plot of the rank-ordered optimization minima for the
    labeled runs keywords. Only the top 'tail' entries are kept.
    """
    fig, ax = plt.subplots()
    for i, (key, run) in enumerate(runs.items()):
        nll = run.nlls_
        succ = run.success_
        if downsample is not None:
            idx = np.random.choice(np.arange(len(nll)), downsample)
            nll = nll[idx]
            succ = succ[idx]
        if only_success:
            keep = succ >= 1
            nll = nll[keep]
            succ = succ[keep]
        q = np.quantile(nll, tail)
        nlls = nll[nll < q]
        sort_idx = np.argsort(nlls)
        y = nlls[sort_idx]
        x = 2*i + succ[nll < q].astype('int')
        cols = mpl.cm.get_cmap('Paired')(x)
        x = np.array(list(reversed(range(len(y)))))
        if x_percent:
            x = x / len(x)
        ax.scatter(x, y, s=1, label=key, c=cols)
    ax.set_ylabel("negative log-likelihood")
    if x_percent:
        ax.set_xlabel("rank (proportion of total)")
    else:
        ax.set_xlabel("rank")
    ax.legend()
    ax.semilogy()

class OptimResult:
    def __init__(self, nlls, thetas, success, starts=None):
        # order from best to worst
        idx = np.argsort(nlls)
        self.rank_ = idx
        self.nlls_ = nlls[idx]
        self.thetas_ = thetas[idx]
        self.success_ = success[idx]
        self.starts_ = starts[idx]

    @property
    def stats(self):
        succ = Counter(self.success_)
        return {NL_OPT_CODES[k]: n for k, n in succ.items()}

    @property
    def pass_idx(self):
        return self.success_ >= 1

    @property
    def thetas(self):
        # get only the successful thetas
        return self.thetas_[self.pass_idx]

    @property
    def theta(self):
        return self.thetas[0]

    @property
    def nlls(self):
        # get only the successful nlls
        return self.nlls_[self.pass_idx]

    @property
    def nll(self):
        return self.nlls[0]

    @property
    def frac_success(self):
        x = np.mean([v >= 1 for v in self.success_])
        return x

    def __repr__(self):
        code = NL_OPT_CODES[self.success_[self.pass_idx][0]]
        return ("OptimResult\n"
               f"  termination code: {code}\n"
               f"  stats: {self.stats} (prop success: {np.round(self.frac_success, 2)*100}%)\n"
               f"  negative log-likelihood = {self.nll}\n"
               f"  theta = {self.theta}")

