import multiprocessing
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#import tqdm.notebook as tqdm
import tqdm.autonotebook as tqdm
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

def run_optims(workerfunc, starts, progress=True, ncores=50):
    """
    Parallel optiomization.

    Returns an OptimResult, which contains all the optimizations
    from each random start.
    """
    nstarts = len(starts)
    ncores = ncores if ncores is not None else 1
    ncores = min(nstarts, ncores)
    if ncores > 1:
        with multiprocessing.Pool(ncores) as p:
            if progress:
                res = list(tqdm.tqdm(p.imap(workerfunc, starts),
                                     total=nstarts))
            else:
                res = list(p.imap(workerfunc, starts))
    else:
        # should be refactored TODO
        if progress:
            res = list(tqdm.tqdm(map(workerfunc, starts), total=nstarts))
        else:
            res = list(map(workerfunc, starts))

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


def constraint_matrix(nt, nf, fixed_mu=False):
    """
    Build a constrain matrix A for the simplex model.
    The constraint matrix is dot'd with the θ vector
    and compared to the contraints.

    The dimensionality of A is nparams x nconstraints.
    For the simplex model, the number of contraints is the
    number of features -- the DFE of each feature must sum
    to one. The dot product A·θ should return the sums of
    the DFE weights down each column.

    """
    offset = 2 - int(fixed_mu)
    nparams = nt*nf + offset
    A = np.zeros((nf, nparams))
    for i in range(nf):
        W = A[i, offset:].reshape((nt, nf))
        W[:, i] = 1.
    return A


def inequality_constraint_functions(nt, nf, log10_W_bounds, mu=None):
    """
    Return two functions for testing whether inequality constraints 
    are met (for nlopt).  NOTE: due to an unfortunate naming accident
    w = μW in the command line code.

    The constraint is that:

       l < μW < u

    where l and u are the lower and upper bounds for w (in code), or
    M (in math), the product of μW.

    These are rearranged to give explicit lower and upper bounds, that must
    be less than zero:

       l - μW < 0
       μW - u < 0

    This also works if the fixed mutation model is used.
    """
    fixed_mu = mu is not None
    A = constraint_matrix(nt, nf, fixed_mu=fixed_mu)
    lower, upper = 10**log10_W_bounds[0], 10**log10_W_bounds[1]

    def func_l(result, x, grad):
        if not fixed_mu:
            u = x[1]
        else:
            u = mu
        M = lower - (u * A.dot(x))
        for i in range(nf):
            result[i] = M[i]

    def func_u(result, x, grad):
        if not fixed_mu:
            u = x[1]
        else:
            u = mu
        M = (u * A.dot(x)) - upper
        for i in range(nf):
            result[i] = M[i]

    return func_l, func_u


def equality_constraint_function(nt, nf, fixed_mu=False):
    """
    Ensure that all the DFE weights sum to 1. Returns a function
    for nlopt to test this (will be equal to zero, within nlopt's
    tolerance, if contraint is met).
    """
    A = constraint_matrix(nt, nf, fixed_mu)

    def func(result, x, grad):
        M = A.dot(x)
        for i in range(nf):
            result[i] = M[i] - 1.
    return func


def nlopt_softmax_worker(start, func, nt, nf, bounds, 
                         log10_W_bounds, 
                         constraint_tol=1e-3, xtol_rel=1e-3,
                         maxeval=1000000, algo='ISRES'):
    """
    not for fixed mu (TODO)
    """
    # get the nlopt optimiziation routine
    nlopt_algo = getattr(nlopt, algo)
    nparams = nt * nf + 2

    opt = nlopt.opt(nlopt_algo, nparams)
    opt.set_min_objective(func)

    # set the bounds of all parameters
    nbounds = len(bounds[0])
    softmax_bounds = [-1e-4] * nbounds, [1e4] * nbounds
    # pi0
    softmax_bounds[0][0] = bounds[0][0]
    softmax_bounds[1][0] = bounds[1][0]
    # mu
    softmax_bounds[0][1] = bounds[0][1]
    softmax_bounds[1][1] = bounds[1][1]
    lb, ub = softmax_bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # set the x relative tolerance
    opt.set_xtol_rel(xtol_rel)

    # set max number of evaluations
    opt.set_maxeval(maxeval)
    assert start.size == nparams
    mle = opt.optimize(start)
    nll = opt.last_optimum_value()
    success = opt.last_optimize_result()
    return nll, mle, success



def nlopt_simplex_worker(start, func, nt, nf, bounds, 
                         log10_W_bounds, mu=None,
                         constraint_tol=1e-3, xtol_rel=1e-3,
                         maxeval=1000000, algo='ISRES'):
    """
    Use nlopt to do constrained (inequality to bound DFE weights
    and equality to enforce the simplex) and bounded optimization
    for the simplex model (possibly with fixed mu)
    """
    fixed_mu = mu is not None
    # we have one fewer parameter to optimize over with fixed mu
    offset = 1 + int(not fixed_mu)
    nparams = nt * nf + offset

    # get the nlopt optimiziation routine
    nlopt_algo = getattr(nlopt, algo)

    opt = nlopt.opt(nlopt_algo, nparams)
    opt.set_min_objective(func)

    # inequality constraint for l < μW < u
    # NOTE: these bounds are for the product μW, 
    # so they are *wider* than MU_BOUNDS. These should 
    # be bounded by the B interpolation range.
    hl, hu = inequality_constraint_functions(nt, nf, mu=mu,
                                             log10_W_bounds=log10_W_bounds)
    # tolerances for inequality constraint
    tols = np.repeat(constraint_tol, nf)

    # specify the simplex inequality constraints, that each entry be
    # 0 ≤ W ≤ 1 (up to the tolerance)
    opt.add_inequality_mconstraint(hl, tols)
    opt.add_inequality_mconstraint(hu, tols)

    # add the equality constraint -- that DFE must sum to 1
    ce = equality_constraint_function(nt, nf, fixed_mu)
    opt.add_equality_mconstraint(ce, tols)

    # set the bounds of all parameters
    lb, ub = bounds
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    # set the x relative tolerance
    opt.set_xtol_rel(xtol_rel)

    # set max number of evaluations
    opt.set_maxeval(maxeval)
    assert start.size == nparams
    mle = opt.optimize(start)
    nll = opt.last_optimum_value()
    success = opt.last_optimize_result()
    return nll, mle, success


def optim_plot(only_success=True, logy=False, tail=0.5, x_percent=False, downsample=None, **runs):
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
    if logy:
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

    def is_succes(self):
        """
        Check that the MLE optimization passed.
        """
        return self.success_[0]

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

