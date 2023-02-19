"""
Notes on MLE optimization.

Initially I tried optimizing over the linearly-interpolated B grid,
which would eventually find a solution, but the results were noisy 
across runs and took a long time to converge to an optima. When doing 
this I tried various parameterizations

    - Free mutation model: each DFE x μ term was unique and only 
       bounded to be positive
    - Simplex model: a variation of what we use now (below) 


Random notes:
 - nlopt has an issue with LD_LBFGS failing on some random starts. 
   Googling around, this also happens with Julia's nlopt library so 
   seems to be a higher-level issue, but I could not confirm this.
   So we use scipy.minimize.

 - global nlopt algorithms like GN_ISRES would be good to try 
    againt softmax simplex models, but they require finite 
    bounds. I tried putting an artificial bound on this but 
    it did not seem to work.

SimplexModels:
  - softmax, which allows for unconstrained simplex optimization
  - simplex with inequality and equality constraints

I experimented with the constrained simplex model in notebooks, but it seemed
to be finicky. Generally, turning a constrained problem into an unconstrained
one is advised.

See the `optim_tests/` directory and the `notebooks/mle_diagnostics.ipynb`
notebook for more on this. Generally, I find softmax parameterization
with nlopt's BOBYQA out performs everything else.

"""


import warnings
import multiprocessing
from collections import Counter
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import nlopt
from scipy.optimize import minimize

# nlopt has different error messages; we use this for scipy too
# which uses booleans we convert to integers
NL_OPT_CODES = {1:'success', 2:'stopval reached', 3:'ftol reached',
                4:'xtol reached', 5:'max eval', 6:'maxtime reached', -1:'failure',
               -2:'invalid args', -3:'out of memory', -4:'forced stop',
                0: 'scipy failure'}


def extract_opt_info(x):
    """
    General function that for both scipy and nlopt.
    """
    is_scipy = hasattr(x, 'fun')
    if is_scipy:
        nll, res, success = x.fun, x.x, x.success
        success = -1 if not success else 1
    else:
        nll, res, success = x
        # 0 case is the try/except around nlopt.optimize
        assert success == 0 or success in NL_OPT_CODES.keys()
    return nll, res, success


def array_all(x):
    return tuple([np.array(a) for a in x])


def run_optims(workerfunc, starts, progress=True, ncores=50,
               return_raw=False):
    """
    Parallel optimization.

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
    if return_raw:
        return nlls, thetas, success
    return OptimResult(nlls, thetas, success, np.array(starts))


def scipy_softmax_worker(start, func, nt, nf, 
                         bounds,
                         method='L-BFGS-B'):
    """
    Main wrapper around scipy optimization via optimize.minimize.
    """
    nparams = nt*nf + 2
    res = minimize(func, start, bounds=bounds, method=method, options={'maxiter':1e6})
    nll = res.fun
    mle = res.x
    mle = convert_softmax(mle, nt, nf)
    success = int(res.success)  # this is so it matches nlopt
    return nll, mle, success


def nlopt_softmax_worker(start, func, nt, nf, bounds,
                         method, xtol_rel=1e-3,
                         constraint_tol=1e-11,
                         maxeval=1000000):
    """
    nlopt softmax wrapper
    """
    nparams = nt*nf + 2
    method = getattr(nlopt, method)
    opt = nlopt.opt(method, nparams)
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
    mle = convert_softmax(mle, nt, nf)
    return nll, mle, success


def convert_softmax(theta_sm, nt, nf):
    """
    Given an MLE θ in softmax space for W, convert it back
    """
    theta = np.copy(theta_sm)
    with np.errstate(under='ignore'):
        theta[2:] = softmax(theta[2:].reshape(nt, nf), axis=0).flat
    return theta


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


def optim_diagnotics_plot(fit, top_n=100, figsize=None,
                          pi_scale=1e3, mu_scale=1e8,
                          add_nll=False, filter_success=True,
                          cmap='viridis'):
    """
    Thanks to Nate Pope for this visualization suggestion!
    """
    opt = fit.optim 
    features = fit.features
    nt, nf, t = fit.nt, fit.nf, fit.t
    nlls = opt.nlls_
    thetas = opt.thetas_
    success = opt.success_.astype('bool')

    if filter_success:
        nlls = nlls[success]
        thetas = thetas[success]
        success = success[success]
    if len(thetas) < top_n:
        msg = "top_n < number of optimization results, truncating!"
        warnings.warn(msg)
        top_n = len(thetas)
    dfes = []
    mu_pi0 = []
 
    for i in range(top_n):
        dfes.append(thetas[i][2:].reshape(nt, nf))

    # mu_pi0 = np.stack(mu_pi0)
    dfes = np.stack(dfes)

    fig, ax = plt.subplots(ncols=1, nrows=nf+2 + int(add_nll),
                           figsize=figsize, sharex=True, 
                           height_ratios=[1.2]*nf + [1]*(2+add_nll))

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    for i in range(nf):
        #ax[i].imshow(dfes[:, :, i].T, cmap='inferno')
        x = np.arange(top_n)
        ax[i].pcolormesh(dfes[:, :, i].T, cmap=cmap, norm=norm)
        ax[i].set_ylabel(f"{features[i]}")
        ax[i].set_yticks(np.arange(nt)+0.5, 
                         [f"${10}^{{{x}}}$" for x in np.log10(1e-12+t).astype(int)])
        ax[i].xaxis.set_visible(False)
        ax[i].tick_params(axis='y', which='major', labelsize=4, 
                          width=0.5, length=3)
        ax[i].set_aspect('auto')
        i += 1

    # label size for line plots
    line_lsize = 6
    line_lw = 1

    ax[i].set_ylabel(f"$\pi_0$ ($\\times^{{{int(-np.log10(pi_scale))}}}$)")
    ax[i].plot(np.arange(len(thetas))[:top_n], 
               [x[0]*pi_scale for x in thetas][:top_n],
               linewidth=line_lw, c='0.22')
    ax[i].scatter(np.arange(len(thetas))[:top_n], 
               [x[0]*pi_scale for x in thetas][:top_n],
                  c=[{True: 'k', False: 'r'}[x] for x in success][:top_n],
                  s=2, zorder=2)
 
    ax[i].tick_params(axis='both', which='major', labelsize=line_lsize)

    i += 1
    ax[i].set_ylabel(f"$\mu$ ($\\times^{{{int(-np.log10(mu_scale))}}}$)")
    ax[i].plot(np.arange(len(thetas))[:top_n], 
               [x[1]*mu_scale for x in thetas][:top_n],
               linewidth=line_lw,
               c='0.22')

    ax[i].tick_params(axis='both', which='major', labelsize=line_lsize)
    i += 1

    if add_nll:
        y = np.sort(nlls)[:top_n]
        nlls_scale = int(-np.log10(np.max(y)))
        ax[i].step(np.arange(len(nlls))[:top_n], 
                   y*10**nlls_scale, c='0.22')
        ax[i].set_ylabel(f'NLL ($\\times 10^{{{nlls_scale}}}$)')
        ax[i].set_xlabel('rank')
        ax[i].ticklabel_format(useOffset=False)
        #plt.tight_layout()
    return fig, ax


class OptimResult:
    def __init__(self, nlls, thetas, success, starts=None):
        prop_success = np.mean(success)
        if prop_success <= 0.9:
            msg = f"only {np.round(prop_success*100, 2)}% optimizations terminated successfully!"
            warnings.warn(msg)
        # order from best to worst
        idx = np.argsort(nlls)
        self.rank_ = idx
        self.nlls_ = nlls[idx]
        self.thetas_ = thetas[idx]
        self.success_ = success[idx]
        self.starts_ = starts[idx]

    def is_success(self):
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
        # works for both scipy and nlopt
        x = np.mean([v >= 1 for v in self.success_])
        return x

    def __repr__(self):
        code = NL_OPT_CODES[self.success_[self.pass_idx][0]]
        return ("OptimResult top result:\n"
               f"  termination code: {code}\n"
               f"  stats: {self.stats} (prop success: {np.round(self.frac_success, 2)*100}%)\n"
               f"  negative log-likelihood = {self.nll}\n"
               f"  theta = {self.theta}")

