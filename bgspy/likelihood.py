## likelihood.py -- functions for likelihood stuff
import os
import warnings
import re
from copy import copy
from collections import defaultdict
import pickle
import itertools
import tqdm
from si_prefix import si_format
from scipy.special import softmax
from scipy.stats import linregress
from tabulate import tabulate
from functools import partial
import numpy as np
from ctypes import POINTER, c_double, c_ssize_t, c_int
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# no longer needed
# HAS_JAX = False
# try:
#     import jax.numpy as jnp
#     HAS_JAX = True
# except ImportError:
#     pass

from scipy import interpolate
from bgspy.utils import signif, load_pickle, coefvar, center_and_scale
from bgspy.data import pi_from_pairwise_summaries, GenomicBinnedData
from bgspy.optim import run_optims, scipy_softmax_worker
from bgspy.optim import nlopt_softmax_worker, nlopt_softmax_fixedmu_worker
from bgspy.plots import model_diagnostic_plots, predict_chrom_plot
from bgspy.plots import resid_fitted_plot, get_figax
from bgspy.plots import chrom_resid_plot
from bgspy.bootstrap import moving_block_bins, jackknife_stderr


# load the library (relative to this file in src/)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
likclib = np.ctypeslib.load_library("likclib", LIBRARY_PATH)

#likclib = np.ctypeslib.load_library('lik', 'bgspy.src.__file__')

# mutation rate hard bounds
# these are set for human. I'm targetting the uncertainty 
# with the mutational slowdown, e.g. see Moorjani et al 2016
# the upper bound is *very* permissive
MU_BOUNDS = tuple(np.log10((1e-9,  1e-7)))


# π0 bounds: lower is set by lowest observed π in humans
# highest is based on B lowest is 0.02, e.g. π = π0 Β,
# 1e-4 = 0.005 B
PI0_BOUNDS = tuple(np.log10((0.0005, 0.005)))  # this is fairly permissive
# see https://twitter.com/jkpritch/status/1600296856999047168/photo/1
# NOTE: these were old bounds based on human data. For simulations
# we need very permissive bounds -- e.g. all range seen
# https://iiif.elifesciences.org/lax/67509%2Felife-67509-fig2-v3.tif/full/1500,/0/default.jpg
#PI0_BOUNDS = tuple(np.log10((1e-4, 1e-1))) 

# -------- random utility/convenience functions -----------

def param_names(t, features, fixed_mu=False):
    """
    Label the simplex model parameters.
    """
    sels, feats = np.meshgrid(np.log10(t), features)
    sels, feats = itertools.chain(*sels.tolist()), itertools.chain(*feats.tolist())
    ps = ["pi0"]
    if not fixed_mu:
        ps.append("mu")
    return ps + [f"W[{int(s)},{f}]" for s, f in zip(*(sels, feats))]

def R2(x, y):
    """
    Based on scipy.stats.linregress
    https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_mstats_common.py#L22-L209
    """
    complete_idx = ~(np.isnan(x) | np.isnan(y))
    x = x[complete_idx]
    y = y[complete_idx]
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=True).flat
    try:
        return ssxym / np.sqrt(ssxm * ssym)
    except FloatingPointError:
        return np.nan


def check_bounds(x, lb, ub):
    assert np.all((x >= lb) & (x <= ub))


def W_summary(W, features, nt, nf, t, cis=None):
    """
    pretty-print the W matrix, optionally with CIs
    """
    header = ()
    if features is not None:
        header = [''] + features
    W = W.reshape((nt, nf))
    t = t[:, None]
    nt, nf = nt, nf
    rows = []
    theta_i = 2
    if cis is not None:
        lower, upper = cis
    for i in range(nt):
        row = [t[i]]
        for j in range(nf):
            entry = f"{W[i, j]:.3f}"
            if cis is not None:
                l, u = lower[theta_i], upper[theta_i]
                entry += f" ({l:.4f}, {u:.4f})"
            row.append(entry)
            theta_i += 1
        rows.append(row)
    return tabulate(rows, headers=header)


# -------- main likelihood methods -----------

def mut_curve(x, a):
    """
    This is the parametric form of the mutation grid -- this was
    inferred and turned out to be exact over the points, given the
    one free-parameter. This makes some sense given that the form of
    the reduction is exp(-V/2 Q^2) and V is ~linear in mutation rate.
    """
    return np.exp(-a * x)


def fit_B_curve_params(b, w):
    """
    Five the mutation curve for each DFE x feature combination.
    Reduces the dimensionality of whatever the grid is to a single
    parameter.
    """
    # TODO: fixing this, but shouldn't be global. Something upstream
    # messed up the data
    np.seterr(under='ignore')
    nx, nw, nt, nf = b.shape
    params = np.empty((nx, 1, nt, nf))
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                #popt, pcov = curve_fit(mut_curve, w, 
                #                       np.exp(b[i, :, j, k]),
                #                       maxfev=int(1e6))
                slope = linregress(w, b[i, :, j, k]).slope
                #assert(len(popt) == 1)
                #params[i, 0, j, k] = popt[0]
                params[i, 0, j, k] = -slope

    # next we check that the residuals aren't unusually high
    # the threshold here is set just by some EDA
    resid = []
    for li in range(b.shape[0]):
        grid = np.exp(b[li, :, :, :])
        func = np.exp(-w[:, None, None] * params[li, :, :, :])
        e2 = (grid-func)**2
        resid.append(e2.mean())
    resid = np.array(resid)
    msg = "unusually high residual in B'(μ) fitting"
    assert np.all(resid < 5e-3), msg
    return params


def negll_softmax(theta, Y, B, w):
    """
    Softmax version of the simplex model wrapper for negll_c().

    For scipy, i.e. no grad argument, so must be wrapped for use
    with nlopt.
    """
    nx, nw, nt, nf = B.shape
    # get out the W matrix, which on the optimization side, is over all reals
    sm_theta = np.copy(theta)
    W_reals = theta[2:].reshape(nt, nf)
    with np.errstate(under='ignore'):
        W = softmax(W_reals, axis=0)
    assert np.allclose(W.sum(axis=0), np.ones(W.shape[1]))
    sm_theta[2:] = W.flat
    return negll_c(sm_theta, Y, B, w, version=3)


def negll_softmax_nlopt(theta, grad, Y, B, w):
    """
    The nlopt wrapper to the above function, since
    nlopt requires a grad second argument even if 
    not known.
    """
    return negll_softmax(theta, Y, B, w)


def negll_softmax_fixedmu_nlopt(theta, grad, Y, B, w, mu):
    """
    The nlopt wrapper to the above function, since
    nlopt requires a grad second argument even if 
    not known.
    """
    new_theta = np.empty(len(theta)+1)
    new_theta[0] = theta[0]
    new_theta[1] = mu
    new_theta[2:] = theta[1:]
    return negll_softmax(new_theta, Y, B, w)


def negll_simplex(theta, Y, B, w):
    """
    Simplex model wrapper for negll_c().
    """
    return negll_c(theta, Y, B, w, version=3)


def bounds_simplex(nt, nf, log10_pi0_bounds, log10_mu_bounds,
                   mu=None, softmax=True, bounded_softmax=False,
                   global_softmax_bound=1e4,
                   paired=True):
    """
    Return the bounds on for optimization under the simplex model
    model. If paired=True, the bounds are zipped together for each
    parameter, for scipy.

    If softmax=True, all W entries are in the reals, with bounds
    -Inf, Inf. If a global optimization algorithm is being used,
    bounded_softmax should be set to True; a high bound is chosen,
    assuming that things are started on N(0, 1).
    """
    l = [10**log10_pi0_bounds[0]]
    u = [10**log10_pi0_bounds[1]]
    #l = [(1000)]
    #u = [(100_000)]
    if mu is None:
        l += [10**log10_mu_bounds[0]]
        u += [10**log10_mu_bounds[1]]
    # if we use softmax, it should technically be unbounded
    # but this handles the case where global optimization is used,
    # which nlopt requires bounds for -- we set these to something
    # relatively large
    softmax_bound = np.inf if not bounded_softmax else global_softmax_bound
    lval = 0 if not softmax else -softmax_bound
    uval = 1 if not softmax else softmax_bound
    l += [lval]*nf*nt
    u += [uval]*nf*nt
    lb = np.array(l)
    ub = np.array(u)
    assert np.all(lb < ub)
    if paired:
        return list(zip(lb, ub))
    return lb, ub


def random_start_simplex(nt, nf, log10_pi0_bounds, 
                         log10_mu_bounds, mu=None,
                         softmax=True):
    """
    Create a random start position, uniform over the bounds for π0
    and μ, and Dirichlet under the DFE weights for W, under the simplex model.
    """
    mu_is_fixed = mu is not None
    if not mu_is_fixed:
        mu = np.random.uniform(10**log10_mu_bounds[0],
                               10**log10_mu_bounds[1], 1)
    pi0 = np.random.uniform(10**log10_pi0_bounds[0],
                            10**log10_pi0_bounds[1], 1)

    fixed_mu_offset = 1 + int(not mu_is_fixed)
    nparams = nt*nf + fixed_mu_offset
    theta = np.empty(nparams)
    if softmax:
        #theta[0] = np.random.uniform(1000, 100_000, 1)
        theta[0] = pi0
        if not mu_is_fixed:
            theta[1] = mu
        theta[fixed_mu_offset:] = np.random.normal(0, 1, nt*nf)
        return theta

    W = np.empty((nt, nf))
    for i in range(nf):
        W[:, i] = np.random.dirichlet([1.] * nt)
        assert np.abs(W[:, i].sum() - 1.) < 1e-5
    theta = np.empty(nt*nf + fixed_mu_offset)
    theta[0] = pi0
    if mu_is_fixed:
        theta[1] = mu
    theta[fixed_mu_offset:] = W.flat
    bounds = bounds_simplex(nt, nf, log10_pi0_bounds=log10_pi0_bounds,
                            log10_mu_bounds=log10_mu_bounds, mu=mu)
    check_bounds(theta, *bounds)
    return theta


def predict_simplex(theta, logB, w, mu=None, use_grid=False):
    """
    Prediction function for SimplexModel and FixedMutationModel.

    TODO: this should be ported to C, it's very slow.
    """
    fixed_mu = mu is not None
    nx, nw, nt, nf = logB.shape
    # mut weight params
    if not fixed_mu:
        pi0, mu, W = theta[0], theta[1], theta[2:]
    else:
        pi0, W = theta[0], theta[1:]
    W = W.reshape((nt, nf))
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                if use_grid:
                    # this is deprecated, can be removed in future
                    logBw[i] += np.interp(mu*W[j, k], w, logB[i, :, j, k])
                else:
                    logBw[i] += -mu*W[j, k]*logB[i, 0, j, k]
    return pi0*np.exp(logBw)


# -------- main likelihood classes -----------

class BGSLikelihood:
    """
    Y ~ π0 B(w)

    (note Bs are stored in log-space.)
    bins
    """

    def __init__(self,
                 Y, w, t, logB, bins=None, features=None,
                 log10_pi0_bounds=PI0_BOUNDS, log10_mu_bounds=MU_BOUNDS):

        # this was an experiment to test if it made a difference
        # to add a truly neutral class. Because the default lowest
        # wμ point (1e-7) is so close to zero, 
        add_neutral_class = False
        if add_neutral_class:
            nx, nw, nt, nf = logB.shape
            logBn = np.zeros((nx, nw, nt+1, nf), dtype=logB.dtype)
            logBn[:, :, 1:, :] = logB
            logB = logBn
            t = np.array([0] + t.tolist())
        self.w = w
        self.t = t
        if bins is not None:
            # TODO: this is disabled for now because it throws an error with
            # notebooks that's uncessary (even when bins really is
            # GenomicBinnedData)
            assert isinstance(bins, GenomicBinnedData)
            assert bins.nbins() == Y.shape[0]
        self.bins = bins
        self.log10_pi0_bounds = log10_pi0_bounds
        # the bounds for mu alone
        self.log10_mu_bounds = log10_mu_bounds
        self._indices_fit = None
        self._chrom_loo_fit = None
        self._chrom_fit = None
        self._fixed_mu = None

        try:
            assert logB.ndim == 4
            #assert logB.shape[1] == w.size # TODO (low priority)
            assert logB.shape[2] == t.size
        except AssertionError:
            msg = "B dimensions ({logB.shape}) do not match (supplied w and t dimensions)"
            raise AssertionError(msg)

        # labels for the features
        self.features = features

        # check the data and bins
        try:
            logB.shape[0] == Y.shape[0]
            Y.shape[1] == 2
        except AssertionError:
            raise AssertionError("Y shape is incorrect")
        self.Y = Y

        self.logB = logB
        self.theta_ = None

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

    def predict_B_at_pos(self, chrom, pos, **kwargs):
        """
        Predict B from this fit, via interpolation.

        TODO (low priority): sometimes outputs B > 1 due to
        interpolation issues.
        """
        defaults = {'kind': 'quadratic',
                    'assume_sorted': True,
                    'bounds_error': False,
                    'copy': False}
        kwargs = {**defaults, **kwargs}
        mids = self.bins.midpoints()
        idx = self.bins.chrom_indices(chrom)
        y = self.predict(B=True)
        func = interpolate.interp1d(mids[chrom], y[idx],
                                    fill_value=(y[0], y[-1]),
                                    **kwargs)
        return func(pos)

    def _load_optim(self, optim_res, index=None):
        """
        Taken an OptimResult() object.
        """
        self.optim = optim_res
        if index is None:
            # load in the best values
            self.theta_ = optim_res.theta
            self.nll_ = optim_res.nll
        else:
            self.theta_ = optim_res.thetas_[index]
            self.nll_ = optim_res.nlls_[index]

    def jackknife_chrom(self, chrom, **kwargs):
        """
        Do the jackknife, leaving out the specified chromosome.
        """
        msg = "bins attribute must be set to a GenomicBinnedData object"
        assert self.bins is not None, msg
        chrom_idx = set(self.bins.chrom_indices(chrom))
        assert len(chrom_idx)
        all_idx = set(range(self.Y.shape[0]))
        in_sample_idx = np.array(list(all_idx.difference(chrom_idx)))
        # return the fit directly
        obj = self.fit(**kwargs, _indices=in_sample_idx)
        obj._chrom_loo_fit = chrom
        return obj

    def jackknife_block(self, blocksize, blocknum=None,
                        calc_loo_R2=True, **kwargs):
        """
        Use the block-jackknife to estimate uncertainty. This 
        will use blocksize-sized blocks (in number of consecutive
        windows, so the physical size of the block will vary with
        the window size.

        TODO: this could be refactored a bit to remove redundancy.
        """
        # generate the blocks
        blocks = moving_block_bins(self.bins, blocksize)

        if blocknum is not None:
            # the jackknife out-sample
            out_block_idx = blocks.pop(blocknum)
            # combine the rest of the blocks
            block_indices = np.array(list(set(itertools.chain(*blocks))))
            # fit the model on these indices
            obj = self.fit(**kwargs, _indices=block_indices)
            obj._indices_fit = block_indices
            # store metadata 
            obj._blocksize = blocksize
            obj._blocknum = blocknum
            obj._blocksize_bp = blocksize*self.bins.width
            # for out-sample prediction
            obj._out_block_indices = out_block_idx
            if calc_loo_R2:
                r2 = obj.R2(_indices = out_block_idx)
                obj._block_r2 = r2  # TODO could weight by num idx but minor
            return obj
        
        # we're going to do this for all blocks (mostly for testing)
        jackknife_results = []
        r2s = []
        for i, out_block_idx in tqdm.tqdm(enumerate(blocks)):
            # note that this is not the most efficient here...
            block_indices = copy(blocks)
            block_indices.pop(i)  # drop this block
            # because of the moving block, there are numerous redundant indices
            # we remove here with set
            block_indices = np.array(list(set(itertools.chain(*block_indices))))
            obj = self.fit(**kwargs, _indices=block_indices, progress=False)
            obj._indices_fit = block_indices
            # store metadata 
            obj._blocksize = blocksize
            obj._blocknum = blocknum
            obj._blocksize_bp = blocksize*self.bins.width
            # for out-sample prediction
            obj._out_block_indices = out_block_idx
            if calc_loo_R2:
                r2 = obj.R2(_indices = out_block_idx)
                r2s.append(r2)  # TODO could weight by num idx but minor
            jackknife_results.append(obj)
        return jackknife_results, r2s

    def load_jackknives(self, fit_dir, label=False, trim=None):
        """
        Load the jackknife results by setting attributes.
        """
        files = [f for f in os.listdir(fit_dir) if f.endswith('.pkl')]
        if not len(files):
            warnings.warn(f"no .pkl model files found in {fit_dir}")
            return None
        join = os.path.join
        fits = [load_pickle(join(fit_dir, f))['mbp'] for f in files]
        thetas = np.stack([f.theta_ for f in fits])
        rgx = re.compile(r'jackknife_([\w.]+)\.pkl')
        indices = [rgx.match(f).groups()[0] for f in files]
        nlls = np.stack([f.nll_ for f in fits])
        r2s = [f._block_r2 for f in fits]

        if label:
            fits = dict(zip(indices, fits))
        self.jack_fits_ = fits
        self.jack_nlls_ = nlls
        self.jack_thetas_ = thetas
        self.jack_indices = indices
        self.jack_r2s = r2s
        self.jackknife_stderr(trim=trim)

    def load_loo(self, loo_dir):
        """
        Load the Leave-one-out chromosome directory.

        TODO this could be merged with load_jackknives and 
        made to be a more general function with code outside the
        class.
        """
        files = [f for f in os.listdir(loo_dir) if f.endswith('.pkl')]
        if not len(files):
            warnings.warn(f"no .pkl model files found in {loo_dir}")
            return None
        join = os.path.join
        fits = [load_pickle(join(loo_dir, f))['mbp'] for f in files]
        thetas = np.stack([f.theta_ for f in fits])
        rgx = re.compile(r'loo_([\w.]+)\.pkl')
        chroms = [rgx.match(f).groups()[0] for f in files]
        nlls = np.stack([f.nll_ for f in fits])

        fits = dict(zip(chroms, fits))
        self.loo_fits_ = fits
        self.loo_nlls_ = nlls
        self.loo_thetas_ = thetas
        self.loo_chroms_ = chroms

    #def ci(self, method='quantile'):
    #    assert self.boot_thetas_ is not None, "bootstrap() has not been run"
    #    if method == 'quantile':
    #        lower, upper = pivot_ci(self.boot_thetas_, self.theta_)
    #    elif method == 'percentile':
    #        lower, upper = percentile_ci(self.boot_thetas_)
    #    else:
    #        raise ValueError("improper bootstrap method")
    #    return np.stack((lower, self.theta_, upper)).T

    def save(self, filename):
        """
         Pickle this object.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


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

    def dfe_plot2(self, figax=None, figsize=None):
        """
        """
        nf = self.nf
        fig, ax = get_figax(figax, nrows=nf, 
                            figsize=figsize, sharex=True)
        xt = np.log10(self.t)

        for i in range(nf):
            feat = self.features[i]
            ax[i].bar(xt, self.mle_W[:, i], align='edge', label=feat)
            if i < nf-1:
                ax[i].xaxis.set_visible(False)
            #ax[i].set_title(feat)
            ax[i].text(0.5, 0.9, feat, 
                       horizontalalignment='center',
                       verticalalignment='top',
                       transform=ax[i].transAxes, fontsize=8)
        xtl = [f"$10^{{{int(x)}}}$" for x in xt]
        ax[nf-1].set_xticks(np.log10(self.t), xtl, fontsize=5)
        #ax.set_ylabel('probability')
        #ax.legend()

    def W_stderrs(self):
        """
        A dictionary of standard errors for each feature.
        """
        do_se = self.sigma_ is not None
        fixed_mu_offset = 1 + int(not self._fixed_mu is not None)
        if do_se:
            ses = self.sigma_[fixed_mu_offset:].reshape(self.mle_W.shape)
            se = defaultdict(list)
        else:
            se =None
        mean = defaultdict(list)
        for i, feature in enumerate(self.features):
            for j, t in enumerate(self.t):
                if do_se:
                    se[feature].append(ses[j,i])
                mean[feature].append(self.mle_W[j,i])
        return mean, se

    def normal_draw(self):
        """
        For approximate parametric bootstrap. Out of bounds draws of W are 
        set to 0, 1.
        """
        draw = np.random.normal(self.theta_, self.sigma_)
        # truncate the bounds of W
        W = draw[2:]
        W[W < 0] = 0
        W[W > 1] = 1
        return draw[0], draw[1], W.reshape((self.nt, self.nf))

         
    def W_df(self):
        mean, se = self.W_stderrs()
        t = self.t
        means, errors = self.W_stderrs()
        df_means = pd.DataFrame(means)
        df_means['t'] = t
        df_means_melted = pd.melt(df_means, id_vars='t', var_name='category', value_name='mean')
        df_merged = df_means_melted
        
        if errors is not None:
            df_errors = pd.DataFrame(errors)
            df_errors['t'] = t
            df_errors_melted = pd.melt(df_errors, id_vars='t', var_name='category', value_name='std_err')
            df_merged = pd.merge(df_means_melted, df_errors_melted, on=['t', 'category'])
        return df_merged

    def dfe_plot(self, add_legend=True, legend_kwargs={}, figax=None, ylabel='probability', barplot_kwargs={}):
        """
        Plot a boxplot of all features.
        """
        df = self.W_df()
        fig, ax = get_figax(figax)
        sns.barplot(data=df, x='t', y='mean', hue='category', ax=ax, **barplot_kwargs)
        xticks = ax.get_xticks()
        xlabels = ax.get_xticklabels()
        ax.set_xticks(xticks, [f"$10^{{{int(np.log10(float(x.get_text())))}}}$" for x in xlabels])
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if add_legend:
            ax.legend(**legend_kwargs)
        return fig, ax

    def pi(self):
        pi = pi_from_pairwise_summaries(self.Y)
        return pi

    def R2(self, _indices=None, **kwargs):
        """
        The R² value of the predictions against actual results.

        _indices: the indices of values to use, e.g. for cross-validation or LOO
              bootstrapping.
        """
        pred_pi = self.predict(**kwargs)
        pi = pi_from_pairwise_summaries(self.Y)
        if _indices is None:
            return R2(pred_pi, pi)
        return R2(pred_pi[_indices], pi[_indices])

    def jackknife_stderr(self, use_loo_chrom=False, trim=None, iqr_factor=2):
        """
        Calculate the jackknife standard errors.
        """
        # the MLE fit -- currently we use the bootstrap mean
        #theta = self.theta_
        if not use_loo_chrom:
            msg = "SimplexModel.jack_thetas_ is None, load jackknife results first"
            assert self.jack_thetas_ is not None, msg
            thetas = self.jack_thetas_
        else:
            msg = ("use_loo_chrom=True set but SimplexModel.loo_thetas_ is None, "
                   "load leave-one-out chromosome reults first")
            assert self.loo_thetas_ is not None, msg
            thetas = self.loo_thetas_

        if trim:
            if trim.upper() != "IQR":
                if isinstance(trim, float):
                    lower, upper = trim, 1-trim
                else:
                    lower, upper = trim
                l, u = np.quantile(thetas, (lower, upper), axis=0)
            else:
                q1, q3 = np.percentile(thetas, (25, 75), axis=0)
                iqr = q3 - q1
                l, u = q1 - iqr_factor*iqr, q3 + iqr_factor*iqr
            thetas = np.where((thetas < l) | (thetas > u), np.nan, thetas)

        sigma_jack, n = jackknife_stderr(thetas)
        self.sigma_trim_ = trim
        self.sigma_ = sigma_jack
        self.sigma_n_ = n
        self.std_error_type = 'loo' if use_loo_chrom else 'moving block jackknife'
        return sigma_jack



        Tni = thetas
        n = Tni.shape[0]
        Tn = Tni.mean(axis=0)
        Q = np.sum((Tni - Tn[None, :])**2, axis=0)
        sigma2_jack = (n-1)/n * Q
        sigma_jack = np.sqrt(sigma2_jack)
        self.sigma_ = sigma_jack
        self.sigma_n_ = n
        self.std_error_type = 'loo' if use_loo_chrom else 'moving block jackknife'
        return sigma_jack

    def loo_R2(self, return_raw=False):
        """
        Leave-one-out chromosome R2.
        """
        assert hasattr(self, 'loo_thetas_') and self.loo_thetas_ is not None
        loo_thetas = self.loo_thetas_
        n = loo_thetas.shape[0]
        pi = pi_from_pairwise_summaries(self.Y)
        r2s = dict()
        weights = dict()
        for i in range(n):
            loo_chrom = self.loo_chroms_[i]
            chrom_idx = self.bins.chrom_indices(loo_chrom)
            theta = loo_thetas[i, ...]
            chrom_b = self.logB_fit[chrom_idx, ...]
            pred_pi = predict_simplex(theta, chrom_b, self.w)
            r2s[loo_chrom] = R2(pred_pi, pi[chrom_idx])
            weights[loo_chrom] = len(pred_pi)
        self.loo_chrom_r2s = r2s
        self.loo_chrom_weights = weights
        if return_raw: 
            return r2s, weights
        return np.average(list(r2s.values()), weights=list(weights.values()))

    def resid(self):
        pred_pi = self.predict()
        pi = pi_from_pairwise_summaries(self.Y)
        return pi - pred_pi

    def resid_fitted_plot(self, *args, **kwargs):
        return resid_fitted_plot(self, *args, **kwargs)

    def diagnostic_plots(self):
        return model_diagnostic_plots(self)

    def chrom_resid_plot(self, figax=None):
        return chrom_resid_plot(self, figax)

    def predict_plot(self, chrom, figax=None, predict_kwargs=None, 
                     **kwargs):
        return predict_chrom_plot(self, chrom, figax=figax, 
                                  predict_kwargs=predict_kwargs,
                                  **kwargs)

    def spatial_resid(self, chrom):
        idx = self.bins.chrom_indices(chrom)
        x = self.bins.midpoints(filter_masked=True)[chrom]
        y = self.resid()
        y_all = self.bins.merge_filtered_data(y)

        bins = self.bins.flat_bins(filter_masked=False)
        chrom_idx = np.array([i for i, (c, s, e) in enumerate(bins) if c == chrom])

        x, y = np.array(self.bins.midpoints(filter_masked=False)[chrom]), center_and_scale(y_all[chrom_idx])
        return x, y

    def spatial_error(self, chrom, figax=None):
        fig, ax = get_figax(figax)
        x, y = self.spatial_resid(chrom)
        ax.plot(x, y)
        ax.axhline(0, linestyle='dashed', c='0.44', zorder=-1)

    def scatter_plot(self, figax=None, highlight_chrom=None, chrom_cols=False, **scatter_kwargs):
        fig, ax = get_figax(figax)
        pred_pi = self.predict()
        pi = pi_from_pairwise_summaries(self.Y)
        if chrom_cols:
            ax.scatter(pred_pi, pi, s=1, c=self.bins.chrom_ints())
        else:
            ax.scatter(pred_pi, pi, c='0.22', s=1)
        if highlight_chrom is not None:
            idx = self.bins.chrom_indices(highlight_chrom)
            ax.scatter(pred_pi[idx], pi[idx], s=2, c='r')
        ax.axline((0, 0), slope=1)
        ax.set_ylabel('predicted $\\hat{\pi}$')
        ax.set_xlabel('observed $\pi$')

    def coefvar(self, use_B=True):
        """
        Get the coefficient of variation for predictions and pi.
        """
        pred_cfs = dict()
        data_cfs = dict()
        for chrom in self.bins.seqlens:
            bins = self.bins.flat_bins(filter_masked=False)
            chrom_idx = np.array([i for i, (c, s, e) in enumerate(bins) if c == chrom])
            predicts = self.predict(B=use_B)
            pi = self.pi()
            predicts_full = self.bins.merge_filtered_data(predicts)
            pi_full = self.bins.merge_filtered_data(pi)
            y = predicts_full[chrom_idx]
            pred_cfs[chrom] = np.nanvar(y)
            data_cfs[chrom] = coefvar(pi_full[chrom_idx])
        return data_cfs, pred_cfs

    @property
    def mle_pi0(self):
        return self.theta_[0]

    def to_npz(self, filename):
        np.savez(filename, logB=self.logB, w=self.w, t=self.t, Y=self.Y_,
                 bounds=self.bounds())

    @staticmethod
    def from_npz(filename):
        d = np.load(filename)
        bounds = d['bounds']
        obj = BGSLikelihood(d['w'], d['t'], d['logB'],
                           (np.log10(bounds[0, 0]), np.log10(bounds[0, 1])))
        return obj

    def W_to_tsv(self, filename, ndigits=3):
        """
        Write W to TSV.
        """
        W = self.mle_W.T
        with open(filename, 'w') as f:
            for i, feat in enumerate(self.features):
                dfe = ','.join(map(str, np.round(W[i, :], ndigits)))
                f.write(f"{feat}\t{dfe}\n")

    def __repr__(self):
        rows = [f"{type(self).__name__} (interpolated w): {self.nw} x {self.nt} x {self.nf}"]
        rows.append(f"  w grid: {signif(self.w)} (before interpolation)")
        rows.append(f"  t grid: {signif(self.t)}")
        bs = self.bins.width
        rows.append(f"  window size: {si_format(bs)}bp")
        return "\n".join(rows)


class SimplexModel(BGSLikelihood):
    """
    BGSLikelihood Model with the DFE matrix W on a simplex, free
    π0 and free μ (within bounds) by default. μ can be fixed too.

    There are two main way to parameterize the simplex for optimization.

      1. No reparameterization: 0 < W < 1, with constrained optimization.
      2. Softmax: the columns of W are optimized over all reals,
          and mapped to the simplex space with softmax.

    Initially we tried no reparameterization, but this was slow and
    fragile. Softmax seems to work quite well so is the implementation
    here.
    """
    def __init__(self, Y, w, t, logB, bins=None,
                 features=None, 
                 log10_pi0_bounds=PI0_BOUNDS,
                 log10_mu_bounds=MU_BOUNDS):
        super().__init__(Y=Y, w=w, t=t, logB=logB,
                         bins=bins, features=features,
                         log10_pi0_bounds=log10_pi0_bounds,
                         log10_mu_bounds=log10_mu_bounds)
        # fit the exponential over the grid of mutation points
        self._fit_B_curve()
        # set all model estimates, etc. to None
        self._reset()

    def _fit_B_curve(self):
        self.logB_fit = fit_B_curve_params(self.logB, self.w)

    def _reset(self):
        # this is filled in with later model fits
        self.theta_ = None
        self.nll_ = None
        self.jack_fits_ = None
        self.jack_nlls_ = None
        self.jack_thetas_ = None
        self.jack_indices = None
        self.jack_r2s = None
        self.loo_fits_ = None
        self.loo_nlls_ = None
        self.loo_thetas_ = None
        self.loo_chroms_ = None
        self.sigma_ = None
        self.sigma_n_ = None
        self.std_error_type = None

    @staticmethod
    def from_data(file, use_classic_B=False, **kwargs):
        """
        Load a model data pickle file, e.g. from the command line
        tool bgspy data.
        """
        dat = load_pickle(file)
        Y, bins = dat['Y'], dat['bins']
        features = dat['features']
        if not use_classic_B:
            b = dat['bp']
        else:
            try:
                b = dat['b']
            except KeyError:
                msg = f"B is not computed in {file}, but use_classic_B=True"
                raise KeyError(msg)
        w, t = dat['w'], dat['t']
        obj = SimplexModel(w=w, t=t, logB=b, Y=Y,
                           bins=bins, features=features,
                           **kwargs)
        if 'md' in dat:
            # package metadata, e.g. from sims
            obj.metadata = dat['md']
        return obj

    def param_dict(self):
        """
        Return parameters in a dictionary with their named keys.
        """
        pn = param_names(self.t, self.features, self._fixed_mu)
        return dict(zip(pn, self.theta_.tolist()))

    def random_start(self, mu=None):
        """
        Random starts, on a linear scale for μ and π0.
        """
        start = random_start_simplex(self.nt, self.nf, mu=mu,
                                     log10_pi0_bounds=self.log10_pi0_bounds,
                                     log10_mu_bounds=self.log10_mu_bounds)
        return start

    def bounds(self, paired=True, global_softmax_bound=False, mu=None):
        """
        By default, we assume scipy minimize. Scipy expects zipped
        lists. For nlopt, (lower, upper) bounds are needed, so set 
        paired = False. For global nlopt routines, we need fixed
        bounds for softmax; we use 1e4 since these shouldn't get 
        too large, but as a warning, note that this can cause 
        optimization issues.
        """
        return bounds_simplex(self.nt, self.nf,
                              log10_pi0_bounds=self.log10_pi0_bounds,
                              log10_mu_bounds=self.log10_mu_bounds,
                              mu=mu,
                              bounded_softmax=global_softmax_bound,
                              paired=paired)

    def fit(self, starts=1, ncores=None,
            mu=None, chrom=None, 
            progress=True, _indices=None,
            engine='nlopt', method='LN_BOBYQA'):
        """
        Fit likelihood models with numeric optimization (using nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        chrom: only fit specified chromosome.
        _indices: indices to include in fit; this is primarily internal, for
                  bootstrapping
        """
        # the model is being re-fit; we need to 
        # clear out any older state from jackknife stuff, etc
        # as it's no longer valid
        self._reset()
        mu_is_fixed = mu is not None
        self._fixed_mu = mu
        WORKERS = {'nlopt': nlopt_softmax_worker, 
                   'scipy': scipy_softmax_worker,
                   'fixed': nlopt_softmax_fixedmu_worker}
        NEGLL_FUNCS = {'nlopt': negll_softmax_nlopt,
                       'fixed': negll_softmax_fixedmu_nlopt,
                       'scipy': negll_softmax}
        if isinstance(starts, int):
            starts = [self.random_start(mu=mu) for _ in range(starts)]

        if ncores is not None:
            # don't request more cores than we need
            ncores = min(len(starts), ncores)

        Y = self.Y
        assert self.logB_fit is not None
        logB = self.logB_fit

        # if we have pre-specified indices, we're doing
        # jackknife or leaving a chrosome out
        resample = _indices is not None
        if resample:
            msg = "currently only whole-genome jackknife supported"
            assert chrom is None, msg
            Y = Y[_indices, ...]
            logB = logB[_indices, ...]

        # chromosome-specific fit
        if chrom is not None:
            idx = self.bins.chrom_indices(chrom)
            Y = Y[idx, ...]
            logB = logB[idx, ...]

        # set up optimization function closure, and the 
        # optimization worker.
        if mu_is_fixed:
            # if μ is fixed, we used a special fixed nlopt routine
            engine = 'fixed'
        negll_func = NEGLL_FUNCS[engine]

        if mu_is_fixed:
            nll_func = partial(negll_func, Y=Y, B=logB,
                               w=self.w, mu=mu)
        else:
            nll_func = partial(negll_func, Y=Y, B=logB, w=self.w)
        worker_func = WORKERS[engine]

        is_nlopt = 'nlopt' == engine or mu_is_fixed
        is_nlopt_global = method.startswith('GN_') and is_nlopt
        bounds = self.bounds(paired=not is_nlopt, mu=mu,
                             global_softmax_bound=is_nlopt_global)
        worker = partial(worker_func, 
                         func=nll_func,
                         bounds=bounds,
                         nt=self.nt, nf=self.nf, method=method)

        # run the optimization routine on muliple starts
        res = run_optims(worker, starts, ncores=ncores,
                         progress=progress)

        if resample:
            # We don't clobber the existing results since 
            # this ins't a fit, so we return copy this object and 
            # load them there.
            obj = copy(self)
            obj._load_optim(res)
            obj._indices_fit = _indices
            return obj

        if chrom is not None:
            obj = copy(self)
            obj._load_optim(res)
            obj._chrom_fit = chrom
            return obj

        self._load_optim(res)
        return res

    @property
    def mle_mu(self):
        """
        Extract out the mutation rate, μ.
        """
        if self._fixed_mu is not None:
            warnings.warn("mu is fixed! returning fixed value")
            return self._fixed_mu
        return self.theta_[1]

    @property
    def mle_W(self):
        """
        Extract out the W matrix.
        """
        # if we have a free μ, we change the offset
        i = 1 + int(self._fixed_mu is None)
        return self.theta_[i:].reshape(self.nt, self.nf)

    def summary(self, index=None):
        if self.theta_ is None:
            return "\n[not fit yet]\n"  # not fit yet
        if index is None:
            pi0 = self.mle_pi0  # FIX TODO (to handle fixed, low priority)
            mu = self.mle_mu
            W = self.mle_W
            R2 = self.R2()
        else:
            theta = self.optim.thetas[index, ...]
            pi0 = theta[0]
            mu = theta[1] if self._fixed_mu is None else self._fixed_mu
            W = theta[2:]
            R2 = self.R2(theta=theta)

        cis = None
        if hasattr(self, 'sigma_') and self.sigma_ is not None:
            cis = self.sigma_ci()

        Ne = pi0 / (4*mu)
        if self.theta_ is None:
            return "no model fit."
        base_rows = ""
        base_rows += "\n\nML estimates:"
        if index is not None:
            base_rows += f"WARNING: for non-MLE (optimization {index})"
            base_rows += ", interpret CIs at your own risk!"
        # hasattr used for back-compatability; can be removed in future
        if hasattr(self, '_chrom_fit') and self._chrom_fit is not None:
            base_rows += f" (chromosome {self._chrom_fit} only)\n"
        elif hasattr(self, '_chrom_loo_fit') and self._chrom_loo_fit is not None:
            base_rows += f" (chromosome {self._chrom_loo_fit} left out)\n"
        elif hasattr(self, '_indices_fit') and self._indices_fit is not None:
            perc = np.round(100*len(self._indices_fit) / self.Y.shape[0], 2)
            base_rows += f" (at pre-spcified indices {perc}% of total)\n"
        else:
            base_rows += f" (whole genome)\n"

        # jackknife info
        if hasattr(self, 'jack_thetas_') and self.jack_thetas_ is not None:
            base_rows += f"number jackknife samples: {self.jack_thetas_.shape[0]}\n"

        base_rows += f"standard error method: {self.std_error_type}\n"
        base_rows += f"negative log-likelihood: {self.nll_}\n"
        nstarts = self.optim.thetas.shape[0]
        frac = np.round(self.optim.frac_success, 2)*100
        base_rows += f"number of successful starts: {nstarts} ({frac}% total)\n"
        base_rows += f"π0 = {pi0:0.6g}"
        if cis is not None:
            l, u = cis[0][0], cis[1][0]
            base_rows += f" ({l:.5f}, {u:.5f})\n"
        else:
            base_rows += "\n"
        pi = pi_from_pairwise_summaries(self.Y.sum(axis=0))
        base_rows += f"π  = {pi:0.6g}\n"
        fixed_str = "(FIXED)" if self._fixed_mu is not None else ""
        base_rows += f"μ_del  = {mu:0.4g} {fixed_str}"
        if cis:
            l, u = cis[0][1], cis[1][1]
            base_rows += f" ({l:.3g}, {u:.3g})\n"
        else:
            base_rows += "\n"

        #base_rows += f"Ne (del) = {int(Ne):,} (implied from π0 and μ)\n"
        base_rows += f"Ne = {int(pi0 / (4 * 1e-8)):,} (if μ=1e-8), "
        base_rows += f"Ne = {int(pi0 / (4 * 2e-8)):,} (if μ=2e-8)\n"
        base_rows += f"R² = {np.round(100*R2, 4)}% (in-sample)"
        if hasattr(self, 'loo_thetas_') and self.loo_thetas_ is not None:
            loo_R2 = self.loo_R2()
            base_rows += f"  {np.round(100*loo_R2, 4)}% (out-sample)"
        base_rows += "\n"

        base_rows += "W = \n"
        base_rows += W_summary(W, self.features, 
                               self.nt, self.nf, self.t, cis)
        return base_rows

    def __repr__(self):
        base_rows = super().__repr__()
        base_rows += self.summary()
        return base_rows

    def sigma_ci(self, factor=1):
        """
        """
        theta = self.theta_
        sigma = self.sigma_
        lower, upper = theta - factor*sigma, theta + factor*sigma
        return lower, upper

    def predict(self, optim=None, theta=None, B=False):
        """
        Predicted π from the best fit (if optim = None). If optim is 'random', a
        random MLE optimization is chosen (e.g. to get a senes of how much
        variation there is across optimization). If optim is an integer,
        this rank of optimization results is given (e.g. optim = 0 is the
        best MLE).
        """
        logB_fit = self.logB_fit
        if theta is not None:
            return predict_simplex(theta, logB_fit, self.w,
                                   mu=self._fixed_mu)
        if optim is None:
            theta = np.copy(self.theta_)
        else:
            # predict off a certain optimization result
            # this is deprecated, from before the optimization became
            # much more stable. It still me be useful later though 
            # for debugging.
            thetas = self.optim.thetas_
            if optim == 'random':
                theta = np.copy(thetas[np.random.randint(0, thetas.shape[0]), :])
            else:
                theta = np.copy(thetas[optim])
        if B:
            # rescale so B is returned, π0 = 1
            theta[0] = 1.
        return predict_simplex(theta, logB_fit, self.w,
                               mu=self._fixed_mu)


# -------- older likelihood functions -----------
# these are primarily legacy but used for testing


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

def negll_c(theta, Y, logB, w, two_alleles=False,
            version=3):
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
    negloglik_argtypes = (POINTER(c_double), POINTER(c_double),
                          POINTER(c_double), POINTER(c_double),
                          POINTER(c_double),
                          # weird type for dims/strides
                          POINTER(np.ctypeslib.c_intp),
                          POINTER(np.ctypeslib.c_intp),
                          c_int)
    negloglik_restype = c_double

    likclib.negloglik.argtypes = negloglik_argtypes
    likclib.negloglik.restype = negloglik_restype

    likclib.negloglik2.argtypes = negloglik_argtypes
    likclib.negloglik2.restype = negloglik_restype

    likclib.negloglik3.argtypes = negloglik_argtypes
    likclib.negloglik3.restype = negloglik_restype

    args = (theta_ptr, nS_ptr, nD_ptr, logB_ptr, w_ptr,
            logB.ctypes.shape, logB.ctypes.strides,
            int(two_alleles))

    if version == 1:
        return likclib.negloglik(*args)
    elif version == 2:
        return likclib.negloglik2(*args)
    elif version == 3:
        return likclib.negloglik3(*args)
    else:
        raise ValueError("unsupported version")




