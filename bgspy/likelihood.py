## likelihood.py -- functions for likelihood stuff
import os
import copy
import pickle
import itertools
import tqdm
from scipy.special import softmax
from scipy.optimize import curve_fit
from tabulate import tabulate
from functools import partial
import numpy as np
from ctypes import POINTER, c_double, c_ssize_t, c_int

# no longer needed
# HAS_JAX = False
# try:
#     import jax.numpy as jnp
#     HAS_JAX = True
# except ImportError:
#     pass

from scipy import interpolate
from bgspy.utils import signif, load_pickle
from bgspy.data import pi_from_pairwise_summaries, GenomicBinnedData
from bgspy.optim import run_optims, scipy_softmax_worker
from bgspy.plots import model_diagnostic_plots, predict_chrom_plot
from bgspy.plots import resid_fitted_plot, get_figax
from bgspy.plots import chrom_resid_plot
from bgspy.bootstrap import process_bootstraps, pivot_ci, percentile_ci

# load the library (relative to this file in src/)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
likclib = np.ctypeslib.load_library("likclib", LIBRARY_PATH)

#likclib = np.ctypeslib.load_library('lik', 'bgspy.src.__file__')

# mutation rate hard bounds
# these are set for human. I'm targetting the uncertainty 
# with the mutational slowdown, e.g. see Moorjani et al 2016
MU_BOUNDS = tuple(np.log10((0.9e-8,  5e-8)))


# π0 bounds: lower is set by lowest observed π in humans
# highest is based on B lowest is 0.02, e.g. π = π0 Β,
# 1e-4 = 0.005 B
PI0_BOUNDS = tuple(np.log10((0.0005, 0.005)))  # this is fairly permissive
# see https://twitter.com/jkpritch/status/1600296856999047168/photo/1

# -------- random utility/convenience functions -----------

def param_names(t, features):
    """
    Label the simplex model parameters.
    """
    sels, feats = np.meshgrid(np.log10(t), features)
    sels, feats = itertools.chain(*sels.tolist()), itertools.chain(*feats.tolist())
    return ["pi0", "mu"] + [f"W[{int(s)},{f}]" for s, f in zip(*(sels, feats))]

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


def check_bounds(x, lb, ub):
    assert np.all((x >= lb) & (x <= ub))


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
    for i in tqdm.tqdm(range(nx)):
        for j in range(nt):
            for k in range(nf):
                popt, pcov = curve_fit(mut_curve, w, np.exp(b[i, :, j, k]))
                assert(len(popt) == 1)
                params[i, 0, j, k] = popt[0]
    return params

def negll_softmax(theta, Y, B, w):
    """
    Softmax version of the simplex model wrapper for negll_c().

    For scipy, i.e. no grad argument.
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


def negll_simplex(theta, Y, B, w):
    """
    Simplex model wrapper for negll_c().
    """
    return negll_c(theta, Y, B, w, version=3)


def bounds_simplex(nt, nf, log10_pi0_bounds=PI0_BOUNDS,
                   log10_mu_bounds=MU_BOUNDS,
                   softmax=False, bounded_softmax=False,
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


def random_start_simplex(nt, nf, pi0=None, mu=None,
                         log10_pi0_bounds=PI0_BOUNDS,
                         log10_mu_bounds=MU_BOUNDS, 
                         softmax=True):
    """
    Create a random start position, uniform over the bounds for π0
    and μ, and Dirichlet under the DFE weights for W, under the simplex model.
    """
    if pi0 is None:
        pi0 = np.random.uniform(10**log10_pi0_bounds[0], 
                                10**log10_pi0_bounds[1], 1)
    if mu is None:
        mu = np.random.uniform(10**log10_mu_bounds[0],
                               10**log10_mu_bounds[1], 1)

    nparams = nt*nf + 2
    theta = np.empty(nparams)
    if softmax:
        theta[0] = pi0
        theta[1] = mu
        theta[2:] = np.random.normal(0, 1, nt*nf)
        return theta

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


def predict_simplex(theta, logB, w, mu=None):
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
    # interpolate B(w)'s
    logBw = np.zeros(nx, dtype=float)
    for i in range(nx):
        for j in range(nt):
            for k in range(nf):
                logBw[i] += np.interp(mu*W[j, k], w, logB[i, :, j, k])
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
                 log10_pi0_bounds=PI0_BOUNDS, 
                 log10_mu_bounds=MU_BOUNDS):
        self.w = w
        self.t = t
        if bins is not None:
            # TODO: this is disabled for now because it throws an error with
            # notebooks that's uncessary (even when bins really is GenomicBinnedData)
            assert isinstance(bins, GenomicBinnedData)
            assert bins.nbins() == Y.shape[0]
        self.bins = bins
        self.log10_pi0_bounds = log10_pi0_bounds
        # the bounds for mu alone
        self.log10_mu_bounds = log10_mu_bounds

        try:
            assert logB.ndim == 4
            #assert logB.shape[1] == w.size # TODO FIX
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
 

    def load_bootstraps(self, nlls, thetas):
        """
        Load the bootstrap results by setting attributes.
        """
        self.boot_nlls_ = nlls
        self.boot_thetas_ = thetas

    def bootstrap(self, nboot, blocksize, **kwargs):
        """
        Resample bin indices and fit each time.
        """
        msg = "bins attribute must be set to a GenomicBinnedData object"
        assert self.bins is not None, msg
        res = []
        for b in tqdm.trange(nboot):
            idx = self.bins.resample_blocks(blocksize=blocksize,
                                            filter_masked=True)
            res.append(self.fit(**kwargs, _indices=idx))
        nlls, thetas = process_bootstraps(res)
        self.boot_nlls_ = nlls
        self.boot_thetas_ = thetas
        return nlls, thetas

    def jackknife(self, njack=None, chunks=None, **kwargs):
        """
        Do the jackknife.

	If you want to compute across a cluster, set the chunks tuple,
        (nchunks, chunk_id)

        If you want to recycle the MLE θ, pass a starts keyword argument
        with the MLE vector (optionally repeated for multiple starts.
        """
        msg = "bins attribute must be set to a GenomicBinnedData object"
        assert self.bins is not None, msg
        res = []
        idx = np.arange(self.Y.shape[0])
        if chunks is None:
            # what samples are we going to remove?
            jack_idx = np.random.choice(idx, njack, replace=False)
        else:
            rng = np.arange(*remove_range)
            if njack is not None:
                njack = len(rng)
            else:
                assert njack <= len(rng), "len(njack) must be <= len(range)"
            jack_idx = np.random.choice(rng, njack, replace=False)

        for j in tqdm.tqdm(jack_idx):
            jidx = list(set(idx).difference(jack_idx))
            res.append(self.fit(**kwargs, _indices=jidx))

        nlls, thetas = process_bootstraps(res)
        self.boot_nlls_ = nlls
        self.boot_thetas_ = thetas
        return nlls, thetas

    def ci(self, method='quantile'):
        assert self.boot_thetas_ is not None, "bootstrap() has not been run"
        if method == 'quantile':
            lower, upper = pivot_ci(self.boot_thetas_, self.theta_)
        elif method == 'percentile':
            lower, upper = percentile_ci(self.boot_thetas_)
        else:
            raise ValueError("improper bootstrap method")
        return np.stack((lower, self.theta_, upper)).T

    def save(self, filename):
        """
         Pickle this object.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename):
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

    def loo_chrom_R2(self, out_sample_chrom=None, loo_fits_dir=None, **fit_kwargs):
        """
        Estimate R2 by leave-one-out of a whole chromosome, run a new fit,
        and then estimate of R2 of the out sample chromosome.

        If out_sample_chrom is None, this loops over all chromosomes. However,
        to accomodate running things in parallel on a cluster, out_sample_chrom can
        bet set manually.

        If fits_dir is specified, this will write each LOO fit to this directory.

        The ML estimate can be recycled via the **fit_kwargs.
        """
        if out_sample_chrom is None:
            all_chroms = self.bins.bins().keys()
        else:
            assert out_sample_chrom in self.bins.keys(), f"LOO chrom {chrom} not in bins!"
            all_chroms = [out_sample_chrom]

        r2s = []
        for chrom in all_chroms:
            in_sample = self.bins.chrom_indices(chrom, exclude=True)
            out_sample = self.bins.chrom_indices(chrom, exclude=False)
            print(f"fitting, leaving out {chrom}")
            fit_optim = self.fit(**fit_kwargs, _indices=in_sample)
            # mimic a new fit
            new_fit = copy.copy(self)
            new_fit._load_optim(fit_optim)
            r2s.append(new_fit.R2(_idx=out_sample))
            # monkey patch in some idx
            new_fit._out_sample = out_sample
            new_fit._in_sample = in_sample
            if loo_fits_dir is not None:
                if not os.path.exists(loo_fits_dir):
                    os.makedirs(loo_fits_dir)
                fpath = os.path.join(loo_fits_dir, f"mle_loo_{chrom}.pkl")
                new_fit.save(fpath)
        return np.array(r2s)

    def dfe_plot(self, figax=None):
        """
        TODO
        """
        fig, ax = get_figax(figax)
        xt = np.log10(self.t)

        nf = self.nf
        pad = 0.03
        w = 1/nf - pad  # width of bars, for each feature with allowance
        hw = w/2
        for i in range(nf):
            feat = self.features[i]
            ax.bar(xt - 1/(nf/2) + i/nf, self.mle_W[:, i], align='edge', width=w, label=feat)
        ax.set_xticks(np.log10(self.t), [f"$10^{{{int(x)}}}$" for x in xt])
        ax.set_ylabel('probability')
        ax.legend()

    def R2(self, _idx=None, **kwargs):
        """
        The R² value of the predictions against actual results.

        _idx: the indices of values to use, e.g. for cross-validation or LOO
              bootstrapping.
        """
        pred_pi = self.predict(**kwargs)
        pi = pi_from_pairwise_summaries(self.Y)
        if _idx is None:
            return R2(pred_pi, pi)
        return R2(pred_pi[_idx], pi[_idx])

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

    def predict_plot(self, chrom, ratio=True, label='prediction', figax=None):
        return predict_chrom_plot(self, chrom, ratio=ratio,
                                  label=label, figax=figax)

    def scatter_plot(self, figax=None, **scatter_kwargs):
        fig, ax = get_figax(figax)
        pred_pi = self.predict()
        pi = pi_from_pairwise_summaries(self.Y)
        ax.scatter(pi, pred_pi)

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
        rows = [f"MLE (interpolated w): {self.nw} x {self.nt} x {self.nf}"]
        rows.append(f"  w grid: {signif(self.w)} (before interpolation)")
        rows.append(f"  t grid: {signif(self.t)}")
        return "\n".join(rows)


class SimplexModel(BGSLikelihood):
    """
    BGSLikelihood Model with the DFE matrix W on a simplex, free 
    π0 and free μ (within bounds).

    There are two main way to parameterize the simplex for optimization.

      1. No reparameterization: 0 < W < 1, with constrained optimization.
      2. Softmax: the columns of W are optimized over all reals,
          and mapped to the simplex space with softmax.
    
    Initially we tried no reparameterization, but this was slow and 
    fragile. Softmax seems to work quite well.
    """
    def __init__(self, Y, w, t, logB, bins=None,
                 features=None, 
                 log10_pi0_bounds=PI0_BOUNDS,
                 log10_mu_bounds=MU_BOUNDS):
        super().__init__(Y=Y, w=w, t=t, logB=logB,
                         bins=bins, features=features,
                         log10_pi0_bounds=log10_pi0_bounds,
                         log10_mu_bounds=log10_mu_bounds)
        self.start_pi0_ = None
        self.start_mu_ = None
        # fit the exponential over the grid of mutation points
        self.logB_fit = fit_B_curve_params(logB, w)

    def random_start(self, pi0=None, mu=None):
        """
        Random starts, on a linear scale for μ and π0.
        """
        start = random_start_simplex(self.nt, self.nf,
                                     pi0=pi0, mu=mu,
                                     log10_pi0_bounds=self.log10_pi0_bounds,
                                     log10_mu_bounds=self.log10_mu_bounds)
        if pi0 is not None:
            self.start_pi0_ = pi0
        if mu is not None:
            self.start_mu_ = mu
        return start

    def bounds(self):
        return bounds_simplex(self.nt, self.nf,
                              self.log10_pi0_bounds,
                              self.log10_mu_bounds,
                              softmax=True,
                              global_softmax_bound=False,
                              paired=True)

    def fit(self, starts=1, ncores=None, 
            start_pi0=None, start_mu=None,
            chrom=None, progress=True, _indices=None):
        """
        Fit likelihood models with numeric optimization (using nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        start_pi0/start_mu: starting values for pi0 and mu.
        _indices: indices to include in fit; this is primarily internal, for
                  bootstrapping
        """
        if isinstance(starts, int):
            starts = [self.random_start(pi0=start_pi0, mu=start_mu) for _
                      in range(starts)]

        if ncores is not None:
            # don't request more cores than we need
            ncores = min(len(starts), ncores)

        Y = self.Y
        assert self.logB_fit is not None
        logB = self.logB_fit
         
        # if we have pre-specified indices, we're doing 
        # jackknife or bootstrap
        bootstrap = _indices is not None
        if bootstrap:
            msg = "currently only whole-genome bootstrap/jackknife supported"
            assert chrom is None, msg
            Y = Y[_indices, ...]
            logB = logB[_indices, ...]

        if chrom is not None:
            idx = self.bins.chrom_indices(chrom)
            Y = Y[idx, ...]
            logB = logB[idx, ...]

        nll_func = partial(negll_softmax, Y=Y, B=logB, w=self.w)

        worker = partial(scipy_softmax_worker, 
                         func=nll_func, 
                         bounds=self.bounds(), 
                         nt=self.nt, nf=self.nf, method='L-BFGS-B')

        # run the optimization routine on muliple starts
        res = run_optims(worker, starts, ncores=ncores, 
                         progress=progress)

        if bootstrap:
            # we don't clobber the existing results since this ins't a fit.
            return res

        if chrom is not None:
            obj = copy.copy(self)
            obj._load_optim(res)
            obj._chrom_fit = chrom
            return obj

        self._load_optim(res)

    @property
    def mle_mu(self):
        """
        Extract out the mutation rate, μ.
        """
        return self.theta_[1]

    @property
    def mle_W(self):
        """
        Extract out the W matrix.
        """
        return self.theta_[2:].reshape(self.nt, self.nf)
    
    def summary(self, index=None):
        if self.theta_ is None:
            return "no model fit."
        base_rows = ""
        base_rows += "\n\nSimplex model ML estimates:"
        if hasattr(self, '_chrom_fit') and self._chrom_fit is not None:
            base_rows += f" (chromosome {self._chrom_fit} only)\n"
        else:
            base_rows += f" (whole genome)\n"
        base_rows += f"negative log-likelihood: {self.nll_}\n"
        base_rows += f"π0 = {self.mle_pi0}\n"
        base_rows += f"μ = {self.mle_mu}\n"
        base_rows += f"R² = {np.round(100*self.R2(), 4)}\n"
        header = ()
        if self.features is not None:
            header = [''] + self.features

        W = self.mle_W.reshape((self.nt, self.nf))
        tab = np.concatenate((self.t[:, None], np.round(W, 3)), axis=1)
        base_rows += "W = \n" + tabulate(tab, headers=header)
        return base_rows

    def __repr__(self):
        base_rows = super().__repr__()
        base_rows += self.summary()
        return base_rows

    def predict(self, optim=None, theta=None, B=False):
        """
        Predicted π from the best fit (if optim = None). If optim is 'random', a
        random MLE optimization is chosen (e.g. to get a senes of how much
        variation there is across optimization). If optim is an integer,
        this rank of optimization results is given (e.g. optim = 0 is the
        best MLE).
        """
        if theta is not None:
            return predict_simplex(theta, self.logB, self.w)
        if optim is None:
            theta = np.copy(self.theta_)
        else:
            thetas = self.optim.thetas_
            if optim == 'random':
                theta = np.copy(thetas[np.random.randint(0, thetas.shape[0]), :])
            else:
                theta = np.copy(thetas[optim])
        if B:
            # rescale so B is returned, π0 = 1
            theta[0] = 1.
        return predict_simplex(theta, self.logB, self.w)


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

