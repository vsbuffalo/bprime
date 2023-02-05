## likelihood.py -- functions for likelihood stuff
import os
import copy
import pickle
import warnings
import tqdm.autonotebook as tqdm
#import tqdm.notebook as tqdm
from scipy.special import softmax
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
from scipy.optimize import minimize
from bgspy.genome import Genome
from bgspy.data import GenomeData
from bgspy.utils import signif, load_seqlens, load_pickle
from bgspy.data import pi_from_pairwise_summaries, GenomicBinnedData
from bgspy.optim import run_optims, nlopt_mutation_worker, nlopt_simplex_worker, nlopt_softmax_worker
from bgspy.plots import model_diagnostic_plots, predict_chrom_plot
from bgspy.plots import resid_fitted_plot, get_figax
from bgspy.plots import chrom_resid_plot
from bgspy.bootstrap import process_bootstraps, pivot_ci, percentile_ci
from bgspy.models import BGSModel
from bgspy.sim_utils import mutate_simulated_tree


# load the library (relative to this file in src/)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
likclib = np.ctypeslib.load_library("likclib", LIBRARY_PATH)

#likclib = np.ctypeslib.load_library('lik', 'bgspy.src.__file__')
AVOID_CHRS = set(('M', 'chrM', 'chrX', 'chrY', 'Y', 'X'))

# mutation rate hard bounds
# these are set for human. I'm targetting the uncertainty 
# with the mutational slowdown, e.g. see Moorjani et al 2016
MU_BOUNDS = tuple(np.log10((0.9e-8,  5e-8)))


# π0 bounds: lower is set by lowest observed π in humans
# highest is based on B lowest is 0.02, e.g. π = π0 Β,
# 1e-4 = 0.005 B
PI0_BOUNDS = tuple(np.log10((0.0005, 0.005)))  # this is fairly permissive
# see https://twitter.com/jkpritch/status/1600296856999047168/photo/1

def fit_likelihood(seqlens_file=None, recmap_file=None, counts_dir=None,
                   sim_tree_file=None, sim_mu=None,
                   neut_file=None, access_file=None, fasta_file=None,
                   bs_file=None,
                   model='free',
                   chrom=None,
                   mu=None,
                   fit_outfile=None,
                   ncores=70,
                   nstarts=200,
                   loo_nstarts=100,
                   loo_chrom=False,
                   loo_fits_dir=None,
                   save_data=None,
                   premodel_data=None,
                   only_save_data=True,
                   window=1_000_000, outliers=(0.0, 0.995),
                   recycle_mle=False,
                   bp_only=False,
                   only_autos=True,
                   B=None, blocksize=20,
                   J=None,
                   r2_file=None,
                   bootjack_outfile=None,
                   name=None,
                   fit_file=None,
                   thresh_cM=0.3, verbose=True):
    """
    Wrapper function for a standard BGS analysis. This also allow for one to
    take a fit object, and use it for bootstrapping across a cluster.

    Many of the defaults are tailored for my human genetics data (sorry, use
    at your own caution for other organisms).

    Note: when both B and B' are pickled as a tuple, it is always in this order.

    name: if None, inferred from seqlens file, e.g. '<name>_seqlens.tsv'
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    already_fit = fit_file is not None
    if not already_fit:
        #  --------- load data, build summary statistics ----------
        if not premodel_data:
            # infer the genome name if not supplies
            name = seqlens_file.replace('_seqlens.tsv', '') if name is None else name
            seqlens = load_seqlens(seqlens_file)
            if chrom is not None:
                # only include this one chromosome
                seqlens = {c: l for c, l in seqlens.items() if c == chrom}
            if only_autos:
                seqlens = {c: l for c, l in seqlens.items() if c not in AVOID_CHRS}
            vprint("-- loading genome --")
            g = Genome(name, seqlens=seqlens)
            g.load_recmap(recmap_file)
            gd = GenomeData(g)
            is_sim = sim_tree_file is not None
            if counts_dir is not None:
                assert not is_sim, "set either counts directory or sim tree file"
                gd.load_counts_dir(counts_dir)
            else:
                assert counts_dir is None, "set either counts directory or sim tree file"
                assert chrom is not None, "chrom needs to be specified when loading from a treeseq"
                assert sim_mu is not None, "set a mutation rate to turn treeseqs to counts"
                ts = mutate_simulated_tree(sim_tree_file, rate=sim_mu)
                gd.load_counts_from_ts(ts, chrom=chrom)


            gd.load_neutral_masks(neut_file)
            gd.load_accessibile_masks(access_file)
            gd.load_fasta(fasta_file, soft_mask=True)
            gd.trim_ends(thresh_cM=thresh_cM)

            # bin the diversity data
            vprint("-- binning pairwise diversity --")
            bgs_bins = gd.bin_pairwise_summaries(window,
                                                 filter_accessible=True,
                                                 filter_neutral=True)

            del gd  # we don't need it after this, it's summarized
            # mask bins that are outliers
            bgs_bins.mask_outliers(outliers)

            # genome models
            vprint("-- loading Bs --")
            gm = BGSModel.load(bs_file)

            # features -- load for labels
            features = list(gm.segments.feature_map.keys())

            # handle if we're doing B too or just B'
            m_b = None
            if gm.Bs is None:
                warnings.warn(f"BGSModel.Bs is not set, so not fitting classic Bs (this is likely okay.)")
                bp_only = True  # this has to be true now, no B

            # bin Bs
            if not bp_only:
                vprint("-- binning B --")
                b = bgs_bins.bin_Bs(gm.BScores)

            # B' is always fit
            vprint("-- binning B' --")
            bp = bgs_bins.bin_Bs(gm.BpScores)

            # get the diversity data
            vprint("-- making diversity matrix Y --")
            Y = bgs_bins.Y()

            # save testing data at this point if specified
            if save_data is not None:
                vprint("-- saving pre-fit model data --")
                with open(save_data, 'wb') as f:
                    dat = {'bgs_bins': bgs_bins, 'Y': Y, 
                           'bp': bp, 'gm': gm, 'features': features}
                    if not bp_only:
                        dat['b'] = b
                    pickle.dump(dat, f)
                if only_save_data:
                    return
        else:
            # we have pre-model data already crunched we can load!
            dat = load_pickle(premodel_data)
            bgs_bins, Y, bp, gm = (dat['bgs_bins'], dat['Y'], 
                                   dat['bp'], dat['gm'])
            features = dat['features']
            b = dat.get('b', None)

        #  --------- fit the model ----------
        if model == 'simplex':
            # fit the simplex model
            if not bp_only:
                vprint("-- fitting B simplex model --")
                m_b = SimplexModel(w=gm.w, t=gm.t, logB=b, Y=Y,
                                    bins=bgs_bins, features=features)
                m_b.fit(starts=nstarts, ncores=ncores, algo='ISRES')

            # now to the B'
            vprint("-- fitting B' simplex model --")
            m_bp = SimplexModel(w=gm.w, t=gm.t, logB=bp, Y=Y,
                                bins=bgs_bins, features=features)
            m_bp.fit(starts=nstarts, ncores=ncores, algo='ISRES')
        elif model == 'fixed':
            assert isinstance(mu, float), "mu must be a float if model='fixed'"
            # fit the simplex model
            if not bp_only:
                vprint("-- fitting B fixed model --")
                m_b = FixedMutationModel(w=gm.w, t=gm.t, logB=b, Y=Y,
                                         bins=bgs_bins, features=features)
                m_b.fit(starts=nstarts, mu=mu, ncores=ncores, algo='ISRES')

            # now to the B'
            vprint("-- fitting B' fixed model --")
            m_bp = FixedMutationModel(w=gm.w, t=gm.t, logB=bp, Y=Y,
                                      bins=bgs_bins, features=features)
            m_bp.fit(starts=nstarts, mu=mu, ncores=ncores, algo='ISRES')
        else:
            # free mutation / default
            # fit the simplex model
            if not bp_only:
                vprint("-- fitting B free model --")
                m_b = FreeMutationModel(w=gm.w, t=gm.t, logB=b, Y=Y,
                                        bins=bgs_bins, features=features)
                m_b.fit(starts=nstarts, ncores=ncores, algo='ISRES')

            # now to the B'
            vprint("-- fitting B' free model --")
            m_bp = FreeMutationModel(w=gm.w, t=gm.t, logB=bp, Y=Y,
                                     bins=bgs_bins, features=features)
            m_bp.fit(starts=nstarts, ncores=ncores, algo='ISRES')


        # save the fitted model
        obj = {'m_b': m_b, 'm_bp': m_bp}
        with open(fit_outfile, 'wb') as f:
            pickle.dump(obj, f)
        vprint("-- model saved --")

    else:
        # model already fit, we load it for downstream stuff 
        with open(fit_file, 'rb') as f:
            obj = pickle.load(f)
            m_b, m_bp = obj.get('m_b', None), obj['m_bp']  # always expect B'


    #  --------- bootstrap ----------
    bootstrap = B is not None
    msg = "cannot do both bootstrap and jackknife"
    if bootstrap:
        assert J is None, msg
        vprint('note: recycling MLE for bootstrap')
        starts_b = nstarts if not recycle_mle else [m_b.theta_] * nstarts
        starts_bp = nstarts if not recycle_mle else [m_bp.theta_] * nstarts

        nlls_b, thetas_b = None, None # in case only-Bp
        if not bp_only:
            print("-- bootstrapping B --")
            nlls_b, thetas_b = m_b.bootstrap(nboot=B, blocksize=blocksize,
                                             starts=starts_b, ncores=ncores)
        print("-- bootstrapping B' --")
        nlls_bp, thetas_bp = m_bp.bootstrap(nboot=B, blocksize=blocksize,
                                            starts=starts_bp, ncores=ncores)
        if bootjack_outfile is not None:
            np.savez(bootjack_outfile, 
                     nlls_b=nlls_b, thetas_b=thetas_b,
                     nlls_bp=nlls_bp, thetas_bp=thetas_bp)
            print("-- bootstrapping results saved --")
            return

    #  --------- jackknife ----------
    jackknife = J is not None
    if jackknife:
        assert B is None, msg
        vprint('note: recycling MLE for jackknife')
        starts_b = nstarts if not recycle_mle else [m_b.theta_] * nstarts
        starts_bp = nstarts if not recycle_mle else [m_bp.theta_] * nstarts

        nlls_b, thetas_b = None, None  # in case only-Bp
        if not bp_only:
            print("-- jackknife B --")
            nlls_b, thetas_b = m_b.jackknife(njack=J, 
                                             starts=starts_b, ncores=ncores)
        print("-- jackknife B' --")
        nlls_bp, thetas_bp = m_bp.jackknife(njack=J,
                                            starts=starts_bp, ncores=ncores)
        if bootjack_outfile is not None:
            np.savez(bootjack_outfile, 
                     nlls_b=nlls_b, thetas_b=thetas_b,
                     nlls_bp=nlls_bp, thetas_bp=thetas_bp)
            print("-- jackknife results saved --")
            return

    #  --------- leave-one-out ----------
    if loo_chrom is not False:
        # can be False (don't do LOO), True (do LOO for all chroms), or string
        # chromosome name (do LOO, exluding chromosome)
        starts_b = nstarts if not recycle_mle else [m_b.theta_] * nstarts
        starts_bp = nstarts if not recycle_mle else [m_bp.theta_] * nstarts
        b_r2 = None
        if loo_chrom is True:
           # iterate through everything
           out_sample_chrom = None
        else:
           # single chrom specified...
           out_sample_chrom = loo_chrom
        if not bp_only:
            print("-- leave-one-out R2 estimation for B --")
            b_r2 = m_b.loo_chrom_R2(starts=loo_nstarts,
                                    out_sample_chrom=out_sample_chrom,
                                    loo_fits_dir=loo_fits_dir,
                                    ncores=ncores)
        print("-- leave-one-out R2 estimation for B' --")
        bp_r2 = m_bp.loo_chrom_R2(starts=loo_nstarts,
                                  out_sample_chrom=out_sample_chrom,
                                  loo_fits_dir=loo_fits_dir,
                                  ncores=ncores)

        np.savez(r2_file, b_r2=b_r2, bp_r2=bp_r2)

def get_out_sample(idx, n):
    """
    Get the bootstrap indices that randomly weren't sampled.
    """
    return np.array(list(set(np.arange(n)).difference(idx)))


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


def bounds_mutation(nt, nf, log10_pi0_bounds=PI0_BOUNDS,
                    log10_mu_bounds=MU_BOUNDS, paired=False):
    """
    Return the bounds on for optimization under the free mutation
    model. If paired=True, the bounds are zipped together for each
    parameter.
    """
    l = [10**log10_pi0_bounds[0]]
    u = [10**log10_pi0_bounds[1]]
    for i in range(nt):
        for j in range(nf):
            l += [10**log10_mu_bounds[0]]
            u += [10**log10_mu_bounds[1]]
    lb = np.array(l)
    ub = np.array(u)
    assert np.all(lb < ub)
    if paired:
        return list(zip(lb, ub))
    return lb, ub


def bounds_simplex(nt, nf, log10_pi0_bounds=PI0_BOUNDS,
           log10_mu_bounds=MU_BOUNDS,
           softmax=False, bounded_softmax=False, global_bound=1e3,
           paired=False):
    """
    Return the bounds on for optimization under the simplex model
    model. If paired=True, the bounds are zipped together for each
    parameter.

    If softmax=True, all W entries are in the reals, with bounds
    -Inf, Inf. If a global optimization algorithm is being used,
    bounded_softmax should be set to True; a high bound is chosen,
    assuming that things are started on N(0, 1).
    """
    l = [10**log10_pi0_bounds[0]]
    u = [10**log10_pi0_bounds[1]]
    l += [10**log10_mu_bounds[0]]
    u += [10**log10_mu_bounds[1]]
    softmax_bound = np.inf if not bounded_softmax else global_bound
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


def bounds_fixed_mutation(nt, nf, log10_pi0_bounds=PI0_BOUNDS):
    """
    Return the bounds on for optimization under the simplex model
    model with fixed mutations.
    """
    l = [10**log10_pi0_bounds[0]]
    u = [10**log10_pi0_bounds[1]]
    l += [0.]*nf*nt
    u += [1.]*nf*nt
    lb = np.array(l)
    ub = np.array(u)
    assert np.all(lb < ub)
    return lb, ub

def random_start_mutation(nt, nf,
                          log10_pi0_bounds=PI0_BOUNDS,
                          log10_mu_bounds=MU_BOUNDS):
    """
    Create a random start position log10 uniform over the bounds for π0
    and all the mutation parameters under the free mutation model.
    """
    pi0 = np.random.uniform(10**log10_pi0_bounds[0], 
                            10**log10_pi0_bounds[1], 1)
    W = np.empty((nt, nf))
    for i in range(nt):
        for j in range(nf):
            W[i, j] = np.random.uniform(10**log10_mu_bounds[0], 
                                        10**log10_mu_bounds[1])
    theta = np.empty(nt*nf + 1)
    theta[0] = pi0
    theta[1:] = W.flat
    return theta


def random_start_mutation_log10(nt, nf,
                                log10_pi0_bounds=PI0_BOUNDS,
                                log10_mu_bounds=MU_BOUNDS):
    """
    Create a random start position log10 uniform over the bounds for π0
    and all the mutation parameters under the free mutation model.
    """
    pi0 = 10**np.random.uniform(log10_pi0_bounds[0], log10_pi0_bounds[1], 1)
    W = np.empty((nt, nf))
    for i in range(nt):
        for j in range(nf):
            W[i, j] = 10**np.random.uniform(log10_mu_bounds[0], log10_mu_bounds[1])
    theta = np.empty(nt*nf + 1)
    theta[0] = pi0
    theta[1:] = W.flat
    return theta


def random_start_simplex(nt, nf, log10_pi0_bounds=PI0_BOUNDS,
                         log10_mu_bounds=MU_BOUNDS, softmax=False):
    """
    Create a random start position, uniform over the bounds for π0
    and μ, and Dirichlet under the DFE weights for W, under the simplex model.
    """
    pi0 = np.random.uniform(10**log10_pi0_bounds[0], 
                            10**log10_pi0_bounds[1], 1)
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


def random_start_simplex_log10(nt, nf, log10_pi0_bounds=PI0_BOUNDS,
                               log10_mu_bounds=MU_BOUNDS):
    """
    Create a random start position log10 uniform over the bounds for π0
    and μ, and uniform under the DFE weights for W, under the simplex model.
    """
    pi0 = 10**np.random.uniform(log10_pi0_bounds[0], log10_pi0_bounds[1], 1)
    mu = 10**np.random.uniform(log10_mu_bounds[0], log10_mu_bounds[1], 1)
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


def random_start_fixed_mutation(nt, nf, log10_pi0_bounds=PI0_BOUNDS):
    """
    Create a random start position log10 uniform over the bounds for π0
    and μ, and uniform under the DFE weights for W, under the simplex model,
    but for a fixed mutation rate.
    """
    pi0 = np.random.uniform(10**log10_pi0_bounds[0], 
                            10**log10_pi0_bounds[1], 1)
    W = np.empty((nt, nf))
    for i in range(nf):
        W[:, i] = np.random.dirichlet([1.] * nt)
        assert np.abs(W[:, i].sum() - 1.) < 1e-5
    theta = np.empty(nt*nf + 1)
    theta[0] = pi0
    theta[1:] = W.flat
    return theta


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


def penalized_negll_c(theta, Y, logB, w, mu0, r):
    """
    A thin wrapper over negll_c() that imposes a penalty of the form:

     l*(θ) = l(θ | x) - r (μ - μ0)^2 / 2

    where r can be thought of as the precision (1/variance) -- this is
    essentially a Gaussian prior.
    """
    nll = negll_c(theta, Y, logB, w)
    mu = theta[1]
    return nll + r/2 * (mu - mu0)**2

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


def negll_mutation(theta, Y, logB, w):
    nS = Y[:, 0]
    nD = Y[:, 1]
    nx, nw, nt, nf = logB.shape
    # mut weight params
    pi0, W = theta[0], theta[1:]
    mu = 1.0
    W = W.reshape((nt, nf))
    # interpolate B(w)'s
    ll = 0.
    for i in range(nx):
        logBw_i = 0.
        for j in range(nt):
            for k in range(nf):
                logBw_i += np.interp(mu*W[j, k], w, logB[i, :, j, k])
        log_pibar = np.log(pi0) + logBw_i
        ll += nD[i]*log_pibar + nS[i]*np.log1p(-np.exp(log_pibar))
    return -ll

def check_bounds(x, lb, ub):
    assert np.all((x >= lb) & (x <= ub))

def negll_c(theta, Y, logB, w, two_alleles=False):
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
    likclib.negloglik.argtypes = (POINTER(c_double), POINTER(c_double),
                              POINTER(c_double), POINTER(c_double),
                              POINTER(c_double),
                              # weird type for dims/strides
                              POINTER(np.ctypeslib.c_intp),
                              POINTER(np.ctypeslib.c_intp),
                              c_int)
    likclib.negloglik.restype = c_double
    return likclib.negloglik(theta_ptr, nS_ptr, nD_ptr, logB_ptr, w_ptr,
                             logB.ctypes.shape, logB.ctypes.strides, int(two_alleles))

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

def predict_freemutation(theta, logB, w):
    """
    """
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


def rescale_freemutation_thetas(thetas):
    """

    """
    pass

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
            assert logB.shape[1] == w.size
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

    def _load_optim(self, optim_res):
        """
        Taken an OptimResult() object.
        """
        self.optim = optim_res
        # load in the best values
        self.theta_ = optim_res.theta
        self.nll_ = optim_res.nll

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

def negll_freemut(Y, B, w, two_alleles=False):
    """
    This is a closure around data; returns a negative log- likelihood
    function around the data and a few fixed parameters (B and w).

    The core part is negll_c().
    """
    def func(theta):
        new_theta = np.full(theta.size + 1, np.nan)
        theta = np.copy(theta)
        new_theta[0] = theta[0]
        # fix mutation rate to one and let W represent mutation rates to various classes
        new_theta[1] = 1.
        new_theta[2:] = theta[1:] # times mutation rates
        #print("-->", theta, new_theta)
        return negll_c(new_theta, Y, B, w, two_alleles)
    return func


def negll_freemut_full(theta, grad, Y, B, w, two_alleles=False):
    """
    Like negll_freemut() but not a closure.

    grad is for nlopt and should be set to None if SciPy is being used.
    """
    new_theta = np.full(theta.size + 1, np.nan)
    new_theta[0] = theta[0]
    # fix mutation rate to one and let W represent mutation rates to various classes
    new_theta[1] = 1.
    new_theta[2:] = theta[1:] # times mutation rates
    #print("-->", theta, new_theta)
    return negll_c(new_theta, Y, B, w, two_alleles)


def negll_softmax_full(theta, grad, Y, B, w):
    """
    Softmax version of the simplex model wrapper for negll_c().

    grad is required for nlopt.
    """
    nx, nw, nt, nf = B.shape
    # get out the W matrix, which on the optimization side, is over all reals
    sm_theta = np.copy(theta)
    W_reals = theta[2:].reshape(nt, nf)
    with np.errstate(under='ignore'):
        W = softmax(W_reals, axis=0)
    assert np.allclose(W.sum(axis=0), np.ones(W.shape[1]))
    sm_theta[2:] = W.flat
    return negll_c(sm_theta, Y, B, w)


def negll_simplex_full(theta, grad, Y, B, w):
    """
    Simplex model wrapper for negll_c().

    grad is required for nlopt.
    """
    return negll_c(theta, Y, B, w)


def negll_simplex_fixed_mutation_full(theta, grad, Y, B, w, mu):
    """
    Simplex model wrapper for negll_c(), with fixed mutation.

    grad is required for nlopt.
    """
    # insert the fixed μ
    new_theta = np.zeros(theta.size + 1)
    new_theta[0] = theta[0]
    new_theta[1] = mu
    new_theta[2:] = theta[1:]
    return negll_c(new_theta, Y, B, w)


class FreeMutationModel(BGSLikelihood):
    def __init__(self, Y, w, t, logB, bins=None,
                 features=None, 
                 log10_pi0_bounds=PI0_BOUNDS):
        super().__init__(Y=Y, w=w, t=t, logB=logB, features=features,
                         bins=bins, log10_pi0_bounds=log10_pi0_bounds)

    def random_start(self):
        """
        Random starts
        """
        return random_start_mutation(self.nt, self.nf,
                                     self.log10_pi0_bounds,
                                     self.log10_mu_bounds)

    def bounds(self, paired=False):
        return bounds_mutation(self.nt, self.nf,
                               self.log10_pi0_bounds,
                               self.log10_mu_bounds, paired=paired)

    def fit(self, starts=1, ncores=None, algo='ISRES', two_alleles=False,
            _indices=None):
        """
        Fit likelihood models with mumeric optimization (either scipy or nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        algo: the algorithm to use
        two_alleles: do the two alleles correction (i.e. not infinite sites)
        """
        algo = algo.upper()
        algos = {'L-BFGS-B':'scipy', 'ISRES':'nlopt',
                 'NELDERMEAD':'nlopt', 'NEWUOA':'nlopt'}
        assert algo in algos, f"algo must be in {algos}"
        engine = algos[algo]

        if isinstance(starts, int):
            starts = [self.random_start() for _ in range(starts)]
        assert isinstance(starts, list), "starts must be a list of θs or integer of random starts"
        ncores = 1 if ncores is None else ncores
        ncores = min(len(starts), ncores) # don't request more cores than we need

        Y = self.Y
        logB = self.logB
        bootstrap = _indices is not None
        if bootstrap:
            Y = Y[_indices, ...]
            logB = logB[_indices, ...]

        if engine == 'scipy':
            nll = partial(negll_freemut_full, grad=None, Y=Y,
                          B=logB, w=self.w, two_alleles=two_alleles)
            worker = partial(minimize, nll, bounds=self.bounds(paired=True),
                             method=algo, options={'eps':1e-9})
        elif engine == 'nlopt':
            nll = partial(negll_freemut_full, Y=Y,
                          B=logB, w=self.w, two_alleles=two_alleles)
            worker = partial(nlopt_mutation_worker,
                             func=nll, nt=self.nt, nf=self.nf,
                             bounds=self.bounds(), algo=algo)
        else:
            raise ValueError("engine must be 'scipy' or 'nlopt'")

        # run the optimization routine on muliple starts
        res = run_optims(worker, starts, ncores=ncores,
                         progress=not bootstrap)

        if bootstrap:
            # we don't clobber the existing results since this ins't a fit.
            # instead, we return the OptimResult directly
            return res
        self._load_optim(res)

    @property
    def mle_M(self):
        """
        Extract out the M matrix.
        """
        return self.theta_[1:].reshape((self.nt, self.nf))

    @property
    def mle_mu(self):
        """
        Get the mutation rates by summing the columns of the M matrix.
        """
        M = self.mle_M.reshape((self.nt, self.nf))
        return M.sum(axis=0)

    @property
    def mle_W(self):
        """
        Normalized W matrix (e.g. a DFE)
        """
        M = self.mle_M.reshape((self.nt, self.nf))
        W = M / M.sum(axis=0)
        return W

    def dfe_table(self):
        rows = []
        Wc = self.mle_W
        for i, feature in enumerate(self.features):
            rows.append("\t".join([feature, ','.join([str(round(x, 3)) for x in Wc[:, i].tolist()])]))
        return "\n".join(rows)

    @property
    def nll(self):
        return self.nll_

    def predict(self, optim=None, theta=None, B=False):
        """
        Predicted π from the best fit (if optim = None). If optim is 'random', a
        random MLE optimization is chosen (e.g. to get a senes of how much
        variation there is across optimization). If optim is an integer,
        this rank of optimization results is given (e.g. optim = 0 is the
        best MLE).
        """
        if theta is not None:
            return predict_freemutation(theta, self.logB, self.w)
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
        return predict_freemutation(theta, self.logB, self.w)

    def __repr__(self):
        base_rows = super().__repr__()
        if self.theta_ is not None:
            base_rows += "\n\nFree-mutation model ML estimates:\n"
            base_rows += f"negative log-likelihood: {self.nll_}\n"
            base_rows += f"π0 = {self.mle_pi0}\n"
            base_rows += f"R² = {np.round(100*self.R2(), 4)}\n"
            W = self.mle_W.reshape((self.nt, self.nf))
            Wc = W / W.sum(axis=0)
            tab = np.concatenate((self.t[:, None], np.round(Wc, 3)), axis=1)
            header = ()
            if self.features is not None:
                header = [''] + self.features

            base_rows += "W = \n" + tabulate(tab, headers=header) + "\n"
            base_rows += "μ = \n" + tabulate(W.sum(axis=0)[None, :], headers=header[1:])
        return base_rows


class SimplexModel(BGSLikelihood):
    """
    BGSLikelihood Model with the DFE matrix W on a simplex, free 
    π0 and free μ (within bounds).

    There are two main way to parameterize the simplex for optimization.

      1. No reparameterization: 0 < W < 1
      2. Softmax: the columns of W are optimized over all reals,
          and mapped to the simplex space with softmax.

    Note that the B grid sets the bounds around μW, since this product
    cannot fall outside the interpolation range. For bounded μ,
    this implies a lower and upper DFE 

    l < μW < u: l, u are the interpolation bounds

    l/μ_upper < W < u/μ_lower

    The C likelihood routine will always return B=1 when w is 
    below the lower interpolation point (corresponding to w≈0). So
    the lower bound doesn't need to be set. 

    """
    def __init__(self, Y, w, t, logB, bins=None,
                 features=None, 
                 log10_pi0_bounds=PI0_BOUNDS,
                 log10_mu_bounds=MU_BOUNDS):
        super().__init__(Y=Y, w=w, t=t, logB=logB,
                         bins=bins, features=features,
                         log10_pi0_bounds=log10_pi0_bounds,
                         log10_mu_bounds=log10_mu_bounds)
        self.start_pi0 = None
        self.start_mu = None
        # the bounds of W are set by B grid interpolation range
        # and mutation rate bounds
        self.W_bounds = (np.min(w)/(10**MU_BOUNDS[0]),
                         min(1, np.max(w)/(10**MU_BOUNDS[1])))
        assert self.W_bounds[0] > 0
        assert self.W_bounds[0] <= 1


    def random_start(self, softmax=False):
        """
        Random starts, on a linear scale for μ and π0.

        Note: if the attributes self.start_pi0 and/or self.start_mu
        are set, these fixed start points *replace* the corresponding
        start element in this random array.
        """
        start = random_start_simplex(self.nt, self.nf,
                                     self.log10_pi0_bounds,
                                     self.log10_mu_bounds, 
                                     softmax=softmax)
        if self.start_pi0 is not None:
            start[0] = self.start_pi0
        if self.start_mu is not None:
            start[1] = self.start_mu

        return start

    def bounds(self, softmax=False, global_bound=False):
        return bounds_simplex(self.nt, self.nf, 
                              self.log10_pi0_bounds,
                              self.log10_mu_bounds, 
                              softmax=softmax,
                              global_bound=global_bound,
                              paired=False)

    def fit(self, starts=1, ncores=None, 
            algo='GN_ISRES', start_pi0=None, start_mu=None,
            softmax=False,
            progress=True, _indices=None):
        """
        Fit likelihood models with numeric optimization (using nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        algo: the optimization algorithm to use.
        _indices: indices to include in fit; this is primarily internal, for
                  bootstrapping
        """
        # TODO: now I just directly pass the algo to nlopt, no checking
        # e.g. for testing. Clean up interface later.
        if start_pi0 is not None:
            self.start_pi0 = start_pi0
        if start_mu is not None:
            self.start_mu = start_mu

        if isinstance(starts, int):
            starts = [self.random_start(softmax=softmax) for _
                      in range(starts)]

        if ncores is not None:
            # don't request more cores than we need
            ncores = min(len(starts), ncores)

        Y = self.Y
        logB = self.logB
        bootstrap = _indices is not None
        if bootstrap:
            Y = Y[_indices, ...]
            logB = logB[_indices, ...]

        # wrap the negloglik function around the data and B grid
        negll_func = negll_simplex_full if not softmax else negll_softmax_full
        nll = partial(negll_func, Y=Y,
                      B=logB, w=self.w)

        global_bound = algo.startswith('GN')
        # make the main simplex worker for parallelization
        worker_func = nlopt_simplex_worker if not softmax else nlopt_softmax_worker
        worker = partial(worker_func,
                         func=nll, nt=self.nt, nf=self.nf,
                         log10_W_bounds=np.log10(self.W_bounds),
                         bounds=self.bounds(softmax=softmax, global_bound=global_bound), 
                         algo=algo)

        # run the optimization routine on muliple starts
        res = run_optims(worker, starts, ncores=ncores, progress=progress)

        if bootstrap:
            # we don't clobber the existing results since this ins't a fit.
            return res
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

    def __repr__(self):
        base_rows = super().__repr__()
        if self.theta_ is not None:
            base_rows += "\n\nSimplex model ML estimates:\n"
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

class FixedMutationModel(BGSLikelihood):
    def __init__(self, Y, w, t, logB, bins=None,
                 features=None, log10_pi0_bounds=(-5, -1)):
        super().__init__(Y=Y, w=w, t=t, logB=logB, bins=bins,
                         features=features, log10_pi0_bounds=log10_pi0_bounds)

    def random_start(self):
        """
        Random starts
        """
        return random_start_fixed_mutation(self.nt, self.nf,
                                           self.log10_pi0_bounds)

    def bounds(self):
        return bounds_fixed_mutation(self.nt, self.nf,
                                     self.log10_pi0_bounds)

    def fit(self, mu, starts=1, ncores=None, algo='ISRES'):
        """
        Fit likelihood models with mumeric optimization (using nlopt).

        starts: either an integer number of random starts or a list of starts.
        ncores: number of cores to use for multiprocessing.
        algo: the optimization algorithm to use.
        """
        algo = algo.upper()
        algos = {'ISRES':'nlopt', 'NELDERMEAD':'nlopt'}
        assert algo in algos, f"algo must be in {algos}"
        engine = algos[algo]

        if isinstance(starts, int):
            starts = [self.random_start() for _ in range(starts)]
        # don't request more cores than we need
        ncores = min(len(starts), ncores)

        nll = partial(negll_simplex_fixed_mutation_full, Y=self.Y,
                      B=self.logB, w=self.w, mu=mu)
        worker = partial(nlopt_simplex_worker, mu=mu,
                         func=nll, nt=self.nt, nf=self.nf,
                         bounds=self.bounds(), algo=algo)
        res = run_optims(worker, starts, ncores=ncores)
        self.mu = mu
        self._load_optim(res)

    @property
    def mle_W(self):
        """
        Extract out the W matrix.
        """
        return self.theta_[1:]

    def __repr__(self):
        base_rows = super().__repr__()
        if self.theta_ is not None:
            base_rows += "\n\nFixed-Mutation Simplex model ML estimates:\n"
            base_rows += f"negative log-likelihood: {self.nll_}\n"
            base_rows += f"π0 = {self.mle_pi0}\n"
            base_rows += f"μ = {self.mu} (fixed)\n"
            base_rows += f"R² = {np.round(100*self.R2(), 4)}\n"
            header = ()
            if self.features is not None:
                header = [''] + self.features
            W = self.mle_W.reshape((self.nt, self.nf))
            tab = np.concatenate((self.t[:, None], np.round(W, 3)), axis=1)
            base_rows += "W = \n" + tabulate(tab, headers=header)
        return base_rows


        return base_rows

    def predict(self, optim=None, theta=None, mu=None):
        """
        Predicted π from the best fit (if optim = None). If optim is 'random', a
        random MLE optimization is chosen (e.g. to get a senes of how much
        variation there is across optimization). If optim is an integer,
        this rank of optimization results is given (e.g. optim = 0 is the
        best MLE).
        """
        mu = self.mu if mu is None else mu
        if theta is not None:
            return predict_simplex(theta, self.logB, self.w, mu)
        if optim is None:
            theta = self.theta_
        else:
            thetas = self.optim.thetas_
            if optim == 'random':
                theta = thetas[np.random.randint(0, thetas.shape[0]), :]
            else:
                theta = thetas[optim]
        return predict_simplex(theta, self.logB, self.w, mu)


