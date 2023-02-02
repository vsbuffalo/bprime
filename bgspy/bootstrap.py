import os
import warnings
from collections import namedtuple
import itertools
import numpy as np

BootstrapResults = namedtuple('BootstrapResults',
                              ('b_nll', 'b_theta',
                               'bp_nll', 'bp_theta'))


def resample_blocks(bins, blocksize, nsamples=None, exclude_chroms=None):
    """
    Note that because mask filters can block out large regions of chromosome,
    e.g. chr22, the number of possible blocks may not reflect the chromosome
    length. Blocks are always created from the first complete position.
    """
    # we need chromosome weights -- we sample a chromosome, then
    # create blocks on those
    chroms = list(bins.keys())
    n_map = {c: len(v) for c, v in bins.items()}

    msg = f"blocksize must be less than smallest chromosome size's number of blocks ({min(n_map.values())})"
    assert np.all(np.array(list(n_map.values())) >= blocksize), msg

    n = sum(n_map.values())
    weights_map = {c: len(v) for c, v in bins.items()}
    weights = np.array(list(weights_map.values()))
    offset = np.cumsum(list(weights))
    assert offset[-1] == n

    # drop the last cumsum (past the last chrom)
    offset_map = dict(zip(chroms, [0] + offset[:-1].tolist()))

    # number of samples
    nsamples = int(n/blocksize) if nsamples is None else nsamples

    # exclude chromosomes if necessary for out-sample prediction
    if exclude_chroms is not None:
        chroms = [c for c in chroms if c not in exclude_chroms]
        weights = np.array([w for w, c in zip(weights, chroms) if c not in exclude_chroms])

    # sample chromosomes for each block first, weighted by their number of
    # blocks
    chrom_samples = np.random.choice(chroms, size=nsamples,
                                     p=weights/weights.sum(),
                                     replace=True)
    indices = []
    for chrom in chrom_samples:
        # these are in terms of the number of bins per chromosome...
        off = offset_map[chrom]
        start_idx = off + np.random.randint(0, n_map[chrom]-blocksize, 1)
        # so they need to be offset by the total number of bins per chromosome.
        indices.extend(itertools.chain(*[list(range(i, i+blocksize)) for i in start_idx]))
    return np.array(indices)


def extract_bootstraps(optim_results, warn_frac_success=0.8):
    """
    Extract out relevant parts of optimization results, and
    warn if the success rate is low.
    """
    for res in optim_results:
        if not res.is_succes():
            yield None, None

        if res.frac_success < warn_frac_success:
            a = np.round(res.frac_success*100, 2)
            b = warn_frac_success * 100
            msg = f"OptimResult has a success rate {a} < warn_frac_success ({b}%))"
            warnings.warn(msg)
        nll = res.nll
        theta = res.theta
        yield nll, theta

def process_bootstraps(optim_results):
    nlls, thetas = [], [],
    failed = 0
    for nll, theta in extract_bootstraps(optim_results):
        if nll is None or theta is None:
            failed += 1
            continue
        nlls.append(nll)
        thetas.append(theta)

    nlls = np.stack(nlls).T
    thetas = np.stack(thetas).T
    return nlls, thetas

def percentile_ci(boot_thetas, alpha=0.05, axis=0):
    eps = np.nanquantile(boot_thetas, (alpha/2, 1-alpha/2), axis=axis)
    return eps

def pivot_ci(boot_thetas, theta, alpha=0.05, axis=0):
    """
    Use axis is 1 for MLE thetas from boostrap samples.
    """
    eps = np.quantile(boot_thetas, (alpha/2, 1-alpha/2), axis=axis)
    return np.array([2*theta - eps[1], 2*theta - eps[0]])

def load_from_bs_dir(bootstrap_dir):
    """
    Load bootstrap results from the .npz file created 
    from the bootstrap command line option. 
    Because of the way parallelization was done, 
    this these assume there are B and B' results in each 
    .npz.
    """
    strap_files = os.listdir(bootstrap_dir)
    nlls_b, thetas_b = [], []
    nlls_bp, thetas_bp = [], []
    for f in strap_files:
        d = np.load(os.path.join(bootstrap_dir, f))
        nlls_b.append(d['nlls_b'])
        nlls_bp.append(d['nlls_bp'])
        thetas_b.append(d['thetas_b'])
        thetas_bp.append(d['thetas_bp'])

    b_boot_nlls = np.concatenate(nlls_b, axis=0)
    b_boot_thetas = np.concatenate(thetas_b, axis=1).T
    bp_boot_nlls = np.concatenate(nlls_bp, axis=0)
    bp_boot_thetas = np.concatenate(thetas_bp, axis=1).T

    return BootstrapResults(b_boot_nlls, b_boot_thetas, bp_boot_nlls, bp_boot_thetas)
