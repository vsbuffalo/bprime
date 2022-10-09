import itertools
import numpy as np


def resample_blocks(bins, blocksize, nsamples=None):
    """
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

def pivot_ci(boot_thetas, theta, alpha=0.05, log=True):
    if log:
        boot_thetas = np.log10(boot_thetas)
        theta = np.log10(theta)
    eps = np.quantile(boot_thetas, (alpha/2, 1-alpha/2), axis=1)
    return 10**(2*theta - eps[1]), 10**(2*theta - eps[0])
