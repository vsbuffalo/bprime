import sys
sys.path.extend(['../']) # for bprime modules

import os
from multiprocessing import Pool
import numpy as np
import msprime
import tqdm
import pickle
import click
import json
from bprime.samplers import UniformSampler
from bprime.utils import read_params, signif
from bprime.theory import bgs_segment

PARAMS = ('mu', 's', 'rbp', 'recfrac', 'N', 'L')
# 4bp of tracking, 1bp recombination spacer, 4bp of selected region
# (note though that the B function is modeling Lbp of selected region)
WINS = [0, 4, 5, 10]

#bgs_ranges = {'mu': (1e-8, 1e-7, False),
#              's': (1e-3, 1e-1, False),
#              'r': (1e-7, 1e-9, False),
#              'N': (50, 10_000, False),
#              'L': (10, 100_000, False),
#              'nreps': (1, 1, False)}
#

def bgs_rec_runner(param, nreps=1):
    """
    Run msprime with N scaled by B.

    Note: nreps is set to 1, as exploring averaging over replicate runs
    did not work as well as just drawing more from the sampler.
    """
    mu, s, rf, rbp, N, L = [param[k] for k in PARAMS]
    N = int(N)
    L = int(L)
    #B = bgs_rec(mu, s, r, L)
    B = bgs_segment(mu, s, rf, rbp, L)
    rate = msprime.RateMap(position=WINS, rate=[0, rf, 0])
    # get the π in the tracking region
    Bhats = [msprime.sim_ancestry(N, population_size=B*N, recombination_rate=rate).diversity(windows=WINS, mode='branch')[0]/(4*N)
             for _ in range(nreps)]
    return Bhats

@click.command()
@click.argument('configfile', required=True)
@click.option('--outfile', default=None,
              type=click.File('wb'),
              help='output file (default: <configfile>_sims.npz)')
@click.option('--nsamples', default=None, type=int, help='number of samples')
@click.option('--ncores', default=1, help='number of cores to use')
@click.option('--seed', default=1, help='random seed to use')
def sim_bgs(configfile, outfile=None, nsamples=10_000, ncores=1, seed=1):
    """
    Simulate BGS under a rescaled neutral coalescent model with msprime.

    This is separate from (1) proper forward sims, and (2) simulating a
    two deme structured coalescent model. This is meant as a minimum test
    of learning a function from coalescent times.

    Note: s here is the homozygous selection coefficient. It is automatically
    rescaled to s/2 in the simulations to match the case where h=1/2. BGS
    theory only considers the heterozygous selection coefficient.
    """
    with open(configfile) as f:
        config = json.load(f)

    ranges = read_params(config)

    try:
        total = nsamples if nsamples is not None else config['nsamples']
    except KeyError:
        raise KeyError(f"configfile '{configfile}' does npt specify nsamples"
                        " and --nsamples not set via command line")

    sampler = UniformSampler(ranges, total=total, seed=seed)

    print(sampler)

    if outfile is None:
        basename = os.path.splitext(os.path.basename(configfile))[0]
        outfile = f"{basename}_sims.npz"

    with Pool(ncores) as p:
        y = np.array(list(tqdm.tqdm(p.imap(bgs_rec_runner, sampler),
                                    total=sampler.total)))
    X, features = sampler.as_matrix()
    assert(len(y) == X.shape[0])
    targets = ['Bhat']
    np.savez(outfile, X=X, y=y, features=features, targets=targets)

if __name__ == "__main__":
    sim_bgs()
