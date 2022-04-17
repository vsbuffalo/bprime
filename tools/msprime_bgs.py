import sys
sys.path.extend(['../']) # for bprime modules

import os
from multiprocessing import Pool
import numpy as np
from math import floor
import msprime
import tqdm
import pickle
import click
import json
from bprime.samplers import Sampler
from bprime.utils import signif
from bprime.sim_utils import read_params
from bprime.theory import bgs_segment, bgs_rec, BGS_MODEL_PARAMS

def bgs_msprime_runner(param, func='bgs_segment', nreps=1):
    """
    Run msprime with N scaled by B.

    Note: nreps is set to 1, as exploring averaging over replicate runs
    did not work as well as just drawing more from the sampler.
    """
    N = int(param['N'])
    kwargs = {k: param[k] for k in BGS_PARAMS['bgs_rec']}
    B = bgs_rec(**kwargs)
    Ne = B*N
    Bhats = [msprime.sim_ancestry(N, population_size=Ne).diversity(mode='branch')/(4*N)
             for _ in range(nreps)]
    return Bhats

@click.command()
@click.argument('configfile', required=True)
@click.option('--outfile', default=None,
              type=click.File('wb'),
              help='output file (default: <configfile>_sims.npz)')
@click.option('--nsamples', default=None, type=int, help='number of samples')
@click.option('--ncores', default=1, help='number of cores to use')
@click.option('--func', default='simple', help="the BGS function to use ('segment', 'rec')")
@click.option('--seed', default=1, help='random seed to use')
def sim_bgs(configfile, outfile=None, nsamples=10_000, ncores=1, func='simple', seed=1):
    """
    Simulate BGS under a rescaled neutral coalescent model with msprime.

    This is separate from (1) proper forward sims, and (2) simulating a
    two deme structured coalescent model. This is meant as a minimum test
    of learning a function from coalescent times.

    Note: s here is the homozygous selection coefficient. It is automatically
    rescaled to s/2 in the simulations to match the case where h=1/2. BGS
    theory only considers the heterozygous selection coefficient.

    There are two BGS functions that can be used. Setting --segment uses the
    'segment' BGS model which is more complicated than then simple model.
    """
    with open(configfile) as f:
        config = json.load(f)

    try:
        func_params = BGS_MODEL_PARAMS[func]
    except KeyError:
        raise KeyError("--func must be either ', '.join(BGS_MODEL_PARAMS.keys())")

    ranges, _ = read_params(config, add_rep=False)

    json_params = set(tuple(k for k in ranges.keys() if k != 'N'))
    needed_params = set(BGS_MODEL_PARAMS[func])
    missing = needed_params.difference(json_params)
    excess = json_params.difference(needed_params)
    if len(missing):
        raise ValueError(f"missing params in JSON for '{func}': {missing}")
    if len(excess):
        raise ValueError(f"excess params in JSON for '{func}': {excess}")

    try:
        total = nsamples if nsamples is not None else config['nsamples']
    except KeyError:
        raise KeyError(f"configfile '{configfile}' does npt specify nsamples"
                        " and --nsamples not set via command line")

    sampler = Sampler(ranges, total=total, seed=seed, add_seed=False)

    print(sampler)

    if outfile is None:
        basename = os.path.splitext(os.path.basename(configfile))[0]
        outfile = f"{basename}_sims.npz"

    # get the right BGS simulation function
    runner_func = partial(bgs_msprime_runner, func=func)
    if ncores > 1:
        with Pool(ncores) as p:
            y = np.array(list(tqdm.tqdm(p.imap(runner_func, sampler),
                                        total=sampler.total)))
    else:
        y = np.array(list(tqdm.tqdm((runner_func(s) for s in sampler),
                                     total=sampler.total)))

    X, features = sampler.as_matrix()
    assert(len(y) == X.shape[0])
    targets = ['Bhat']
    np.savez(outfile, X=X, y=y, features=features, targets=targets)

if __name__ == "__main__":
    sim_bgs()

