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
from functools import partial
from bprime.samplers import Sampler, target_rejection_sampler
from bprime.utils import signif
from bprime.sim_utils import read_params
from bprime.theory import bgs_segment, bgs_rec, BGS_MODEL_PARAMS, BGS_MODEL_FUNCS

def bgs_msprime_runner(param, model, nreps=5):
    """
    Run msprime with N scaled by B.

    Note: nreps is set to 1, as exploring averaging over replicate runs
    did not work as well as just drawing more from the sampler.
    """
    N = int(param['N'])
    kwargs = {k: param[k] for k in BGS_MODEL_PARAMS[model]}
    B_func = BGS_MODEL_FUNCS[model]
    B = B_func(**kwargs)
    # protect against B = 0 leading to Ne = 0
    Ne = max(B*N, np.finfo(float).tiny)
    Bhats = np.mean([msprime.sim_ancestry(N, population_size=Ne).diversity(mode='branch')/(4*N)
             for _ in range(nreps)])
    return Bhats

@click.command()
@click.argument('configfile', required=True)
@click.option('--outfile', default=None,
              type=click.File('wb'),
              help='output file (default: <configfile>_sims.npz)')
@click.option('--nsamples', default=None, type=int, help='number of samples')
@click.option('--nreps', default=None, help='number of simulations to average over per parameter set')
@click.option('--ncores', default=1, help='number of cores to use')
@click.option('--model', default='simple', help="the BGS function to use ('bgs_segment', 'bgs_rec')")
@click.option('--reject/--no-reject', default=False, help="rejection sample the target")
@click.option('--seed', default=1, help='random seed to use')
def sim_bgs(configfile, outfile=None, nsamples=10_000, nreps=1,
            ncores=1, model='simple', reject=False, seed=1):
    """
    Simulate BGS under a rescaled neutral coalescent model with msprime.

    This is separate from (1) proper forward sims, and (2) simulating a
    two deme structured coalescent model. This is meant as a minimum test
    of learning a function from coalescent times.

    There are two BGS functions that can be used. Setting --segment uses the
    'segment' BGS model which is more complicated than then simple model.

    Note that the --nsamples and --nreps override 'nsamples' and 'nreps'
    in the JSON config file!
    """
    with open(configfile) as f:
        config = json.load(f)

    try:
        func_params = BGS_MODEL_PARAMS[model]
        func = BGS_MODEL_FUNCS[model]
    except KeyError:
        raise KeyError("--func must be either ', '.join(BGS_MODEL_PARAMS.keys())")

    def func2(**kwargs):
        # wrap the BGS model function such that non-used fixed params
        # that go to sims (like N) aren't included here (since those model
        # funcs don't support them
        kwargs = {k: v for k, v in kwargs.items() if k in func_params}
        return func(**kwargs)

    ranges, _ = read_params(config, add_rep=False)

    json_params = set(tuple(k for k in ranges.keys() if k != 'N'))
    needed_params = set(BGS_MODEL_PARAMS[model])
    missing = needed_params.difference(json_params)
    excess = json_params.difference(needed_params)
    if len(missing):
        raise ValueError(f"missing params in JSON for '{model}': {missing}")
    if len(excess):
        raise ValueError(f"excess params in JSON for '{model}': {excess}")

    # set the nsamples and nreps -- CL args override config
    try:
        total = nsamples if nsamples is not None else config['nsamples']
    except KeyError:
        raise KeyError(f"configfile '{configfile}' does not specify nsamples"
                        " and --nsamples not set via command line")

    try:
        nreps = nreps if nreps is not None else config['nreps']
    except KeyError:
        nreps = 1

    sampler = Sampler(ranges, total=total, seed=seed, add_seed=False)

    print(sampler)

    # rejection sampling if either in config or param line
    reject = config.get('reject', False) or reject

    total = sampler.total
    if reject:
        sampler.total = None # turn this into an infinite sampler
        sampler = target_rejection_sampler(func2, sampler, n=total)

    if outfile is None:
        basename = os.path.splitext(os.path.basename(configfile))[0]
        outfile = f"{basename}_sims.npz"

    # get the right BGS simulation function and wrap it
    runner_func = partial(bgs_msprime_runner, nreps=nreps, model=model)
    if ncores > 1:
        with Pool(ncores) as p:
            y = np.array(list(tqdm.tqdm(p.imap(runner_func, sampler),
                                        total=total)))
    else:
        y = np.array(list(tqdm.tqdm((runner_func(s) for s in sampler),
                                     total=total)))

    if not reject:
        X, features = sampler.as_matrix()
    else:
        X = np.array([list(v.values()) for v in sampler])
        features = list(sampler[0].keys())
    assert(len(y) == X.shape[0])
    targets = ['Bhat']
    np.savez(outfile, X=X, y=y, features=features, targets=targets)

if __name__ == "__main__":
    sim_bgs()

