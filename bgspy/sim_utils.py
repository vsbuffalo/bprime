## sim_utils.py -- common utility functions for msprime and slim
import warnings
import itertools
import operator
from collections import defaultdict
import numpy as np
import pyslim
import tqdm
from bgspy.utils import get_files, random_seed

def fixed_params(params):
    """
    Iterate through the parameters and return the fixed parameters,
    either because a grid only has one element or lower/upper are
    the same.
    """
    fixed = dict()
    for key, param in params.items():
        if param['dist']['name'] == 'fixed':
            fixed[key] = param['dist']['val']
        else:
            if param['dist']['low'] == param['dist']['high']:
                warnings.warn(f"parameter '{key}' is fixed implicitly!")
                fixed[key] = param['dist']['low']
    return fixed

def param_grid(params, add_seed=False, rng=None):
    """
    Generate a Cartesian product parameter grid from a
    dict of grids per parameter, optionally adding the seed.
    """
    grid = []
    for param, values in params.items():
        if len(values):
            grid.append([(param, v) for v in values])
        else:
            grid.append([(param, '')])
    def convert_to_dict(x):
        # and package seed if needed
        x = dict(x)
        if add_seed:
            x['seed'] = random_seed(rng)
        return x
    return map(convert_to_dict, itertools.product(*grid))

def read_params(config):
    """
    Grab the parameter ranges from a configuration dictionary.

    There is some polymorphic behavior here, depending on whether a
    parameter grid is used, or whether sampling is done. For parameter
    grids, entries have an array with name "grid". For sampling, "lower",
    "upper", and "log10" are defined.

    Returns a dict of either param->(lower, uppper, log10, type) tuples
    (for sampling case) or param->[grid] for the grid case.
    """
    params = {}
    types = {}
    for param, vals in config['params'].items():
        assert param != "rep", "invalid param name 'rep'!"
        val_type = {'float': float, 'int': int, 'str':str}.get(vals['type'], None)
        is_grid = "grid" in vals
        if is_grid:
            params[param] = [val_type(v) for v in vals['grid']]
        else:
            # distribution functions specified
            assert 'dist' in vals
            if vals['dist']['name'] != 'fixed':
                assert 'low' in vals['dist'], f"'low' bound of dist must be set in key '{param}'"
                assert 'high' in vals['dist'], f"'high' bound of dist must be set in key '{param}'"
            params[param] = vals
        types[param] = val_type
    return params, types

def get_bounds(params):
    """
    Get the domain and scale (log10 or linear) for each parameter,
    from the type of density it is.
    """
    domain = dict()
    for key, param in params.items():
        is_fixed = param['dist']['name'] == 'fixed'
        is_log10 = param['dist']['name'].startswith('log10_')
        if is_fixed:
            low, high = param['dist']['val'], param['dist']['val']
        else:
            low, high = param['dist']['low'], param['dist']['high']
        domain[key] = (low, high, is_log10)
    return domain


def calc_b_from_treeseqs(file,  width=1000, recrate=1e-8, seed=None):
    """
    Recapitate trees and calc B for whole-chromosome BGS simulations

    Note that this computes this in windows [0, width, 2*width, ...],
    *not* at focal positions like B.

    There is slightly less optimal, since we should center these on the focal
    positions and do width-sized window around the focal points, but the
    difference is likely insignicant.
    """
    ts = pyslim.load(file)
    md = ts.metadata['SLiM']['user_metadata']
    region_length = md['region_length'][0]
    N = md['N'][0]
    #recmap = load_recrates('../data/annotation/rec_100kb_chr10.bed', ts.sequence_length)
    rts = pyslim.recapitate(ts, recombination_rate=recrate, sequence_length=ts.sequence_length,
                            ancestral_Ne=N, random_seed=seed)
    length = int(ts.sequence_length)
    neut_positions = np.linspace(0, length, length // width).astype(int)
    # extract the specified simulation parameters from the ts metadata
    params = {k: md[k][0] for k in md.keys()}
    B = rts.diversity(mode='branch', windows=neut_positions) / (4*N)
    return params, neut_positions, B

def load_b_chrom_sims(dir, progress=True, **kwargs):
    """
    Load a batch of BGS simulations for an entire chromosome, and store the
    positions and arary of Bs (across simulation replicates!) in a dictionary
    with (sh, mu) parameters.


    WARNING: these should be the only varying parameters across these
    simulations!

    The results are grouped by these parameters (e.e. across replicates),
    and combined into arrays.

    **kwargs are passed to calc_b_from_treeseqs().
    """
    tree_files = get_files(dir, suffix='.tree')
    sims = defaultdict(list)

    if progress:
        tree_files = tqdm.tqdm(tree_files)

    # this is a check to make sure no other  parameters are varying
    # across these simulations other than mu and sh
    unique_keys = defaultdict(set)
    for file in tree_files:
        sim_params, pos, b = calc_b_from_treeseqs(file, **kwargs)
        # merge s and h into sh since that's what we care about now
        sim_params['sh'] = sim_params.pop('s') * sim_params.pop('h')
        for param in sim_params:
            unique_keys[param].add(sim_params[param])
        param_key = (sim_params['sh'], sim_params['mu'])
        sims[param_key].append((pos, b))

    # now, let's check to make sure that only sh and mu vary
    for param in unique_keys:
        if len(unique_keys[param]) > 1:
            if param in ('sh', 'mu'):
                continue
            raise ValueError(f"key '{param}' has multiple values!")

    for key, res in sims.items():
        # get the position and Bs; skip 0 in pos so they're the same size
        pos = list(map(operator.itemgetter(0), res))[1:]
        b = list(map(operator.itemgetter(1), res))
        sims[key] = np.stack(pos)[0, :], np.stack(b).T
    return sims
