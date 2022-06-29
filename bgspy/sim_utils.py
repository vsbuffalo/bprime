## sim_utils.py -- common utility functions for msprime and slim
import os
import re
import warnings
from functools import partial
import itertools
import operator
from collections import defaultdict
import numpy as np
import pyslim
import tqdm
import multiprocessing
from bgspy.utils import get_files, random_seed, bin_chrom

SIM_REGEX = re.compile(r'chrombgs_chr10_N1000_mu(?P<mu>[^_]+)_sh(?P<sh>[^_]+)_chr10_seed\d+_rep(?P<rep>[^_]+)_treeseq.tree')

def fixed_params(params):
    """
    Iterate through the parameters and return the fixed parameters,
    either because a grid only has one element or lower/upper are
    the same.
    """
    fixed = dict()
    for key, param in params.items():
        if 'grid' in param:
            # handle grid params
            if len(param['grid']) == 1:
                fixed[key] = param['grid'][0]
        else:
           # it's a distribution parameter set
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
        if 'grid' in param:
            is_fixed = len(param['grid']) == 1
            is_log10 = False
            low, high = np.min(param['grid']), np.max(param['grid'])
        else:
            is_fixed = param['dist']['name'] == 'fixed'
            is_log10 = param['dist']['name'].startswith('log10_')
            if is_fixed:
                low, high = param['dist']['val'], param['dist']['val']
            else:
                low, high = param['dist']['low'], param['dist']['high']
        domain[key] = (low, high, is_log10)
    return domain


def calc_b_from_treeseqs(file, width=1000, recrate=1e-8, seed=None):
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
    neut_positions = bin_chrom(length+1, width).astype(int)
    # extract the specified simulation parameters from the ts metadata
    params = {k: md[k][0] for k in md.keys()}
    B = rts.diversity(mode='branch', windows=neut_positions) / (4*N)
    return params, neut_positions, B


def parse_sim_filename(file):
    res = SIM_REGEX.match(os.path.basename(file)).groupdict()
    res['mu'] = float(res['mu'])
    res['sh'] = float(res['sh'])
    res['rep'] = int(res['rep'])
    return res

def load_b_chrom_sims(dir, progress=True, ncores=None, **kwargs):
    """
    Load a batch of BGS simulations for an entire chromosome, and store the
    positions and array of Bs (across simulation replicates!) in a dictionary
    with (sh, mu) parameters.


    WARNING: these should be the only varying parameters across these
    simulations!

    The results are grouped by these parameters (e.e. across replicates),
    and combined into arrays.

    **kwargs are passed to calc_b_from_treeseqs().
    """
    tree_files = get_files(dir, suffix='.tree')
    params = [parse_sim_filename(x) for x in tree_files]
    mus = set(x['mu'] for x in params)
    shs = set(x['sh'] for x in params)
    reps = set(x['rep'] for x in params)

    # read one file in to get the number of positions
    _, pos, b = calc_b_from_treeseqs(tree_files[0], **kwargs)
    npos = len(pos)

    # allocate matrix for results
    # we do one less than position since these are window boundaries
    # including start
    X = np.full((npos-1, len(mus), len(shs), len(reps)), np.nan, dtype=np.single)

    # for indices
    mu_lookup = {mu: i for i, mu in enumerate(sorted(mus))}
    sh_lookup = {sh: i for i, sh in enumerate(sorted(shs))}

    if ncores is None or ncores == 1:
        if progress:
            tree_files = tqdm.tqdm(tree_files)
        for file in tree_files:
            sim_params, pos, b = calc_b_from_treeseqs(file, **kwargs)
            rep = parse_sim_filename(file)['rep']
            mu, sh = sim_params['mu'], sim_params['sh']
            X[:, mu_lookup[mu], sh_lookup[sh], rep] = b
    else:
        with multiprocessing.Pool(ncores) as p:
            func = partial(calc_b_from_treeseqs, **kwargs)
            res = list(tqdm.tqdm(p.imap(func, tree_files), total=len(tree_files)))
            for (sim_params, pos, b), file in zip(res, tree_files):
                rep = parse_sim_filename(file)['rep']
                mu, sh = sim_params['mu'], sim_params['sh']
                X[:, mu_lookup[mu], sh_lookup[sh], rep] = b
    
    mu = np.fromiter(mu_lookup.keys(), dtype=float)
    sh = np.fromiter(sh_lookup.keys(), dtype=float)
    return mu, sh, pos, X
