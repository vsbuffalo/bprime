## sim_utils.py -- common utility functions for msprime and slim
import warnings
import itertools
import numpy as np

SEED_MAX = 2**32-1

def random_seed(rng=None):
    
    if rng is None:
        return np.random.randint(0, SEED_MAX)
    return rng.integers(0, SEED_MAX)

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

def param_grid(params, seed=False):
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
    out = list(map(dict, itertools.product(*grid)))
    if not seed:
        return out
    for entry in out:
        entry['seed'] = random_seed()
    return out

def read_params(config, add_rep=True):
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
    if add_rep and is_grid:
        nreps = int(config['nreps'])
        params["rep"] = list(range(nreps))
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



