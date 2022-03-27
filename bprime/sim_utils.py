## sim_utils.py -- common utility functions for msprime and slim

import itertools
import numpy as np

def random_seed():
    return np.random.randint(0, 2**63)

def fixed_params(params):
    """
    Iterate through the parameters and return the fixed parameters,
    either because a grid only has one element or lower/upper are
    the same.
    """
    fixed = dict()
    for key, param in params.items():
        if isinstance(param, tuple):
            lower, upper, _ = param
            if lower == upper:
                fixed[key] = lower
        elif isinstance(param, list):
            if len(param) == 1:
                fixed[key] = param[0]
        else:
            raise ValueError("param items must be tuple of list")
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


def infer_types(params):
    "peek at the first element's type"
    param_types = {}
    for k, v in params.items():
        if isinstance(v, tuple):
            # this is a range, with lower, upper, log10
            assert len(v) == 3
            assert type(v[0]) == type(v[1])
            param_types[k] = type(v[0])
        elif isinstance(v, list):
            # this is a grid
            param_types[k] = type(v[0])
    return param_types


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
    for param, vals in config['params'].items():
        assert param != "rep", "invalid param name 'rep'!"
        val_type = {'float': float, 'int': int, 'str':str}.get(vals['type'], None)
        is_grid = "grid" in vals
        if is_grid:
            assert("lower" not in vals)
            assert("upper" not in vals)
            assert("log10" not in vals)
            params[param] = [val_type(v) for v in vals['grid']]
        else:
            lower, upper = vals['lower'], vals['upper']
            log10 = vals['log10']
            params[param] = (val_type(lower), val_type(upper), log10)
    if add_rep and is_grid:
        nreps = int(config['nreps'])
        params["rep"] = list(range(nreps))
    return params


