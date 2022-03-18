from math import floor, log10
import numpy as np

def index_cols(cols):
    """
    For extracting columns (more safely than remembering indices)
    """
    index = {c: i for i, c in enumerate(cols)}
    def get(*args):
        return tuple(index[c] for c in args)
    return get

def read_params(config):
    """
    Grab the parameter ranges from a configuration dictionary.
    """
    params = {}
    for param, vals in config['params'].items():
      lower, upper = vals['lower'], vals['upper']
      log10 = vals['log10']
      type = {'float': float, 'int': int}.get(vals['type'], None)
      params[param] = (type(lower), type(upper), log10, type)
    return params

def signif(x, digits=4):
    return np.round(x, digits-int(floor(log10(abs(x))))-1)


