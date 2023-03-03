# slim.py -- helpers for snakemake/slim sims

import os
import copy
from math import ceil
from collections import defaultdict
import itertools
import warnings
import numpy as np
from bgspy.utils import signif
from bgspy.sim_utils import param_grid, read_params, random_seed



def filename_pattern(dir, base, params, split_dirs=False, seed=False, rep=False):
    """
    Create a filename pattern with wildcares in braces (for Snakemake)
    from a basename 'base' and list of parameters. If 'seed' or 'rep' are
    True, these will be added in manually.

    Example:
      input: base='run', params=['h', 's']
      output: 'run_h{h}_s{s}'
    """
    param_str = [v + '{' + v + '}' for v in params]
    if seed:
        param_str.append('seed{seed}')
    if rep:
        param_str.append('rep{rep}')
    if split_dirs:
        base = os.path.join(dir, '{subdir}', base)
    else:
        base = os.path.join(dir, base)
    pattern = base + '_'.join(param_str)
    return pattern


def slim_call(param_types, script, slim_cmd="slim", 
              add_seed=False, add_rep=False, manual=None):
    """
    Create a SLiM call prototype for Snakemake, which fills in the
    wildcards based on the provided parameter names and types (as a dict).

    param_types: dict of param_name->type entries
    slim_cmd: path to SLiM
    seed: bool whether to pass in the seed with '-s <seed>' and add a
            seed between 0 and 2^63
    manual: a dict of manual items to pass in
    """
    call_args = []
    for p, val_type in param_types.items():
        is_str = val_type is str
        if is_str:
            # silly escapes...
            val = f'\\"{{wildcards.{p}}}\\"'
        else:
            val = f"{{wildcards.{p}}}"
        call_args.append(f"-d {p}={val}")
    add_on = ''
    if manual is not None:
        # manual stuff
        add_on = []
        for key, val in manual.items():
            if isinstance(val, str):
                add_on.append(f'-d {key}=\\"{val}\\"')
            else:
                add_on.append(f'-d {key}={val}')
        add_on = ' ' + ' '.join(add_on)
    if add_seed:
        call_args.append("-s {wildcards.seed}")
    if add_rep:
        call_args.append("-d rep={wildcards.rep}")
    full_call = f"{slim_cmd} " + " ".join(call_args) + add_on + " " + script
    return full_call


class SlimRuns():
    def __init__(self, config):
        self.config = config
        self.dir = config['dir']
        self.fixed = config['fixed']
        self.params = config['params']
        self.nreps = config['nreps']

    def generate(self):
        keys = list(self.params.keys())
        param_grid(self.params)

