# slim.py -- helpers for snakemake/slim sims

import os
from os.path import join
import copy
from math import ceil
from collections import defaultdict
import itertools
import warnings
import numpy as np
from bgspy.utils import signif
from bgspy.sim_utils import param_grid, random_seed



def filename_pattern(suffix, rep, seed, params=None):
    """
    Build a filename from parts.
    If params is set, the variable parameters will be
    used to construct a more unique ID.
    """
    # for Snakemake's wildcards:
    if params is not None:
        param_str = '_'.join([v + '{' + v + '}' for v in params]) + '_'
    else:
        param_str = ''
    pattern = f"{param_str}rep{rep}_seed{seed}_{suffix}"
    return pattern


def slim_call(params, script, slim_cmd="slim",
              output=None, add_seed=True, add_rep=True,
              manual=None):
    """
    Create a SLiM call prototype for Snakemake, which fills in the
    wildcards based on the provided parameter names and types (as a dict).

    param: dict of sample parameters, so types can be inferred
    slim_cmd: path to SLiM
    output: a dict of output names, values
    add_seed: bool whether to pass in the seed with '-s <seed>' 
    manual: a dict of manual items to pass in
    """
    param_types = {k: type(v) for k, v in params.items()}
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
    if output is not None:
        for name, file in output.items():
            call_args.append(f"-d \"{name}='{{output.{name}}}'\"")
    full_call = f"{slim_cmd} " + " ".join(call_args) + add_on + " " + script
    return full_call

def create_directories(path):
    if not os.path.exists(path):
        os.makedirs(path)

def params_to_dirs(params, basedir=None, create=False):
    """
    params: full expanded version
    This makes the directories too.
    """
    dir = join(*[f"{k}__{v}" for k, v in params.items()])
    if basedir is not None:
        dir = join(basedir, dir)
    if create:
        create_directories(dir)
    return dir


class SlimRuns():
    """
    This is a class to do aid in setting up multiple SLiM simulations.

    It uses a fairly generation YAML format.

    The goal is to store simulations in a way where we can quickly find them if we 
    need them. The format I use, after some experimentation is:

      dir / name / N / mu / N_mu_

      base run directory / the run name / free_param_1 / \
          free_param_2 / {free_param_1}_{free_param_2}_{rep}_{seed}_filetype.out

     This is the parameter-to-directory scheme, and it puts some mild constraints
     on what can be a free parameter (e.g. we can't have file paths become the name
     of a sub directory).

     - dir: base directory to store the simulations.
     - name: the main run name -- *this should be unique to the fixed parameters*
     - fixed: the things that do not change across the simulations, e.g. duration
     - dependencies: files that are also fixed, but will trigger a re-run
     - free: the things that vary across parameters. Note that this should not be
              paths as this would mess up the params-to-directory scheme.

    TODO: 
     - could have the filepaths be named so they can be a directory.
    """
    def __init__(self, config):
        self.config = config
        self.name = config['name']
        self.script = config['script']
        self.nreps = config['nreps']
        self.seed = config['seed']
        self.rng = ng = np.random.default_rng(self.seed)
        self.dir = config['dir']
        self.suffices = config['suffices']
        self.fixed = config.get('fixed', {})
        self.input = config.get('input', {})
        self.variable = config['variable']
        msg = "'rep' cannot be used as a variable parameter"
        assert 'rep' not in self.variable, msg
        self.nreps = config['nreps']
        self.basedir = join(self.dir, self.name)
        
        simdir = config.get('simdir', None)
        if simdir is not None:
            self.simdir = join(self.basedir, simdir)
        else:
            self.simdir = self.basedir

    def generate_targets(self, suffices=None, outdir=None, nreps=None,
                         include_params=False, create=True, reseed=False):
        """
        Generate all the target filenames.
        """
        # if we want to restart from the seed, we do that here
        rng = self.rng if not reseed else np.random.default_rng(self.seed)
        # make all grid combinations of variable parameters
        all_run_params = param_grid(self.variable)
        targets = []
        if suffices is not None:
            assert isinstance(suffices, dict), "suffices must be a dict"
        else:
            suffices = self.suffices
        if outdir is None:
            # use the simulation directory
            outdir = self.simdir
        else:
            # use this directory
            outdir = join(self.basedir, outdir)

        # allow custom nreps
        nreps = self.nreps if nreps is None else nreps 
        for params in all_run_params:
            # generate the directory structure on the 
            # variable params
            dir = params_to_dirs(params, basedir=outdir, create=create)
            for rep in range(nreps):
                # get a random seed
                seed = random_seed(rng)
                for name, suffix in suffices.items():
                    fn_params = None if not include_params else params
                    filename = filename_pattern(suffix, rep, seed, fn_params)
                    ps = dict(rep=rep, seed=seed, suffix=suffix)
                    if include_params:
                        # merge the two dicts
                        ps = ps | params
                    targets.append(join(dir, filename.format(**ps)))
        return targets

    def slim_cmd(self):
        """
        Fixed parameters, input files, and the name are passed
        directly to the SLiM call here, and are not part of the
        wildcards.
        """
        # all the same fixed parameters are passed manually
        manual = dict(**self.fixed, **self.input,
                      name=self.name)
        # we get one sample run, to extract the types
        all_run_params = list(param_grid(self.variable))[0]
        output = self.suffices
        return slim_call(all_run_params, self.script,
                         add_seed=True, add_rep=True,
                         output=output, manual=manual)

    def output_template(self, outdir=None, suffices=None, 
                        include_params=False):
        """
        Create a dictionary of expected output for all the 
        suffices, with wildcards.

        By default, this will produce outputs for the
        suffixes in the config in the base directory set
        by the config. However, sometimes sims results
        are needed for further downstream results; in this
        case set basedir and suffices to something other
        than None.
        """
        param_wildcards = {p: f"{{{p}}}" for p in self.variable}
        outputs = {}
        if suffices is not None:
            assert isinstance(suffices, dict), "suffices must be a dict"
        else:
            suffices = self.suffices
        if outdir is None:
            # use the simulation directory
            outdir = self.simdir
        else:
            # use this directory
            outdir = join(self.basedir, outdir)
           
        fn_params = None if not include_params else params
        for name, suffix in suffices.items():
            filename = filename_pattern(suffix, '{rep}', '{seed}', fn_params)
            dir = params_to_dirs(param_wildcards,
                                 basedir=outdir)
            outputs[name] = join(dir, filename)
        return outputs
