# slim.py -- helpers for snakemake/slim sims
import os
import itertools
import warnings
import numpy as np

def filename_pattern(base, params, seed=False, rep=False):
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
    pattern = base + '_'.join(param_str)
    return pattern


def infer_types(params):
    """
    Infer the types from a dictionary of parameters.
    """
    types = dict()
    for param, values in params.items():
        types[param] = type(values[0])
    return types

def slim_call(param_types, script, slim_cmd="slim", seed=False, manual=None):
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
    if seed:
        call_args.append("-s {wildcards.seed}")
    full_call = f"{slim_cmd} " + " ".join(call_args) + add_on + " " + script
    return full_call

def random_seed():
    return np.random.randint(0, 2**63)

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
    param_types = {}
    for param, vals in config['params'].items():
        assert param != "rep", "invalid param name 'rep'!"
        val_type = {'float': float, 'int': int}.get(vals['type'], None)
        is_grid = "grid" in vals
        if is_grid:
            assert("lower" not in vals)
            assert("upper" not in vals)
            assert("log10" not in vals)
            params[param] = vals['grid']
            param_types[param] = val_type
        else:
            lower, upper = vals['lower'], vals['upper']
            log10 = vals['log10']
            params[param] = (val_type(lower), val_type(upper), log10)
            param_types[param] = val_type
    if add_rep and is_grid:
        params["rep"] = list(range(config['nreps']))
        param_types["rep"] = int
    return params, param_types

def signif(x, digits=4):
    return np.round(x, digits-int(floor(log10(abs(x))))-1)

class SlimRuns(object):
    def __init__(self, config, dir='.', sampler=None, seed=None):
        msg = "runtype must be 'grid' or 'samples'"
        assert config['runtype'] in ['grid', 'samples'], msg
        self.runtype = config['runtype']
        self.name = config['name']
        if self.is_grid:
            self.nreps = config['nreps']
        if self.is_samples:
            self.nsamples = config['nsamples']

        self.script = config['slim']
        msg = f"SLiM file '{self.script}' does not exist"
        assert os.path.exists(self.script), msg

        self.params, self.param_types = read_params(config)
        self.dir = os.path.join(dir, self.name)
        self.basename = os.path.join(self.dir, f"{self.name}_")
        self.seed = seed if seed is not None else random_seed()
        self.sampler_func = sampler
        self.sampler = None

    def generate(self):
        """
        Run the sampler to generate samples or expand out the parameter grid.
        """
        if self.is_grid:
            if self.sampler_func is not None:
                warnings.warn("sampler specified but runtype is grid!")
            self.runs = param_grid(self.params)
        else:
            self.sampler = self.sampler_func(self.params, total=self.nsamples, seed=self.seed)
            self.runs = list(self.sampler)

    @property
    def is_grid(self):
        return self.runtype == 'grid'

    @property
    def is_samples(self):
        return self.runtype == 'samples'

    def slim_call(self, slim_cmd='slim', manual=None):
        """
        Return a string SLiM call, with SLiM variables passed as command
        line arguments and retrieved from SLiM wildcards, e.g.
        slim -v val={wildcards.val} (for use with a structured filename).

        Passes in the name of the run.
        """
        name = {'name': self.name}
        if manual is not None:
            manual = {**name, **manual}
        else:
            manual = name
        return slim_call(self.param_types, self.script,
                         slim_cmd=slim_cmd, manual=manual)

    def slim_commands(self, *args, **kwargs):
        call = self.slim_call(*args, **kwargs).replace("wildcards.", "")
        if self.runs is None:
            raise ValueError("run SlimRuns.generate()")
        for wildcards in self.runs:
            yield call.format(**wildcards)

    @property
    def filename_pattern(self):
        """
        Return the filename pattern with wildcards.
        """
        return filename_pattern(self.basename, self.params.keys())

    def wildcard_output(self, suffix):
        """
        For the 'output' entry of a Snakemake rule, this returns the
        expected output files with wildcards, with suffix attached.
        'suffix' can be a list/tuple of many outputs or a string.
        """
        if isinstance(suffix, (list, tuple)):
            return [f"{self.filename_pattern}_{end}" for end in suffix]
        elif isinstance(suffix, str):
            return f"{self.filename_pattern}_{suffix}"
        else:
            raise ValueError("suffix must be list/tuple or str")

    @property
    def param_order(self):
        "SLiM constructs filename automatically too; this is the param order"
        return ', '.join(self.params.keys())

    def targets(self, suffix):
        """
        Get all targets.
        """
        if self.runs is None:
            raise ValueError("run SlimRuns.generate()")
        if isinstance(suffix, str):
            suffix = [suffix]
        targets = []
        for run_params in self.runs:
            for end in suffix:
                filename = f"{self.filename_pattern}_{end}"
                targets.append(filename.format(**run_params))
        return targets







