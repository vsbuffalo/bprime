# slim.py -- helpers for snakemake/slim sims

import os
import copy
import itertools
import warnings
import numpy as np
from bprime.utils import signif
from bprime.sim_utils import param_grid, read_params, random_seed

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


def slim_call(param_types, script, slim_cmd="slim", add_seed=False,
              add_rep=False, manual=None):
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


def time_grower(start_time, factor=1.8):
    """
    Return a function that grows the time with the number of attempts,
    made to be used with a Snakemake rule, e.g.:
        resources:
            runtime=time_grower(initial_runtime)
    initial run time could be set in the config.json file, e.g.
    use time_grower(config['initial_runtime'])
    """
    def time_limit(wildcards, attempt):
        new_time = start_time*(attempt + factor*(attempt-1))
        days = int(new_time // 24)
        time_left = new_time % 24
        hours = int(time_left // 1)
        minutes = int(60*(time_left - (time_left // 1)))
        return f"{days:02d}-{hours}:{minutes}:00"
    return time_limit


class SlimRuns(object):
    def __init__(self, config, dir='.', sampler=None, split_dirs=False,
                 seed=None):
        msg = "runtype must be 'grid' or 'samples'"
        assert config['runtype'] in ['grid', 'samples'], msg
        self.runtype = config['runtype']
        self.name = config['name']
        self.nreps = config.get('nreps', None)
        if self.is_grid:
            assert self.nreps is not None
        if self.is_samples:
            self.nsamples = config['nsamples']

        self.script = config['slim']
        msg = f"SLiM file '{self.script}' does not exist"
        assert os.path.exists(self.script), msg

        self.params, self.param_types = read_params(config)
        self.add_seed = True
        if split_dirs is not None:
            assert isinstance(split_dirs, int), "split_dirs needs to be int"
            # we need to pass in the subdir
            self.param_types = {'subdir': str, **self.param_types}
        self.dir = os.path.join(dir, self.name)
        self.split_dirs = split_dirs
        self.basename = f"{self.name}_"
        self.seed = seed if seed is not None else random_seed()
        self.sampler_func = sampler
        if sampler is None and self.is_samples:
            raise ValueError("no sampler function specified and runtype='samples'")
        self.sampler = None

    def _generate_runs(self, package_rep=True):
        # package_rep is whether to include 'rep' into sample dict
        if self.nreps is None or self.nreps == 1:
            self.runs = list(self.sampler)
        else:
            self.runs = []
            bad = 10
            for sample in self.sampler:
                for rep in range(self.nreps):
                    # draw nreps samples
                    if package_rep:
                        sample = copy.copy(sample)
                        sample['rep'] = rep
                    self.runs.append(sample)
                bad -= 1
                if not bad:
                    break

    def generate(self):
        """
        Run the sampler to generate samples or expand out the parameter grid.
        """
        if self.is_grid:
            if self.sampler_func is not None:
                warnings.warn("sampler specified but runtype is grid!")
            self.runs = param_grid(self.params)
        else:
            self.sampler = self.sampler_func(self.params, total=self.nsamples,
                                             add_seed=True, seed=self.seed)
            self._generate_runs()


        if self.split_dirs is not None:
            runs = []
            for run in self.runs:
                dir_seed = str(run['seed'])[:self.split_dirs]
                new_run = dict(subdir=dir_seed)
                new_run = {**new_run, **run}
                runs.append(new_run)
            self.runs = runs

    @property
    def has_reps(self):
        return self.nreps is not None or self.nreps > 1

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

        return slim_call(self.param_types, self.script, slim_cmd=slim_cmd,
                         add_seed=self.add_seed, add_rep=self.has_reps,
                         manual=manual)

    def slim_commands(self, *args, **kwargs):
        call = self.slim_call(*args, **kwargs).replace("wildcards.", "")
        print("asd", call)
        if self.runs is None:
            raise ValueError("run SlimRuns.generate()")
        for wildcards in self.runs:
            yield call.format(**wildcards)

    @property
    def filename_pattern(self):
        """
        Return the filename pattern with wildcards.
        """
        return filename_pattern(self.dir, self.basename, self.params.keys(),
                                split_dirs=self.split_dirs is not None,
                                seed=self.add_seed, rep=self.has_reps)

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
        """
        SLiM constructs filename automatically too; this is the order, as a
        string of parameters, to use for the filename_str() function.
        """
        return ', '.join(f"'{v}'" for v in self.params.keys())

    def targets(self, suffix):
        """
        Create a list of targets by using the filename pattern, appending
        the suffixes in 'suffix'.
        """
        if self.runs is None:
            raise ValueError("run SlimRuns.generate()")
        if isinstance(suffix, str):
            suffix = [suffix]
        targets = []
        for run_params in self.runs:
            print(run_params)
            for end in suffix:
                filename = f"{self.filename_pattern}_{end}"
                targets.append(filename.format(**run_params))
        return targets







