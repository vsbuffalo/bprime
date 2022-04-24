import numpy as np
import scipy.stats as stats
from bprime.utils import signif

def discrete_uniform(rng, low, high):
    def func():
        return rng.integers(low, high)
    return func

def uniform(rng, low, high):
    def func():
        return rng.uniform(low, high)
    return func

def log10_uniform(rng, low, high):
    def func():
        return 10**rng.uniform(low, high)
    return func

def log10_truncnormal(rng, low, high, loc, scale):
    "Have to use stats.trucnorm here -- careful to pass on random state"
    a = (low - loc)/scale
    b = (high - loc)/scale
    def func():
        return 10**stats.truncnorm.rvs(a=a, b=b, loc=loc, scale=scale, random_state=rng)
    return func


def truncnormal(rng, low, high, loc, scale):
    "Have to use stats.trucnorm here -- careful to pass on random state"
    def func():
        return stats.truncnorm.rvs(a=low, b=high, loc=loc, scale=scale, random_state=rng)
    return func

def normal(rng, loc, scale):
    def func():
        return rng.normal(loc, scale)
    return func

def mixed(rng, low, high, prob_log=0.5):
    def func():
        use_log = rng.binomial(1, prob_log, 1)
        if use_log:
            return rng.uniform(10**low, 10**high, 1)
        return 10**rng.uniform(low, high, 1)
    return func
    
def fixed(rng, val):
    # rng is ignored
    def func():
        return val
    return func

DISTS = {"fixed": fixed,
         "mixed": mixed,
         "uniform": uniform,
         "normal": normal,
         "truncnormal": truncnormal,
         "log10_truncnormal": log10_truncnormal,
         "discrete_uniform": discrete_uniform,
         "log10_uniform": log10_uniform}

TYPES = {"float": float, "int": int}

def sampler_factory(rng, params, add_seed=False, signif_digits=None, seed_max=2**32):
    samplers = dict()
    types = dict()
    assert 'seed' not in params.keys(), "seed cannot be a parameter name"
    for key, parameters in params.items():
        dist = parameters['dist']['name']
        try:
            distfunc = DISTS[dist]
        except KeyError:
            raise KeyError(f"'{dist}' not in dist functions, {', '.join(DISTS)}")
        dist_params = {k: v for k, v in parameters['dist'].items() if k != 'name'}
        samplers[key] = distfunc(rng=rng, **dist_params)
        try:
            types[key] = TYPES[parameters['type']]
        except KeyError:
            raise KeyError(f"invalid type {parameters['type']}")

    def func():
        sample = {}
        for key, sampler in samplers.items():
            type_converter = types[key]
            s = type_converter(sampler())
            if signif_digits is not None and types[key] is float:
                s = signif(s, signif_digits)
            sample[key] = s
        if add_seed:
            sample['seed'] = rng.integers(0, seed_max)
        return sample
    return func

class Sampler(object):
    def __init__(self, params, total, seed=None, add_seed=True,
                 seed_max=2**32, signif_digits=4):
        assert isinstance(total, int) or total is None
        assert isinstance(seed, int) or seed is None
        assert isinstance(params, dict), "'params' must be a dict"
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.add_seed = add_seed
        self.seed_max = seed_max
        self.params = params
        self.total = total
        self.samples_remaining = total
        self.samples = []
        self.signif_digits = signif_digits
        self.sampler = sampler_factory(self.rng, self.params, add_seed=add_seed,
                                       signif_digits=signif_digits,
                                       seed_max=seed_max)

    def __iter__(self):
        return self

    @property
    def __len__(self):
        return len(self.samples)

    def as_matrix(self, return_cols=True):
        assert len(self.samples) > 0, "No samples have been taken."
        mat = np.array([[row[k] for k in self.params.keys()] for row
                        in self.samples])
        if not return_cols:
            return mat
        return mat, list(self.params.keys())

    def __getitem__(self, key):
        """
        Get the sampled values of key as an array.
        """
        assert len(self.samples) > 0, "No samples have been taken."
        return np.array([x[key] for x in self.samples])

    def generate(self):
        """
        Generate all the samples in place.
        """
        _ = list(self)

    def __next__(self):
        if self.total is not None and self.samples_remaining == 0:
            raise StopIteration
        assert self.total is not None, "total not set"
        assert self.samples_remaining >= 0
        sample = self.sampler()
        self.samples_remaining -= 1
        self.samples.append(sample)
        return sample

    def __repr__(self):
        rows = [f"Sampler with {self.samples_remaining}/{self.total} samples remaining, seed={self.seed}"]
        for key, params in self.params.items():
            dist_name = params['dist']['name']
            dist_params = ', '.join([f"{k}={v}" for k, v in params['dist'].items() if k != 'name'])
            row = f"  {key} ~ {dist_name}({dist_params})"
            rows.append(row)
        return "\n".join(rows)

