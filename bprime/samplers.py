import numpy as np
import scipy.stats as stats
from bprime.utils import signif
from bprime.sim_utils import random_seed

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

def fixed(rng, val):
    # rng is ignored
    def func():
        return val
    return func

DISTS = {"fixed": fixed,
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


def target_rejection_sampler(func, sampler, burnin=1000, n=10000, nbins=50, range=(0, 1)):
    """
    Experimental rejection sampler based on the target density.
    Each sample is a vector X ~ f(), and we compute g(X). We want
    g(X) ~ Unif(range); since we don't know PDF(g(X)), we need to estimate it
    while we do the rejection sampling. We do this initiall with a burnin, then
    run the burnin samples through the rejection sampler. The epdf is updated
    each sample, so gets more accurate.
    """
    rng = sampler.rng
    bins = np.linspace(*range, nbins)
    counts = np.full(nbins, 1)

    # burnin to estimate the PDF of proposal
    burnin_left = burnin
    burnin_samples = []
    while burnin_left:
        sample = next(sampler)
        burnin_samples.append(sample)
        y = func(**sample)
        bin_i = np.digitize(y, bins)
        counts[bin_i] += 1
        burnin_left -= 1
    epdf = counts/burnin

    # now, with an estimate of the PDF of the proposal, we can
    # proceed as normal
    total = burnin
    nreject = 0
    i = 0
    c = 1.1
    accepted = []
    ys = np.zeros(n)
    nremain = n
    target_prob = 1/nbins
    sampler = itertools.chain(burnin_samples, sampler)
    while nremain:
        try:
            sample = next(sampler)
        except StopIteration:
            break
        y = func(**sample)
        bin_i = np.digitize(y, bins)
        counts[bin_i] += 1 # always re-estimate the epdf
        proposal_prob = counts[bin_i] / total
        ratio = target_prob/proposal_prob
        U = rng.uniform(0, 1)
        accept = U <= ratio/c
        if accept:
            accepted.append(sample)
            ys[i] = y
            i += 1
            nremain -= 1
        else:
            nreject += 1
        total +=1
        c = max(c, ratio)
        print(f"nremain: {nremain}, reject: {np.round(100*nreject/total, 2)}%, total: {total}, ratio: {ratio}, c: {c}", end='\r')
    #return accepted, ys, bins, counts, total
    return accepted

class Sampler(object):
    def __init__(self, params, total=None, seed=None, add_seed=True,
                 seed_max=2**32, signif_digits=4):
        assert isinstance(total, int) or total is None
        assert isinstance(seed, int) or seed is None
        assert isinstance(params, dict), "'params' must be a dict"
        if seed is None:
            seed = random_seed()
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
        assert self.samples_remaining is None or self.samples_remaining >= 0
        sample = self.sampler()
        if self.total is not None:
            self.samples_remaining -= 1
        self.samples.append(sample)
        return sample

    def __repr__(self):
        remain = f"{self.samples_remaining}/{self.total}" if self.total is not None else '∞'
        rows = [f"Sampler with {remain} samples remaining, seed={self.seed}"]
        for key, params in self.params.items():
            dist_name = params['dist']['name']
            dist_params = ', '.join([f"{k}={v}" for k, v in params['dist'].items() if k != 'name'])
            row = f"  {key} ~ {dist_name}({dist_params})"
            rows.append(row)
        return "\n".join(rows)

