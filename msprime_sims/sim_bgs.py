from multiprocessing import Pool
import numpy as np
import msprime
import tqdm
import pickle


def bgs_rec(mu, s, r, L):
    return np.exp(-L * mu/(s*(1+(1-s)*r/s)**2))

class UniformSampler(object):
    def __init__(self, ranges, total=None, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.ranges = ranges
        self.total = total
        self.samples_remaining = total
        self.samples = []

    def __iter__(self):
        return self

    @property
    def __len__(self):
        return len(self.samples)

    def __next__(self):
        if self.total is not None and self.samples_remaining == 0:
            raise StopIteration
        assert(self.samples_remaining >= 0)
        param = []
        for key, (lower, upper, log10) in self.ranges.items():
            if lower == upper:
                # fixed parameter
                if log10:
                    param.append(10**lower)
                else:
                    param.append(lower)
                continue
            sample = np.random.uniform(lower, upper)
            if log10:
                sample = 10**sample
            param.append(sample)
        self.samples_remaining -= 1
        self.samples.append(param)
        return param

bgs_ranges = {'mu': (-8, -7, True),
              's': (-3, -1, True),
              'r': (-8, -8, True),
              'N': (3, 3, True),
              'L': (2, 5, True),
              'nreps': (200, 200, False)}


#bgs_sampler = UniformSampler(bgs_ranges, total=100, seed=1)
bgs_sampler = UniformSampler(bgs_ranges, total=100_000, seed=1)

def bgs_rec_runner(param):
    mu, s, r, N, L, nreps = param
    B = bgs_rec(mu, s, r, L)
    Bhats = [msprime.sim_ancestry(N, population_size=B*N).diversity(mode='branch')/(4*N)
             for _ in range(nreps)]
    return Bhats


NCORES = 70
with Pool(NCORES) as p:
    results = list(tqdm.tqdm(p.imap(bgs_rec_runner, bgs_sampler), total=bgs_sampler.total))

with open("bgs_rec.pkl", 'wb') as f:
    pickle.dump((bgs_sampler.samples, results), f)
