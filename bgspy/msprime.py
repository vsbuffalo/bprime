import numpy as np
import msprime
from bgspy.utils import Bhat, random_seed 
from bgspy.samplers import Sampler

def msprime_spike(params, N, total, reps, tracklen=10, seed=None):
    """
    Experimental: spike in L=0 neutral sims

    If L = 0, there is no selection, so this is invariant to all other params.
    """
    params['L'] = {'dist': {'name': 'fixed', 'val': 0}, 'type': 'int'}
    expected_features = ('mu', 'sh', 'L', 'rbp', 'rf')
    # put things in the right order
    params = {k: params[k] for k in expected_features}
    sampler = Sampler(params=params, total=total, seed=seed)
    y = []
    keys = []
    rep_col = []
    for sample in sampler:
        # we make a key; this is different than the slim ones (not a filename)
        key = '_'.join([f"{v}{k}" for k, v in sample.items()])
        for rep in range(reps):
            ts = msprime.sim_ancestry(samples=2*N, population_size=N, ploidy=2, 
                                      recombination_rate=0, sequence_length=tracklen,
                                      random_seed=random_seed(sampler.rng))
            pi = ts.diversity(mode='branch')
            bhat = Bhat(pi, N)
            # this must match the rows in process_tree_file
            y_row = (pi, bhat, 0, 0, 0)
            y.append(y_row)
            keys.append(key)
            rep_col.append(rep)
            
    X, features = sampler.as_matrix()
    X = np.repeat(X, reps, axis=0)
    X = np.append(X, np.array(rep_col)[:, None], axis=1)
    targets = ('pi', 'Bhat', 'Ef', 'Vf', 'load')
    assert tuple(features) == expected_features, "feature mismatch!"
    features.append('rep')
    return np.array(X), np.array(y), features, targets, keys
    

    


