import os
import sys
import tqdm
import numpy as np

def bootstrap_mean(X, nboot):
    # bootstrap last axis
    shape = tuple(list(X.shape)[:-1] + [nboot])
    boot = np.full(shape, np.nan, dtype=float)
    for b in tqdm.tqdm(range(nboot)):
        idx = np.random.randint(0, nreps, nreps)
        boot[..., b] = X[..., idx].mean(axis=len(X.shape)-1)
    return boot
 

if __name__ == "__main__":
    filename = sys.argv[1]
    d = np.load(filename)
    X = d['X']
    nreps = X.shape[3]
    mean = X.mean(axis=3)
    sd = X.std(axis=3)

    boot = bootstrap_mean(X, 200)
    np.savez(filename.replace('.npz', '_boot.npz'), pos=pos, mu=mu, sh=sh, mean=mean, sd=sd, boot=boot)
 
