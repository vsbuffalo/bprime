import os
import sys
import tqdm
import numpy as np


from bgspy.sim_utils import load_b_chrom_sims


if __name__ == "__main__":
    ncores = 6
    mu, sh, pos, X = load_b_chrom_sims(sys.argv[1], ncores=ncores)

    nreps = X.shape[3]
    mean = X.mean(axis=3)
    sd = X.std(axis=3)
    np.savez(sys.argv[2], mu=mu, sh=sh, pos=pos, X=X, nreps=nreps, mean=mean, sd=sd)

   

 
