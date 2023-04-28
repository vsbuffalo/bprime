import os
import sys
import tqdm
import defopt
import numpy as np
from bgspy.sim_utils import load_b_chrom_sims

def main(simdir: str, output: str, *, ncores: int=1):
   """
   Process Simulation TreeSeqs, calculating empirical B.

   :param simdir: directory to simulation results
   :param output: output .npz file
   :param ncores: number of cores to use
   """
   mu, sh, pos, X, files, pis, r2sum = load_b_chrom_sims(simdir, ncores=ncores)

   nreps = X.shape[3]
   mean = X.mean(axis=3)
   sd = X.std(axis=3)
   np.savez(output, mu=mu, sh=sh, pos=pos, X=X, nreps=nreps,
            mean=mean, sd=sd, files=files, pis=pis, r2sum=r2sum)




if __name__ == "__main__":
    defopt.run(main)


