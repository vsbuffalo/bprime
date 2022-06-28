import os
import sys
import numpy as np


from bgspy.sim_utils import load_b_chrom_sims


if __name__ == "__main__":
    X = load_b_chrom_sims(sys.argv[1])
    np.save(sys.argv[2], X)
    

 
