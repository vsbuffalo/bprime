"""
Clean up and combine the African-American 
recmap (why people don't use standard formats is a mystery!)

Note that this file's data column is cumulative rates,
so I take the difference between adjacent markers for local
estimates
"""
import sys
from os.path import join

dir = sys.argv[1]

chroms = [f"chr{c}" for c in range(1, 23)] + ['chrX']
files = [f"AAmap.{c}.txt" for c in chroms]

for chrom, file in zip(chroms, files):
    with open(join(dir, file)) as f:
        header = next(f)
        assert header.startswith("Physical")
        line = next(f)
        pos, rate = line.strip().split()
        #last_pos, last_rate = int(pos), float(rate)
        for line in f:
            pos, rate = line.strip().split()
            pos, rate = int(pos), float(rate)
            #assert pos > last_pos
            # convert from cM to Morgans
            row = map(str, (chrom, pos, rate))
            print('\t'.join(row))
            #last_pos, last_rate = pos, rate
