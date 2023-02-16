"""
Clean up and combine the African-American 
recmap (why people don't use standard formats is a mystery!)
"""
import sys
from os.path import join

dir = sys.argv[1]

chroms = [f"chr{c}" for c in range(1, 23)]
files = [f"AAmap.{c}.txt" for c in chroms]

for chrom, file in zip(chroms, files):
    with open(join(dir, file)) as f:
        header = next(f)
        assert header.startswith("Physical")
        for line in f:
            pos, rate = line.strip().split()
            print('\t'.join((chrom, pos, str(int(pos)+1), rate)))
