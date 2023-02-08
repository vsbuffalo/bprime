import sys
import numpy as np
from collections import Counter
from bgspy.utils import readfile

total_bp = 0
counts = Counter()

for line in readfile(sys.argv[1]):
    chrom, start, end, feature = line.strip().split('\t')
    start, end = int(start), int(end)
    width = end-start
    counts[feature] += width
    total_bp += width


for key, n in counts.items():
    print(f"{key}\t{n}\t{np.round(n/total_bp, 6)}")


