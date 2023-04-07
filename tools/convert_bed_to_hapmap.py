import sys
from bgspy.utils import readfile

last_chrom = None
last_pos = None
print("Chromosome\tPosition(bp)\tRate(cM/Mb)")
for line in readfile(sys.argv[1]):
    if line.startswith('#') or line.startswith('Chr'):
        continue
    chrom, start, end, rate, cumrate = line.strip().split('\t')
    start, end = int(start), int(end)
    if last_pos is not None and (start != last_pos and (chrom == last_chrom)):
        raise ValueError(f"recmap is not full coverage, cannot convert ({chrom}:{start}-{end})")
    print(f"{chrom}\t{start}\t{rate}")
    last_chrom = chrom
    last_pos = end
