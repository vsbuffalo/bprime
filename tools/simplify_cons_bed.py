import sys

# in order of most to least conserved
priority = ('phastcons', 'CDS', 'exon', 'UTR', 'intron', 'pseudogene')
order = {f: i for i, f in enumerate(priority)}

for line in sys.stdin:
    chrom, start, end, feature = line.strip().split('\t')
    if int(end) - int(start) < 1:
        # skip zero-width elements
        continue
    features = feature.split(',')
    idx = [order[f] for f in features]
    row = "\t".join([chrom, start, end, priority[min(idx)]])
    print(row)
