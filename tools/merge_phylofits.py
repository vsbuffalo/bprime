import sys
import os
import pandas as pd
import re
from bgspy.utils import readfq, readfile
from bgspy.utils import get_branch_length, read_phylofit

PF_DIR = sys.argv[1]
KEEP_CHROMS = [f'chr{x}' for x in range(1, 23)]

chroms, starts, ends = [], [], []
rates, lens = [], []
genes = []

for file in os.listdir(PF_DIR):
    loc = re.match('(?P<ucsc_id>[^_]+)_(?P<chrom>chr[0-9X_]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-]).mod', file)
    fasta_file = './cds_alns/' + file.replace('.mod', '.fa')
    name, seq, _ = next(readfq(readfile(fasta_file)))
    seqlen = len(seq)

    if loc is None:
        continue
    gene, chrom, start, end, strand = loc.groups()

    if chrom not in KEEP_CHROMS:
        continue

    rate = get_branch_length(read_phylofit(os.path.join(PF_DIR, file))['tree'], 'hg38')
    chroms.append(chrom)
    starts.append(int(start))
    ends.append(int(end))
    rates.append(rate)
    lens.append(seqlen)
    genes.append(gene)

pf = pd.DataFrame(dict(gene=genes, chrom=chroms, start=starts, end=ends,
                        rate=rates, len=lens))
pf = pf.sort_values(by=['chrom', 'start', 'end'])
pf.to_csv(sys.argv[2], sep='\t', header=False, index=False)

