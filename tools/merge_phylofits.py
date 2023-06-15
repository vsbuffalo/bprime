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
    if not file.endswith('.mod'):
        continue
    loc = re.match('(?P<ucsc_id>[^_]+)_(?P<chrom>chr[0-9X_]+):(?P<start>\d+)-(?P<end>\d+)(?P<strand>[+-]).mod', file)
    fasta_file = './cds_alns/' + file.replace('.mod', '.fa')
    # read the first sequence in the FASTA alignment file (human) and get
    # the length for downstream weighting
    name, seq, _ = next(readfq(readfile(fasta_file)))
    seqlen = len(seq)

    if loc is None:
        continue
    gene, chrom, start, end, strand = loc.groups()

    if chrom not in KEEP_CHROMS:
        continue

    pf = read_phylofit(os.path.join(PF_DIR, file))
    rate = get_branch_length(pf['tree'])['hg38']
    chroms.append(chrom)
    starts.append(int(start))
    ends.append(int(end))
    rates.append(rate)
    lens.append(seqlen)
    genes.append(gene)

pf = pd.DataFrame(dict(chrom=chroms, start=starts, end=ends,
                        gene=genes,
                        rate=rates, seqlen=lens))
pf = pf.sort_values(by=['chrom', 'start', 'end'])
pf.to_csv(sys.argv[2], sep='\t', header=False, index=False)

