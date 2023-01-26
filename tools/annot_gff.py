import sys
from collections import defaultdict
from bgspy.utils import read_gff

# there's an ID mapping file
gene_map = defaultdict(list)
tx_map = dict()

for line in open(sys.argv[2]):
    gene_id, tx_id = line.strip().split('\t')
    gene_map[gene_id].append(tx_id)
    tx_map[tx_id] = gene_id

out_prefix = sys.argv[3]
open_fh = dict()

for entry in read_gff(sys.argv[1], parse_attributes=True):
    feature = entry['feature']
    if feature in ('CDS', 'five_prime_UTR', 'three_prime_UTR'):
        cols = [entry['seqname'], entry['start'], entry['end']]
        tx_id = entry['attribute']['Parent'].replace('transcript:', '')
        try:
            gene_id = tx_map[tx_id]
        except KeyError:
            # skip a row not in the mapping, e,g. non-canonical
            continue

        chrom, start, end = cols
        start, end = int(start), int(end)
         
        feature = 'UTR' if feature in ('five_prime_UTR', 'three_prime_UTR') else feature

        if end-start == 0:
            # skip zero-bp entries
            continue
        name = f"{feature}_{gene_id}"
        cols.append(name)

        if chrom not in open_fh:
            open_fh[chrom] = open(f"{out_prefix}_{chrom}.bed", 'w')

        open_fh[chrom].write('\t'.join(cols) + '\n')
