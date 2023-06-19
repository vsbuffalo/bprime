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

encountered_genes = set()

for entry in read_gff(sys.argv[1], parse_attributes=True):
    feature = entry['feature']
    # genes only!
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

        if end-start == 0:
            # skip zero-bp entries
            continue
        gene_feature_id = entry['feature'] + '_' + gene_id
        cols.append(gene_feature_id)
        encountered_genes.add((chrom, gene_feature_id))
        name = f"{feature}_{gene_id}"
        print('\t'.join(cols))

with open(sys.argv[3], 'w') as f:
    for cols in encountered_genes:
        f.write('\t'.join(cols) + '\n')
