import sys
import pickle
from bgspy.utils import readfq, readfile

fp = readfile(sys.argv[1])

keep_chroms = list(map(str, range(1, 23))) + ['X']

cds_lens = dict()

for name, seq, _ in readfq(fp, name_only=False):
    gene_id = [x for x in name.split(' ') if x.startswith('gene:')][0]
    gene_id = gene_id.replace('gene:', '')
    gene_id = gene_id.split('.')[0]
    loc = [x for x in name.split(' ') if x.startswith('chromosome:GRCh38')]
    if not len(loc):
        continue
    chrom, start, end, _ = loc[0].replace('chromosome:GRCh38:', '').split(':')
    if chrom not in keep_chroms:
        continue
    tx_id = name.split(' ')[0]
    print('\t'.join(map(str, ['chr'+chrom, start, end, gene_id, tx_id, len(seq)])))

    *_, curr_len = cds_lens.get(gene_id, (None, None, None, 0))
    if curr_len < len(seq):
        cds_lens[gene_id] = 'chr'+chrom, int(start), int(end), len(seq)

with open(sys.argv[2], 'wb') as f:
    pickle.dump(cds_lens, f)
