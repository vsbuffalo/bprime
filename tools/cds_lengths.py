import sys
import pickle
from collections import Counter, defaultdict
import numpy as np
from bgspy.utils import readfq, readfile
from CAI import CAI, relative_adaptiveness

START = 'ATG'
STOP = ['TGA', 'TAG', 'TAA']

keep_chroms = list(map(str, range(1, 23))) + ['X']

missing_start = 0
incorrect_length = 0
invalid = 0

cleaned_codons = dict()
locs = dict()

fp = readfile(sys.argv[1])
for name, seq, _ in readfq(fp, name_only=False):
    gene_id, tx_id, chrom, start, end = name.split('|')
    seq = seq.upper()

    if chrom not in keep_chroms:
        continue

    if len(seq) % 3:
        incorrect_length += 1

    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    stops = [i for i, c in enumerate(codons) if c in STOP]
    if not len(stops):
        invalid += 1
        continue
    first_stop = min(stops)
    codons = codons[:first_stop]

    first_codon = codons.pop(0)
    is_invalid = False
    while first_codon != START:
        missing_start += 1
        if not len(codons):
            # no start codon, so ignored entirely
            is_invalid = True
            invalid += 1
            break
        first_codon = codons.pop(0)
    if is_invalid:
        continue

    # reinsert the start codon (CAI examples have this!)
    codons.insert(0, 'ATG')

    assert all([len(x)==3 for x in codons])
    assert '_' not in codons
    cleaned_codons[gene_id] = ''.join(codons)

    locs[gene_id] = (chrom, int(start), int(end))

# reference set (all CDS)
ref = list(cleaned_codons.values())

weights = relative_adaptiveness(sequences=ref)

for gene_id, seq in cleaned_codons.items():
    chrom, start, end = locs[gene_id]
    cai = CAI(seq, weights=weights)
    gc = sum(x in 'GC' for x in seq) / len(seq)
    print('\t'.join(map(str, ['chr'+chrom, start, end, gene_id, cai, gc, len(seq)])))


