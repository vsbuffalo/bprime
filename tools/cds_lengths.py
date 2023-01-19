"""
Summarize the CDS (i.e. canonical ensembl), calculating GC, GC3, and lengths. For a
join to Urrichio et al data.

"""
import sys
import pandas as pd
import pickle
from collections import Counter, defaultdict
import numpy as np
from bgspy.utils import readfq, readfile
import Bio.Data.CodonTable as CT
from CAI import CAI, relative_adaptiveness


ST = CT.standard_dna_table
START = ST.start_codons
STOP = ST.stop_codons


def num_syn_nonsyn(codon):
    NUCS = ['A', 'T', 'C', 'G']
    MUTS = {'A': ('T', 'C', 'G'),
            'T': ('A', 'C', 'G'),
            'C': ('T', 'A', 'G'),
            'G': ('A', 'T', 'C')}

    AA = ST.forward_table[codon]
    assert all((c in NUCS) for c in codon)
    f = np.array([0, 0, 0])
    for i, base in enumerate(codon):
        for alt in MUTS[base]:
            mut_codon = codon[:i] + alt + codon[(i + 1):]
            if mut_codon in ST.stop_codons:
                continue
            mut_AA = ST.forward_table[mut_codon]
            syn = mut_AA == AA
            f[i] += syn
    S = np.sum((f / 3))
    N = 3-S
    return S, N

num_syn_nonsyn_table = dict()

for codon in ST.forward_table.keys():
    num_syn_nonsyn_table[codon] = num_syn_nonsyn(codon)


keep_chroms = list(map(str, range(1, 23))) + ['X']

missing_start = 0
incorrect_length = 0
invalid = 0

cleaned_codons = dict()
locs = dict()
gc3 = dict()
num_syn = dict()
num_nonsyn = dict()

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
    while first_codon not in START:
        missing_start += 1
        if not len(codons):
            # no start codon, so ignored entirely
            is_invalid = True
            invalid += 1
            break
        first_codon = codons.pop(0)
    if is_invalid:
        continue

    # reinsert the start codon (CAI examples have this)
    codons.insert(0, 'ATG')

    # get the gc3
    gc3[gene_id] = sum([c[2] in 'GC' for c in codons]) / len(codons)

    # count the syn/non-syn AAs
    S, N = zip(*[num_syn_nonsyn_table[c] for c in codons])
    num_syn[gene_id] = np.sum(S)
    num_nonsyn[gene_id] = np.sum(N)

    assert all([len(x)==3 for x in codons])
    assert '_' not in codons
    cleaned_codons[gene_id] = ''.join(codons)

    locs[gene_id] = (chrom, int(start), int(end))


# reference set (all CDS)
ref = list(cleaned_codons.values())

weights = relative_adaptiveness(sequences=ref)

rows = []
for gene_id, codons in cleaned_codons.items():
    chrom, start, end = locs[gene_id]
    cai = CAI(codons, weights=weights)
    gc = sum(x in 'GC' for x in codons) / len(codons)
    data = dict(chrom='chr'+chrom, start=start, end=end,
                gene_id=gene_id, cai=cai, gc=gc, gc3=gc3[gene_id],
                len=len(codons),
                S=num_syn[gene_id], N=num_nonsyn[gene_id])
    rows.append(data)

pd.DataFrame(rows).to_csv(sys.argv[2], sep='\t', index=False)


