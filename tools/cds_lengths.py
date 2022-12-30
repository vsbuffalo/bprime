import sys
import pickle
from collections import Counter, defaultdict
import numpy as np
from bgspy.utils import readfq, readfile

START = 'ATG'

CODON_TBL = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
    'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}

AA_CODONS = {c: a for c, a in CODON_TBL.items() if a not in ('_', )}
AAs = set([a for c, a in CODON_TBL.items() if a != '_'])

SYN_TBL = defaultdict(set)
for codon, aa in CODON_TBL.items():
    SYN_TBL[aa].add(codon)

STOP_TBL = set([k for k, v in CODON_TBL.items() if v == '_'])

fp = readfile(sys.argv[1])

keep_chroms = list(map(str, range(1, 23))) + ['X']

obs_codon_counts = Counter()
obs_cds_codon_counts = defaultdict(Counter)

missing_start = 0
incorrect_length = 0
invalid = 0

cleaned_codons = dict()

for name, seq, _ in readfq(fp, name_only=False):
    gene_id, tx_id, chrom, start, end = name.split('|')
    seq = seq.upper()

    if chrom not in keep_chroms:
        continue

    if len(seq) % 3:
        incorrect_length += 1

    codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
    stops = [i for i, c in enumerate(codons) if c in STOP_TBL]
    if not len(stops):
        invalid += 1
        continue
    first_stop = min(stops)
    codons = codons[:first_stop]

    start = codons.pop(0)
    is_invalid = False
    while start != START:
        missing_start += 1
        if not len(codons):
            # no start codon, so ignored entirely
            is_invalid = True
            invalid += 1
            break
        start = codons.pop(0)
    if is_invalid:
        continue

    assert all([len(x)==3 for x in codons])
    assert '_' not in codons
    x = Counter(codons)
    obs_codon_counts.update(x)
    obs_cds_codon_counts[gene_id] = x

    cleaned_codons[gene_id] = codons


obs_aa_counts = Counter()
for codon in AA_CODONS:
    counts = obs_codon_counts.get(codon, 0)
    obs_aa_counts[CODON_TBL[codon]] += counts

ngenes = len(obs_cds_codon_counts)

X = np.zeros((ngenes, len(AAs), len(AA_CODONS)))

AA_CODON_idx = {c: i for i, c in enumerate(AA_CODONS)}
AA_idx = {aa: i for i, aa in enumerate(set(AA_CODONS.values()))}
AA_rev_idx = {i: aa for i, aa in enumerate(set(AA_CODONS.values()))}

AA_codon_num = {aa: len(SYN_TBL[aa]) for aa in AA_idx}
n = np.fromiter(AA_codon_num.values(), dtype=int)

for i, (gene, codon_counts) in enumerate(obs_cds_codon_counts.items()):
    for codon, counts in codon_counts.items():
        X[i, AA_idx[CODON_TBL[codon]], AA_CODON_idx[codon]] = counts

# sum accross all genes
C = X.sum(axis=0)

# make a codon table
codon_freqs = (C / (C.sum(axis=1))[:, None])

codon_table = defaultdict(dict)
codon_table_alt = defaultdict(dict)

for aa, i in sorted(AA_idx.items(), key=lambda x: x[0]):
    for c, j in sorted(AA_CODON_idx.items(), key=lambda x: x[0]):
        codon_table[c][aa] = codon_freqs[i, j]
        codon_table_alt[aa][c] = codon_freqs[i, j]

major_codons = {aa: max(c, key=c.get) for aa, c in codon_table_alt.items()}

# calculate the CAIs

for gene_id, codons in obs_cds_codon_counts.items():
    # put in order
    counts = np.array([codons[c] for c in AA_CODON_idx.keys()])
    fi = counts/counts.sum()

    # get the AAs
    aas = [CODON_TBL[c] for c in codons]




    # print('\t'.join(map(str, ['chr'+chrom, start, end, gene_id, tx_id, len(seq)])))


