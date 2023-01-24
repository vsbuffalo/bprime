"""
Split the multiple alignment files of CDS from UCSC.
"""
import numpy as np
import sys
import re
import os
from collections import defaultdict
from bgspy.utils import readfq, readfile, parse_region

KEEP_SPECIES = set(['hg38', 'panPan1', 'panTro4', 'gorGor3', 'ponAbe2'])
KEEP_CHROMS = set([f'chr{c}' for c in range(1, 23)])
UCSC_NAMES = True

try:
    in_fasta = readfile(sys.argv[1])
    out_dir = sys.argv[2]
except:
    raise ValueError("usage: cds_aligned.fa out_dir/ [id_map_file.tsv] [cds.fa]")


# def compare_gapped_cds(ref, gapped, seed_len=6):
#     gaps = gapped.split('-')
#     gap_seeds = [g[:(seed_len+1)] for g in gaps]
#     idx = []
#     for seed in gap_seeds:
#         try:
#             idx.append(ref.index(seed))
#         except ValueError:
#             idx.append(None)


cds = defaultdict(lambda: defaultdict(list))

n = 0
for name, seq, comment in readfq(in_fasta, name_only=False):
    if UCSC_NAMES:
        aln, _, _, _, region = name.split(' ')
        gene, species, _, _ = aln.split('_')
        gene, tmp = gene.split('.')
    else:
        gene_species, _, _, _, region = name.split(' ')
        nm, id, species, _, _  = gene_species.split('_')
        gene = f"{nm}_{id}"
    if species not in KEEP_SPECIES:
        continue
    region = parse_region(region, with_strand=True)
    cds[gene][species].append((region, seq))
    n += 1

use_map = False
if len(sys.argv) >= 4:
    use_map = True
    id_map = defaultdict(list)
    for line in readfile(sys.argv[3]):
        ids = line.split('\t')
        if ids[0] == 'Gene stable ID':
            # header line
            continue
        # remove all those version, etc
        ids = [x.split('.')[0].strip() for x in ids]
        ensembl_gene, ensembl_gene_v, ensembl_tx, ensembl_tx_v, ucsc, refseq = ids
        refseq = refseq.strip()
        if UCSC_NAMES:
            key = ucsc
        else:
            key = refseq
        if not len(key):
            continue
        id_map[key].append(ensembl_gene)

# if a cds
# NOTE: I validated this against the CDS, but annoyingly it's hard to find
# the right CDS file to use with the MA. Many validations by hand showed
# things look good.
use_cdsfa = False
if len(sys.argv) == 5:
    use_cdsfa = True
    ref_cds = dict()
    for name, seq, comment in readfq(readfile(sys.argv[4]), name_only=False):
        gene_id = name.split('|')[0]
        ref_cds[gene_id] = seq

total = 0
no_id = 0
for name in cds:
    new_name = name
    if use_map:
        if name in id_map:
            names = list(set(id_map[name]))
            # if there are multiple names, take the earlier ID
            # in a few hand-inspections, duplicates look like novel genes that
            # are misannoated?
            order = np.array([int(n.replace('ENSG', '')) for n in names])
            new_name = names[np.argmin(order)]

    human = cds[name]['hg38']

    start = min([r[1] for r, s in human])
    end = max([r[2] for r, s in human])

    chrom = human[0][0][0]
    if chrom not in KEEP_CHROMS:
        continue
    strand = human[0][0][3]
    seqfile = f"{new_name}_{chrom}:{start}-{end}{strand}.fa"

    with open(os.path.join(out_dir, seqfile), 'w') as f:
        for species in cds[name]:
            species_ranges, species_seqs = [], []
            # we concatenate
            for region, seq in cds[name][species]:
                species_seqs.append(seq)
            concat_seqs = '-'.join(species_seqs)
            if species == 'hg38' and use_cdsfa:
                if new_name in ref_cds:
                    # could do seq comparison here
                    pass
                else:
                    no_id += 1
            f.write(f">{species} {new_name}\n{concat_seqs}\n")
            total += 1

print(f"total: {total}\nno ID found: {no_id}")
