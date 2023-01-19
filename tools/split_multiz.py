import sys
import re
import os
from collections import defaultdict
from bgspy.utils import readfq, readfile

KEEP_SPECIES = set(['hg38', 'panPan1', 'panTro4', 'gorGor3', 'ponAbe2'])

try:
    in_fasta = readfile(sys.argv[1])
    out_dir = sys.argv[2]
except:
    raise ValueError("usage: cds_aligned.fa out_dir/ [id_map_file.tsv]")

def parse_region_withstrand(x):
    res = re.match(r'(chr[^:]+):(\d+)-(\d+)([+-])', region)
    if res is None:
        return res
    chrom, start, end, strand = res.groups()
    return chrom, int(start), int(end), strand

cds = defaultdict(lambda: defaultdict(list))

n = 0
for name, seq, comment in readfq(in_fasta, name_only=False):
    aln, _, _, _, region = name.split(' ')
    gene, tmp = aln.split('.')
    _, species, _, _ = tmp.split('_')
    if species not in KEEP_SPECIES:
        continue
    region = parse_region_withstrand(region)
    cds[gene][species].append((region, seq))
    n += 1

use_map = False
if len(sys.argv) == 4:
    use_map = True
    id_map = defaultdict(list)
    for line in readfile(sys.argv[3]):
        ensembl, ucsc, *_ = line.split('\t')
        if not len(ucsc):
            continue
        ucsc = ucsc.split('.')[0]
        id_map[ucsc].append(ensembl)

for name in cds:
    new_name = name
    if use_map:
        if name in id_map:
            names = set(id_map[name])
        new_name = ','.join(names)

    human = cds[name]['hg38']

    start = min([r[1] for r, s in human])
    end = max([r[2] for r, s in human])

    chrom = human[0][0][0]
    strand = human[0][0][3]
    seqfile = f"{name}_{chrom}:{start}-{end}{strand}.fa"

    with open(os.path.join(out_dir, seqfile), 'w') as f:
        for species in cds[name]:
            species_ranges, species_seqs = [], []
            # we concatenate
            for region, seq in cds[name][species]:
                species_seqs.append(seq)
            concat_seqs = ''.join(species_seqs)
            f.write(f">{species} {new_name}\n{concat_seqs}\n")
