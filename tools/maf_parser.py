"""
Read an Ensembl MAF file. There's some additional code here for
later extensions, but the main functionality is to extract out the
multiple alignments for entries that have all nodes,
"""

import sys
import re
import os
import newick
from bgspy.utils import readfile
OUTDIR = "fasta_alns"

nodes = ('homo_sapiens', 'pan_paniscus', 'pan_troglodytes', 'gorilla_gorilla',
         'pongo_abelii', 'macaca_mulatta', 'chlorocebus_sabaeus', 'nomascus_leucogenys',
         'microcebus_murinus', 'macaca_fascicularis')

nodes = set(nodes)

seqs = dict()
trees = dict()
f = readfile(sys.argv[1])

def write_fasta(aligns, filename):
    with open(filename, 'w') as f:
        for aln in aligns:
            species, coords, seq = aln
            seqname, start, end = coords
            name = f"{species} {seqname}:{start}-{end}"
            f.write(f">{name}\n{seq}\n")

for line in f:
    if line.startswith('#'):
        if line.startswith('# tree: '):
            # new entry with the tree
            line = line.strip()
            aligns = list()
            tree_string = line[8:]
            tree = newick.loads(tree_string)
            #trees.append(tree)
            line = next(f)
            assert(line.startswith('# id: '))
            id = line.replace('# id: ', '')
            line = next(f)
            assert(line.startswith('a'))
            line = next(f).strip()
            filename = None
            species_set = set()
            while len(line):
                assert(line.startswith('s'))
                typ, species_seqname, start, size, strand, src_size, aln = re.split(' +', line)
                start, size = int(start), int(size)
                if species_seqname.startswith("ancestral_sequences"):
                    line = next(f).strip()
                    continue
                tmp = re.match(r"(.+)\.([0-9XY]+)", species_seqname).groups()
                assert(len(tmp) == 2)
                species, seqname = tmp
                seqname = f"chr{seqname}"
                #coord = f"{seqname}:{start}-{start+size}"
                end = start+size
                coords = (seqname, start, end)
                if species == 'homo_sapiens':
                    # these coords are the name
                    filename = f"{seqname}:{start}-{end}.fa"
                assert(typ == 's')
                aligns.append((species, coords, aln))
                species_set.add(species)
                line = next(f).strip()

            if species_set == nodes:
                # all species there
                write_fasta(aligns, os.path.join(OUTDIR, filename))
            seqs[id] = aligns
        else:
            # header info
            continue
