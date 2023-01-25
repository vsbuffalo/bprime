"""
DEPRECATED -- this is too fragile. Trying a different way to filter
out phastcons from CDS

For a directory of FASTA multiple alignments, named for their
region, extract our the region.
"""

from ncls import NCLS
import numpy as np
import os
import sys
from collections import defaultdict
from bgspy.utils import parse_region, readfq, load_bed_annotation

bed_file = sys.argv[1]
cds_dir = sys.argv[2]
out_dir = sys.argv[3]

bed_ranges = load_bed_annotation(bed_file)

cds_files = [f for f in os.listdir(cds_dir) if f.endswith('.fa')]

# ENSG00000143341_chr1:185734780-186189878+.fa

def load_fasta(filename):
    gene_id, region = os.path.basename(filename).split('_')
    chrom, start, end, strand = parse_region(region, with_strand=True)
    species_seqs = dict()
    for species_name, seq, _ in readfq(open(filename), name_only=False):
        species, name = species_name.split(' ')
        species_seqs[species] = seq.upper()
    return (chrom, start, end, strand), species_seqs, name


cds_ranges = defaultdict(list)
seq_ids = dict()
cds_ranges_id_map = dict()

range_id = 0
for filename in cds_files:
    region, seqs, gene_id = load_fasta(os.path.join(cds_dir, filename))
    chrom, start, end, stand = region
    cds_ranges[chrom].append((start, end, range_id))
    seq_ids[range_id] = seqs
    cds_ranges_id_map[range_id] = (chrom, start, end, gene_id)
    range_id += 1

range_db = dict()
for chrom, ranges in cds_ranges.items():
    starts, ends, ids = list(map(lambda x: np.array(x, dtype=int), zip(*ranges)))
    range_db[chrom] = NCLS(starts, ends, ids)

for chrom, (query_ranges, feature_type) in bed_ranges.ranges.items():
    query_starts, query_ends = list(map(lambda x: np.array(x, dtype=int), zip(*query_ranges)))
    query_idx = np.arange(len(query_starts))

    if chrom not in range_db:
        continue

    query_ids, db_ids = range_db[chrom].all_overlaps_both(query_starts, query_ends, query_idx)
    for query_id, db_id in zip(query_ids.tolist(), db_ids.tolist()):
        assert db_id in cds_ranges_id_map
        db_chrom, db_start, db_end, gene_id = cds_ranges_id_map[db_id]
        assert db_end > db_start

        db_seq = seq_ids[db_id]['hg38']

        query_start, query_end = query_starts[query_id], query_ends[query_id]
        assert query_end > query_start

        db_width = db_end-db_start + 1
        seqlen = len(db_seq)

        # the seq start can't over run the left side
        seq_start = max(db_start-query_start, 0)
        seq_end = min(db_end-query_start, seqlen)

        assert seq_end > seq_start

        s, e = seq_start + query_start, seq_end + query_start
        with open(os.path.join(out_dir, f"{gene_id}_{chrom}:{s}-{e}.fa"), 'w') as f:
            for species, seq in seq_ids[db_id].items():
                assert len(seq) == seqlen, "f{species} has a different length sequence!"
                f.write(f">{species}\n{seq[seq_start:seq_end+1]}\n")



