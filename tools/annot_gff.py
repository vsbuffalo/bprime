import sys
from bgspy.utils import read_gff


for entry in read_gff(sys.argv[1], parse_attributes=True):
    if entry['feature'] == 'CDS':
        cols = [entry['seqname'], entry['start'], entry['end']]
        name = entry['attribute']['Parent'].replace('transcript:', '')
        chrom, start, end = cols
        start, end = int(start), int(end)
        if end-start == 0:
            continue
        name += f"_{chrom}:{start}-{end}"
        cols.append(name)
        sys.stdout.write('\t'.join(cols) + '\n')


