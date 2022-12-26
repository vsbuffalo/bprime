import sys
from collections import defaultdict
import pickle
from bgspy.utils import readfile

genes = dict()
transcripts = dict()
gff_file = readfile(sys.argv[1])

def parse_keyvals(keyval_str):
    keyvals = dict([x.split('=') for x in keyval_str.split(';')])
    return keyvals

for line in gff_file:
    if line.startswith('#'):
        continue
    chrom, _, feature, start, end, _, _, _, info = line.strip().split('\t')
    if feature == 'gene':
        keyvals = parse_keyvals(info)
        key = keyvals['ID'].replace('gene:', '')
        genes[key] = (set(), 0)
        continue

    if feature == 'mRNA'
        keyvals = parse_keyvals(info)
        key = keyvals['ID'].replace('transcript:', '')
        transcripts[key] = set()
        continue

    if feature == 'exon'
        keyvals = parse_keyvals(info)
        key = keyvals['Parent'].replace('transcript:', '')
        start, end = int(start), int(end)
        transcripts[key].add(end-start)
        continue

   
        

with open('gene_coords.pkl', 'wb') as f:
    pickle.dump(genes, f)
