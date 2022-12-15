import sys
import pickle
from bgspy.utils import readfile

genes = dict()

for line in readfile(sys.argv[1]):
    if line.startswith('#'):
        continue
    chrom, _, feature, start, end, _, _, _, info = line.strip().split('\t')
    if feature != 'gene':
        continue
    keyvals = dict([x.split('=') for x in info.split(';')])
    key = keyvals['ID'].replace('gene:', '')
    start, end = int(start), int(end)
    genes[key] = (chrom, start, end)

with open('gene_coords.pkl', 'wb') as f:
    pickle.dump(genes, f)
