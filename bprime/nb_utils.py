import re
import os
from collections import defaultdict
from bprime.learn import LearnedFunction, LearnedB

def stem(x):
    return x.replace('.h5', '')

def parse_fitname(file, run):
    match = re.match(r'msprime_bgs_\w+_(?P<n64>\d+)n64_(?P<n32>\d+)n32_(?P<activ>(relu|elu|tanh))activ_fit_(?P<rep>\d+)rep'.format(run=run), file)
    assert match is not None, file
    return match.groupdict()

def load_learnedfuncs_in_dir(dir):
    """
    Load all the serialized LearnedFunction objects in a directory, e.g.
    after a batch of training replicates are run across different architectures.
    """
    files = [f for f in os.listdir(dir) if f.endswith('.h5')]
    out = defaultdict(lambda: defaultdict(list))
    for file in files:
        file_stem = stem(file)
        key = parse_fitname(file_stem, run='simple')
        lf = LearnedFunction.load(os.path.join(dir, file_stem))
        bf = LearnedB(model=lf.metadata['model'])
        bf.func = lf
        out[(int(key['n64']), int(key['n32']))][key['activ'].append(bf)
    return out

