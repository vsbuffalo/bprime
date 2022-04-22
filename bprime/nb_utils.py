import re
import os
from collections import defaultdict
from bprime.learn import LearnedFunction, LearnedB

PATH = r'\w+_(?P<n128>\d+)n128_(?P<n64>\d+)n64_(?P<n32>\d+)n32_(?P<n8>\d+)n8_(?P<activ>(relu|elu|tanh))activ_fit_(?P<rep>\d+)rep'

def stem(x):
    return x.replace('.h5', '')

def parse_fitname(file):
    match = re.match(PATH, file)
    assert match is not None, file
    return match.groupdict()

def load_learnedfuncs_in_dir(dir, max_rep=None):
    """
    Load all the serialized LearnedFunction objects in a directory, e.g.
    after a batch of training replicates are run across different architectures.
    """
    files = [f for f in os.listdir(dir) if f.endswith('.h5')]
    out = defaultdict(lambda: defaultdict(list))
    layers = ['n128', 'n64', 'n32', 'n8']
    for file in files:
        file_stem = stem(file)
        key = parse_fitname(file_stem)
        if max_rep is not None:
            if int(key['rep']) > max_rep:
                continue
        lf = LearnedFunction.load(os.path.join(dir, file_stem))
        bf = LearnedB(model=lf.metadata['model'])
        bf.func = lf
        arch_key = tuple(int(key[layer]) for layer in layers)
        out[arch_key][key['activ']].append(bf)
    return out

