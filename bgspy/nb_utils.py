import re
import os
import pandas as pd
from collections import defaultdict
from bgspy.learn import LearnedFunction, LearnedB

PATH = r'\w+_(?P<n128>\d+)n128_(?P<n64>\d+)n64_(?P<n32>\d+)n32_(?P<n8>\d+)n8_(?P<n4>\d+)n4_(?P<n2>\d+)n2_(?P<nx>\d+)nx_(?P<activ>(relu|elu|tanh))activ_(?P<outactive>(sigmoid|relu))outactiv_(?P<balance>(False|True))balance_fit_(?P<rep>\d+)rep'

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
    # as a check, we keep track of which activations we've seen -- if
    # ignore_activ is True, we want to make sure there aren't more than one
    seen_activs = set()
    layers = ['n128', 'n64', 'n32', 'n8', 'nx']
    rows = []
    for file in files:
        file_stem = stem(file)
        keys = parse_fitname(file_stem)
        arch = {layer: int(keys[layer]) for layer in layers}
        rep = int(keys.pop('rep'))
        if max_rep is not None:
            if rep > max_rep:
                continue
        lf = LearnedFunction.load(os.path.join(dir, file_stem))
        bf = LearnedB(model=lf.metadata['model'])
        bf.func = lf

        row = {'key': stem(os.path.basename(file)), **arch, **keys, 'bf': bf,
               'mae': bf.func.test_mae(), 'mse': bf.func.test_mse()}
        rows.append(row)
    return pd.DataFrame(rows)

