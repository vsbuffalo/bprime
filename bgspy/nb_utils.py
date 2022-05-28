import re
import os
import pandas as pd
from collections import defaultdict
from bgspy.learn import LearnedFunction, LearnedB

LAYERS = [8, 4, 2, 'x']

LAYER_COMPONENT = '_'.join([f"(?P<n{i}>\d+)n{i}" for i in LAYERS])

PATH = rf'\w+_{LAYER_COMPONENT}_(?P<weightl2>[^w]+)weightl2_(?P<biasl2>[^b]+)biasl2_(?P<activ>(relu|elu|tanh))activ_(?P<outactive>(sigmoid|relu))outactiv_fit_(?P<rep>\d+)rep'

def stem(x):
    return x.replace('.h5', '')

def parse_fitname(file):
    match = re.match(PATH, file)
    assert match is not None, file
    return match.groupdict()

def load_learnedfuncs_in_dir(dir):
    """
    Load all the serialized LearnedFunction objects in a directory, e.g.
    after a batch of training replicates are run across different architectures.
    """
    files = [f for f in os.listdir(dir) if f.endswith('.h5')]
    # as a check, we keep track of which activations we've seen -- if
    # ignore_activ is True, we want to make sure there aren't more than one
    seen_activs = set()
    rows = []
    for file in files:
        file_stem = stem(file)
        keys = parse_fitname(file_stem)
        layers = [f"n{i}" for i in LAYERS]
        arch = {layer: int(keys[layer]) for layer in layers}
        rep = int(keys.pop('rep'))
        lf = LearnedFunction.load(os.path.join(dir, file_stem), load_model=False)
        bf = LearnedB(model=lf.metadata['model'])
        bf.func = lf

        row = {'key': stem(os.path.basename(file)), **arch, **keys, 'bf': bf}
               #'mae': bf.func.test_mae(), 'mse': bf.func.test_mse()}
        rows.append(row)
    return pd.DataFrame(rows)

