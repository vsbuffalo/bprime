import os
import tskit as tsk
import numpy as np

def trees_to_training_data(dir, params, windows, suffix="recap.tree"):
    """
    Load the tree files with a given suffix from a directory,
    parse out the parameters from metadata, and return the
    collated results.
    """
    tree_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(suffix)]
    X, y = [], []
    for tree_file in tree_files:
        ts = tsk.load(tree_file)
        md = ts.metadata['SLiM']['user_metadata']
        #region_length = int(md['region_length'][0]) + 1 # TODO fix we re-run sims
        #assert(region_length  == ts.sequence_length)
        X.append(tuple(converter(md[p][0]) for p, converter in params.items()))
        wins = windows + [ts.sequence_length]
        pi = ts.diversity(mode='branch', windows=wins)
        y.append((pi[0], 0.25*pi[0]/float(md['N'][0])))
        #y.append(pi[0])

    X_dtypes = list(params.items())
    X = np.stack(X)
    y = np.array(y)
    return X, y




