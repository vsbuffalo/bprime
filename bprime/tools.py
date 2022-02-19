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
        region_length = int(md['region_length'][0]) - 1
        print(region_length, ts.sequence_length)
        assert(region_length == ts.sequence_length)
        X.append([converter(md[p][0]) for p, converter in params.items()])
        pi = ts.diversity(mode='branch', windows=np.linspace(0, region_length, 10))
        y.append(pi)

    X_dtypes = list(params.items())
    X = np.array(X, dtype=X_dtypes)
    y = np.array(y)
    return X, y




