import os
import re
from itertools import groupby
from collections import defaultdict
import tskit as tsk
import numpy as np
import tqdm

FILENAME_RE = "(.*)_rep(\d+)_{suffix}"

def param_basename(filename, suffix="recap.tree"):
    """
    Extract the part of the filename that only contains parameters.
    """
    pattern = FILENAME_RE.format(suffix=suffix)
    match = re.match(pattern, os.path.basename(filename)).groups()[0]
    return match

def trees_to_aggregate_training_data(dir, params, windows, suffix="recap.tree",
                                      progress=False):
    """
    Load the tree files with a given suffix from a directory,
    parse out the parameters from metadata, and return the
    collated results.
    """
    tree_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(suffix)]
    X, y = [], []
    grouped_replicates = defaultdict(list)
    for tree_file in tree_files:
        key = param_basename(tree_file)
        grouped_replicates[key].append(tree_file)
    if progress:
        grouped_iter = tqdm.tqdm(grouped_replicates.items())
    else:
        grouped_iter = grouped_replicates.items()
    for key, tree_file_replicates in grouped_iter:
        y_row = []
        X_row = None
        for tree_file in tree_file_replicates:
            ts = tsk.load(tree_file)
            md = ts.metadata['SLiM']['user_metadata']
            region_length = int(md['region_length'][0])
            #assert(region_length  == ts.sequence_length)
            # all replicates should have the same parameters; we check that here
            param_row = tuple(converter(md[p][0]) for p, converter in params.items())
            if X_row is None:
                X_row = param_row
            else:
                assert(X_row == param_row)
            wins = windows + [ts.sequence_length]
            pi = ts.diversity(mode='branch', windows=wins)
            Ef = float(md['Ef'][0])
            Vf = float(md['Vf'][0])
            load = float(md['fixed_load'][0])
            y_row.append((pi[0], 0.25*pi[0]/float(md['N'][0]), Ef, Vf, load))
        y_row_ave = np.stack(y_row).mean(axis=0).tolist()
        y_row_var = np.stack(y_row).var(axis=0).tolist()
        y.append(y_row_ave + y_row_var + [len(y_row)])
        X.append(X_row)

    X_dtypes = list(params.items())
    X = np.array(X)#, dtype=X_dtypes)
    y = np.array(y)
    return X, y



def trees_to_training_data(dir, params, windows, suffix="recap.tree",
                                      progress=False):
    """
    Load the tree files with a given suffix from a directory,
    parse out the parameters from metadata, and return the
    collated results.
    """
    tree_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(suffix)]
    X, y = [], []
    grouped_replicates = defaultdict(list)
    for tree_file in tree_files:
        ts = tsk.load(tree_file)
        md = ts.metadata['SLiM']['user_metadata']
        region_length = int(md['region_length'][0])
        #assert(region_length  == ts.sequence_length)
        # all replicates should have the same parameters; we check that here
        param_row = tuple(converter(md[p][0]) for p, converter in params.items())
        X.append(param_row)
        wins = windows + [ts.sequence_length]
        pi = ts.diversity(mode='branch', windows=wins)
        Ef = float(md['Ef'][0])
        Vf = float(md['Vf'][0])
        load = float(md['fixed_load'][0])
        y.append((pi[0], 0.25*pi[0]/float(md['N'][0]), Ef, Vf, load))

    X_dtypes = list(params.items())
    X = np.array(X)#, dtype=X_dtypes)
    y = np.array(y)
    return X, y




