import sys
sys.path.extend(['..', '../bprime'])

from functools import partial
import multiprocessing
import warnings
import os
import click
import numpy as np
import tskit
import pyslim
import msprime
import tqdm
from bprime.theory import BGS_MODEL_PARAMS

# we mirror the BGS segment model
DEFAULT_FEATURES = BGS_MODEL_PARAMS['bgs_segment']

def Bhat(pi, N):
    """
    Branch statistics π is 4N (e.g. if μ --> 1)
    If there's a reduction factor B, such that
    E[π] = 4BN, a method of moments estimator of
    B is Bhat = π / 4N.
    """
    return 0.25 * pi / N

def get_files(dir, suffix):
    """
    Recursively get files.
    """
    all_files = set()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if suffix is not None:
                if not file.endswith(suffix):
                    continue
            all_files.add(os.path.join(root, *dirs, file))
    return all_files

def process_tree_file(tree_file, features, recap='auto'):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # so many warnings
        ts = tskit.load(tree_file)
        md = ts.metadata['SLiM']['user_metadata']
        needs_recap = max(t.num_roots for t in ts.trees()) > 1
        if (recap is True) or (recap == 'auto' and needs_recap):
            ts = pyslim.slim_tree_sequence.SlimTreeSequence(ts)
            ts = pyslim.recapitate(ts, recombination_rate=0,
                                   ancestral_Ne=md['N'][0]).simplify()
    nroots = max(t.num_roots for t in ts.trees())
    assert(nroots == 1)
    region_length = int(md['region_length'][0])
    L = int(md['L'][0])
    tracklen = int(md['tracklen'][0])
    N = int(md['N'][0])
    # the SLiM script metadata reports the het sel coef for convenience
    # let's check s h = reported sh
    s, h, sh = float(md['s'][0]), float(md['h'][0]), float(md['sh'][0])
    np.testing.assert_almost_equal(s*h, sh)

    # tracking vs selected regions
    wins = [0, tracklen, tracklen + L + 1]
    pi = ts.diversity(mode='branch', windows=wins)
    Ef = float(md['Ef'][0])
    Vf = float(md['Vf'][0])
    ngens = int(md['generations'][0])
    load = float(md['fixed_load'][0])

    # get features from metadata
    X = tuple(md[f][0] for f in features)
    # get targets and other data
    tracking_pi = pi[0]
    y = (tracking_pi, Bhat(tracking_pi, N), Ef, Vf, load)
    return X, y

def trees2training_data(dir, features, recap='auto', progress=True,
                        ncores=None, suffix="recap.tree"):
    # this will recap automatically with rec rate 0
    tree_files = get_files(dir, suffix)
    X, y = [], []
    if progress:
        tree_files = tqdm.tqdm(tree_files)
    func = partial(process_tree_file, features=features, recap=recap)
    if ncores in (None, 1):
        X, y = zip(*map(func, tree_files))
    else:
        with multiprocessing.Pool(ncores) as p:
            X, y = zip(*list(tqdm.tqdm(p.imap(func, tree_files),
                                       total=len(tree_files))))
    targets = ('pi', 'Bhat', 'Ef', 'Vf', 'load')
    return np.array(X), np.array(y), features, targets


@click.command()
@click.argument('dir')
@click.option('--outfile', default='B_data',
              type=click.Path(writable=True),
              help='path to save data to (exclude extension)')
@click.option('--suffix', default='treeseq.tree', help='tree file suffix')
@click.option('--ncores', default=1, help='number of cores for parallel processing')
@click.option('--recap', default='auto', help='recapitate trees')
@click.option('--features', default=','.join(DEFAULT_FEATURES),
              help='features to extract from SLiM metadata')
def main(dir, outfile, suffix, ncores, recap, features):
    """
    Extract features and targets from tree sequences. If the treeseq isn't recapitated,
    it will be recapitated with the pop size in the SLiM metadata, and rec rate = 0.
    """
    X, y, features, targets = trees2training_data(dir, features=features.split(','), suffix=suffix, ncores=ncores)
    np.savez(outfile, X=X, y=y, features=features, targets=targets)

if __name__ == "__main__":
    main()
