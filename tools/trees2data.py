import sys
sys.path.extend(['..', '../..', '../bgspy'])

from functools import partial
import multiprocessing
import warnings
import os
import re
import click
import numpy as np
import tskit
import pyslim
import msprime
import tqdm
from bgspy.theory import BGS_MODEL_PARAMS
from bgspy.utils import get_files, Bhat

FILENAME_RE = re.compile("(.*)_seed(\d+)_rep\d+_treeseq\.tree")

# we mirror the BGS segment model
DEFAULT_FEATURES = tuple(list(BGS_MODEL_PARAMS['bgs_segment']) + ['rep'])

def process_tree_file(tree_file, features, recap='auto'):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # so many warnings
        try:
            ts = tskit.load(tree_file)
        except Exception as Error:
            print(f"error reading file {tree_file}! Skipping...")
            return None
        md = ts.metadata['SLiM']['user_metadata']
        needs_recap = max(t.num_roots for t in ts.trees()) > 1
        if (recap is True) or (recap == 'auto' and needs_recap):
            ts = pyslim.slim_tree_sequence.SlimTreeSequence(ts)
            ts = pyslim.recapitate(ts, recombination_rate=0,
                                   ancestral_Ne=md['N'][0]).simplify()
    r2_sum, r2_mean = np.nan, np.nan
    ld_sum, ld_mean = np.nan, np.nan
    num_muts = len(list(ts.mutations()))
    if num_muts:
        idx = np.triu_indices(num_muts, k=1)
        try:
            r2 = tskit.LdCalculator(ts).r2_matrix()[idx]
            r2_sum = r2.sum()
            r2_mean = r2.mean()
        except:
            # this is due to infinite sites issues -- rare exception
            pass
        ld = np.cov(ts.genotype_matrix())
        if ld.ndim == 0:
            ld_sum, ld_mean = ld, ld
        else:
            idx = np.triu_indices(ld.shape[0], k=1)
            ld_sum = ld[idx].sum()
            ld_mean = ld[idx].mean()
    nroots = max(t.num_roots for t in ts.trees())
    assert(nroots == 1)
    region_length = int(md['region_length'][0])
    L = int(md['L'][0])
    tracklen = int(md['tracklen'][0])
    N = int(md['N'][0])
    sh = float(md['sh'][0])

    # tracking vs selected regions
    #wins = [0, tracklen, tracklen + L + 1]
    wins = [0, tracklen, ts.sequence_length]
    pi = ts.diversity(mode='branch', windows=wins)
    Ef = float(md['Ef'][0])
    Vf = float(md['Vf'][0])
    ngens = int(md['generations'][0])
    load = float(md['fixed_load'][0])
    nsubs = md['subs']
    #ndels = md['ndel_muts']
    #popfit = md['popfit']

    # get features from metadata
    X = tuple(md[f][0] for f in features)
    # get targets and other data
    tracking_pi = pi[0]
    y = (tracking_pi, Bhat(tracking_pi, N), Ef, Vf, load, nsubs, r2_sum, r2_mean, ld_sum, ld_mean, num_muts)
    return X, y

def filename_key(filename):
    "Removes the replicate number and seed providing a key for parameters."
    match = FILENAME_RE.match(filename)
    assert match is not None
    return match.groups()[0]

def trees2training_data(dir, features, recap='auto', progress=True,
                        ncores=None, suffix="recap.tree"):
    # this will recap automatically with rec rate 0
    tree_files = get_files(dir, suffix)
    X, y = [], []
    if progress:
        tree_files_iter = tqdm.tqdm(tree_files)
    else:
        tree_files_iter = iter(tree_files)
    func = partial(process_tree_file, features=features, recap=recap)
    if ncores in (None, 1):
        res = map(func, tree_files_iter)
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(p.imap(func, tree_files_iter))
    # if any files were skipped, drop the Nones returned here
    drop = [r is None for r in res]
    res = [r for r in res if r is not None]
    X, y = zip(*res)
    targets = ('pi', 'Bhat', 'Ef', 'Vf', 'load', 'nsubs', 'r2_sum', 'r2_mean', 'ld_sum', 'ld_mean', 'ld_nmuts')
    keys = [filename_key(os.path.basename(f)) for f, ignore in zip(tree_files, drop) if not ignore]
    assert len(keys) == len(X)
    assert len(keys) == len(y)
    X, y, keys = np.array(X), np.array(y), np.array(keys)
    return X, y, features, targets, keys

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
    X, y, features, targets, keys = trees2training_data(dir, features=features.split(','), suffix=suffix, ncores=ncores)
    np.savez(outfile, X=X, y=y, features=features, targets=targets, keys=keys)

if __name__ == "__main__":
    main()
