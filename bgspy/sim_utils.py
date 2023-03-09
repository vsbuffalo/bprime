## sim_utils.py -- common utility functions for msprime and slim
import os
import re
import pickle
import warnings
from functools import partial
import itertools
import operator
from collections import defaultdict, Counter, namedtuple
import numpy as np
import tskit as tsk
import msprime
import pyslim
import tqdm
import multiprocessing
from bgspy.utils import get_files, random_seed, bin_chrom, readfile

SIM_REGEX = re.compile(r'(?P<name>\w+)_N1000_mu(?P<mu>[^_]+)_sh(?P<sh>[^_]+)_chr10_seed\d+_rep(?P<rep>[^_]+)_treeseq.tree')


def delete_mutations(ts):
    return ts.delete_sites([s.id for s in ts.sites()])


def mutate_simulated_tree(ts, rate, seed=None,
                          remove_existing_mutations=True):
    """
    Given a TreeSequence of a simulated tree, remove all selected
    sites (if remove_existing_mutations=True)
    """
    if isinstance(ts, str):
        ts = tsk.load(ts)
    if remove_existing_mutations:
        ts = delete_mutations(ts)
    return msprime.sim_mutations(ts, rate=rate, random_seed=seed, 
                                 model=msprime.BinaryMutationModel())

def get_counts_from_ts(ts_dict):
    """
    Return an ref/alt allele counts matrix from a tree
    sequence with mutations.
    """
    seqlens = dict()
    all_counts = dict()
    for chrom, ts in ts_dict.items():
        sl = int(ts.sequence_length)
        seqlens[chrom] = sl
        num_deriv = np.zeros(sl)
        for var in ts.variants():
            nd = (var.genotypes > 0).sum()
            num_deriv[int(var.site.position)] = nd
        ntotal = np.repeat(ts.num_samples, sl)
        nanc = ntotal - num_deriv
        counts = {chrom: np.stack((nanc, num_deriv)).T}
        all_counts[chrom] = counts
    return all_counts, seqlens


def param_grid(params):
    """
    Generate a Cartesian product parameter grid from a
    dict of grids per parameter.
    """
    grid = []
    for param, values in params.items():
        if len(values):
            grid.append([(param, v) for v in values])
        else:
            grid.append([(param, '')])
    def convert_to_dict(x):
        # and package seed if needed
        x = dict(x)
        return x
    return map(convert_to_dict, itertools.product(*grid))


def calc_b_from_treeseqs(file, width=1000, recrate=1e-8, seed=None):
    """
    Recapitate trees and calc B for whole-chromosome BGS simulations

    Note that this computes this in windows [0, width, 2*width, ...],
    *not* at focal positions like B.

    There is slightly less optimal, since we should center these on the focal
    positions and do width-sized window around the focal points, but the
    difference is likely insignicant.
    """
    ts = tsk.load(file)
    md = ts.metadata['SLiM']['user_metadata']
    N = md['N'][0]
    #recmap = load_recrates('../data/annotation/rec_100kb_chr10.bed', ts.sequence_length)
    rts = pyslim.recapitate(ts, recombination_rate=recrate,
                            sequence_length=ts.sequence_length,
                            ancestral_Ne=N, random_seed=seed)
    length = int(ts.sequence_length)
    neut_positions = bin_chrom(length, width).astype(int)
    # extract the specified simulation parameters from the ts metadata
    params = {k: md[k][0] for k in md.keys()}
    B = rts.diversity(mode='branch', windows=neut_positions) / (4*N)
    return params, neut_positions, B


def parse_sim_filename(file, types=None, basedir='runs'):
    """
    Assumes: 
     basedir / name / [variable params] / rep(rep)_seed(seed)_treeseq.tree
    types is a dictionary of types

    """
    parts = file.split('/')
    assert parts[0] == basedir
    res = dict()
    res['name'] = parts[1]
    file = parts[-1]
    params = parts[2:-1]
    for param in params:
        key, val  = param.split('__')
        if types is not None:
            val = types[val](val)
        res[key] = val
    rep, seed = re.match('rep(\d+)_seed(\d+)_treeseq.tree', file).groups()
    res['rep'] = int(rep)
    res['seed'] = int(seed)
    return res


def load_b_chrom_sims(dir, progress=True, ncores=None, **kwargs):
    """
    Load a batch of BGS simulations for an entire chromosome, and store the
    positions and array of Bs (across simulation replicates!) in a dictionary
    with (sh, mu) parameters.

    WARNING: these should be the only varying parameters across these
    simulations!

    The results are grouped by these parameters (e.e. across replicates),
    and combined into arrays.

    **kwargs are passed to calc_b_from_treeseqs().
    """
    tree_files = get_files(dir, suffix='.tree')
    params = [parse_sim_filename(x) for x in tree_files]
    mus = set(float(x['mu']) for x in params)
    shs = set(float(x['sh']) for x in params)
    reps = set(int(x['rep']) for x in params)

    # read one file in to get the number of positions
    _, pos, b = calc_b_from_treeseqs(tree_files[0], **kwargs)
    npos = len(pos)

    # allocate matrix for results
    # we do one less than position since these are window boundaries
    # including start
    X = np.full((npos-1, len(mus), len(shs), len(reps)), np.nan, dtype=np.single)

    # for indices
    mu_lookup = {float(mu): i for i, mu in enumerate(sorted(mus))}
    sh_lookup = {float(sh): i for i, sh in enumerate(sorted(shs))}
    if ncores is None or ncores == 1:
        if progress:
            tree_files_iter = tqdm.tqdm(tree_files)
        for file in tree_files_iter:
            sim_params, pos, b = calc_b_from_treeseqs(file, **kwargs)
            rep = parse_sim_filename(file)['rep']
            mu = sim_params['mu']
            sh = sim_params['sh']
            X[:, mu_lookup[mu], sh_lookup[sh], rep] = b
    else:
        with multiprocessing.Pool(ncores) as p:
            func = partial(calc_b_from_treeseqs, **kwargs)
            res = list(tqdm.tqdm(p.imap(func, tree_files), total=len(tree_files)))
            for (sim_params, pos, b), file in zip(res, tree_files):
                rep = parse_sim_filename(file)['rep']
                mu, sh = sim_params['mu'], sim_params['sh']
                X[:, mu_lookup[mu], sh_lookup[sh], rep] = b

    mu = np.fromiter(mu_lookup.keys(), dtype=float)
    sh = np.fromiter(sh_lookup.keys(), dtype=float)
    return mu, sh, pos, X, tree_files


def process_substitution_files(dir, outfile, suffix='sub.tsv.gz'):
    """
    Collate all the substitution files in a directory.

    For a single DFE per chromosome (currently), e.g. one sel coef
    only free parms are mu and sel coef.

    Returns: dict of dict of numpy counts per basepair.
    """
    all_files = get_files(dir, suffix=suffix)
    #results = defaultdict(dict) 
    results = dict()
    only_chrom = None # for checking we only have one chromosome

    for filename in tqdm.tqdm(all_files):
        with readfile(filename) as f:
            for line in f:
                if line.startswith('#'):
                    md = line[1:].strip().strip(';').split(';')
                    md = dict([tuple(x.split('=')) for x in md])
                    continue
                chrom, pos, sel, mtype, h = line.strip().split('\t')
                if only_chrom is None:
                    only_chrom = chrom
                assert only_chrom == chrom
                sel = float(sel)
                mu = float(md['mu']) 
                if mu not in results:
                    results[mu] = dict()
                if sel not in results[mu]:
                    results[mu][sel] = Counter()
                results[mu][sel][int(pos)] += 1

    with open(outfile, 'wb') as f:
        pickle.dump(results, f)


 
