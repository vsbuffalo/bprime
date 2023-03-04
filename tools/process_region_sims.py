import numpy as np
import pandas as pd
import multiprocessing
import tskit
import pyslim
import sys
import os
import tqdm
import statsmodels.api as sm
from bgspy.utils import get_files, bin_chrom

def force_infinite_sites(tr):
    bad_sites = np.array([int(x.id) for x in tr.sites() if len(x.mutations) > 1])
    if len(bad_sites) > 0:
        return tr.delete_sites(bad_sites)
    return tr

def middle_element(x):
    "get the middle element of an odd-lengthened array, average two if even"
    if len(x) % 2 == 0:
       return x[len(x)//2]
    return x[int(len(x)//2 -1 ):int(len(x)//2 + 1)]


def load_and_recap(treefile, burnin=2000):
    ts = tskit.load(treefile)
    md = {k: v[0] for k, v in ts.metadata['SLiM']['user_metadata'].items()}
    N = md['N']
    rbp = md['rbp']
    rts = pyslim.recapitate(ts, recombination_rate=rbp,
                            sequence_length=ts.sequence_length,
                            ancestral_Ne=N)
    # LD
    r2_sum, ld_sum, ldn = 0, 0, 0
    if rts.num_mutations > 1:
        ld_calc = tskit.LdCalculator(force_infinite_sites(rts))
        r2_sum = np.nansum(np.triu(ld_calc.r2_matrix(), k=1))
        gm = rts.genotype_matrix()
        try:
            ld_triu = np.triu(np.cov(gm), k=1)
            ld_sum = np.nansum(ld_triu.sum())
        except TypeError:
            ld_sum = 0
        ldn = gm.shape[0]

    # B and log file stats
    B = ts.diversity(mode='branch') / (4*N)
    bins = bin_chrom(ts.sequence_length, 1000) # 1kb windows
    B_wins = ts.diversity(windows=bins, mode='branch') / (4*N)
    B_middle = middle_element(B_wins).mean()
    logfile = treefile.replace('_treeseq.tree', '_log.tsv.gz')
    d = pd.read_csv(logfile, sep='\t', comment='#')
    last_gen = np.max(d['cycle'])
    res = list(d.loc[d['cycle'] == last_gen, ].itertuples(index=False))[0]._asdict()
    R = 0
    if np.any(d['s'] == 0):
        # WARNING: this estimate is deprecated. See the region sim notebook for
        # a less biased estimator. This one is biased because the default
        # burnin is too little
        x, y = d['cycle'], d['s']
        x, y = x[x > burnin], y[x > burnin]
        X = sm.add_constant(x)
        fit = sm.OLS(y, X).fit()
        R = fit.params[1]
    res['R'] = R
    res['B'] = B
    #res['B_wins'] = B_wins
    res['B_middle'] = B_middle
    res['ratchet'] = d.loc[:, ('cycle', 's')].values
    # put in params
    res['sh'] = md['sh']
    res['N'] = N
    res['rbp'] = rbp
    res['mu'] = md['mu']
    res['U'] = md['U']
    res['r2sum'] = r2_sum
    res['ldsum'] = ld_sum
    res['ldn'] = ldn
    return res

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError("usage: process_region_sims.py simdir/ outfile.pkl")
    resdir = sys.argv[1]
    outfile = sys.argv[2]
    ncores = 20
    #ncores = 1
    all_files = get_files(resdir, "_treeseq.tree")
    #import pdb;pdb.set_trace()
    if ncores == 1 or ncores is None:
        res = []
        for file in tqdm.tqdm(all_files):
            res.append(load_and_recap(file))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(load_and_recap, all_files), total=len(all_files)))
    pd.DataFrame(res).to_pickle(outfile)
