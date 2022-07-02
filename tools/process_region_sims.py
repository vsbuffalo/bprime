import numpy as np
import pandas as pd
import multiprocessing
import tskit
import pyslim
import sys
import os
import tqdm
import statsmodels.api as sm
from bgspy.utils import get_files

def force_infinite_sites(tr):
    bad_sites = np.array([int(x.id) for x in tr.sites() if len(x.mutations) > 1])
    if len(bad_sites) > 0:
        return tr.delete_sites(bad_sites)
    return tr


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
    logfile = treefile.replace('_treeseq.tree', '_log.tsv.gz')
    d = pd.read_csv(logfile, sep='\t', comment='#')
    last_gen = np.max(d.generation)
    res = list(d.loc[d.generation == last_gen, ].itertuples(index=False))[0]._asdict()
    R = 0
    if np.any(d['s'] == 0):
        x, y = d['generation'], d['s']
        x, y = x[x > burnin], y[x > burnin]
        X = sm.add_constant(x)
        fit = sm.OLS(y, X).fit()
        R = fit.params[1]
    res['R'] = R
    res['B'] = B
    res['ratchet'] = d.loc[:, ('generation', 's')].values
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
    resdir = sys.argv[1]
    outfile = sys.argv[2]
    ncores = 20
    #ncores = 1
    all_files = get_files(resdir, "_treeseq.tree")
    if ncores == 1 or ncores is None:
        res = []
        for file in tqdm.tqdm(all_files):
            res.append(load_and_recap(file))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(load_and_recap, all_files), total=len(all_files)))
    pd.DataFrame(res).to_pickle(outfile)
