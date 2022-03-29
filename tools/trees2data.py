import os
import click
import numpy as np
import tskit
import pyslim
import tqdm

def Bhat(pi, N):
    """
    Branch statistics π is 4N (e.g. if μ --> 1)
    If there's a reduction factor B, such that
    E[π] = 4BN, a method of moments estimator of
    B is Bhat = π / 4N.
    """
    return 0.25 * pi / N

def trees2training_data(dir, features, recap='auto',
                        progress=True, suffix="recap.tree"):
    tree_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(suffix)]
    X, y = [], []
    if progress:
        tree_files = tqdm.tqdm(tree_files)
    for tree_file in tree_files:
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
        seglen = int(md['seglen'][0])
        tracklen = int(md['tracklen'][0])
        N = int(md['N'][0])
        #assert(region_length  == ts.sequence_length)


        # tracking vs selected regions
        wins = [0, tracklen, tracklen + seglen + 1]
        pi = ts.diversity(mode='branch', windows=wins)
        Ef = float(md['Ef'][0])
        Vf = float(md['Vf'][0])
        ngens = int(md['generations'][0])
        load = float(md['fixed_load'][0])

        # get features from metadata
        X.append(tuple(md[f][0] for f in features))
        # get targets and other data
        tracking_pi = pi[0]
        y.append((tracking_pi, Bhat(tracking_pi, N), Ef, Vf, load))
    targets = ('pi', 'Bhat', 'Ef', 'Vf', 'load')
    return np.array(X), np.array(y), features, targets


@click.command()
@click.argument('dir')
@click.option('--outfile', default='B_data',
              type=click.Path(writable=True),
              help='path to save data to (exclude extension)')
@click.option('--recap', default='auto', help='recapitate trees')
@click.option('--suffix', default='recap.tree', help='tree file suffix')
@click.option('--features', default='N,s,h,mu,rf,rbp,seglen',
              help='features to extract from metadata')
def main(dir, outfile, recap, suffix, features):
    X, y, features, targets = trees2training_data(dir, features=features.split(','), suffix=suffix)
    np.savez(outfile, X=X, y=y, features=features, targets=targets)

if __name__ == "__main__":
    main()
