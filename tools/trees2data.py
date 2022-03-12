import os
import click
import numpy as np
import tskit
import tqdm

def trees2training_data(dir, features, progress=True, suffix="recap.tree"):
    tree_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(suffix)]
    X, y = [], []
    if progress:
        tree_files = tqdm.tqdm(tree_files)
    for tree_file in tree_files:
        ts = tskit.load(tree_file)
        md = ts.metadata['SLiM']['user_metadata']
        region_length = int(md['region_length'][0])
        seglen = int(md['seglen'][0])
        tracklen = int(md['tracklen'][0])
        #assert(region_length  == ts.sequence_length)

        wins = [0, tracklen, tracklen + seglen + 1]
        pi = ts.diversity(mode='branch', windows=wins)
        Ef = float(md['Ef'][0])
        Vf = float(md['Vf'][0])
        ngens = int(md['generations'][0])
        load = float(md['fixed_load'][0])

        # get features from metadata
        X.append(tuple(md[f][0] for f in features))
        # get targets and other data
        y.append((pi[0], 0.25*pi[0]/float(md['N'][0]), Ef, Vf, load))
    return np.array(X), np.array(y), features


@click.command()
@click.argument('dir')
@click.option('--outfile', default='B_data',
              type=click.Path(writable=True),
              help='path to save data to (exclude extension)')
@click.option('--suffix', default='recap.tree', help='tree file suffix')
@click.option('--features', default='N,s,h,mu,recfrac,rbp,seglen',
              help='features to extract from metadata')
def main(dir, outfile, suffix, features):
    X, y, features = trees2training_data(dir, features=features.split(','), suffix=suffix)
    np.savez(outfile, X=X, y=y, features=features)

if __name__ == "__main__":
    main()
