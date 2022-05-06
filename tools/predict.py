import os
import numpy as np
import itertools
import re
import click
import glob
import keras

CHROM_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+).npy')
CHUNK_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+)_(?P<i>\d+)_(?P<lidx>\d+)_(?P<uidx>\d+).npy')

@click.command()
@click.argument('chunkfile', required=True)
@click.option('--chunk-dir', required=True, help='chunks directory')
@click.option('--out-dir', required=True, help='output directory')
@click.option('--h5-file', required=True, help='HDF5 file of keras model')
@click.option('--constrain', default=True,
              help='whether to compute B using segments along the entire '
              'chromosome, or whether to use the supplied slices in filename')
def predict(chunkfile, chunk_dir, out_dir, h5_file, constrain):
    infofile = os.path.join(chunk_dir, 'chunk_info.npz')
    seg_dir = os.path.join(chunk_dir, 'segments')
    info = np.load(infofile)

    model = keras.models.load_model(h5_file)

    # chunk to process
    chunk_parts = CHUNK_MATCHER.match(chunkfile).groupdict()
    chrom = chunk_parts['chrom']
    seg_file = glob.glob(os.path.join(seg_dir), f"{chrom}_*")

    # deal with the focal position chunk file
    chrom_parts = CHROM_MATCHER.match(seg_file).groupdict()

    lidx, uidx = int(chunk_parts['lidx']), int(chunk_parts['uidx'])
    chunk_i = int(chunk_parts['i'])
    focal_positions = np.load(chunkfile)

    # segment and info stuff
    Sm = np.memmap(seg_file)
    mean, scale = info['mean'], info['scale']
    w, t = info['w'], info['t']
    islog = info['islog']

    if not constrain:
        lidx, uidx = 0, Sm.shape[0]

    # get relevant part of segment matrix (possibly all of it)
    S = Sm[lidx:uidx, :]

    # Now build up X, expanding out wt grid. We do this
    # once per chunk, since we can just replace that last column
    mesh = np.array(list(itertools.product(w, t)))
    nmesh, nsegs = w*t, S.shape[0]
    X = np.empty((nsegs*nmesh, 5))
    X[:2, :] = np.tile(mesh, nsegs)
    X[2:4, :] = np.tile(S, nmesh)

    def transfunc(x, feature, mean, scale):
        if info[f"islog_{feature}"]:
            x = log10(x)
        return (x-mean)/scale

    # apply necessary log10 transforms, and center and scale
    X[:, 0] = transfunc(X[:, 0], 'mu', mean[0], scale[0])
    X[:, 1] = transfunc(X[:, 1], 'sh', mean[1], scale[1])
    X[:, 2] = transfunc(X[:, 2], 'L', mean[2], scale[2])
    X[:, 3] = transfunc(X[:, 2], 'rbp', mean[3], scale[3])

    # now calculate the recomb distances
    # Bs.append()
    Bs = np.empty(len(focal_positions))
    for i, f in enumerate(focal_positions):
        rf = dist_to_segment(f, S[:, 2:4])
        X[:, 5] = transfunc(rf, 'rf', mean[4], scale[4])
        B[i] = np.sum(np.log10(model.predict(X)))
        # Bs.append(B)

    outdir = make_dirs(out_dir)
    outfile = join(outdir, chunkfile)
    print(B)
    np.save(outfile, B)


if __name__ == "__main__":
    predict()

