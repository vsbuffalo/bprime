"""
Note -- can we speed up prediction by grid'ing
all the rfs and counting occurences, then predictin on the small
rf grid?
"""
import os
import numpy as np
import itertools
import re
import click
import glob
import gpustat
import tensorflow as tf
from tensorflow import keras
from bgspy.utils import dist_to_segment, make_dirs

# for playing nice on GPUs
stats = gpustat.GPUStatCollection.new_query()
ids = map(lambda gpu: int(gpu.entry['index']), stats)
ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)

CHROM_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+).npy')
CHUNK_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+)_(?P<i>\d+)_(?P<lidx>\d+)_(?P<uidx>\d+).npy')

@click.command()
@click.argument('chunkfile', required=True)
@click.option('--chunk-dir', required=True, help='chunks directory')
@click.option('--out-dir', required=True, help='output directory')
@click.option('--h5-file', required=True, help='HDF5 file of keras model')
@click.option('--constrain/--no-constrain', default=True,
              help='whether to compute B using segments along the entire '
              'chromosome, or whether to use the supplied slices in filename')
def predict(chunkfile, chunk_dir, out_dir, h5_file, constrain):
    infofile = os.path.join(chunk_dir, 'chunk_info.npz')
    seg_dir = os.path.join(chunk_dir, 'segments')
    info = np.load(infofile)

    model = keras.models.load_model(h5_file)

    # chunk to process
    chunk_parts = CHUNK_MATCHER.match(os.path.basename(chunkfile)).groupdict()
    chrom = chunk_parts['chrom']
    name = chunk_parts['name']
    seg_file = os.path.join(seg_dir, f"{name}_{chrom}.npy")
    assert len(seg_file), "no appropriate chromosome segment file found!"

    # deal with the focal position chunk file
    chrom_parts = CHROM_MATCHER.match(os.path.basename(seg_file)).groupdict()

    lidx, uidx = int(chunk_parts['lidx']), int(chunk_parts['uidx'])
    chunk_i = int(chunk_parts['i'])
    focal_positions = np.load(chunkfile)

    # segment and info stuff
    Sm = np.load(seg_file)
    mean, scale = info['mean'], info['scale']
    w, t = info['w'], info['t']

    assert 0 <= lidx <= Sm.shape[0]
    assert 0 <= uidx <= Sm.shape[0]
    assert lidx < uidx

    if not constrain:
        lidx, uidx = 0, Sm.shape[0]

    # get relevant part of segment matrix (possibly all of it)
    S = Sm[lidx:uidx, :]

    # Now build up X, expanding out wt grid. We do this
    # once per chunk, since we can just replace that last column
    mesh = np.array(list(itertools.product(w, t)))
    nw, nt = w.size, t.size
    nmesh, nsegs = nw*nt, S.shape[0]
    X = np.empty((nsegs*nmesh, 5), dtype='f8')
    X[:, :2] = np.tile(mesh, (nsegs, 1))
    X[:, 2:4] = np.tile(S[:, :2], (nmesh, 1))

    def transfunc(x, feature, mean, scale):
        lower, upper = info[f"bounds_{feature}"]
        x[x < lower] = lower
        x[x > upper] = upper
        if info[f"islog_{feature}"]:
            assert np.all(x > 0)
            x = np.log10(x)
        return (x-mean)/scale

    Xp = np.copy(X)
    # apply necessary log10 transforms, and center and scale
    for j, feature in enumerate(('mu', 'sh', 'L', 'rbp')):
        X[:, j] = transfunc(X[:, j], feature, mean[j], scale[j])

    # now calculate the recomb distances
    # Bs.append()
    B = np.empty((nw, nt, focal_positions.shape[0]), dtype='f8')
    for i, f in enumerate(focal_positions):
        if i > 5:
            break
        p = np.round(i/len(focal_positions) * 100, 2)
        print(f"{i}/{len(focal_positions)}, {p}%", end='\r')
        rf = dist_to_segment(f, S[:, 2:4])
        X[:, 4] = np.tile(transfunc(rf, 'rf', mean[4], scale[4]), nmesh)
        # note: at some point, we'll want to see how many are nans
        b = model.predict(X).reshape((nw, nt, -1))
        out_of_bounds = np.logical_or(b > 1, b <= 0)
        b[out_of_bounds] = np.nan
        bp = np.nansum(np.log10(b), axis=2)
        B[:, :, i] = bp
        __import__('pdb').set_trace()

    outdir = make_dirs(out_dir)
    outfile = os.path.join(outdir, os.path.basename(chunkfile))
    np.save(outfile, B)


if __name__ == "__main__":
    predict()

