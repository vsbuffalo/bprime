"""
Note -- can we speed up prediction by grid'ing
all the rfs and counting occurences, then predictin on the small
rf grid?
"""
USE_GPU = False
import os
from os.path import join
import numpy as np
import itertools
import re
import glob
import pickle
import click
import tqdm
if USE_GPU:
    try:
        import gpustat
        GPUSTAT_AVAIL = True
    except:
        GPUSTAT_AVAIL = False
import tensorflow as tf
from tensorflow import keras
from bgspy.utils import dist_to_segment, make_dirs, haldanes_mapfun

# for playing nice on GPUs
if USE_GPU and GPUSTAT_AVAIL:
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)

if not USE_GPU:
    # limit CPU usage
    MAX_CPU = 10
    os.environ["OMP_NUM_THREADS"] = f"{MAX_CPU}"
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{MAX_CPU}"
    os.environ["TF_NUM_INTEROP_THREADS"] = f"{MAX_CPU}"
    tf.config.threading.set_inter_op_parallelism_threads(MAX_CPU)
    tf.config.threading.set_intra_op_parallelism_threads(MAX_CPU)
    tf.config.set_soft_device_placement(True)

CHROM_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+).npy')
CHUNK_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+)_(?P<i>\d+)_(?P<lidx>\d+)_(?P<uidx>\d+).npy')

@click.command()
@click.argument('chunkfile', required=True)
@click.option('--input-dir', required=True, help='main prediction directory')
@click.option('--h5', required=True, help='HDF5 file of keras model', multiple=True)
@click.option('--constrain/--no-constrain', default=True,
              help='whether to compute B using segments along the entire '
              'chromosome, or whether to use the supplied slices in filename')
@click.option('--progress/--no-progress', default=True, help='whether to display progress bar')
def predict(chunkfile, input_dir, h5, constrain, progress):
    h5_files = h5
    out_dir = make_dirs(join(input_dir, 'preds'))
    chunk_dir = join(input_dir, 'chunks')
    infofile = join(input_dir, 'info.npz')
    seg_dir = join(input_dir, 'segments')
    info = np.load(infofile)

    models = {f: keras.models.load_model(f) for f in h5_files}

    # chunk to process
    chunk_parts = CHUNK_MATCHER.match(os.path.basename(chunkfile)).groupdict()
    chrom = chunk_parts['chrom']
    name = chunk_parts['name']
    seg_file = join(seg_dir, f"{name}_{chrom}.npy")
    assert len(seg_file), "no appropriate chromosome segment file found!"

    # deal with the focal position chunk file
    chrom_parts = CHROM_MATCHER.match(os.path.basename(seg_file)).groupdict()

    lidx, uidx = int(chunk_parts['lidx']), int(chunk_parts['uidx'])
    chunk_i = int(chunk_parts['i'])
    # contains two columns: the physical and map positions of focal sites
    sites_chunk = np.load(chunkfile)

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
    X[:, :2] = np.repeat(mesh, nsegs, axis=0)
    X[:, 2:4] = np.tile(S[:, :2], (nmesh, 1))

    def transfunc(x, feature, mean, scale):
        x = np.copy(x)
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
    nsites = sites_chunk.shape[0]
    nmodels = len(h5_files)
    B = np.empty((nw, nt, nsites, nmodels), dtype='f8')
    #np.array(2 * [f"{i}-{j}" for i, j in itertools.product(range(5), range(4))]).reshape((5, 4, -1))
    sites_indices = range(nsites)
    if progress:
        sites_indices = tqdm.tqdm(sites_indices)

    for i in sites_indices:
        #p = np.round(i/len(focal_positions) * 100, 2)
        #print(f"{i}/{len(focal_positions)}, {p}%", end='\r')
        f = sites_chunk[i, 1] # get map position
        rf = dist_to_segment(f, S[:, 2:4])
        #rf = haldanes_mapfun(rf)
        X[:, 4] = np.tile(transfunc(rf, 'rf', mean[4], scale[4]), nmesh)
        # note: at some point, we'll want to see how many are nans
        for j, model in enumerate(models.values()):
            b = model.predict(X).reshape((-1, nw, nt))
            # for debugging:
            #np.savez("out.npz", X=X, Xp=Xp, rf=rf, f=f, Sm=Sm, b=b)
            #__import__('pdb').set_trace()
            out_of_bounds = np.logical_or(b > 1, b <= 0)
            b[out_of_bounds] = np.nan
            bp = np.nansum(np.log10(b), axis=0)
            B[:, :, i, j] = bp

    chrom_out_dir = make_dirs(out_dir, chrom)
    outfile = join(chrom_out_dir, os.path.basename(chunkfile))
    np.save(outfile, B)

if __name__ == "__main__":
    predict()

