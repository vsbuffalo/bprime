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
from bgspy.theory import bgs_segment
from bgspy.predict import new_predict_matrix, inject_rf, predictions_to_B_tensor

HALDANE = False
FIX_BOUNDS = True

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
@click.option('--output-xps/--no-xps', default=False, help='output unformatted chunk matrices for debugging')
def predict(chunkfile, input_dir, h5, constrain, progress, output_xps):
    h5_files = h5
    out_dir = make_dirs(input_dir, 'preds')
    chunk_dir = make_dirs(input_dir, 'chunks')
    infofile = make_dirs(input_dir, 'info.npz')
    seg_dir = make_dirs(input_dir, 'segments')
    info = np.load(infofile)

    models = {f: keras.models.load_model(f) for f in h5_files}
    print(f"h5 file order: {list(models.keys())}")

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
    # columns are L, rbp, seg map start, seg map end
    S = Sm[lidx:uidx, :]

    # Now build up X, expanding out wt grid. We do this
    # once per chunk, since we can just replace that last column
    X = new_predict_matrix(w, t, S[:, 0], S[:, 1])
    # old stuff for reference:
    # mesh = np.array(list(itertools.product(w, t)))
    nw, nt = w.size, t.size
    nmesh, nsegs = nw*nt, S.shape[0]
    # X = np.empty((nsegs*nmesh, 5), dtype='f8')
    # X[:, :2] = np.repeat(mesh, nsegs, axis=0)
    # X[:, 2:4] = np.tile(S[:, :2], (nmesh, 1))

    def transfunc(x, feature, mean, scale):
        x = np.copy(x)
        if FIX_BOUNDS:
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
    # plus one is for BGS theory
    B = np.empty((nw, nt, nsites, nmodels+1), dtype='f8')
    #np.array(2 * [f"{i}-{j}" for i, j in itertools.product(range(5), range(4))]).reshape((5, 4, -1))
    sites_indices = np.arange(nsites)
    if progress:
        sites_iter = tqdm.tqdm(sites_indices)
    else:
        sites_iter = sites_indices

    # optionally output the prediction matrices (for debugging)
    if output_xps:
        # draw a random focal site
        i = np.random.choice(sites_indices)
        f = sites_chunk[i, 1]
        rf = dist_to_segment(f, S[:, 2:4])
        if HALDANE:
            rf = haldanes_mapfun(rf)
        Xp = inject_rf(rf, Xp, nmesh)
        xpr_dir = make_dirs(input_dir, 'xps', chrom)
        outfile = join(xpr_dir, os.path.basename(chunkfile))
        np.save(outfile, Xp)
        return

    #model = models[list(models.keys())[0]] # FOR DEBUG

    for i in sites_iter:
        #p = np.round(i/len(focal_positions) * 100, 2)
        #print(f"{i}/{len(focal_positions)}, {p}%", end='\r')
        f = sites_chunk[i, 1] # get map position
        rf = dist_to_segment(f, S[:, 2:4])
        if HALDANE:
            rf = haldanes_mapfun(rf)
        rf = transfunc(rf, 'rf', mean[4], scale[4])
        X = inject_rf(rf, X, nmesh)

        # let's calc B theory as a check!
        Xp = inject_rf(rf, Xp, nmesh)
        bp_theory = predictions_to_B_tensor(bgs_segment(*Xp.T), nw, nt, nsegs)
        B[:, :, i, 0] = bp_theory

        for j, model in enumerate(models.values(), start=1):
            b = predictions_to_B_tensor(model.predict(X), nw, nt, nsegs, nan_bounds=False)
            # for debugging:
            #np.savez("out.npz", X=X, Xp=Xp, rf=rf, f=f, Sm=Sm, b=b)
            #__import__('pdb').set_trace()
            #bp = np.nansum(np.log10(b), axis=0)
            B[:, :, i, j] = b

    # save real output
    chrom_out_dir = make_dirs(out_dir, chrom)
    outfile = join(chrom_out_dir, os.path.basename(chunkfile))
    np.save(outfile, B.squeeze())

if __name__ == "__main__":
    predict()

