"""
Note -- can we speed up prediction by grid'ing
all the rfs and counting occurences, then predictin on the small
rf grid?
"""
USE_GPU = False
import os
import json
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
@click.option('--constrain/--no-constrain', default=True,
              help='whether to compute B using segments along the entire '
              'chromosome, or whether to use the supplied slices in filename')
@click.option('--progress/--no-progress', default=True, help='whether to display progress bar')
@click.option('--output-xps/--no-xps', default=False, help='output unformatted chunk matrices for debugging')
def predict(chunkfile, input_dir, constrain, progress, output_xps):
    """
    
    Note: bounds are determined for a group of models (e.g. all trained on the same
    simulation data, which sets the boudns). Centering and scaling parameters, and
    what features are log transfomed are model fit specific.
    """
    out_dir = make_dirs(input_dir, 'preds')
    chunk_dir = make_dirs(input_dir, 'chunks')
    infofile = make_dirs(input_dir, 'info.json')
    seg_dir = make_dirs(input_dir, 'segments')
    with open(infofile, 'r') as f:
        info = json.load(f)

    models = info['models']
  
    model_h5s = {m: keras.models.load_model(v['filepath']) for m, v in models.items()}

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
    means = {m: np.array(v['mean']) for m, v in models.items()}
    scales = {m: np.array(v['scale']) for m, v in models.items()}
    islogs = {m: v['log'] for m, v in models.items()}
    w, t = np.array(info['w']), np.array(info['t'])

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

    def transfunc(x, feature, mean, scale, islog):
        # we need the feature to get global bounds for this feature
        # The islog mean/scale are feature and model specific
        # so we pass those in
        x = np.copy(x)
        if FIX_BOUNDS:
            lower, upper = info["bounds"][feature]
            x[x < lower] = lower
            x[x > upper] = upper
        if islog:
            assert np.all(x > 0)
            x = np.log10(x)
        return (x-mean)/scale

    # this is the unmodified transformed copy, e.g. for BGS theory
    Xp = np.copy(X)

    # apply necessary log10 transforms, and center and scale
    # for *each* model, which can have different centering/scaling
    # params
    Xs = {m: np.copy(X) for m in models.keys()}
    for model in models.keys():
        for j, feature in enumerate(('mu', 'sh', 'L', 'rbp')):
            Xs[model][:, j] = transfunc(X[:, j], feature, means[model][j], 
                                        scales[model][j], islogs[model][feature])

    # now calculate the recomb distances
    nsites = sites_chunk.shape[0]

    nmodels = len(models)

    # plus one is for BGS theory
    B = np.empty((nw, nt, nsites, nmodels+1), dtype='f8')
    sites_indices = np.arange(nsites)

    if progress:
        sites_iter = tqdm.tqdm(sites_indices)
    else:
        sites_iter = sites_indices

    # optionally output the prediction matrices (for debugging) only
    # and don't continue
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

        # let's calc B theory as a check!
        Xp = inject_rf(rf, Xp, nmesh)
        bp_theory = predictions_to_B_tensor(bgs_segment(*Xp.T), nw, nt, nsegs)
        B[:, :, i, 0] = bp_theory

        # now do the DNN prediction stuff
        for j, (model_name, model) in enumerate(model_h5s.items(), start=1):
            X = Xs[model_name]
            mean, scale = means[model_name][4], scales[model_name][4] # 4 = rf
            islog = islogs[model_name]['rf']
            rf = transfunc(rf, 'rf', mean, scale, islog)
            X = inject_rf(rf, X, nmesh)
            b = predictions_to_B_tensor(model.predict(X), nw, nt, nsegs, nan_bounds=False)
            B[:, :, i, j] = b

    # save real output
    chrom_out_dir = make_dirs(out_dir, chrom)
    outfile = join(chrom_out_dir, os.path.basename(chunkfile))
    np.save(outfile, B.squeeze())

if __name__ == "__main__":
    predict()

