import os
from os.path import join, basename
import json
import tqdm
from itertools import product
import numpy as np
import tensorflow as tf
from tensorflow import keras
from bgspy.theory import bgs_segment
from bgspy.utils import dist_to_segment, make_dirs, haldanes_mapfun
from bgspy.learn import LearnedFunction
from bgspy.theory import BGS_MODEL_PARAMS
from bgspy.parallel import MapPosChunkIterator
from bgspy.utils import BScores


def new_predict_matrix(w, t, L, rbp):
    """
    Columns are mu, s, L, rbp, rf
    """
    assert L.shape[0] == rbp.shape[0], "L and rbp have different sizes!"
    if L.ndim == 1:
        L = L[:, None]
    if rbp.ndim == 1:
        rbp = rbp[:, None]
    mesh = np.array(list(product(w, t)))
    nw, nt = w.size, t.size
    nmesh, nsegs = nw*nt, L.shape[0]
    X = np.empty((nsegs*nmesh, 5), dtype='f8')
    X[:, :2] = np.repeat(mesh, nsegs, axis=0)
    # fill in the L end rbp columns
    X[:, 2] = np.tile(L, (nmesh, 1)).squeeze()
    X[:, 3] = np.tile(rbp, (nmesh, 1)).squeeze()
    return X


def inject_rf(rf, X, nmesh):
    """
    Put the rf in X's last column, tiling it.
    """
    if rf.ndim == 1:
        rf = rf[:, None]
    X[:, 4] = np.tile(rf, (nmesh, 1)).squeeze()
    return X

def predictions_to_B_tensor(y, nw, nt, nsegs, nan_bounds=False):
    """
    Take the product (actually, log sum) across all segments.


    Prediction on each row of the matrix X leads to Bs that are in order,
    such that all segment's Bs are aligned for a combination of μ, s.
    We need reshape these so that we end up with a nw x nt by nsegs
    matrix, which can then be log-summed.
    """
    if nan_bounds:
        out_of_bounds = np.logical_or(y > 1, y <= 0)
        y[out_of_bounds] = np.nan

    y_segs_mat = y.reshape((nw*nt, nsegs))
    return np.exp(np.sum(np.log(y_segs_mat), axis=1).reshape((nw, nt)))

def write_predinfo(dir, model_files, w_grid, t_grid, step, nchunks, max_map_dist):
    models = dict()
    for filepath in model_files:
        func = LearnedFunction.load(filepath)
        # make sure this is the right bgs model
        model_name = os.path.basename(filepath)
        features = BGS_MODEL_PARAMS['bgs_segment']
        islog = {f: func.logscale[f] for f in features}
        bounds = {f: func.get_bounds(f) for f in features}
        filepath = filepath + '.h5' if not filepath.endswith('.h5') else filepath
        models[model_name] = dict(filepath=filepath,
                                  log=islog, bounds=bounds,
                                  mean=func.scaler.mean_.tolist(),
                                  scale=func.scaler.scale_.tolist())

    json_out = dict(dir=dir, w=w_grid.tolist(), t=t_grid.tolist(), step=step,
                    nchunks=nchunks, max_map_dist=max_map_dist, bounds=bounds,
                    models=models)
    jsonfile = join(dir, "info.json")
    with open(jsonfile, 'w') as f:
        json.dump(json_out, f)

def predict_chunk(sites_chunk, model_info_dict, segment_matrix,
                  bounds, w, t, lidx=None, uidx=None,
                  use_haldane=False, output_xps=False,
                  dont_predict=False, progress=True):
    FIX_BOUNDS = True # for debugging
    # let's alias some stuff for convienence
    models = model_info_dict
    Sm = segment_matrix

    model_h5s = {m: keras.models.load_model(v['filepath']) for m, v in models.items()}

    # get the centering and scaling parameters
    means = {m: np.array(v['mean']) for m, v in models.items()}
    scales = {m: np.array(v['scale']) for m, v in models.items()}
    islogs = {m: v['log'] for m, v in models.items()}

    if lidx is None or uidx is None:
        # unconstrained -- use all segments on a chromosome
        lidx, uidx = 0, Sm.shape[0]

    assert 0 <= lidx <= Sm.shape[0]
    assert 0 <= uidx <= Sm.shape[0]
    assert lidx < uidx


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
            lower, upper = bounds[feature]
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

    #model = models[list(models.keys())[0]] # FOR DEBUG

    Xps = []
    for i in sites_iter:
        f = sites_chunk[i, 1] # get map position
        rf = dist_to_segment(f, S[:, 2:4])
        if use_haldane:
            rf = haldanes_mapfun(rf)

        # build up the original, untransformed matrix for theory prediction
        # and debugging/testing (see tests/test_predict.py). This can be used
        # see if predictions on these Xps through LearnedFunction.predict()
        # match the results below
        Xp = inject_rf(rf, Xp, nmesh)

        if output_xps:
            Xps.append(Xp)
            if dont_predict:
                continue

        # let's calc B theory as a check!
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

    if not output_xps:
        return B
    return B, Xps

def load_predictions(genome, path, chroms=None):
    """
    Load the predictions from a path to a prediction directory.
    """
    info_file = join(path, 'info.json')
    with open(info_file) as f:
        info = json.load(f)
    chunks = MapPosChunkIterator(genome, w_grid=info['w'], t_grid=info['t'],
                                 step=info['step'], nchunks=info['nchunks'])
    chrom_dirs = os.listdir(join(path, 'preds'))
    if chroms is not None:
        if isinstance(chroms, str):
            chroms = set([chroms])
        chrom_dirs = [c for c in chrom_dir if c in chroms]

    B_chunks, pos_chunks = defaultdict(list), defaultdict(list)
    for chrom in chrom_dirs:
        chrom_pred_dir = join(path, 'preds', chrom)
        chrom_chunk_dir = join(path, 'chunks', chrom)
        files = os.listdir(chrom_pred_dir)
        files = sorted(files, key=lambda x: int(basename(x).split('_')[2]))
        ids = [int(basename(x).split('_')[2]) for x in files]
        for file in files:
            # this exploids the 1-to-1 correspodence between site chunk
            # and results npy files
            sites_chunk = np.load(join(chrom_chunk_dir, file))
            B_pos = sites_chunk[:, 0]
            Bs = np.load(join(chrom_pred_dir, file))
            B_chunks[chrom].append(Bs)
            pos_chunks[chrom].append(B_pos)
    Bs, B_pos = dict(), dict()
    for chrom in B_chunks:
        assert len(B_chunks[chrom]) == len(pos_chunks[chrom])
        Bs[chrom] = np.concatenate(B_chunks[chrom], axis=2)
        positions = np.concatenate(pos_chunks[chrom])
        B_pos[chrom] = positions
        assert np.all(positions == np.sort(positions))
    w = np.array(info['w'])
    t = np.array(info['t'])
    return BScores(Bs, B_pos, w, t, info['step'])


