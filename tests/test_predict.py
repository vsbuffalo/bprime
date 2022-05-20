import os
import json
import pytest
from collections import defaultdict
from itertools import product
import click
from click.testing import CliRunner
import shutil
import tensorflow as tf
import numpy as np

from bgspy.predict import new_predict_matrix, inject_rf, predictions_to_B_tensor
from bgspy.predict import predict_chunk
from bgspy.learn import LearnedFunction
from bgspy.theory import bgs_segment

NW, NT = 2, 3
NSEGS = 3

@pytest.fixture
def X():
    w, t = np.arange(NW), np.arange(NT)
    L = np.array([10, 11, 12])
    rbp = np.array([20, 21, 22])
    assert len(L) == NSEGS
    assert len(rbp) == NSEGS
    X = new_predict_matrix(w, t, L, rbp)
    return X

def test_new_pred_matrix(X):
    # expected: repeat each (μ, s) pair nsegs times each
    # with the segment stuff (L, rbp) filled in cols 2 and 3
    X_expected = np.array([[ 0.,  0., 10., 20.,  0.],
                           [ 0.,  0., 11., 21.,  0.],
                           [ 0.,  0., 12., 22.,  0.],
                           [ 0.,  1., 10., 20.,  0.],
                           [ 0.,  1., 11., 21.,  0.],
                           [ 0.,  1., 12., 22.,  0.],
                           [ 0.,  2., 10., 20.,  0.],
                           [ 0.,  2., 11., 21.,  0.],
                           [ 0.,  2., 12., 22.,  0.],
                           [ 1.,  0., 10., 20.,  0.],
                           [ 1.,  0., 11., 21.,  0.],
                           [ 1.,  0., 12., 22.,  0.],
                           [ 1.,  1., 10., 20.,  0.],
                           [ 1.,  1., 11., 21.,  0.],
                           [ 1.,  1., 12., 22.,  0.],
                           [ 1.,  2., 10., 20.,  0.],
                           [ 1.,  2., 11., 21.,  0.],
                           [ 1.,  2., 12., 22.,  0.]])
    np.testing.assert_equal(X[:, :4], X_expected[:, :4])

def test_inject_rf(X):
    rf = np.array([1, 2, 3])
    Xrf = inject_rf(rf, X, NW * NT)
    Xrf_expected = np.array([[ 0.,  0., 10., 20.,  1.],
                             [ 0.,  0., 11., 21.,  2.],
                             [ 0.,  0., 12., 22.,  3.],

                             [ 0.,  1., 10., 20.,  1.],
                             [ 0.,  1., 11., 21.,  2.],
                             [ 0.,  1., 12., 22.,  3.],

                             [ 0.,  2., 10., 20.,  1.],
                             [ 0.,  2., 11., 21.,  2.],
                             [ 0.,  2., 12., 22.,  3.],

                             [ 1.,  0., 10., 20.,  1.],
                             [ 1.,  0., 11., 21.,  2.],
                             [ 1.,  0., 12., 22.,  3.],

                             [ 1.,  1., 10., 20.,  1.],
                             [ 1.,  1., 11., 21.,  2.],
                             [ 1.,  1., 12., 22.,  3.],

                             [ 1.,  2., 10., 20.,  1.],
                             [ 1.,  2., 11., 21.,  2.],
                             [ 1.,  2., 12., 22.,  3.]])
    np.testing.assert_equal(X, Xrf_expected)

def test_predictions_to_B_tensor(X):
    rf = np.array([1, 2, 3])
    Xrf = inject_rf(rf, X, NW * NT)
    y_fake = Xrf.sum(axis=1)
    fake_tensor = predictions_to_B_tensor(y_fake, NW, NT, NSEGS)
    #print(y_fake)
    tensor_expected = np.array([[31*34*37, 32*35*38, 33*36*39],
                                [42560., 46332., 50320.]])
    np.testing.assert_allclose(fake_tensor, tensor_expected)

@pytest.fixture
def fake_prediction_chunk_data():
    chunkfile = "hg38_chr10_933_61305_92693_sample.npy"
    test_infofile = "test_data/fake_dnnb/info.json"
    test_chunkfile = f"test_data/fake_dnnb/chunks/chr10/{chunkfile}"
    seg_file = "test_data/fake_dnnb/segments/hg38_chr10.npy"
    with open(test_infofile) as f:
        info = json.load(f)
    sites_chunk = np.load(test_chunkfile)
    segment_matrix = np.load(seg_file)
    return info, sites_chunk, segment_matrix, 61305, 92693

def test_predict_chunk(fake_prediction_chunk_data):
    """
    Test the command line tool (which has fewer checks) against the
    LearnedFunction.
    """
    info, sites_chunk, segment_matrix, lidx, uidx = fake_prediction_chunk_data
    models = info['models']
    assert all(os.path.exists(v['filepath']) for v in models.values()), "missing model .h5 test data"
    w = np.array(info['w'])
    t = np.array(info['t'])
    bounds = info['bounds']

    gpus = tf.config.list_physical_devices('GPU')
    device = '/CPU:0'
    if gpus:
        device = '/device:GPU:2'

    with tf.device(device):
        Bo, Xps = predict_chunk(sites_chunk, models, segment_matrix,
                                bounds, w, t, lidx, uidx, output_xps=True)
    assert Bo.shape[2] == sites_chunk.shape[0], "incorrect B tensor dimension"
    assert Bo.shape[0] == len(w), "incorrect B tensor dimension (w)"
    assert Bo.shape[1] == len(t), "incorrect B tensor dimension (t)"

    test_learnedfuncs = ['test_data/bmap_hg38_reps_0n128_0n64_0n32_0n8_2nx_eluactiv_fit_0rep',
                        'test_data/bmap_hg38_reps_0n128_0n64_0n32_0n8_2nx_eluactiv_fit_1rep']

    manual_B_pred_parts = defaultdict(list)
    manual_B_preds = defaultdict(list)
    manual_theory = []
    with tf.device(device):
        for j, model_file in enumerate(test_learnedfuncs):
            func = LearnedFunction.load(model_file)
            # each Xp is for ONE focal site... all segments "near" it
            for Xp in Xps:
                # each of these is for ONE focal site
                Bm = func.predict(Xp)
                Bm_theory = bgs_segment(*Xp.T)
                manual_theory.append(Bm_theory)

                # manually reshape things to avoid bugs
                res = defaultdict(list)
                res_theory = defaultdict(list)
                for i, (mu, s) in enumerate(zip(*Xp[:, :2].T)):
                    res[(mu, s)].append(Bm[i])
                    res_theory[(mu, s)].append(Bm_theory[i])

                # convert all lists of Bs across segments to numpy array
                res = {k: np.array(v) for k, v in res.items()}
                res_theory = {k: np.array(v) for k, v in res_theory.items()}

                manual_B_pred_parts[j].append(res)

                # now, convert to matrix, taking product across segments
                B = np.empty(shape=(len(w), len(t)))
                Btheory = np.empty(shape=(len(w), len(t)))
                for wi, ww in enumerate(w):
                    for ti, tt in enumerate(t):
                        B[wi, ti] = np.exp(np.log(res[(ww, tt)]).sum())
                        Btheory[wi, ti] = np.exp(np.log(res_theory[(ww, tt)]).sum())

                # append the prediction for this focal site
                manual_B_preds[j].append(B)
                # manual_theory.append(Btheory)

    # manually build up the manual prediction tensor
    Bm = np.empty((len(w), len(t), sites_chunk.shape[0], len(models)))
    for j in range(2):
        for f in range(5):
            # which focal site
            Bm[:, :, f, j] = manual_B_preds[j][f]

    assert np.testing.assert_allclose(Bm, Bo[..., 1:])
    assert False



