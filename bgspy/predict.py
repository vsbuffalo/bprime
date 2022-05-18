from itertools import product
import numpy as np


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


