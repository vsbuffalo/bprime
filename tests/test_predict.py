import pytest
from itertools import product
import numpy as np

from bgspy.predict import new_predict_matrix, inject_rf, predictions_to_B_tensor

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
