import numpy as np
from bgspy.theory import bgs_segment, bgs_rec, bgs_segment_sc16_manual_vec
from bgspy.classic import _BSC16_segment_lazy


def test_bgs_segment():
    mu, s, L, rbp, rf = 1e-8, 0.1, 1000, 1e-8, 1e-9
    np.testing.assert_allclose(bgs_segment(mu, s, L, rbp, rf), 0.99990001)

def test_bgs_rec():
    mu, s, L, rbp = 1e-8, 0.1, 1000, 1e-8
    np.testing.assert_allclose(bgs_rec(mu, s, L, rbp), 0.99990001)

def test_bgs_segment_sc16_manual_vec():
    w = np.array([1.00e-10, 3.16e-10, 1.00e-09, 3.16e-09, 1.00e-08, 3.16e-08])
    t = np.array([0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316, 0.1])

    L = np.array([1000, 1000])
    rbp = np.array([1e-8, 1e-8])
    # takes float L and rbp, since vectorized over mu, sh grid
    N = 1000
    a = []
    for i in range(2):
        a.append(bgs_segment_sc16_manual_vec((float(L[i]), float(rbp[i])),
                                             w, t, haploid_N=2*N))
    b = _BSC16_segment_lazy(w, t, L, rbp, N)
    np.testing.assert_allclose(np.moveaxis(np.stack(a), 0, 3), np.stack(b))


