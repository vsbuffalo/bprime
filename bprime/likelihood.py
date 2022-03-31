## likelihood.py -- functions for likelihood stuff
import numpy as np

def loglik2(pi0, B, y):
    # y is nsame, ndiff as cols
    # TODO does pi0 matter?
    assert(np.all(pi0 <= 0.5))
    pi = np.exp(np.add.outer(np.log(pi0), B))
    assert(np.all(pi <= 0.5))
    pi[pi==0] = np.nextafter(0, 1)
    #return y
    a = np.log(pi)*y[None, :, 1, None, None, None]
    b = np.log1p(-pi)*y[None, :, 0, None, None, None]
    assert(np.all(np.isfinite(a) | np.isnan(a)))
    assert(np.all(np.isfinite(b) | np.isnan(b)))
    ll = a + b
    #__import__('pdb').set_trace()
    return ll

def loglik(pi0, B, y):
    # y is nsame, ndiff as cols
    # TODO does pi0 matter?
    pi = pi0*np.exp(B).squeeze()
    assert(np.all(pi <= 0.5))
    pi[pi==0] = np.nextafter(0, 1)
    #return y
    nD = y[:, 1, None, None]
    nS = y[:, 0, None, None]
    a = np.log(pi)*nD
    b = np.log1p(-pi)*nS
    assert(np.all(np.isfinite(a) | np.isnan(a)))
    assert(np.all(np.isfinite(b) | np.isnan(b)))
    ll = a + b
    return ll

def loglik_pi0(B, y):
    # y is nsame, ndiff as cols
    R = np.exp(B).squeeze()
    R[R==0] = np.nextafter(0, 1)
    #return y
    nD = y[:, 1, None, None]
    nS = y[:, 0, None, None]
    def obj(pi0):
        pi = pi0*R
        pi[pi==0] = np.nextafter(0, 1)
        a = np.log(pi)*nD
        b = np.log1p(-pi)*nS
        assert(np.all(np.isfinite(a) | np.isnan(a)))
        assert(np.all(np.isfinite(b) | np.isnan(b)))
        ll = a + b
        return (-ll.sum(axis=0)).min()
    return obj

#def loglike_deriv_pi0(B, y, minimize=True):
#    R = np.exp(B).squeeze()
#    nD = y[:, 1, None, None]
#    nS = y[:, 0, None, None]
#    a = -1 if minimize else 1
#    fun = lambda x: a*(nS/x - R * nD / (1-R*x)).sum(axis=0)
#    return fun

def num_nonpoly(neut_pos, bins, masks):
    """
    Get the number of non-polymorphic sites from
    """
    chroms = bins.keys()
    # find window indices of all neutral SNPs
    idx = {c: np.digitize(neut_pos[c], bins[c])-1 for c in chroms}
    # count the indices (SNPs) per window
    poly_counts = {c: Counter(idx[c].tolist()) for c in chroms}
    npoly = {c: np.array([poly_counts[c][i] for i, _ in enumerate(e)]) for c, e in bins.items()}
    # what's the width of the neutral regions
    widths = {c: [masks[c][a:b].sum() for a, b in zip(e[:-1], e[1:])] for c, e in bins.items()}
    #__import__('pdb').set_trace()
    nfixed = {c: widths[c]-npoly[c][1:] for c in bins.keys()}
    return nfixed


def calc_loglik_components(b, Y, neut_pos, neut_masks, nchroms):
    """
    Interpolate the B values at midpoints, and sum the
    number of same and different pairs of these B windows. Also computes
    pi in the windows.
    """
    chroms = b.pos.keys()
    # interpolate B at the midpoints of the steps
    interpol = {c: interpolate.interp1d(b.pos[c], b.B[c], copy=False,
                                        assume_sorted=True, axis=0)
                for c in chroms}
    midpoints = {c: 0.5*(b.pos[c][1:]+b.pos[c][:-1]) for c in chroms}
    midpoint_Bs = {c: interpol[c](m) for c, m in midpoints.items()}
    # get the number of positions that are not polymorphic in the window
    nonpoly = num_nonpoly(neut_pos, b.pos, neut_masks)
    # next, we need to calculate the components of diversity (n_same,
    # n_diff)
    Y_binned = dict()
    pi_win = dict()
    n = nchroms
    with np.errstate(divide='ignore', invalid='ignore'): # for pi_win
        for chrom in chroms:
            nfixed = nonpoly[chrom]
            nsame = binned_statistic(neut_pos[chrom],
                                    Y[chrom][:, 0], np.sum,
                                    bins=b.pos[chrom]).statistic
            ndiff = binned_statistic(neut_pos[chrom],
                                    Y[chrom][:, 1], np.sum,
                                    bins=b.pos[chrom]).statistic
            nsame_fixed = nfixed * n*(n-1)/2
            Y_binned[chrom] = np.stack((ndiff + nsame_fixed, ndiff)).T
            pi_win[chrom] = ndiff / (ndiff + nsame_fixed)
    return Y_binned, midpoint_Bs, pi_win
