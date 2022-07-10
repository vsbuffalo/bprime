## likelihood.py -- functions for likelihood stuff
from collections import Counter
import numpy as np
from scipy.stats import binned_statistic
from scipy import interpolate
from bgspy.utils import bin_midpoints

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

def num_nonpoly(neut_pos, bins, masks):
    """
    Get the number of non-polymorphic sites that overlap the neutral regions
    specified by the boolean masks. This is calculated as:
        # neutral masked sites in bin - # polymorphic sites in bins

    neut_pos: chrom dict of positions of the neutral polymorphic sites
    bins: chrom dict of bin end positions
    masks: chrom dict of neutral regions as boolean masks
    """
    chroms = bins.keys()

    # validate the neutral sites
    for chrom in chroms:
        for pos in neut_pos[chrom]:
            msg = f"Position {pos} is not in the '{chrom}' neutral mask!"
            assert masks[chrom][pos] == 1, msg

    # find window indices of all neutral SNPs
    idx = {c: np.digitize(neut_pos[c], bins[c])-1 for c in chroms}
    # count the indices (SNPs) per window
    poly_counts = {c: Counter(idx[c].tolist()) for c in chroms}
    npoly = {c: np.array([poly_counts[c][i] for i, _ in enumerate(e)]) for c, e in bins.items()}
    # what's the width of the neutral regions overlapping the bins?
    widths = {c: [masks[c][a:b].sum() for a, b in zip(e[:-1], e[1:])] for c, e in bins.items()}
    #__import__('pdb').set_trace()
    nfixed = {c: widths[c]-npoly[c][:-1] for c in bins.keys()}
    return nfixed

def binned_data_reduction():
    """
    Reduces the site-level data into binned data summaries for the binned-
    likelihood approach.

    Currently we don't account in fixed site's depth, we just assume the total
    number of chromosomes sequenced
    """


def calc_loglik_components(b, Y, Y_pos, neut_masks, nchroms, chrom_bins):
    """
    The MLE for parameters is always calculated per-window (here the window bins
    defined in chrom_bins). This function aggregates the site-level data (the
    number of diff/same sites in Y), counts the number of neutral sites that
    could be polymorphic form the masks (neut_masks), and interpolates the Bs
    at the window midpoints. It also computes windowed π in these windows from
    the data, as a check.

    b: BScores object
    Y: DAC matrix
    Y_pos: neutral region masks
    nchroms: how many chromosomes were sequenced
    bins: dictionary of window endpoints
    """
    chroms = b.pos.keys()

    win_midpoints = bin_midpoints(chrom_bins)
    win_Bs = {c: b.B_at_pos(c, x) for c, x in win_midpoints.items()}

    # get the number of positions that are not polymorphic in the window
    nonpoly = num_nonpoly(Y_pos, chrom_bins, neut_masks)

    # next, we need to calculate the components of diversity (n_same, n_diff)
    Y_binned = dict()
    pi_win = dict()
    n = nchroms
    with np.errstate(divide='ignore', invalid='ignore'): # for pi_win
        for chrom in chroms:
            nfixed = nonpoly[chrom]
            nsame = binned_statistic(Y_pos[chrom],
                                     Y[chrom][:, 0], np.sum,
                                     bins=chrom_bins[chrom]).statistic
            ndiff = binned_statistic(Y_pos[chrom],
                                     Y[chrom][:, 1], np.sum,
                                     bins=chrom_bins[chrom]).statistic
            # for each fixed site, it adds (n choose 2) same combinations
            nsame_fixed = nfixed * n*(n-1)/2
            Y_binned[chrom] = np.stack((nsame + nsame_fixed, ndiff)).T
            pi_win[chrom] = ndiff / (ndiff + nsame + nsame_fixed)
    return Y_binned, win_Bs, pi_win

class WindowedMLE:
    """
    nchroms: number of chromosomes sequenced.

    Note: currently the fixed
    """
    def __init__(self, Y, Y_pos, B, w, t, neut_masks, chrom_bins):
        self.bins = chrom_bins
        self.Y = Y
        # validate Y, Y_pos
        ytot = sum([len(Y_pos[c]) for c in Y_pos.keys()])
        assert Y.shape[0] == ytot, "Y and Y_pos are not equal lengths"
        parts = calc_loglik_components(B, Y, Y_pos, neut_masks, nchroms)
        self.Y_binned =
        self.B = B
        self.w = w
        self.t = t
        self.ll = ll

    @property
    def w_mle(self):
        return self.w[np.where(self.lls == np.nanmax(self.lls))[0]]

    @property
    def t_mle(self):
        return self.t[np.where(self.lls == np.nanmax(self.lls))[1]]

    def __repr__(self):
        return f"MLE Fit: w={self.w_mle}, t={self.t_mle}"



