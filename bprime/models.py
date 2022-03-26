"""

I_B: number of annotation groups
μ_D(i_B): is the rate of deleterious mutations for annotation group i_B

f(t | i_B): is the DFE, the probability of heterozygote selection effect t in
annotation category.

G: the size of the grid of selection coefficients, t.
M: the size of the grid of del mutation rates.

# Parameter grid
DFE grid (G x I_B): each entry is the rate of del mutations into this
annotation class with this selection coefficient. Sums over columns are the bp
mutation rate per annotation class. Marginalizing over columns (summing rows)
gives the total DFE.

E16 use a weight grid, w(t_g | i_B), which is the rate of del mutations given
seletion coefficient t_g, for annotation i_B. The total sum is the rate of del
mutations.


All positions in the genome x = {0, 1, ..., L} are assigned to a annotation
group. The set a_B(i_B) is the collection of sites in with membership in
annotation set i_B.
"""

from collections import defaultdict, namedtuple, Counter
import pickle
import itertools
import functools
import multiprocessing
import ctypes
import tqdm
import time
import numpy as np
import allel
from scipy import interpolate
from scipy.stats import binned_statistic, binned_statistic_dd
from scipy.optimize import minimize_scalar

from bprime.utils import bin_chrom
from bprime.utils import RecMap, readfile, load_dacfile
from bprime.utils import process_feature_recombs, load_bed_annotation
from bprime.utils import process_feature_recombs2
from bprime.utils import haldanes_mapfun, load_seqlens, chain_dictlist
from bprime.utils import ranges_to_masks, sum_logliks_over_chroms

# this dtype allows for simple metadata storage
Bdtype = np.dtype('float32', metadata={'dims': ('site', 'w', 't', 'f')})
BScores = namedtuple('BScores', ('B', 'pos', 'w', 't', 'step'))
BinnedStat = namedtuple('BinnedStat', ('statistic', 'wins', 'nitems'))

BCALC_EINSUM_PATH = ['einsum_path', (0, 2), (0, 1)]

# The rec fraction between the segment and the focal netural site.
# If this is None, 0 rec fractions (fully linked) are allowed. This
# looks to cause pathologies, where as t -> 0, for reasonable mutation,
# etc parameters, B asymptotes to 1-e ~ 0.37.
MIN_RF = None

def log10_grid(log10_min, log10_max, n):
    return np.linspace(log10_min, log10_max, n)

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

def num_nonpoly(neut_pos, bins, masks):
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

def B_segment_lazy(rbp, L, t):
    """
    TODO check rbp = 0 case
    rt/ (b*(-1 + t) - t) * (b*(-1 + t) + r*(-1 + t) - t)
    """
    r = rbp*L
    a = -t*L # numerator -- ignores u
    b = (1-t)**2  # rf^2 terms
    c = 2*t*(1-t)+r*(1-t)**2 # rf terms
    d = t**2 + r*t*(1-t) # constant terms
    return a, b, c, d

def calc_B_chunk_worker(args):
    map_positions, chrom_segments, mut_grid, features_matrix, segment_parts = args
    a, b, c, d = segment_parts
    Bs = []
    for f in map_positions:
        # ---f-----L______R----------
        # ---------L______R-------f--
        is_left = (chrom_segments[:, 0] - f) > 0
        is_right = (f - chrom_segments[:, 1]) > 0
        is_contained = (~is_left) & (~is_right)
        dists = np.zeros(is_left.shape)
        dists[is_left] = np.abs(f-chrom_segments[is_left, 0])
        dists[is_right] = np.abs(f-chrom_segments[is_right, 1])
        assert len(dists) == chrom_segments.shape[0], (len(dists), chrom_segments.shape)
        # TODO fix; this should be end-specific
        # Every segment left of this position (index less than this)
        # should have its distance to this position measured to its end
        # point.

        #rf = -0.5*np.expm1(-dists)[None, :]
        rf = dists
        #rf[dists > 0.5] = 0.5
        if MIN_RF is not None:
            rf[rf < MIN_RF] = MIN_RF
        if np.any(b + rf*(rf*c + d) == 0):
            raise ValueError("divide by zero in calc_B_chunk_worker")
        x = a/(b*rf**2 + c*rf + d)
        assert(not np.any(np.isnan(x)))
        B = np.einsum('ts,w,sf->wtf', x, mut_grid,
                      features_matrix, optimize=BCALC_EINSUM_PATH)
        #B = np.flip(np.flip(B, axis=0), axis=1)
        Bs.append(B)
    return Bs

def share_array(x):
    """
    Convert a numpy array to a multiprocessing.Array
    """
    n = x.size
    cdtype = np.ctypeslib.as_ctypes_type(x.dtype)
    shared_array_base = multiprocessing.Array(cdtype, n)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj()).reshape(x.shape)
    np.copyto(shared_array, x)
    return shared_array

class ChunkIterator(object):
    def __init__(self, seqlens, recmap, segments, features_matrix,
                 segment_parts, mut_grid, step, nchunks):
        # pre-compute the positins to calculate B at for all chromosomes
        # and the segment parts
        chrom_pos = {c: bin_chrom(l, step) for c, l in seqlens.items()}
        chrom_mpos = {c: recmap.lookup(c, p, cummulative=True) for c, p
                      in chrom_pos.items()}
        chrom_idx = {c: segments.index[c] for c in seqlens}
        chrom_segments = {c: share_array(segments.map_pos[idx, :]) for c, idx
                          in chrom_idx.items()}
        chrom_features = {c: share_array(features_matrix[idx, :]) for c, idx
                          in chrom_idx.items()}
        chrom_segparts = {}
        for chrom in seqlens:
            idx = chrom_idx[chrom]
            # share the following arrays across processes, to save memory
            # (these should not be changed!)
            chrom_segparts[chrom] = tuple(map(share_array,
                                              (segment_parts[0][:, idx], segment_parts[1],
                                               segment_parts[2][:, idx], segment_parts[3][:, idx])))
        self.chrom_pos_chunks = {c: np.array_split(p, nchunks) for c, p
                                 in chrom_pos.items()}
        self.chrom_mpos_chunks = {c: np.array_split(m, nchunks) for c, m
                                  in chrom_mpos.items()}
        self.mpos_iter = chain_dictlist(self.chrom_mpos_chunks)
        self.chrom_segments = chrom_segments
        self.chrom_features = chrom_features
        self.chrom_segparts = chrom_segparts
        self.mut_grid = mut_grid
        self.nchunks = nchunks

    def __iter__(self):
        return self

    @property
    def total(self):
        return sum(map(len, list(itertools.chain(self.chrom_mpos_chunks.values()))))

    def __next__(self):
        next_chunk = next(self.mpos_iter)
        chrom, mpos_chunk = next_chunk
        return (mpos_chunk, self.chrom_segments[chrom], self.mut_grid,
                self.chrom_features[chrom], self.chrom_segparts[chrom])

    def collate(self, results):
        Bs = defaultdict(list)
        B_pos = defaultdict(list)
        pos_iter = chain_dictlist(self.chrom_pos_chunks)
        for res in results:
            try:
                chunk = next(pos_iter)
            except StopIteration:
                break
            chrom, pos = chunk
            Bs[chrom].extend(res)
            B_pos[chrom].extend(pos)
        for chrom in Bs:
            Bs[chrom] = np.array(Bs[chrom], dtype='float64')
            assert(B_pos[chrom] == sorted(B_pos[chrom]))
            B_pos[chrom] = np.array(B_pos[chrom], dtype='uint32')
        return Bs, B_pos

def calc_B_parallel(segments, segment_parts, features_matrix, mut_grid,
                    recmap, seqlens, step, nchunks=1000, ncores=2):

    chunks = ChunkIterator(seqlens, recmap, segments, features_matrix,
                           segment_parts, mut_grid, step, nchunks)
    print(f"Genome divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    debug = False
    if debug:
        res = []
        for chunk in tqdm.tqdm(chunks):
            res.append(calc_B_chunk_worker(chunk))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(calc_B_chunk_worker, chunks),
                                 total=chunks.total))
    return chunks.collate(res)


def calc_B(segments, segment_parts, features_matrix, mut_grid,
           recmap, seqlens, step):
    """
    segments: A Segments named tuple.
    segment_parts: a tuple of pre-computed segment parts.
    recmap: a RecMap object.
    features_matrix: a matrix of which segments belong to what annotation class.
    """
    Bs = defaultdict(list)
    Bpos = defaultdict(list)
    chroms = seqlens.keys()
    for chrom in chroms:
        optimal_einsum_path = None
        end = seqlens[chrom]
        print(f"calculating B for {chrom}:", flush=True)
        positions = bin_chrom(end, step)
        Bpos[chrom] = positions
        # pre-compute the map positions for each physical position
        map_pos = recmap.lookup(chrom, positions, cummulative=True)
        assert(len(positions) == len(map_pos))
        # get the segments on this chromosome
        idx = segments.index[chrom]
        nsegs = len(idx)
        chrom_segments = segments.map_pos[idx, :]

        # alias the segments parts to shorter names
        a, b, c, d = (segment_parts[0][:, idx], segment_parts[1],
                      segment_parts[2][:, idx], segment_parts[3][:, idx])
        total_sites = len(positions)
        t0 = time.time()
        F = features_matrix[idx, :]

        # pre-compute the optimal path -- this shouldn't vary accros positions
        if optimal_einsum_path is None:
            print(f"computing optimal contraction with np.einsum_path()")
            focal_map_pos = np.random.choice(map_pos)
            # as an approx, this only considers dist to start of segement
            #rf = -0.5*np.expm1(-np.abs(chrom_segments[:, 0] - focal_map_pos))[None, :]
            rf = np.abs(chrom_segments[:, 0] - focal_map_pos)[None, :]
            if MIN_RF is not None:
                rf[rf < MIN_RF] = MIN_RF
            x = a/(b*rf**2 + c*rf + d)
            #__import__('pdb').set_trace()
            optimal_einsum_path = np.einsum_path('ts,w,sf->wtf', x,
                                                 mut_grid, F, optimize='optimal')
            print(optimal_einsum_path[0])
            print(optimal_einsum_path[1])

        i = 0
        t0, t5, t10, t50 = 0, 0, 0, 0
        tq = tqdm.tqdm(zip(map_pos, positions), total=total_sites)
        #xs = list()
        for f, pos in tq:
            is_left = (chrom_segments[:, 0] - f) > 0
            is_right = (f - chrom_segments[:, 1]) > 0
            is_contained = (~is_left) & (~is_right)
            dists = np.zeros(is_left.shape)
            dists[is_left] = np.abs(f-chrom_segments[is_left, 0])
            dists[is_right] = np.abs(f-chrom_segments[is_right, 1])
            assert(len(dists) == chrom_segments.shape[0])

            #rf = -0.5*np.expm1(-dists)[None, :]
            rf = dists
            assert(not np.any(np.isnan(rf)))
            #hrf = haldanes_mapfun(np.abs(segment_posts - f))
            #assert(np.allclose(rf, hrf))
            x = a/(b*rf**2 + c*rf + d)
            #xs.append(x)
            B = np.einsum('ts,w,sf->wtf', x, mut_grid,
                          F, optimize=optimal_einsum_path[0])
            # TOOD einsum seems to flip these -- need to be flipped back
            #import pdb;pdb.set_trace()
            #B = np.flip(B, axis=1)
            #B = np.outer(x.sum(axis=1), mut_grid[:, None])
            #__import__('pdb').set_trace()
            if i > 1:
                frac_chng = 1-np.mean(np.isclose(B, Bs[chrom][-1]))
                rel_diff = np.median((B - Bs[chrom][-1])/Bs[chrom][-1])
                # how big is the diff?
                t0 += int(rel_diff < 0.05)
                t5 += int(0.05 < rel_diff < 0.1)
                t10 += int(0.1 <= rel_diff < 0.5)
                t50 += int(rel_diff >= 0.5)
                # percent values changed: {np.round(100*frac_chng, 2)}%,
                med_chng = np.abs(np.round(100*rel_diff, 2))
                msg = f"{chrom}: step {int(step/1e3)}kb, median change {med_chng:6.2f}%, ndiffs {t0} (x < 5%) {t5} (5% < x < 10%), {t10} (10% < x < 50%), {t50} (x > 50%))"
                tq.set_description(msg)
            i += 1
            Bs[chrom].append(B)
            #if i > 10:
            #    import pdb;pdb.set_trace()

        t1 = time.time()
        #print(f"chromosome {chrom} complete.", flush=True)
        diff = t1 - t0
        sites_per_hour = total_sites / (diff / (60**2))
        #print(f"{total_sites} sites took {np.round(diff/(60), 4)} minutes.\nA 3Gbp genome would take ~{np.round((3e9/step) / sites_per_hour,2)} hours.")
    return Bs, Bpos#, xs

class BGSModel(object):
    def __init__(self, recmap=None, features=None, seqlens=None,
                 t_grid=None, w_grid=None):
        # main genome data needed to calculate B
        self.recmap = recmap
        self.seqlens = seqlens
        self.segments = None
        self._segment_parts = None
        # stuff for B
        self.Bs = None
        self.B_pos = None
        self.step = None

        # B parameters
        self.t = t_grid
        self.w = w_grid

        # variation data, etc to calculate likelihoods
        self.dacfile = None
        self.metadata = None #
        self._indices = None
        self.neut_pos = None
        self.neut_pos_map = None
        # neutral region info
        self.neut_masks = None

        self.nloci = None
        self.ac = None

        # grid of pairwise same/diff counts
        self.Y = None
        #likelihood
        self.pi0_ll = None
        self.pi0_grid = None
        self.pi0i_mle = None
        if features is not None:
            # process the segments, e.g. splitting by rec rate
            self.segments = process_feature_recombs(features.ranges, self.recmap)

    def load_dacfile(self, dacfile, neut_regions):
        neut_masks = ranges_to_masks(neut_regions, self.seqlens)
        self.neut_masks = neut_masks
        res = load_dacfile(dacfile, neut_masks)
        self.neut_pos, self._indices, self.ac, self.neut_pos_map, self.metadata = res
        self.nloci = len(list(*itertools.chain(self.neut_pos.values())))
        ac = self.ac
        assert(self.nloci == ac.shape[0])
        an = ac.sum(axis=1)
        # the number of chromosomes at fixed sites is set
        # to the highest number of chromosomes in the sample (TODO, change?)
        self.nchroms = an.max()
        n_pairs = an * (an - 1) / 2
        n_same = np.sum(ac * (ac - 1) / 2, axis=1)
        n_diff = n_pairs - n_same
        self.Y = dict()
        m = np.stack((n_same, n_diff)).T
        for chrom, indices in self._indices.items():
            self.Y[chrom] = m[indices, :]

    def pi(self, width=None):
        """
        Return pairwise diversity calculated in windows of 'width', or
        if width is None, using the same windows as B.
        """
        # note, scikit-allel uses 1-based coordinates
        pis = dict()
        b = self.BScores
        for chrom in self.neut_pos:
            ac = self.ac
            mask = self.neut_masks[chrom]
            pos = np.array(self.neut_pos[chrom]) + 1
            if width is None:
                bins = self.B_bins[chrom]
            else:
                bins = bin_chrom(self.seqlens[chrom], width)
            # again, allel uses 1-based
            wins = np.stack((bins[:-1]+1, bins[1:])).T
            pis[chrom] = allel.windowed_diversity(pos, ac, windows=wins,
                                                  is_accessible=mask)
        return pis

    def gwpi(self):
        """
        Genome-wide pi.
        """
        pis = dict()
        for chrom in self.neut_pos:
            ac = self.ac
            mask = self.neut_masks[chrom]
            # allel is 1-indexed
            pos = np.array(self.neut_pos[chrom]) + 1
            pis[chrom] = allel.sequence_diversity(pos, ac, is_accessible=mask)
        x = list(pis.values())
        y = [self.seqlens[c] for c in pis.keys()]
        pi_bar = np.average(x, weights=y)
        assert(np.isfinite(pi_bar) and not np.isnan(pi_bar))
        return pi_bar

    def bin_B(self, width):
        """
        Bin B into genomic windows with width 'width'.
        """
        if width < 10*self.step:
            msg = f"bin width ({width}) <= 10*step size ({self.step}); recommended it's larger"
            raise ValueError(msg)
        B = self.BScores
        binned_B = dict()
        chroms = B.pos.keys()
        for chrom in chroms:
            bins = bin_chrom(self.seqlens[chrom], width)
            n = bins.shape[0] - 1 # there are one less windows than binends
            wins = np.stack((bins[:-1], bins[1:])).T
            idx = np.digitize(B.pos[chrom], bins[1:]) # ignore zero
            grps = np.unique(idx)
            shape = (n, B.B[chrom].shape[1], B.B[chrom].shape[2])
            means = np.full(shape, np.nan)
            nitems = np.zeros(n)
            for i in range(n):
                y = B.B[chrom][i == idx, :, :]
                nitm = (i == idx).sum()
                if nitm > 0:
                    means[i, :, :] = np.nanmean(y, axis=0).squeeze()
                nitems[i] = nitm
            binned_B[chrom] = BinnedStat(means, wins, nitems)
        return binned_B

    def loglikelihood(self, pi0=None, pi0_bounds=None, pi0_grid=None):
        b = self.BScores
        Y_binned, midpoint_Bs, pi_win = calc_loglik_components(b, self.Y, self.neut_pos, self.neut_masks, self.nchroms)
        # for pi0, merge all chromosomes
        bs = np.stack(list(*midpoint_Bs.values()), axis=0)
        ys = np.stack(list(*Y_binned.values()), axis=0)
        chroms = list(b.pos.keys())
        nset = sum([x is not None for x in (pi0, pi0_bounds, pi0_grid)])
        if nset > 1:
            raise ValueError("no more than one pi0, pi0_grid, pi0_bounds can be set")
        if nset == 0:
            raise ValueError("set either pi0, pi0_bounds, or pi0_grid")
        if pi0_bounds is not None:
            print(f"using bounded {pi0_bounds} optimizion for π0...")
            f = loglik_pi0(bs, ys)
            optim = minimize_scalar(f, bounds=(0, 0.1), method='bounded')
            print(optim)
            self.optim = optim
            pi0 = optim.x
        if pi0_grid is not None:
            lls = [loglik(p, bs, ys).sum(axis=0) for p in tqdm.tqdm(pi0_grid)]
            lls_mat = np.stack(lls)
            max_idx = np.unravel_index(np.argmax(lls_mat), lls_mat.shape)
            #__import__('pdb').set_trace()
            pi0 = pi0_grid[max_idx[0]]
            self.pi0_ll = lls_mat
            self.pi0i_mle = max_idx[0]
            self.pi0_grid = pi0_grid
        ll = loglik(pi0, bs, ys)
        lls = ll.sum(axis=0)
        self.wi_mle, self.ti_mle = np.where(lls == np.nanmax(lls))
        self.pi_win = pi_win
        self.pi0 = pi0
        self.ll = ll
        return ll, pi0

    def _calc_segments(self):
        """
        """
        L = np.diff(self.segments.ranges, axis=1).squeeze()
        rbp = self.segments.rates
        #min_rbp = 0 # here as test; doesn't make much of a difference
        #rbp[rbp == 0] = min_rbp
        # turn this into a column vector for downstream
        # operations
        t = self.t[:, None]
        nfeats = len(self.segments.feature_map)
        nsegs = len(self.segments.features)
        F = np.zeros(shape=(nsegs, nfeats), dtype='bool')
        # build a one-hot matrix of features
        np.put_along_axis(F, self.segments.features[:, None], 1, axis=1)
        self.F = F
        self._segment_parts = B_segment_lazy(rbp, L, t)
        #print([x.shape for x in self._segment_parts])

    @property
    def BScores(self):
        Bs = {c: b for c, b in self.Bs.items()}
        return BScores(Bs, self.B_pos, self.w, self.t, self.step)

    def BScores_interpolater(self, feature_idx, **kwargs):
        defaults = {'kind': 'quadratic',
                    'assume_sorted': True,
                    'bounds_error': False,
                    'copy': False}
        kwargs = {**defaults, **kwargs}
        interpols = defaultdict(dict)
        x = self.B_pos
        Bs = {c: b for c, b in self.Bs.items()}
        for i, w in enumerate(self.w):
            for j, t in enumerate(self.t):
                for chrom in Bs:
                    y = Bs[chrom][:, i, j, feature_idx]
                    func = interpolate.interp1d(x[chrom], y,
                                                fill_value=(y[0], y[-1]),
                                                **kwargs)
                    interpols[chrom][(w, t)] = func
        return interpols

    def save_B(self, filename):
        if self.Bs is None or self.B_pos is None:
            raise ValueError("B scores not yet calculated.")
        with open(filename, 'wb') as f:
            pickle.dump(self.BScores, f)

    def load_B(self, filename):
        with open(filename, 'rb') as f:
            B = pickle.load(f)
            self.Bs, self.B_pos, self.w, self.t, self.step = B.B, B.pos, B.w, B.t, B.step

    @property
    def B_bins(self):
        if self.step is None:
            raise ValueError("step is not set, cannot calculate bins")
        chroms = self.B_pos.keys()
        return {c: bin_chrom(self.seqlens[c], self.step) for c in chroms}

    def calc_B(self, step=10_000, ncores=None, nchunks=None):
        if ncores is not None and nchunks is None:
            raise ValueError("if ncores is set, nchunks must be specified")
        self.step = step
        if self._segment_parts is None:
            print(f"pre-computing segment contributions...\t", end='')
            self._calc_segments()
            print(f"done.")
        if ncores is None:
            Bs, B_pos = calc_B(self.segments, self._segment_parts, self.F,
                               self.w, self.recmap, self.seqlens, step=step)
        else:
            Bs, B_pos = calc_B_parallel(self.segments, self._segment_parts,
                                        self.F, self.w, self.recmap,
                                        self.seqlens,step=step, nchunks=nchunks,
                                        ncores=ncores)
        stacked_Bs = {chrom: np.stack(x).astype(Bdtype) for chrom, x in Bs.items()}
        self.Bs = stacked_Bs
        self.B_pos = B_pos
        #self.xs = xs

