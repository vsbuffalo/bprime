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
import multiprocessing
import pickle
import itertools
import tqdm
import time
import numpy as np
import allel
from scipy import interpolate
from scipy.stats import binned_statistic, binned_statistic_dd
from scipy.optimize import minimize_scalar
import tensorflow as tf

from bprime.utils import bin_chrom
from bprime.utils import RecMap, readfile, load_dacfile
from bprime.utils import process_feature_recombs, load_bed_annotation
from bprime.utils import load_seqlens
from bprime.utils import ranges_to_masks, sum_logliks_over_chroms
from bprime.likelihood import calc_loglik_components, loglik
from bprime.classic import B_segment_lazy, calc_B, calc_B_parallel
from bprime.learn import LearnedB, LearnedFunction, calc_Bp_chunk_worker
from bprime.parallel import BpChunkIterator

# this dtype allows for simple metadata storage
Bdtype = np.dtype('float32', metadata={'dims': ('site', 'w', 't', 'f')})
BScores = namedtuple('BScores', ('B', 'pos', 'w', 't', 'step'))
BinnedStat = namedtuple('BinnedStat', ('statistic', 'wins', 'nitems'))

class BGSModel(object):
    def __init__(self, recmap=None, features=None, seqlens=None,
                 t_grid=None, w_grid=None, split_length=1_000):
        # main genome data needed to calculate B
        self.recmap = recmap
        self.seqlens = seqlens
        self.segments = None
        self._segment_parts = None
        # stuff for B
        self.Bs = None
        self.Bps = None
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
            self.segments = process_feature_recombs(features.ranges, self.recmap, split_length)
            self._calc_features()

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

    def _calc_features(self):
        """
        Create a dummy matrix of features for each segment.
        """
        nfeats = len(self.segments.feature_map)
        nsegs = len(self.segments.features)
        F = np.zeros(shape=(nsegs, nfeats), dtype='bool')
        # build a one-hot matrix of features
        np.put_along_axis(F, self.segments.features[:, None], 1, axis=1)
        self.F = F

    def _calc_segments(self):
        """
        Pre-calculate the segment contributions, for classic B approach
        approach.
        """
        L = np.diff(self.segments.ranges, axis=1).squeeze()
        rbp = self.segments.rates
        #min_rbp = 0 # here as test; doesn't make much of a difference
        #rbp[rbp == 0] = min_rbp
        # turn this into a column vector for downstream
        # operations
        t = self.t[:, None]
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

    def add_learnedB(self, learned_func):
        """
        Append a learned B function.
        """
        self.learned_func = learned_func

    def calc_Bp(self, step=10_000, ncores=None, nchunks=None):
        """
        Calculate B', a B statistic using a learned B function.
        """
        model_file = "../data/dnn_models/fullbgs"
        print(f"loading DNN model from '{model_file}'...\t", end='')
        learned_func = LearnedFunction.load(model_file)
        self.dnnB = learned_func
        print(f"done.")
        chunks = BpChunkIterator(self.seqlens, self.recmap, self.segments,
                                self.F, self.w, np.log10(self.t),
                                 learned_func.X_test_scaler, step, nchunks)
        res = learned_func.model.predict(iter(chunks), use_multiprocessing=True,
                                         workers=70)
        return chunks.collate(res)



