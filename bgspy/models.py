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
import os
import pickle
import warnings
import itertools
import tqdm
import time
import numpy as np
import allel
from scipy import interpolate
from scipy.stats import binned_statistic, binned_statistic_dd
from scipy.optimize import minimize_scalar
import tensorflow as tf

from bgspy.utils import bin_chrom, genome_emp_dists
from bgspy.utils import readfile, load_dacfile
from bgspy.utils import load_seqlens, make_dirs
from bgspy.utils import ranges_to_masks, sum_logliks_over_chroms
from bgspy.utils import Bdtype, BScores, BinnedStat
from bgspy.likelihood import calc_loglik_components, loglik
from bgspy.classic import calc_B, calc_B_parallel
from bgspy.learn import LearnedFunction, LearnedB
from bgspy.parallel import MapPosChunkIterator

class BGSModel(object):
    def __init__(self, genome, t_grid=None, w_grid=None, split_length=1_000):
        # main genome data needed to calculate B
        self.genome = genome
        assert self.genome.is_complete(), "genome is missing data!"
        diff_split_lengths = split_length != genome.split_length
        if self.genome.segments is None or diff_split_lengths:
            if diff_split_lengths:
                warnings.warn("supplied Genome object has segment split lengths that differ from that specified -- resegmenting")
            self.genome.create_segments(split_length=split_length)
        # stuff for B
        self.Bs = None
        self.Bps = None
        self.B_pos = None
        self.step = None

        # B parameters
        self.t = np.sort(t_grid)
        self.w = np.sort(w_grid)

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
        #self._calc_features()  # TODO, not full implemented

        # machine learning stuff
        self.bfunc = None

    @property
    def seqlens(self):
        return self.genome.seqlens

    @property
    def recmap(self):
        return self.genome.recmap

    @property
    def segments(self):
        return self.genome.segments

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
        raise NotImplementedError("only support for one feature type")
        # nfeats = len(self.segments.feature_map)
        # nsegs = len(self.segments.features)
        # F = np.zeros(shape=(nsegs, nfeats), dtype='bool')
        # # build a one-hot matrix of features
        # np.put_along_axis(F, self.segments.features[:, None], 1, axis=1)
        # self.F = F

    @property
    def BScores(self):
        Bs = {c: b for c, b in self.Bs.items()}
        return BScores(Bs, self.B_pos, self.w, self.t, self.step)

    def BScores_interpolater(self, **kwargs):
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
                    # The last dimension is features matrix -- for now, only
                    # one feature is supported
                    y = Bs[chrom][:, i, j, 0]
                    func = interpolate.interp1d(x[chrom], y,
                                                fill_value=(y[0], y[-1]),
                                                **kwargs)
                    interpols[chrom][(w, t)] = func
        return interpols

    def save_B(self, filename):
        if self.Bs is None or self.B_pos is None:
            raise ValueError("B scores not yet calculated.")
        assert filename.endswith('.pkl'), "filename should end in '.pkl'"
        self.BScores.save(filename)

    def load_B(self, filename):
        assert filename.endswith('.pkl'), "filename should end in '.pkl'"
        obj = BScores.load(filename)
        self.Bs, self.B_pos = obj.B, obj.pos
        self.w, self.t, self.step = obj.w, obj.t, obj.step
        return obj

    @property
    def B_bins(self):
        if self.step is None:
            raise ValueError("step is not set, cannot calculate bins")
        chroms = self.B_pos.keys()
        return {c: bin_chrom(self.seqlens[c], self.step) for c in chroms}

    def calc_stats(self, mu, s, subsample_frac=0.01, B_subsample_frac=0.001, step=10_100):
        return genome_emp_dists(self.genome, step=step,
                                mu=mu, s=s,
                                B_subsample_frac=B_subsample_frac,
                                subsample_frac=subsample_frac)

    def calc_B(self, step=10_000, ncores=None, nchunks=None):
        if ncores is not None and nchunks is None:
            raise ValueError("if ncores is set, nchunks must be specified")
        self.step = step
        if self.genome.segments._segment_parts is None:
            print(f"pre-computing segment contributions...\t", end='')
            self.genome.segments._calc_segparts(self.t)
            print(f"done.")
        segment_parts = self.genome.segments._segment_parts
        if ncores is None or ncores <= 1:
            Bs, B_pos = calc_B(self.genome, self.w, step=step)
        else:
            Bs, B_pos = calc_B_parallel(self.genome, self.w,
                                        step=step, nchunks=nchunks,
                                        ncores=ncores)
        stacked_Bs = {chrom: np.stack(x).astype(Bdtype) for chrom, x in Bs.items()}
        self.Bs = stacked_Bs
        self.B_pos = B_pos
        #self.xs = xs

    def load_learnedfunc(self, filepath):
        func = LearnedFunction.load(filepath)
        bfunc = LearnedB(self.t, self.w, genome=self.genome)
        bfunc.func = func
        bfunc.is_valid_grid()
        try:
            bfunc.is_valid_segments()
        except AssertionError as msg:
            warnings.warn("Some segments are out of training bounds.\n"
                          "This is not inherently a problem; their values will be "
                          "set to the boundary values. Message: " + str(msg))

        self.bfunc = bfunc

    def _focal_positions(self, step, nchunks, max_map_dist, progress=True):
        """
        Get the focal positions for each position that B is calculated at, with
        the indices max_map_dist apart.


        nchunks is how many chunks across the genome to break this up
        to (for parallel processing)
        """
        chunks = MapPosChunkIterator(self.genome, self.w, self.t,
                                     step=step, nchunks=nchunks)

        # we iterate over the physical and map position chunks --
        # these are each packaged with their chromosomes
        sites_iter = zip(chunks.pos_iter, chunks.mpos_iter)

        if progress:
            sites_iter = tqdm.tqdm(sites_iter, total=chunks.total)

        for ((chrom, pos_chunk), (_, mpos_chunk)) in sites_iter:
            # for this chunk of map positions, get the segment indices
            # within max_map_dist from the start/end of map positions
            lidx = self.genome.get_segment_slice(chrom, mpos=mpos_chunk[0], map_dist=max_map_dist)[0]
            uidx = self.genome.get_segment_slice(chrom, mpos=mpos_chunk[-1], map_dist=max_map_dist)[1]

            # package physical and map positions together -- this makes
            # stitching these back together safe across distributed computing
            sites_chunk = np.array((pos_chunk, mpos_chunk)).T
            yield chrom, sites_chunk, (lidx, uidx)

    def _build_segment_matrix(self, chrom):
        """
        The columns are the features of the B' training data are
               0,      1,          2,        3
              'L', 'rbp', 'map_start' 'map_end'

        Note: previously this fixed the bounds. But because rf bounds *have*
        to be fixed when the focal sites are filled in, this requires bounds
        checking at the lower level, so it's done entirely during prediction.
        """
        segments = self.genome.segments
        nsegs = self.genome.segments.nsegs[chrom]
        S = np.empty((nsegs, 4))
        S[:, 0] = self.segments.L[chrom]
        S[:, 1] = self.segments.rbp[chrom]
        S[:, 2] = self.segments.mpos[chrom][:, 0]
        S[:, 3] = self.segments.mpos[chrom][:, 1]
        return S

    def _build_segment_matrices(self):
        self._Ss = {}
        for chrom in self.genome.seqlens.keys():
            self._Ss[chrom] = self._build_segment_matrix(chrom)

    def write_prediction_chunks(self, dir, step=1000, nchunks=1000, max_map_dist=0.1):
        """
        Write files necessary for B' prediction across a cluster.
        All files (and later predictions) are stored in the supplied 'dir'.

         - dir/chunks/
         - dir/segments/

        Note that these data are prediction model agnostic, e.g. they work regardless
        of the prediction model. Other information is needed, e.g. whether to 
        transform these features and the parameters for centering/scaling.
        """
        name = self.genome.name
        self.genome._build_segment_idx_interpol()
        self._build_segment_matrices()
        # build main directory if it doesn't exist
        dir = make_dirs(dir)

        chrom_dir = make_dirs(dir, 'segments')
        for chrom, S in self._Ss.items():
            nsegs = self.segments.nsegs[chrom]
            filename = os.path.join(chrom_dir, f"{name}_{chrom}.npy")
            np.save(filename, S.astype('f8'))

        focal_pos_iter = self._focal_positions(step=step, nchunks=nchunks, 
                                               max_map_dist=max_map_dist)

        for i, (chrom, sites_chunk, segslice) in enumerate(focal_pos_iter):
            chunk_dir = make_dirs(dir, 'chunks', chrom)
            lidx, uidx = segslice
            filename = os.path.join(chunk_dir, f"{name}_{chrom}_{i}_{lidx}_{uidx}.npy")
            np.save(filename, sites_chunk)


