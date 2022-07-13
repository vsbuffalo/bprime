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
from scipy.optimize import minimize_scalar
import tensorflow as tf

from bgspy.utils import bin_chrom, bin_chroms, genome_emp_dists
from bgspy.utils import Bdtype, BScores, BinnedStat
from bgspy.likelihood import calc_loglik_components, loglik
from bgspy.classic import calc_B, calc_B_parallel, calc_BSC16_parallel
from bgspy.parallel import MapPosChunkIterator


class BGSModel(object):
    def __init__(self, genome, t_grid=None, w_grid=None, split_length=None):
        # main genome data needed to calculate B
        self.genome = genome
        assert self.genome.is_complete(), "genome is missing data!"
        diff_split_lengths = genome.split_length is not None and split_length != genome.split_length
        if self.genome.segments is None or diff_split_lengths:
            if diff_split_lengths:
                warnings.warn("supplied Genome object has segment split lengths that differ from that specified -- resegmenting")
            self.genome.create_segments(split_length=split_length)
        # stuff for B
        self.Bs = None
        self.B_pos = None
        self.Bps = None
        self.Bp_pos = None
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

    def loglikelihood(self, width, pi0=None, pi0_bounds=None, pi0_grid=None,
                      method='SC16'):
        if method == 'SC16':
            b = self.BpScores
        elif method == 'classic':
            b = self.BScores
        else:
            raise ValueError("method must be either 'SC16' or 'classic'")
        assert b is not None, "BScores are not calculated!"

        # create the windows that are the unit of analysis
        chrom_bins = bin_chroms(self.genome.seqlens, width)
        comps = calc_loglik_components(b, self.Y, self.neut_pos, self.neut_masks,
                                     self.nchroms, chrom_bins)
        Y_binned, midpoint_Bs, pi_win = comps

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

        # main likelihood routine
        ll = loglik(pi0, bs, ys)
        lls = ll.sum(axis=0)
        mle = MLEFit(m.w, m.t, lls, pi0_grid, pi0)
        mle._pi_win = pi_win
        return mle

    def save(self, filename):
        assert filename.endswith('.pkl'), "filename should end in '.pkl'"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @property
    def BScores(self):
        return BScores(self.Bs, self.B_pos, self.w, self.t, self.step)

    @property
    def BpScores(self):
        return BScores(self.Bps, self.Bp_pos, self.w, self.t, self.step)

    def save_B(self, filename):
        """
        Save both B and B'.
        """
        if self.Bs is None or self.B_pos is None:
            raise ValueError("B scores not yet calculated.")
        assert filename.endswith('.pkl'), "filename should end in '.pkl'"
        with open(filename, 'wb') as f:
            pickle.dump({'B': self.BScores, 'Bp': self.BpScores})

    def load_B(self, filename):
        """
        Load both B and B'.
        """
        assert filename.endswith('.pkl'), "filename should end in '.pkl'"
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        # check the consistency
        assert obj['B'].w == obj['Bp'].w
        assert obj['B'].t == obj['Bp'].t
        assert obj['B'].pos == obj['Bp'].pos
        b = obj['B']
        self.Bs, self.B_pos = b, b.pos
        self.w, self.t, self.step = b.w, b.t, b.step
        self.Bs = obj['B'].B
        self.Bps = obj['Bp'].B
        return self

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

    def calc_B(self, step=10_000, recalc_segments=False,
               ncores=None, nchunks=None):
        if ncores is not None and nchunks is None:
            raise ValueError("if ncores is set, nchunks must be specified")
        self.step = step
        if recalc_segments or self.genome.segments._segment_parts is None:
            print(f"pre-computing segment contributions...\t", end='')
            self.genome.segments._calc_segparts(self.w, self.t)
            print(f"done.")
        segment_parts = self.genome.segments._segment_parts
        Bs, B_pos = calc_B_parallel(self.genome, self.w,
                                    step=step, nchunks=nchunks,
                                    ncores=ncores)
        stacked_Bs = {chrom: np.stack(x).astype(Bdtype) for chrom, x in Bs.items()}
        self.Bs = stacked_Bs
        self.B_pos = B_pos
        #self.xs = xs

    def calc_Bp(self, N, step=10_000, recalc_segments=False, ncores=None, nchunks=None):
        if ncores is not None and nchunks is None:
            raise ValueError("if ncores is set, nchunks must be specified")
        self.step = step
        if recalc_segments or self.genome.segments._segment_parts_sc16 is None:
            self.genome.segments._calc_segparts(self.w, self.t, N)
        Bs, B_pos = calc_BSC16_parallel(self.genome, step=step,
                                        nchunks=nchunks, ncores=ncores)
        stacked_Bs = {chrom: np.stack(x).astype(Bdtype) for chrom, x in Bs.items()}
        self.Bps = stacked_Bs
        self.Bp_pos = B_pos
