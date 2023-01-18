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
from scipy.optimize import minimize_scalar
import scipy.stats as stats
import pandas as pd

from bgspy.utils import signif
from bgspy.utils import Bdtype, BScores, BinnedStat, bin_chrom
from bgspy.parallel import calc_B_parallel, calc_BSC16_parallel
from bgspy.substitution import grid_interp_weights

class BGSModel(object):
    """
    BGSModel contains the the segments under purifying selection
    to compute the B and B' values.
    """
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

    @property
    def features(self):
        features = [self.segments.inverse_feature_map[i] for i in range(self.nf)]
        return features

    @property
    def seqlens(self):
        return self.genome.seqlens

    @property
    def recmap(self):
        return self.genome.recmap

    @property
    def segments(self):
        return self.genome.segments

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
    def nf(self):
        return len(self.segments.feature_map)

    @property
    def BScores(self):
        return BScores(self.Bs, self.B_pos, self.w, self.t, self.features, self.step)

    @property
    def BpScores(self):
        return BScores(self.Bps, self.Bp_pos,
                       self.w, self.t, self.features,
                       self.step)

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

    def calc_B(self, step=10_000, recalc_segments=False,
               ncores=None, nchunks=None):
        """
        Calculate classic B values across the genome.
        """
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

    def calc_Bp(self, N, step=100_000, recalc_segments=False,
                ncores=None, nchunks=None, fit=None, rescale=None):
        """
        Calculate new B' values across the genome.


        If fit is not None this is a B' fit that is used for local
        reductions mode.
        """
        use_rescaling = False
        if fit is not None:
            assert rescale is None, "cannot use rescale with fit"
            # load the rescaling factors from a model fit
            # NOTE: this only effects the segment parts! nothing past that
            # in the B' calc
            self.genome.segments.load_rescaling_from_fit(fit)
            use_rescaling = True # require segments to be re-calc'd

        if rescale is not None:
            assert fit is None, "cannot use fit with rescale"
            bp, w, t = rescale
            self.genome.segments.load_rescaling_from_fixed_params(bp, w, t)
            use_rescaling = True # require segments to be re-calc'd

        if ncores is not None and nchunks is None:
            raise ValueError("if ncores is set, nchunks must be specified")
        self.step = step
        if use_rescaling or recalc_segments or self.genome.segments._segment_parts_sc16 is None:
            self.genome.segments._calc_segparts(self.w, self.t, N, ncores=ncores)

        Bs, B_pos = calc_BSC16_parallel(self.genome, step=step, N=N,
                                        nchunks=nchunks, ncores=ncores)
        stacked_Bs = {chrom: np.stack(x).astype(Bdtype) for chrom, x in Bs.items()}
        prop_nan = [np.isnan(s).mean() for s in stacked_Bs.values()]
        if any(x > 0 for x in prop_nan):
            msg = f"some NAN in B'! likely fsolve failed under strong sel"
            warnings.warn(msg)
        self.Bps = stacked_Bs
        self.Bp_pos = B_pos
        self.N = N

    def get_ratchet_rates(self, wi, ti, chrom=None, as_times=False,
                          use_midpoints=True, width=None):
        """
        Get the pre-computed fixation times. Bin in width bins
        if width is not None.
        """
        segments = self.genome.segments
        if chrom is not None:
            # get the ratchet rate for one chromosome
            idx = segments.index[chrom]
        else:
            idx = slice(None)
        midpoints = segments.ranges[idx].mean(axis=1)
        seglens = np.diff(segments.ranges[idx], axis=1).squeeze()
        # elements are V, Vm, T -- we call T = x here
        x = segments._segment_parts_sc16[2][wi, ti, idx]
        features = segments.features[idx]
        if not as_times:
            # convert to ratchet rate
            x = 1/x

        x = x/seglens
        if width is not None:
            bins = bin_chrom(self.seqlens[chrom], width)
            binstats = stats.binned_statistic(midpoints, x,
                                              statistic=np.nanmean,
                                              bins=bins)
            bin_midpoints = 0.5*(bins[:-1] + bins[1:])
            return bin_midpoints, binstats
        else:
            if not use_midpoints:
                midpoints = segments.ranges[idx]
            return midpoints, x, features, seglens

    def get_ratchet_segment_array(self):
        segments = self.genome.segments
        midpoints = segments.ranges.mean(axis=1)
        seglens = np.diff(segments.ranges, axis=1).squeeze()
        # elements are V, Vm, T -- we call T = x here
        T = segments._segment_parts_sc16[2]
        R = 1/T
        r = R/seglens
        return midpoints, r, segments.ranges, seglens

    def ratchet_df(self, fit):
        """
        Output a combined ratchet, for all segments.

        W is the MLE estimate of the DFE weights.
        If this is not specified, then all classes
        are given equal weight.

        TODO: this code should go to substitutions module.
        """
        W = fit.mle_W
        mus = fit.mle_mu
        from bgspy.likelihood import SimplexModel
        if isinstance(fit, SimplexModel):
            # repeat one for each feature (for simplex, these are all same)
            mus = np.repeat(mus, fit.nf)

        F = self.genome.segments.F
        segments = self.genome.segments
        ranges = segments.ranges
        seglens = np.diff(segments.ranges, axis=1).squeeze()
        fmaps = segments.inverse_feature_map

        # get chroms
        chroms = []
        for chrom, indices in segments.index.items():
            chroms.extend([chrom] * len(indices))
        chroms = np.array(chroms)

        T = segments._segment_parts_sc16[2]
        R = 1/T
        #r = R/seglens ## this underflows
        r = np.zeros_like(R)
        np.divide(R, seglens[None, None, :], out=r, where=R > np.finfo(np.float64).tiny * seglens.max())

        pred_rs = []
        pred_Rs = []
        pred_load = []
        chrom_col = []
        start_col, end_col = [], []
        seglen_col = []
        feature_col = []
        for i in range(F.shape[1]):
            # get the substiution rates and other info *for this feature type*
            fidx = F[:, i]
            rs = r[..., fidx]
            Rs = R[..., fidx]
            nsegs = fidx.sum()
            mu = mus[i]

            r_interp = np.zeros(nsegs)
            R_interp = np.zeros(nsegs)
            load_interp = np.zeros(nsegs)
            for k in range(len(self.t)):
                muw = mu*W[k, i]
                j, weight = grid_interp_weights(self.w, muw)

                # we should get the mutation rate back if we weight the mutation
                # grid
                assert np.allclose(weight*self.w[j-1] + (1-weight)*self.w[j], muw)

                r_this_selcoef = weight*rs[j-1, k, :] + (1-weight)*rs[j, k, :]
                R_this_selcoef = weight*Rs[j-1, k, :] + (1-weight)*Rs[j, k, :]
                try:
                    assert np.all(r_this_selcoef <= muw)
                except:
                    n_over = (r_this_selcoef > muw).sum()
                    frac = signif(100*(r_this_selcoef > muw).mean(), 2)
                    msg = (f"{n_over} segments ({frac}%) had estimated substitution rates "
                            "greater than the DFE-weighted mutation rate for the"
                           f" class s={self.t[k]}, feature '{fmaps[i]}'. This "
                            "occurrs most likely due to numeric error in solving "
                            "the nonlinear system of equations. These are set to zero.")
                    warnings.warn(msg)
                    r_this_selcoef[r_this_selcoef > muw] = 0.
                r_interp += r_this_selcoef
                try:
                    assert np.all(r_interp <= mu)
                except:
                    __import__('pdb').set_trace()
                R_interp += R_this_selcoef
                with np.errstate(under='ignore'):
                    load_interp += np.log((1-2*self.t[k]))*R_this_selcoef

            # __import__('pdb').set_trace()
            pred_rs.extend(r_interp)
            pred_Rs.extend(R_interp)
            pred_load.extend(load_interp)

            chrom_col.extend(chroms[fidx])
            start_col.extend(ranges[fidx, 0])
            end_col.extend(ranges[fidx, 1])
            seglen_col.extend(seglens[fidx])
            fs = [fmaps[i]] * fidx.sum()
            feature_col.extend(fs)

        d = pd.DataFrame({'chrom': chrom_col,
                          'start': start_col,
                          'end': end_col,
                          'feature': feature_col,
                          'R': pred_Rs,
                          'r': pred_rs,
                          'seglen': seglen_col,
                          'load': pred_load})
        return d

    def get_ratchet_binned_array(self, chrom, width):
        bins = bin_chrom(self.seqlens[chrom], width)
        nbins = len(bins) - 1
        R = np.full((nbins, len(self.w), len(self.t)), np.nan)
        pos = 0.5*(bins[:-1] + bins[1:])
        for wi, w in enumerate(self.w):
            for ti, t in enumerate(self.t):
                mps, bins = self.get_ratchet_rates(wi, ti, chrom, width=width)
                if pos is not None:
                    assert np.all(mps == pos)
                R[:, wi, ti] = bins.statistic
        return pos, R

    #def fill_Bp_nan(self):
    #    """
    #    Sometimes the B' calculations fail, e.g. due to T = Inf; these
    #    can be backfilled with B since they're the same in this domain.
    #    This isn't done manually as we should check there isn't some
    #    other pathology.

    #    TODO: Deprecated?
    #    """
    #    assert self.Bps is not None, "B' not calculated!"
    #    assert self.Bs is not None, "B not calculated!"
    #    for chrom, Bp in self.Bps.items():
    #        B = self.Bs[chrom]
    #        assert Bp.shape == B.shape, "incompatible dimensions!"
    #        # back fill the values
    #        Bp[np.isnan(Bp)] = B[np.isnan(Bp)]

