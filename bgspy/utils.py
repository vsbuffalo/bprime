from dataclasses import dataclass
import pickle
import sys
import re
import warnings
import gzip
import os
import tqdm
import pyBigWig
from collections import namedtuple, defaultdict, deque, Counter
from functools import partial
import math
import itertools
from math import floor, log10
from scipy import interpolate
from scipy.stats import binned_statistic, spearmanr, pearsonr
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
import newick



SEED_MAX = 2**32-1

# alias this
lowess = sm.nonparametric.lowess

# simple named tuple when we just want to store w/t grids
Grid = namedtuple('Grid', ('w','t'))


# this dtype allows for simple metadata storage
Bdtype = np.dtype('float32', metadata={'dims': ('site', 'w', 't', 'f')})

RecPair = namedtuple('RecPair', ('end', 'rate'))
Segments = namedtuple('Segments', ('ranges', 'rates', 'map_pos', 'features',
                                   'feature_map', 'index'))

# this function is used a lot...
def midpoints(x):
    if isinstance(x, dict):
        return {c: 0.5*(b[1:, ...] + b[:-1, ...]) for c, b in x.items()}
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        return 0.5*(x[1:, ...] + x[:-1, ...])


def is_sorted_array(x):
    return np.all(x[:-1] <= x[1:])


class BinnedStat:
    __slots__ = ('stat', 'bins', 'n')
    def __init__(self, stat, bins, n):
        self.stat = stat
        self.bins = bins
        self.n = n

    def __len__(self):
        return self.stat.shape[0]

    @property
    def midpoints(self):
        return midpoints(self.bins)

    def __repr__(self):
        width = Counter(np.diff(self.bins)).most_common(1)[0][0]
        stat_dim = '×'.join(map(str, self.stat.shape))
        return f"BinnedStat(shape: {stat_dim}, bin width: {width})"

    @property
    def pairs(self):
        # ignores the first bin, which is data for points left of zero
        return (self.midpoints, self.stat)

class BScores:
    def __init__(self, B, pos, w, t, features=None, step=None):
        self.B = B
        self.pos = pos
        assert set(B.keys()) == set(pos.keys()), "different chromosomes between Bs and positions!"
        for chrom in B:
            msg =  f"Bs and positions have different lengths in {chrom}!"
            assert B[chrom].shape[0] == pos[chrom].shape[0], msg
        assert is_sorted_array(w), "w is not sorted"
        assert is_sorted_array(t), "t is not sorted"
        self.w = w
        self.t = t
        self.X = None
        self.sd = None
        self.step = step
        self.features = None
        self._interpolators = None

    @property
    def nf(self):
        "Get the number of features"
        nf = [x.shape[3] for x in self.B.values()]
        assert len(set(nf)) == 1, "inconsistent B matrices across classes!"
        return nf[0]

    @property
    def nt(self):
        return len(self.t)

    @property
    def nw(self):
        return len(self.w)

    def indices(self, w, t):
        return w == self.w, t == self.t

    def __getitem__(self, tup):
        chrom, w, t = tup
        pos = self.pos[chrom]
        assert w in self.w, f"'{w}' is not in w array '{self.w}'"
        assert t in self.t, f"'{t}' is not in w array '{self.t}'"
        wi, ti = self.indices(w, t)
        logBs = self.B[chrom][:, wi, ti, ...].squeeze()
        Bs = np.full_like(logBs, 0.)
        # where clause prevents underflow, which are set to zero.
        Bs = np.exp(logBs, out=Bs, where=-logBs < np.log(np.finfo(logBs.dtype).max))
        return pos, Bs

    def pairs(self, chrom, w, t):
        tup = chrom, w, t
        pos, Bs = self[tup]
        return pos, Bs

    def get_nearest(self, chrom, w, t):
        widx = arg_nearest(w, self.w)
        tidx = arg_nearest(t, self.t)
        return self.pos[chrom], self.B[chrom][:, widx, tidx, ...]

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @classmethod
    def load_npz(self, filepath, chrom, bin_pos=True, is_log=False):
        """
        Load a single chromosome's values from a npz file, e.g. for simulation
        data.

        If bin_pos is true, the positions are the bin edges (e.g. if the
        diversity is calculated from simulated data and binned branch-mode
        diversity is used from tskit).

        By default this assumes the B values (and X and sd) are not logged,
        as this is what the sims return. To be consistent they are logged.
        """
        assert filepath.endswith('.npz'), "filepath must be a Numpy .npz"
        d = np.load(filepath)
        w, t = d['mu'], d['sh']
        pos = d['pos']
        if bin_pos:
            # these are tskit-like bin edges, so the B value (average in the
            # bin) should have the position that's at the middle.
            pos = midpoints(pos)
        X = d['X']
        B = d['mean']
        if not is_log:
            B = np.log(B)
        obj = BScores({chrom: B}, {chrom: pos}, w, t)
        obj.sd = d['sd']
        obj.X = X
        return obj

    def _Xpairs(self, w, t, i, bins=None):
        """
        Mostly for internal stuff -- looking at raw sim data.
        """
        wi, ti = self.indices(w, t)
        out = self.X[:, wi, ti, i]
        only_chrom = list(set(self.pos.keys()))[0]
        pos = self.pos[only_chrom]
        if bins is None:
            return pos, out
        return bin_aggregate(pos, out, np.mean, bins).pairs

    def _build_w_interpolators(self, jax=True, **kwargs):
        """
        DEPRECATED: done in C code
        Build interpolators at each position for each selection coefficient
        across all mutation weights.

        """
        raise ValueError()
        defaults = {'kind': 'quadratic',
                    'assume_sorted': True,
                    'bounds_error': True,
                    'copy': False}
        kwargs = {**defaults, **kwargs}
        interpols = defaultdict(list)
        x = self.pos
        Bs = self.B
        for chrom in Bs:
            npos = len(Bs[chrom])
            pos_level = [None] * npos
            chrom_interpols = pos_level
            for i, B in enumerate(Bs[chrom]):
                t_level = [None] * self.nt
                chrom_interpols[i] = t_level
                for j, t in enumerate(self.t):
                    annot_level = [None] * self.nf
                    chrom_interpols[i][j] = annot_level
                    for k in range(self.nf):
                        # annotation class level
                        if not jax:
                            func = interpolate.interp1d(self.w,
                                                        B[:, j, k],
                                                        **kwargs)
                        else:
                            b = B[:, j, k]
                            #def func(x, y):
                            #    return jnp.interp(x, self.w, b)
                            #func = partial(np.interp, xp=self.w, fp=b)
                            func = self.w, b
                        chrom_interpols[i][j][k] = func
            interpols[chrom] = Bw_interpol(chrom_interpols, npos, self.nt, self.nf, jax)

    def _build_interpolators(self, **kwargs):
        """
        Build positional interpolators for each chromsome and w/t combination.
        """
        defaults = {'kind': 'quadratic',
                    'assume_sorted': True,
                    'bounds_error': False,
                    'copy': False}
        kwargs = {**defaults, **kwargs}
        interpols = defaultdict(dict)
        x = self.pos
        Bs = self.B
        for i, w in enumerate(self.w):
            for j, t in enumerate(self.t):
                for chrom in Bs:
                    # The last dimension is features matrix -- for now, only
                    # one feature is supported
                    y = Bs[chrom][:, i, j].squeeze()
                    func = interpolate.interp1d(x[chrom], y,
                                                fill_value=(y[0], y[-1]),
                                                **kwargs)
                    interpols[chrom][(w, t)] = func
        self._interpolators = interpols

    def B_at_pos(self, chrom, pos):
        if self._interpolators is None:
            print(f"building interpolators...\t", end='\t')
            self._build_interpolators()
            print("done.")
        X = np.full((self.w.size, self.t.size, pos.size), np.nan)
        w = 0
        t = 0
        # __import__('pdb').set_trace()
        for i, w in enumerate(self.w):
            for j, t in enumerate(self.t):
                X[i, j, :] = self._interpolators[chrom][(w, t)](pos)
        return X

    def bin_means(self, bins, merge=False, return_bins=False):
        """
        Average the bins into larger genomic windows.

        If merge is True, this concatenates all the Bs together.

        We use the trapezoid rule, which works out to be the midpoint.

        a     b     c     d     e     f
        |     |     |     |     |     | Bs
        x1      x2        x3       x4    wins
        B0    B1    B2    B3    B4    B5

        This shows that handling non-overlapping ranges is a pain; we
        avoid this by requiring the width to be a multiple of the step size
        (which is usually a much smaller scale, e.g. a few kbp).

        a     b     c     d     e     f
        |     |     |     |     |     | Bs
        x1         x2           x3     wins
        B0    B1    B2    B3    B4    B5

        trapezoid rule: int f(x) dx = w x 0.5 x (B_i + B_{i+1})
        mean: 1/w int f(x) dx = 0.5 x (B_i + B_{i+1})

        window mean for x1:x2: 0.5 * (B0 + B1) + 0.5 * (B1 + B2)

        """
        #msg = "bins must be a chrom dict of bins or a GenomicBins"
        #assert isinstance(bins, (GenomicBins, dict)), msg
        means = dict()
        for chrom in bins.keys():
            y = self.B[chrom]
            x = self.pos[chrom]
            # midpoints work out to be linear interpolation mean, through
            # trapezoid rule. The midpoints are those for the end B position
            # (like bins).
            mp_y = midpoints(y) # B-value midpoints
            mp_x = midpoints(x) # position midpoints
            res = bin_aggregate(mp_x, mp_y, np.mean, bins[chrom], axis=0)
            means[chrom] = res

        if merge:
            return np.concatenate([x.stat for x in means.values()], axis=0)

        if return_bins:
            return means
        # otherwise, we turn into another BScores object for easier downstream
        # use
        B = {c: x.stat for c, x in means.items()}
        pos = {c: x.midpoints for c, x in means.items()}
        return BScores(B, pos, self.w, self.t)


def pretty_percent(x, ndigit=3):
    return np.round(100*x, ndigit)

def facet_wrap(nitems, ncols, **kwargs):
    """
    Mimic ggplot2's facet_wrap for nitems.
    """
    nrows = math.ceil(nitems / ncols)
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, **kwargs)
    return fig, [ax[i[0], i[1]] for i in
                 itertools.product(range(nrows), range(ncols))]

def readfq(fp, name_only=True): # this is a generator function
    """
    Thanks to Heng Li! https://github.com/lh3/readfq/blob/master/readfq.py
    """
    last = None # this is a buffer keeping the last unprocessed line
    while True: # mimic closure; is it a bad idea?
        if not last: # the first record or a record following a fastq
            for l in fp: # search for the start of the next record
                if l[0] in '>@': # fasta/q header line
                    last = l[:-1] # save this line
                    break
        if not last: break
        if name_only:
            name, seqs, last = last[1:].partition(" ")[0], [], None
        else:
            name, seqs, last = last[1:], [], None
        for l in fp: # read the sequence
            if l[0] in '@+>':
                last = l[:-1]
                break
            seqs.append(l[:-1])
        if not last or last[0] != '+': # this is a fasta record
            yield name, ''.join(seqs), None # yield a fasta record
            if not last: break
        else: # this is a fastq record
            seq, leng, seqs = ''.join(seqs), 0, []
            for l in fp: # read the quality
                seqs.append(l[:-1])
                leng += len(l) - 1
                if leng >= len(seq): # have read enough quality
                    last = None
                    yield name, seq, ''.join(seqs); # yield a fastq record
                    break
            if last: # reach EOF before reading enough quality
                yield name, seq, None # yield a fasta record instead
                break

def bin_chrom(end, width, dtype='uint32'):
    """
    Bin a chromsome of length 'end' into bins of 'width', with the last
    bin
    """
    # assumes 0-indexed
    cend = width * np.ceil(end / width) + 1
    bins = np.arange(0, cend, width, dtype=dtype)

    # if there's overrun, make the last bin position the chromosome length
    # (remember, not right inclusive)
    last_pos = end
    if bins[-1] > last_pos:
        bins[-1] = last_pos
    assert np.all(bins < end+1)
    return bins

def bin_chroms(seqlens, width, dtype='uint32'):
    bins = dict()
    for chrom, length in seqlens.items():
        bins[chrom] = bin_chrom(seqlens[chrom], width, dtype=dtype)
    return bins

def center_and_scale(x):
    """
    Center and scale.
    """
    return (x-np.nanmean(x))/np.nanstd(x)

def mean_ratio(x):
    """
    Return the ratio x / mean(x), using NaN mean.
    """
    return x/np.nanmean(x)

def aggregate_site_array(x, bins, func, **kwargs):
    """
    Given a site array (an np.ndarray of length equal to a chromosome)
    calculate some summary of values with func on the specified bins.

    """
    assert x.shape[0] == bins[-1], "bins must range 0, ..., L"
    # we skip the first bin since it's zero (no data to the left)
    vals = np.zeros((len(bins)-1, x.shape[1]))
    n = np.zeros(len(bins)-1)
    assert bins[0] == 0, "first bin should be 0!"
    # first binned skipped is skipped since it's zero
    for i in range(1, len(bins)):
        data_in_bin = x[bins[i-1]:bins[i], ...]
        # the data in between i-1 and i does into i, but
        # the results length is len(bins)-1
        vals[i-1, ...] = func(data_in_bin, **kwargs)
        #assert not np.any(np.isnan(data_in_bin)) # otherwise this changes n
        # NOTE: "missingness" here are allele counts that are
        # equal to zero. This is the total number of complete cases
        n[i-1] = np.sum(np.sum(data_in_bin, axis=1) > 0)
    return BinnedStat(vals, bins, n)

def read_phastcons(filename, seqlens, dtype='float32'):
    """
    Read a UCSC PhastCons formatted TSV.

    Score is scaled to range from 0-1000; we rescale it to 0-1 here.
    """
    phastcons = {c: np.full(seqlens[c], np.nan, dtype=dtype) for c in seqlens}
    with readfile(filename) as f:
        for line in f:
            chrom, start, end, lod, score = line.strip().split('\t')
            start, end = int(start), int(end)
            if chrom not in seqlens:
                continue
            phastcons[chrom][start:end] = float(score)/1000
    return phastcons


def load_cadd_bed_scores(filename, seqlens, progress=True, mode='max',
                         dtype='float32'):
    """
    Load a CADD BED file (see Snakefile, this is usually formed by
    running bedtools merge to combine the scores across variants).
    cols: chrom, start, end, mean, max

    Note on CADD scores/phred:

    Phred-scaled are usually what we want -- these have been normalized to
    *all* SNVs in the genome, and are good for comparison to other annotation
    tracks, etc. However, due to scaling issues and floating cutoffs,
    these scaling makes comparing CADD scores across various SNVs less
    precise (see p. 889 of Rentzsch et al 2019 for more discussion).
    """
    cadd = {c: np.full(seqlens[c], np.nan, dtype=dtype) for c in seqlens}
    last_chrom = None
    assert mode == 'max' or mode =='mean', "mode must be 'mean' or 'max'"
    with readfile(filename) as f:
        for line in f:
            if line.startswith('#'):
                continue
            chrom, pos, _, count, mean, max = line.strip().split('\t')
            mean, max = float(mean), float(max)
            if chrom != last_chrom:
                print(f"reading chromosome {chrom}...")
                last_chrom = chrom
            if chrom not in seqlens:
                break
            score = max if mode == 'max' else mean
            cadd[chrom][int(pos)] = score
    return cadd


def read_bigwig(filename, seqlens, dtype='float32'):
    """
    Load a bigwig into numpy arrays.
    """
    bw = pyBigWig.open(filename)
    bigwig = dict()
    for chrom in seqlens:
        bigwig[chrom] = np.array(bw.values(chrom, 0, seqlens[chrom]),
                                 dtype=dtype)
    return bigwig


def bin_aggregate(pos, values, func, bins, right=False, **kwargs):
    """
    Like scipy.stats.binned_statistic but allows higher dimensions.

    pos: positions
    values: a np.ndarray where the first dimension corresponds to positions
    func: the aggregating function
    bins: the bins to bin positions into
    right: whether to include right position
    **kwargs: kwargs to pass to func
    """
    assert isinstance(bins, np.ndarray), "bins must be an np.ndarray"
    assert isinstance(pos, np.ndarray), "bins must be an np.ndarray"
    assert pos.size == values.shape[0], "pos length and values.shape[0] must be the same"
    nbins = bins.size
    agg = np.full(fill_value=np.nan, shape=(nbins-1, *values.shape[1:]))
    n = np.full(fill_value=0, shape=nbins-1)
    idx = np.digitize(pos, bins, right=right)
    for i in np.unique(idx):
        agg[i-1, ...] = func(values[idx == i, ...], **kwargs)
        n[i-1] = np.sum(idx == i)
    return BinnedStat(agg, bins, n)

def read_npy_dir(dir):
    return {f: np.load(os.path.join(dir, f)) for f in os.listdir(dir)}

def subsampler_factory(frac, init_size=10_000_000, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    draws = rng.uniform(0, 1, init_size).tolist()
    def subsampler(iterable):
        nonlocal draws
        for item in iterable:
            try:
                draw = draws.pop()
            except IndexError:
                # rebuild samples
                draws = rng.uniform(0, 1, init_size).tolist()
                draw = draws.pop()
            if draw <= frac:
                yield item
    return subsampler

def Bhat(pi, N):
    """
    Branch statistics π is 4N (e.g. if μ --> 1)
    If there's a reduction factor B, such that
    E[π] = 4BN, a method of moments estimator of
    B is Bhat = π / 4N.
    Note: this should be called bbar but I didn't and now we're stuck with this
    """
    return 0.25 * pi / N

def random_seed(rng=None):
    if rng is None:
        return np.random.randint(0, SEED_MAX)
    return rng.integers(0, SEED_MAX)

def read_bkgd(file):
    """
    Read the B map output from McVicker's calc_bkgd.
    """
    with open(file) as f:
        d = np.loadtxt(f)
    pos = np.cumsum(d[:, 1])
    B = d[:, 0]/1000
    return np.array((pos, B)).T

def load_bkgd(dir):
    """
    McVicker's calc_bkgd splits chromosomes across files; here
    load each chromsome file in dir using read_bkgd() and put
    then into a dictionary.
    """
    files = os.listdir(dir)
    bkgd = dict()
    for file in files:
        chrom = file.replace('.bkgd', '')
        filepath = os.path.join(dir, file)
        try:
            bkgd[chrom] = read_bkgd(filepath)
        except ValueError:
            continue
            raise ValueError(f"parsing error at {filepath}")
    return bkgd

def load_bkgd_runs(dir):
    """
    For multiple calc_bkgd runs in a directory, load them and store results.
    """
    FILE_REGEX = re.compile(r'calc_bkgd_mu(?P<mu>[^_]+)_s(?P<s>.*)')
    dirs = [f for f in os.listdir(dir) if os.path.isdir(f)]
    results = defaultdict(dict)
    for run in dirs:
        match = FILE_REGEX.match(run)
        if match is None:
            warnings.warn(f"{run} did not match calc_bkgd run regex, skipping...")
            continue
        params = match.groupdict()
        filepath = os.path.join(dir, run)
        print(f"loading {run}...\t", end="")
        bs = load_bkgd(filepath)
        print(f"done.")
        param_tuple = (float(params['mu']), float(params['s']))
        results[param_tuple] = bs
    return results

def interpolate_calc_bkgd(results, width, seqlens, **kwargs):
    """
    """
    defaults = {'kind': 'quadratic',
                'assume_sorted': False,
                'bounds_error': False,
                'copy': False}
    kwargs = {**defaults, **kwargs}

    sels = sorted(set([s for _, s in results.keys()]))
    mus = sorted(set([mu for mu, _ in results.keys()]))
    # put things in a md array and BScores
    Bs, B_pos = dict(), dict()
    nsel = len(sels)
    nmu = len(mus)
    all_chroms = set()
    # get all chroms in the params
    for _, chroms in results.items():
        for chrom in chroms:
            all_chroms.add(chrom)
    for chrom in all_chroms:
        step_pos = bin_chrom(seqlens[chrom], width)
        for i, mu in enumerate(mus):
            for j, s in enumerate(sels):
                if chrom not in Bs:
                    # build empty matrix
                    nloci = len(step_pos)
                    Bs[chrom] = np.full((nloci, nmu, nsel), np.nan)
                    B_pos[chrom] = step_pos
                pos, bs = results[(mu, s)][chrom].T
                func = interpolate.interp1d(pos, bs, fill_value=(bs[0], bs[-1]), **kwargs)
                Bs[chrom][:, i, j] = np.log(func(step_pos))
    return BScores(Bs, B_pos, np.array(mus), np.array(sels), None)


def read_centro(file):
    """
    Read a centromere file from UCSC. All features other than 'acen' are
    discarded.
    """
    chroms = defaultdict(list)
    with open(file) as f:
        for line in f:
            chrom, start, end, arm, feature = line.strip().split('\t')
            if feature != 'acen':
                continue
            chroms[chrom].append(int(start))
            chroms[chrom].append(int(end))
    return {k: tuple(sorted(set(v))) for k, v in chroms.items()}

def get_files(dir, suffix):
    """

    Recursively get files from directories in dir with suffix, e.g. used for
    getting all .tree files across seed subdirectories.
    """
    all_files = set()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if suffix is not None:
                if not file.endswith(suffix):
                    continue
            all_files.add(os.path.join(root, *dirs, file))
    return list(all_files)

@np.vectorize
def signif(x, digits=4):
    if x == 0:
        return 0.
    if np.isnan(x):
        return np.nan
    return np.round(x, digits-int(floor(log10(abs(x))))-1)

def midpoint(x):
    return 0.5*(x[:-1, ...] + x[1:, ...])

def nearest(x, val):
    return np.argmin(np.abs(x-val))

def arg_nearest(val, array):
    """
    Get the index of the closest element in 'array' to 'val'.
    """
    if isinstance(array, list):
        array = np.array(array)
    i = np.argmin(np.abs(val-array))
    return i

def exact_index(val, array, tol=1e-30):
    """
    Get the index of element of 'array' to the point within 'tol' to
    'val'.
    """
    i = np.abs(val - array) < tol
    assert(i.sum() == 1) # only one result
    return i

def sum_logliks_over_chroms(ll_dict):
    """
    For a dictionary of chrom --> loglikelihoods multidimensional
    array, merge the chromosomes / windows and sum over all sites.
    """
    ll = np.stack(list(*ll_dict.values())).sum(axis=0)
    return ll

def get_unique_indices(x):
    "Get the indices of unique elements in a list/array"
    indices = defaultdict(list)
    for i, v in enumerate(x):
        indices[v].append(i)
    first_occur = [i[0] for i in indices.values()]
    return np.array(first_occur, dtype=int)

def make_dirs(*args):
    "Make directory if necessary"
    dir = os.path.join(*args)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def inverse_haldanes_mapfun(rec):
    return -0.5*np.log(1-2*rec)

def haldanes_mapfun(dist):
    return 0.5*(1 - np.exp(-dist))

def dist_to_segment(focal, seg_map_pos, haldane=False):
    """
    Return the map distance between a focal site (in map coords)
    and the segments end and start positions (also in map coords).
    This properly handles if the focal site is left, right or within
    the segment. If it's within, the distance is zero.

    focal: focal site in map coords
    seg_map_pos: an n x 2 array of start, end map coords of each segment

    First, figure out for each segment if this map position is left, right
    or within the segment. This matters for calculating the distance to
    the segment
    ---f-----L______R----------
    ---------L______R-------f--
    """
    assert seg_map_pos.shape[1] == 2, "seg_map_pos needs to be (n x 2)"
    f = focal
    is_left = (seg_map_pos[:, 0] - f) > 0
    is_right = (f - seg_map_pos[:, 1]) > 0
    is_contained = (~is_left) & (~is_right)
    dists = np.zeros(is_left.shape)
    dists[is_left] = np.abs(f-seg_map_pos[is_left, 0])
    dists[is_right] = np.abs(f-seg_map_pos[is_right, 1])
    assert len(dists) == seg_map_pos.shape[0], (len(dists), seg_map_pos.shape)
    if haldane:
        dists = haldanes_mapfun(dists)
    return dists

def rel_error(est, truth):
    return 100*np.abs((est - truth)/truth)

def readfile(filename):
    is_gzip = filename.endswith('.gz')
    if is_gzip:
        return gzip.open(filename, mode='rt')
    return open(filename, mode='r')

def read_bed3(file, keep_chroms=None):
    """
    Read a BED3 file.

    file: the (possibly gzipped) BED file)
    keep_chroms: only keep entries with these chromosomes
    """
    ranges = defaultdict(list)
    with readfile(file) as f:
        for line in f:
            cols = line.strip().split('\t')
            assert(len(cols) >= 3)
            chrom, start, end = cols[:3]
            if keep_chroms is not None and chrom not in keep_chroms:
                continue
            ranges[chrom].append((int(start), int(end)))
    return ranges

def ranges_to_masks(range_dict, seqlens):
    masks = {c: np.full(l, 0, dtype='bool') for c, l in seqlens.items()}
    for chrom, ranges in range_dict.items():
        if chrom not in seqlens:
            warnings.warn(f"sequence {chrom} not in dictionary seqlens, skipping...")
            continue
        for rng in ranges:
            masks[chrom][slice(*rng)] = 1
    return masks

def combine_features(feature_ranges, priority, seqlens):
    """
    Given a dictionary of different features e.g. {'introns': {'chr1: (12, 15)},
    ...} merge these into a mask-like array where the value is given by the
    priority.
    """
    features = {feat: ranges_to_masks(r, seqlens) for feat, r in feature_ranges.items()}
    assert len(set(features.keys()).difference(set(priority))) == 0, "some features keys are not in priority"
    masks = dict()
    for chrom in seqlens:
        merged = np.zeros(seqlens[chrom], dtype='int')
        for i, feature in enumerate(priority, start=1):
            if feature not in features:
                continue
            # only fill unfilled (zero) entries!
            idx = features[feature][chrom] & (merged == 0)
            if idx.sum():
                merged[idx] = i
        masks[chrom] = merged
    return masks


def rle(inarray):
        """
        Run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)

        Source: https://stackoverflow.com/a/32681075/147427 (thx SO)
        """
        ia = np.asarray(inarray)                # force numpy
        n = len(ia)
        if n == 0:
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def masks_to_ranges(mask_dict, return_val=False, labels=None):
    """
    Given the masks from combine_features(), turn these into a chromosome
    range dictionary.

    If label is not None, the value is treated as an digitized index
    (the labels are bins) and the label labels[i-1] are added to the ranges.
    """
    ranges = defaultdict(list)
    for chrom in mask_dict:
        rls, starts, vals = rle(mask_dict[chrom])
        for rl, s, val in zip(rls, starts, vals):
            if val == 0:
                continue
            if labels is not None:
                x = (s, s + rl, labels[val-1])
            elif return_val:
                x = (s, s + rl, val)
            else:
                x = (s, s + rl)
            ranges[chrom].append(x)
    return ranges


def genome_wide_quantiles(chromdict, alphas, subsample_chrom=None,
                          return_samples=False):
    """
    Take a chromdict of values and find the genome-wide quantiles.

    If subsample_chrom is not None, a fraction of each chromosome are sampled.
    """
    if not subsample_chrom:
        cuts = np.nanquantile(list(itertools.chain(*chromdict.values())), alphas)
        return cuts
    else:
        samples = []
        for chrom, values in chromdict.items():
            val = values[~np.isnan(values)]
            seqlen = val.size
            n = int(subsample_chrom * seqlen)
            samples.extend(np.random.choice(val[~np.isnan(val)], n, replace=False))
        cuts = np.nanquantile(samples, alphas)
    if return_samples:
        return cuts, samples
    return cuts

def quantize_track(chromdict, cuts, labels=None, bed_file=None, progress=True):
    """
    Take a conservation track, e.g. CADD, and bin it into quantiles,
    and output to BEd.
    """
    bins = dict()
    chroms = list(chromdict.keys())
    if progress:
        chroms = tqdm.tqdm(chroms)
    for chrom in chroms:
        digitized = np.digitize(chromdict[chrom], cuts)
        if bed_file is None:
            bins[chrom] = digitized
        else:
            ranges = masks_to_ranges({chrom: digitized}, labels=labels)
            write_bed(ranges, bed_file, append=True)
    if bed_file is None:
        return bins

def write_bed(chromdict, filename, compress=True, append=False):
    mode = 'wt'
    if append:
        mode = 'at'
    if compress:
        openfile = gzip.open(filename, mode)
    else:
        openfile = open(filename, mode)
    with openfile as f:
        for chrom, ranges in chromdict.items():
            for vals in ranges:
                f.write("\t".join(map(str, [chrom, *vals])) + "\n")

def load_bed_annotation(file, chroms=None):
    """
    Load a four column BED-(ish) file of chrom, start, end, feature name.
    If chroms is not None, this is the set of chroms to keep annotation for.
    """
    ranges = dict()
    params = []
    # nloci = 0
    all_features = set()
    # index_map = defaultdict(list)
    if chroms is not None:
        assert isinstance(chroms, (set, dict)), "chroms must be None, set, or, dict."
    ignored_chroms = set()
    with readfile(file) as f:
        for line in f:
            if line.startswith('#'):
                params.append(line.strip().lstrip('#'))
                continue
            cols = line.strip().split('\t')
            assert(len(cols) >= 3)
            chrom, start, end = cols[:3]
            if chroms is not None:
                if chrom not in chroms:
                    ignored_chroms.add(chrom)
                    continue
            if len(cols) == 3:
                feature = 'undefined'
            else:
                feature = cols[3]
            if chrom not in ranges:
                ranges[chrom] = ([], [])
            start, end = int(start), int(end)
            if end-start < 1:
                warnings.warn(f"skipping 0-width element {chrom}:{start}-{end})")
                continue
            ranges[chrom][0].append((start, end))
            ranges[chrom][1].append(feature)
            # index_map[chrom].append(nloci)
            all_features.add(feature)
            # nloci += 1

    if len(ignored_chroms):
        print(f"load_bed_annotation(): ignored {', '.join(ignored_chroms)}")
    Annotation = namedtuple('Annotation', ('ranges', 'features'))
    return Annotation(ranges, all_features)

def load_seqlens(file):
    seqlens = dict()
    params = []
    with readfile(file) as f:
        for line in f:
            if line.startswith('#'):
                params.append(line.strip().lstrip('#'))
                continue
            chrom, end = line.strip().split('\t')
            seqlens[chrom] = int(end)
    return seqlens

def group_sum(x, index, axis=0):
    """
    Sum the multidimensional array over groups.
    """
    indices = list(set(index))
    res = []
    for i in indices:
        idx = index == i
        res.append(np.sum(x[idx, ...], axis=axis))
    return np.stack(res)


def contains_overlaps(ranges, check_sorted=True):
    if check_sorted:
        ranges = sorted(ranges)
    ends = list(itertools.chain(*[(s, e) for s, e in ranges]))
    return ends != sorted(ends)


def classify_sites(sites, ranges, types, missing_class=None):
    """
    For a non-overlapping set of ranges, classify each site.

    """
    ranges = sorted(ranges)
    assert(not contains_overlaps(ranges, check_sorted=False))
    assert(len(ranges) == len(types))
    ends = list(itertools.chain(*[(s, e) for s, e in ranges]))
    assert(all(x >= 0 for x in ends))
    type_map = {e: t for (_, e), t in zip(ranges, types)}
    # if the first range doesn't start at zero, then we need to
    # add the start as an empty case
    first_start = ranges[0][0]
    add_zero = first_start > 0
    if add_zero:
        ends = [first_start] + ends
    ends = np.array(ends, dtype='uint64')
    idx = np.digitize(sites, ends)
    n = len(ends)
    # connect bin ends to the types
    # 'missing_class' occur for regions uncovered by ranges; the
    # if/else in this list comp handles the case where
    # np.digitize returns an index >= len(ends), which occurs
    # when the site is out of range on the right
    bins = [type_map.get(ends[i], missing_class)
            if i < n else missing_class for i in idx]
    return bins


def index_dictlist(x):
    """
    For a dictionary of chromosomes --> list of data, this
    creates an index for the indices as if the data was concatenated
    by chaining the list data.
    """
    index = defaultdict(list)
    i = 0
    for chrom, data in x.items():
        for _ in data:
            index[chrom].append(i)
            i += 1
        index[chrom] = np.array(index[chrom], dtype='uint32')
    return index


def parse_param_str(x):
    return dict([y.split('=')  for y in x.lstrip('#').rstrip().split(';')])


# deprecated; see load_seqlens
# def read_seqlens(file, keep_seqs=None):
#     seqlens = {}
#     with open(file, 'r') as f:
#         for line in f:
#             seq, length = line.strip().split('\t')
#             seqlens[seq] = int(length)
#     if keep_seqs is None:
#         return seqlens
#     return {c: seqlens[c] for c in keep_seqs}

def chain_dictlist(x):
    """
    Given a dict where values are lists, chain them into an iterator
    of (key, value) tuples.
    """
    for key, values in x.items():
        for value in values:
            yield (key, value)

def index_cols(cols):
    """
    For extracting columns (more safely than remembering indices)
    """
    index = {c: i for i, c in enumerate(cols)}
    def get(*args):
        return tuple(index[c] for c in args)
    return get

def read_phylofit(filename):
    """
    Read output from PhyloFit.
    """
    data = dict()
    with open(filename) as f:
        for line in f:
            if line.startswith("RATE_MAT"):
                line = next(f).strip()
                rates = list()
                while len(line) and not line.startswith("TREE"):
                    row = [float(x) for x in re.split(' +', line.strip())]
                    rates.append(row)
                    line = next(f).strip()
                assert(line.startswith("TREE"))
                data['tree'] = line.strip().split(': ', 1)[1]
            else:
                key, val = line.strip().split(': ', 1)
                key = key.lower().replace(': ', '')
                data[key]= val
    return data


def get_human_branch_length(tree_str, min_tips=10, lab='homo_sapiens'):
    """
    Sort of a specialty function: take output from PhyloFit and
    get the human branch length. min_tips sets the number of tips,
    otherwise NaN will be returned.
    """
    tree = newick.loads(tree_str)
    assert(isinstance(tree, list))
    assert(len(tree) == 1)
    if len(tree[0].get_leaf_names()) < min_tips:
        print( tree[0].get_leaf_names())
        return np.nan
    return [x.length for x in tree[0].walk() if x.name == lab][0]

def bin2midpoints(x):
    """
    Convenience function to take a BinnedStatistics object and return the
    binned midpoints
    """
    return 0.5*(x.bin_edges[1:] + x.bin_edges[:-1])


def bin2pairs(x):
    """
    Convenience function to take a BinnedStatistics object and return the
    binned midpoints and statistic together as a pair.
    """
    return bin2midpoints(x), x.statistic

def logbins(x, nbins, density=True, remove_nan=True):
    """
    Make log10 bins for a matplotlib histogram.
    Use like:
        plt.hist(*logbins(x, 100))
        plt.semilogx()

    """
    if remove_nan:
        x = x[~np.isnan(x)]
    hist, bins = np.histogram(x, bins=nbins, density=density)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    return x, logbins

def cutbins(x, nbins, method='interval', xrange=None, end_expansion=1.00001):
    """
    Bin data into nbins based on the method.
     - interval: create nbins equally-spaced bins
     - number: try to put approximately equal observations in nbins
     - quantile: estimate empirical quantiles and use those as bin edges.

    All have different advantages and disadvantages.
    """
    if method == 'interval':
        if xrange is None:
            xmin, xmax = np.nanmin(x), end_expansion*np.nanmax(x)
        else:
            xmin, xmax = xrange
        bins = np.linspace(xmin, xmax, nbins)
        return bins
    elif method == 'number':
        npt = len(x)
        bins = np.interp(np.linspace(0, npt, nbins + 1),
                         np.arange(npt),
                         np.sort(x))
        return bins[~np.isnan(bins)]
    elif method == 'quantile':
        return np.nanquantile(x, np.linspace(0, 1, nbins))
    else:
        raise ValueError("method must be 'interval', 'number', or 'quantile'")

def binned_summaries(x, y, nbins, method='interval',
                     funs={'mean': np.nanmean,
                           'sd': np.nanstd,
                           'n': lambda x: np.sum(np.isfinite(x))}):
    """
    """
    bins = cutbins(x, nbins, method)
    cols = defaultdict(list)
    cols['start'] = bins[1:]
    cols['end'] = bins[:-1]
    cols['midpoint'] = 0.5*(bins[1:] + bins[:-1])
    for name, fun in funs.items():
        binstats = binned_statistic(x, y, fun, bins)
        cols[name] = binstats.statistic
    return pd.DataFrame(cols)

def corr(x, y):
    """
    Compute Pearson and Spearman correlations, handles NaN.
    """
    keep = np.isfinite(x) & np.isfinite(y)
    return pearsonr(x[keep], y[keep]), spearmanr(x[keep], y[keep])

