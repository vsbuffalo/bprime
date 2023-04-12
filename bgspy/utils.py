import pickle
import re
import warnings
import gzip
import os
import gc
import sys
import tqdm
from collections import namedtuple, defaultdict, Counter
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
from bgspy.annotation import Annotation


GCs = set([ord(b) for b in 'GCgc'])

SEED_MAX = 2**32-1

# alias this
lowess = sm.nonparametric.lowess

# simple named tuple when we just want to store w/t grids
Grid = namedtuple('Grid', ('w', 't'))


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


def actualsize(input_obj):
    # from https://towardsdatascience.com/the-strange-size-of-python-objects-in-memory-ce87bdfbb97f
    memory_size = 0
    ids = set()
    objects = [input_obj]
    while objects:
        new = []
        for obj in objects:
            if id(obj) not in ids:
                ids.add(id(obj))
                memory_size += sys.getsizeof(obj)
                new.append(obj)
        objects = gc.get_referents(*new)
    return memory_size

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
    """
    Container class for B and B' Maps.
    TODO:
     - rename to BMap
     - move out of utils.py (lol)
    """
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

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def load_npz(filepath, chrom, bin_pos=True, is_log=False):
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
    import pyBigWig
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


def get_files(dir, suffix, as_dict=False):
    """

    Recursively get files from directories in dir with suffix, e.g. used for
    getting all .tree files across seed subdirectories.
    
    """
    all_files = dict()
    for root, dirs, files in os.walk(dir):
        for file in files:
            if suffix is not None:
                if not file.endswith(suffix):
                    continue
            key = os.path.join(root, *dirs, file)
            all_files[key] = os.path.basename(file)
    if as_dict:
        return all_files
    return list(all_files.keys())

@np.vectorize
def signif(x, digits=4):
    if x == 0:
        return 0.
    if np.isnan(x):
        return np.nan
    return np.round(x, digits-int(floor(log10(abs(x))))-1)

def pretty_signif(x, digits=2):
    v = signif(x, digits)
    if int(v) == v:
        return int(v)
    return v

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

def abs_error(est, truth):
    aer = np.abs(truth-est)
    return aer


def rel_error(est, truth, as_percent=True):
    rler = np.abs((truth-est)/truth)
    if as_percent:
        return 100*rler
    return rler

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
            if line.startswith('#'):
                continue
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

    priority: defines the order values are filled in. Values will not be overwritten.
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

def masks_to_ranges(mask_dict, return_val=False, ignore_zeros=False, labels=None):
    """
    Given the masks from combine_features(), turn these into a chromosome
    range dictionary.

    If label is not None, the value is treated as an digitized index
    (the labels are bins) and the label labels[i-1] are added to the ranges.

    include_other: whether to include the uncovered bases

    NOTE: on indexing 

     | 0 | 1 | 2 | 3 | 
      -----------

      l = 3, start = 0, s+l = 3
      This is the non-inclusive right bound, as it should be
    """
    ranges = defaultdict(list)
    if labels is not None:
        assert 'other' not in labels, "'other' is reserved"
    for chrom in mask_dict:
        rls, starts, vals = rle(mask_dict[chrom])
        for rl, s, val in zip(rls, starts, vals):
            if val == 0 and ignore_zeros:
                continue
            if labels is not None:
                lab = labels[val-1] if val > 0 else None
                x = (s, s + rl, lab)
            elif return_val:
                x = (s, s + rl, val)
            else:
                x = (s, s + rl)
            ranges[chrom].append(x)

    for chrom, rngs in ranges.items():
        ranges[chrom] = sorted(rngs)
    return ranges


def inverse_phred(Q):
    """
    Given a Phred score Q, return the probability.
    """
    return 10**(-Q/10)

def phred(P):
    return -10 * np.log10(P)


def quantize_track(chromdict, thresh, label=None, bed_file=None, 
                   flip_inequality=False, progress=True):
    """
    Take a chromdict of percentiles and collect or write to BED the top
    thresh hits.
    """
    bins = dict()
    chroms = list(chromdict.keys())
    if progress:
        chroms = tqdm.tqdm(chroms)
    for chrom in chroms:
        if flip_inequality:
            passed = chromdict[chrom] <= thresh
        else:
            passed = chromdict[chrom] >= thresh
        if bed_file is None:
            bins[chrom] = passed
        else:
            # ignore_zeros=True drops all uncovered basepairs (e.g. those not passed the threshold)
            ranges = masks_to_ranges({chrom: passed}, labels=[label], ignore_zeros=True)
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

def load_seqlens(file, exclude=None):
    seqlens = dict()
    params = []
    with readfile(file) as f:
        for line in f:
            if line.startswith('#'):
                params.append(line.strip().lstrip('#'))
                continue
            chrom, end = line.strip().split('\t')
            if exclude is not None:
                if chrom in exclude:
                    continue
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
                # read the rate matrix and tree
                line = next(f).strip()
                rates = list()
                while len(line) and not line.startswith("TREE"):
                    row = [float(x) for x in re.split(' +', line.strip())]
                    rates.append(row)
                    line = next(f).strip()
                assert(line.startswith("TREE"))
                data['rate'] = np.array(rates)
                tree = line.strip().split(': ', 1)[1]
                data['tree'] = tree
                data['branch_lengths'] = get_branch_length(tree)
            else:
                key, val = line.strip().split(': ', 1)
                key = key.lower().replace(': ', '')
                data[key]= val
    return data


def get_branch_length(tree_str):
    """
    Get the branch length of a species from a newick-string.
    """
    tree = newick.loads(tree_str)
    assert(isinstance(tree, list))
    assert(len(tree) == 1)
    return {x.name: x.length for x in tree[0].walk()}

def bin2midpoints(x):
    """
    Convenience function to take a BinnedStatistics object and return the
    binned midpoints
    """
    return 0.5*(x.bin_edges[1:] + x.bin_edges[:-1])


def bin2pairs(x, use_mean_ratio=False):
    """
    Convenience function to take a BinnedStatistics object and return the
    binned midpoints and statistic together as a pair.
    """
    y = x.statistic
    if use_mean_ratio:
        y = mean_ratio(y)
    return bin2midpoints(x), y

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
        bins =  np.nanquantile(x, np.linspace(0, 1, nbins))
        if xrange is not None:
            # end should include max
            bins[-1] = xrange[1]
            bins[0] = xrange[0]
        return bins
    else:
        raise ValueError("method must be 'interval', 'number', or 'quantile'")

def binned_summaries(x, y, nbins, method='interval',
                     funs={'mean': np.nanmean,
                           'median': np.nanmedian,
                           'sd': np.nanstd,
                           'n': lambda x: np.sum(np.isfinite(x))},
                     remove_nan=True,
                     cut_tails=None):
    """

    cut_tails: if None, no modifications. If this is a tuple of probabilities,
               this will use censor() to remove x and y values that fall outside
               the tails. If a float, tails are (float, 1-float).


    Notes: I find cut_tails is often very important for interval-based binning,
    as the bin-width is based on the total range, and thus very sensitive
    to outliers. I think cut_tails should be used in most of these cases.
    """
    x = np.array(x)
    y = np.array(y)
    if remove_nan:
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
    if cut_tails is not None:
        if isinstance(cut_tails, float):
            cut_tails = (cut_tails, 1-cut_tails)
        idx = censor(x, cut_tails, return_idx=True)
        x, y = x[idx], y[idx]
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
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    keep = np.isfinite(x) & np.isfinite(y)
    return pearsonr(x[keep], y[keep]), spearmanr(x[keep], y[keep])

def censor(x, probs, return_idx=False):
    """
    Trim of tails
    """
    assert len(probs) == 2
    l, u = np.nanquantile(x, probs)
    idx = (x >= l) & (x <= u)
    if return_idx:
        return idx
    return x[idx]


def manual_interp(w, r, muw):
    """
    Manual interpolation of the ratchet array. For
    comaprison with manual interpolation with
    grid_interp_weights.
    """
    nw, nl = r.shape
    res = np.empty(nl)
    for i in range(nl):
        res[i] = interp1d(w, r[:, i])(muw)
    return res

def midpoint_linear_interp(Y, x, x0, replace_bounds=True):
    """
    """
    try:
        assert x[0] <= x0 < x[-1]
    except AssertionError:
        if replace_bounds:
            if x0 >= x:
                # out of upper bound, return last Y
                return Y[-1, :]
            else:
                # out of lower bound, return first Y
                assert x0 < x[0]
                return Y[0, :]
        else:
            raise ValueError("out of bounds x0")

    # not out of bounds, do normal linear interpolation
    j = np.searchsorted(x, x0)
    assert j < len(x)
    l, u = x[j-1], x[j]
    assert l < u
    weight = ((x0-l)/(u - l))
    assert 0 <= weight <= 1
    w = 1-weight # so weighting is w*lower + (1-w)*upper

    assert np.allclose(w*x[j-1] + (1-w)*x[j], x0)

    with np.errstate(under='ignore'):
        y_interp = (w*Y[j-1, :] + (1-w)*Y[j, :])
    return y_interp

def parse_region(x, with_strand=False):
    if with_strand:
        res = re.match(r'(chr[^:]+):(\d+)-(\d+)([+-])', x)
    else:
        res = re.match(r'(chr[^:]+):(\d+)-(\d+)', x)
    if res is None:
        return res
    if with_strand:
        chrom, start, end, strand = res.groups()
    else:
        chrom, start, end = res.groups()
        strand = None
    return chrom, int(start), int(end), strand


def GC(seq):
    seqlen = np.sum(~np.isin(seq, [ord('N'), ord('n')]))
    if isinstance(seq, np.ndarray):
        return sum([x in GCs for x in seq]) / seqlen
    else:
        return sum([x.upper() in 'gcGC' for x in seq]) / seqlen



GFF_COLS = 'seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute'

def read_gff(filename, parse_attributes=False):
    tx = dict()
    with readfile(filename) as f:
        for line in f:
            cols = dict(zip(GFF_COLS, line.strip().split('\t')))
            feature = cols['feature']
            if parse_attributes:
                attrs = dict([tuple(x.split('=')) for x in cols['attribute'].split(';')])
                cols['attribute'] = attrs
            yield cols

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        return pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def chromosome_key(chrom):
    chrom_number = chrom.replace('chr', '')
    if chrom_number.isdigit():
        return int(chrom_number)
    else:
        return {'X': 1e6, 'Y': 1e6+1}[chrom_number]

def argsort_chroms(chromosomes):
    indices = list(range(len(chromosomes)))
    indices.sort(key=lambda i: chromosome_key(chromosomes[i]))
    return indices

def summarize_npz_by_bin(npz_file, bin_width, fun=np.nanmean):
    data = np.load(npz_file)

    binned_averages = {}
    for chrom in data.keys():
        chrom_data = data[chrom]
        pad_size = bin_width - (len(chrom_data) % bin_width)
        padded_chrom_data = np.pad(chrom_data, (0, pad_size), constant_values=np.nan)
        reshaped_chrom_data = padded_chrom_data.reshape(-1, bin_width)
        binned_averages[chrom] = fun(reshaped_chrom_data, axis=1)

        num_bins = len(reshaped_chrom_data)
        bin_start_positions = np.arange(0, num_bins * bin_width, bin_width)
        bin_end_positions = bin_start_positions + bin_width
        bin_ranges[chrom] = np.column_stack((bin_start_positions, bin_end_positions))

    return binned_averages, bin_ranges
