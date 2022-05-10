from dataclasses import dataclass
import pickle
import sys
import re
import warnings
import gzip
import os
from collections import namedtuple, defaultdict, deque
from functools import partial
import itertools
from math import floor, log10
from scipy import interpolate
import numpy as np

# this dtype allows for simple metadata storage
Bdtype = np.dtype('float32', metadata={'dims': ('site', 'w', 't', 'f')})
#BScores = namedtuple('BScores', ('B', 'pos', 'w', 't', 'step'))
BinnedStat = namedtuple('BinnedStat', ('statistic', 'wins', 'nitems'))

RecPair = namedtuple('RecPair', ('end', 'rate'))
Segments = namedtuple('Segments', ('ranges', 'rates', 'map_pos', 'features',
                                   'feature_map', 'index'))

@dataclass
class BScores:
    B: np.ndarray
    pos: np.ndarray
    w: np.ndarray
    t: np.ndarray
    step: (None, int)

    def __getitem__(self, tup):
        chrom, w, t = tup
        pos = self.pos[chrom]
        Bs = 10**self.B[chrom][:, w == self.w, t == self.t, ...].squeeze()
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
                Bs[chrom][:, i, j] = func(step_pos)
    return BScores(Bs, B_pos, mus, sels, None)


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
    return all_files


def signif(x, digits=4):
    if x == 0:
        return 0.
    return np.round(x, digits-int(floor(log10(abs(x))))-1)

def midpoint(x):
    return 0.5*(x[:-1, ...] + x[1:, ...])

def nearest(x, val):
    return np.argmin(np.abs(x-val))

def arg_nearest(val, array):
    """
    Get the index of the closest element in 'array' to 'val'.
    """
    i = np.argmin(np.abs(val-array))
    return i

def exact_indice(val, array, tol=1e-30):
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

def bin_chrom(end, width, dtype='uint32'):
    """
    Bin a chromsome of length 'end' into bins of 'width', with the last
    bin
    """
    # assumes 0-indexed
    cend = width * np.ceil(end / width) + 1
    bins = np.arange(0, cend, width, dtype=dtype)

    # if there's overrun, make one short bin at the end
    last_pos = end - 1
    if bins[-1] > last_pos:
        bins[-1] = last_pos
    assert np.all(bins < end)
    return bins


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


def dist_to_segment(focal, seg_map_pos):
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
    return dists

def haldanes_mapfun(dist):
    return 0.5*(1 - np.exp(-dist))

def rel_error(est, truth):
    return 100*np.abs((est - truth)/truth)

def readfile(filename):
    is_gzip = filename.endswith('.gz')
    if is_gzip:
        return gzip.open(filename, mode='rt')
    return open(filename, mode='r')

def read_bed(file, keep_chroms=None):
    """
    Read a BED3 or BED5 file (strand column ignored).

    file: the (possibly gzipped) BED file)
    keep_chroms: only keep entries with these chromosomes
    """
    ranges = defaultdict(list)
    with readfile(file) as f:
        for line in f:
            cols = line.strip().split('\t')
            assert(len(cols) >= 3)
            chrom, start, end = cols[:3]
            bed4 = len(cols) == 4
            if bed4:
                name = cols[3]
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

def load_bed_annotation(file, chroms=None):
    """
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
            ranges[chrom][0].append((int(start), int(end)))
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


def process_feature_recombs(features, recmap, split_length=None):
    """
    Split the features dictionary (values are a list of range tuples and feature
    labels) by the recombination rate ends. This effectively breaks segments
    into two if the recombination rate changes. If split_length is not None, the
    segments are further split to be a maximum length of split_length. The
    recombination rates are also added to each segment.
    """
    try:
        assert(all(chrom in recmap.rates.keys() for chrom in features.keys()))
    except:
        raise ValueError(f"features contains sequences not in recmap.")

    if split_length is not None:
        split_features = dict()
        for chrom, (ranges, feature_types) in features.items():
            split_ranges, split_feature_types = [], []
            ranges_deque = deque(ranges)
            features_deque = deque(feature_types)
            assert len(ranges) == len(feature_types)
            while True:
                try:
                    row = ranges_deque.popleft()
                    feature = features_deque.popleft()
                except IndexError:
                    break
                start, end = row
                if end - start > split_length:
                    new_row = start, start+split_length
                    split_ranges.append(new_row)
                    split_feature_types.append(feature)
                    # the leftover bit
                    split_row = start+split_length, end
                    ranges_deque.appendleft(split_row)
                    features_deque.appendleft(feature)
                    continue
                split_ranges.append((start, end))
                split_feature_types.append(feature)
            assert len(split_ranges) == len(split_feature_types)
            split_features[chrom] = (split_ranges, split_feature_types)
        features = split_features

    recrates = recmap.rates
    all_features = set()
    index = defaultdict(list)
    chroms = list()
    split_ranges = list()
    split_features = list()
    split_rates = list()

    i = 0
    for chrom, feature_ranges in features.items():
        # feature_ranges is a (range, feature type) tuple
        if chrom not in recrates:
            print(f"{chrom} not in recombination map, skipping.")
            continue
        rec_ends = iter(zip(recrates[chrom].end, recrates[chrom].rate))
        rec_end, rec_rate = next(rec_ends)
        for (start, end), feature_type in zip(*feature_ranges):
            all_features.add(feature_type)
            while rec_end <= start:
                # the <= prevents 0-width'd features (TODO check)
                try:
                     rec_end, rec_rate = next(rec_ends)
                except StopIteration:
                     break
                #print(f"bumping up rec ends ({start}, {end}; {rec_end})")
            if rec_end >= end:
                # this range is not to be split
                split_ranges.append((start, end))
                split_features.append(feature_type)
                split_rates.append(rec_rate)
                index[chrom].append(i)
                chroms.append(chrom)
                i += 1
                continue

            overlaps = start <= rec_end < end
            # this feature overlaps a switch in recombinatino rate
            while overlaps:
                new_ranges = [(start, rec_end)]
                split_ranges.append((start, rec_end))
                split_features.append(feature_type)
                split_rates.append(rec_rate)
                index[chrom].append(i)
                chroms.append(chrom)
                i += 1
                start = rec_end
                try:
                    rec_end, rec_rate = next(rec_ends)
                except StopIteration:
                    break
                overlaps = start <= rec_end < end

            split_ranges.append((start, end))
            split_features.append(feature_type)
            split_rates.append(rec_rate)
            index[chrom].append(i)
            chroms.append(chrom)
            i += 1

        print(f"completed segmenting {chrom}.")

    feature_map = {f: i for i, f in enumerate(sorted(all_features))}
    rm = recmap
    ranges = np.array(split_ranges, dtype='uint32')
    assert(i == len(split_ranges))
    print(f"looking up map positions...\t", end='')
    map_pos = []
    for chrom in index:
        idx = index[chrom]
        map_start = rm.lookup(chrom, ranges[idx, 0], cummulative=True)
        map_end = rm.lookup(chrom, ranges[idx, 1], cummulative=True)
        assert(len(map_start) == len(idx))
        map_pos.append(np.stack((map_start, map_end)).T)
    map_pos = np.concatenate(map_pos, axis=0)
    assert(map_pos.shape[0] == ranges.shape[0])
    print(f"done.")
    rates = np.array(split_rates, dtype='float32')
    features = np.array([feature_map[x] for x in split_features])
    return Segments(ranges, rates, map_pos, features, feature_map, index)

def parse_param_str(x):
    assert(len(x) == 1)
    x = x[0]
    return dict([y.split('=')  for y in x.lstrip('#').rstrip().split(';')])

def load_dacfile(dacfile, neut_masks=None):
    """
    Read derived allele counts datafile. These should be sorted
    (TODO: check) and only in regions that are neutral (e.g. as
    specified by neut_regions, TODO: check).

    Returns:
        - positions: a dict(list) of positions per chromosome.
        - indices: a dict(list) indicating the indices of the DAC list
                   per chromosome.
        - nchroms: array of number of chromosomes per site.
        - dacs: array of the derived allele counts per site.
        - position_map: a dict(list) with keys corresponding to chromosomes
                        and values are a dictionary mapping of positions to
                        indices in the dacs/nchroms arrays.
    """
    params = []
    indices = defaultdict(list)
    positions = defaultdict(list)
    position_map = dict()
    nchroms = []
    dacs = []
    i = 0
    last_chrom = None
    neut_sites = None
    skipped_sites = 0
    with readfile(dacfile) as f:
        for line in f:
            if line.startswith('#'):
                params.append(line.strip().lstrip('#'))
                continue
            chrom, pos, nchrom, dac = line.strip().split('\t')[:4]
            if last_chrom != chrom:
                if neut_masks is not None:
                    neut_sites = set(np.where(neut_masks[chrom])[0])
                last_chrom = chrom
            pos = int(pos)
            if neut_sites is not None and pos not in neut_sites:
                skipped_sites += 1
                continue
            indices[chrom].append(i)
            positions[chrom].append(pos)
            nchroms.append(int(nchrom))
            dacs.append(int(dac))
            i += 1
    if neut_masks is not None:
        print(f"{skipped_sites} sites in non-neutral regions skipped.")
    for chrom in positions:
        try:
            assert(sorted(positions[chrom]) == positions[chrom])
        except AssertionError:
            raise AssertionError(f"positions in {dacfile} are not sorted.")
        position_map[chrom] = {p: i for i, p in enumerate(positions[chrom])}

    # use a minimal dtype for data of this size
    max_val = max(nchroms)**2
    uint_dtype = np.min_scalar_type(max_val)
    dac = np.array(dacs, dtype=uint_dtype)
    nchrom = np.array(nchroms, dtype=uint_dtype)
    # ancestral and derived allele counts
    ac = np.stack((nchrom - dac, dac)).T
    return positions, indices, ac, position_map, parse_param_str(params)

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


