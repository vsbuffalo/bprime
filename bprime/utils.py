import sys
import warnings
import gzip
from collections import namedtuple, defaultdict
from functools import partial
import itertools
from math import floor, log10
from scipy import interpolate
import numpy as np


RecPair = namedtuple('RecPair', ('end', 'rate'))
Segments = namedtuple('Segments', ('ranges', 'rates', 'map_pos', 'features',
                                   'feature_map', 'index'))

def read_bkgd(file):
    with open(file) as f:
        d = np.loadtxt(f)
    pos = np.cumsum(d[:, 1])
    B = d[:, 0]/1000
    return pos, B

def nearest(x, val):
    return np.argmin(np.abs(x-val))

def sum_logliks_over_chroms(ll_dict):
    """
    For a dictionary of chrom --> loglikelihoods multidimensional
    array, merge the chromosomes / windows and sum over all sites.
    """
    ll = np.stack(list(*ll_dict.values())).sum(axis=0)
    return ll

def bin_chrom(end, width, dtype='uint32'):
    # assumes 0-indexed
    cend = width * np.ceil(end / width) + 1
    return np.arange(0, cend, width, dtype=dtype)

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

def load_bed_annotation(file):
    """
    """
    ranges = dict()
    params = []
    nloci = 0
    all_features = set()
    index_map = defaultdict(list)
    with readfile(file) as f:
        for line in f:
            if line.startswith('#'):
                params.append(line.strip().lstrip('#'))
                continue
            cols = line.strip().split('\t')
            assert(len(cols) >= 3)
            chrom, start, end = cols[:3]
            if len(cols) == 3:
                feature = 'undefined'
            else:
                feature = cols[3]
            if chrom not in ranges:
                ranges[chrom] = ([], [])
            ranges[chrom][0].append((int(start), int(end)))
            ranges[chrom][1].append(feature)
            index_map[chrom].append(nloci)
            all_features.add(feature)
            nloci += 1
    Annotation = namedtuple('Annotation', ('ranges', 'index_map', 'features'))
    return Annotation(ranges, index_map, all_features)

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


def process_feature_recombs(features, recmap):
    """
    Split the features dictionary (values are a list of range tuples and feature
    labels) by the recombination rate ends. This effectively breaks segments
    into two if the recombinatio rate changes. The recombination rates are also
    added to each segment.
    """
    try:
        assert(all(chrom in recmap.rates.keys() for chrom in features.keys()))
    except:
        raise ValueError(f"features contains sequences not in recmap.")
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

def create_annotation_grid(variants, variant_indices, annotation, dummy=False):
    # DEPRECATED?
    ranges, index_map, all_features = annotation
    # put into an array
    nfeatures = len(all_features)
    nloci = len(list(itertools.chain(*variants.values())))
    feature_map = {f: i for i, f in enumerate(sorted(all_features), 1)}
    F = np.zeros((nloci, 1+nfeatures), dtype='bool')
    for chrom, (chrom_ranges, chrom_features) in ranges.items():
        chrom_variants = variants[chrom]
        indices = [feature_map[f] for f in chrom_features]
        var_classes = classify_sites(chrom_variants, chrom_ranges,
                                        indices, missing_class=0)
        i = variant_indices[chrom]
        if dummy:
            F[i, var_classes] = np.array(True)
        else:
            F = var_classes
    return F, feature_map

class RecMap(object):
    def __init__(self, mapfile, seqlens, conversion_factor=1e-8):
        self.mapfile = mapfile
        self.conversion_factor = conversion_factor
        self.ends = dict()
        self.rates = None
        self.seqlens = seqlens
        self.cumm_rates = None
        self.params = []
        self.readmap()

    def readmap(self):
        rates = defaultdict(list)
        last_chrom, last_end = None, None
        first_bin = True
        first = True
        is_hapmap = False
        with readfile(self.mapfile) as f:
            for line in f:
                if line.startswith('Chromosome'):
                    print(f"ignoring HapMap header...")
                    is_hapmap = True
                    continue
                if line.startswith('#'):
                    self.params.append(line.strip().lstrip('#'))
                cols = line.strip().split("\t")
                is_hapmap = is_hapmap or len(cols) == 3
                if is_hapmap:
                    if first:
                        print("parsing recmap as HapMap formatted (chrom, end, rate)")
                        first = False
                    chrom, end, rate = line.strip().split("\t")[:3]
                    start = -1
                else:
                    # BED file version
                    if first:
                        print("parsing recmap as BED formatted (chrom, start, end, rate)")
                        first = False
                    chrom, start, end, rate = line.strip().split("\t")[:4]
                if last_chrom is not None and chrom != last_chrom:
                    # propagate the ends list
                    self.ends[last_chrom] = last_end
                    first_bin = True
                if first_bin and start != 0:
                    # missing data up until this point, fill in with an nan
                    start = start if not is_hapmap else 0
                    rates[chrom].append((int(start), np.nan))
                    first_bin = False
                rates[chrom].append((int(end), float(rate)))
                last_chrom = chrom
                last_end = int(end)

        # end of loop, put the last position in ends
        self.ends[last_chrom] = last_end

        cumm_rates = dict()
        for chrom, data in rates.items():
            pos = np.array([p for p, _ in data])
            rate = np.array([r for _, r in data])
            rbp = rate * self.conversion_factor
            rates[chrom] = RecPair(pos, rbp)
            widths = np.diff(pos)
            cumrates = np.nancumsum(rbp[1:]*widths)
            pad_cumrates = np.zeros(cumrates.shape[0]+1)
            pad_cumrates[1:] = cumrates
            cumm_rates[chrom] = RecPair(pos, pad_cumrates)
        self.rates = rates
        self.cumm_rates = cumm_rates
        interpol = {c: interpolate.interp1d(ends, rates, copy=False, assume_sorted=True, bounds_error=False)
                    for c, (ends, rates) in self.cumm_rates.items()}
        self.interpol = interpol

    def lookup(self, chrom, pos, cummulative=False):
        #assert(np.all(0 <= pos <= self.ends[chrom]))
        if np.any(pos > self.seqlens[chrom]):
            bad_pos = pos[pos > self.seqlens[chrom]]
            msg = f"some positions {bad_pos} are greater than sequence length ({self.seqlens[chrom]}"
            warnings.warn(msg)
        if not cummulative:
            lookup_table = self.rates
            if chrom not in lookup_table:
                raise KeyError(f"'{chrom}' not in recombination map.")
            idx = np.digitize(pos, lookup_table[chrom].end)
            if isinstance(idx, np.int64):
                return lookup_table[chrom].rate[idx]
            # TODO, bound checking?
            x = np.array([lookup_table[chrom].rate[i] for i in idx])
        else:
            x = self.interpol[chrom](pos)
            # if there's overrun past the last marker, we just
            # approximate this, saying the position is the last marker
            overrun = pos > self.ends[chrom]
            if any(overrun):
                x[overrun] = self.cumm_rates[chrom].rate[-1]
        assert(not any(np.isnan(x)))
        return x

    @property
    def map_lengths(self):
        return {chrom: x.rate[-1] for chrom, x in self.cumm_rates.items()}

    def build_recmaps(self, positions, cummulative=False):
        map_positions = defaultdict(list)
        for chrom in positions:
            for pos in positions[chrom]:
                map_positions[chrom].append(self.lookup(chrom, pos,
                                                        cummulative=cummulative))
            map_positions[chrom] = np.array(map_positions[chrom])
        return map_positions

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


def read_seqlens(file):
    seqlens = {}
    with open(file, 'r') as f:
        for line in f:
            seq, length = line.strip().split('\t')
            seqlens[seq] = int(length)
    return {c: seqlens[c] for c in keep_seqs}

def chain_dictlist(x):
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


