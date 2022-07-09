import pickle
from dataclasses import dataclass
import numpy as np
from scipy import interpolate
from collections import defaultdict, namedtuple, deque
from bgspy.utils import load_seqlens, load_bed_annotation
from bgspy.recmap import RecMap
from bgspy.classic import B_segment_lazy, BSC16_segment_lazy

@dataclass
class Segments:
    """
    A dataclass for the conserved segment details.
    Contains:
        - ranges
        - rec rates within each segment
        - map positions of start and end
        - features (indexed)
        - map of feature names to indices
        - index, a dict of chrom->indices of the ranges array, which contains
          info for all chroms
    """
    def __init__(self, ranges, rates, map_pos, features, feature_map, index):
        self.ranges = ranges
        self.rates = rates
        self.map_pos = map_pos
        self.features = features
        self.feature_map = feature_map
        self.index = index
        self._segment_parts = None
        self._segment_parts_sc16 = None
        self.F = None
        self._calc_features()

    def __repr__(self):
        nfeats = len(self.feature_map)
        return f"Segments ({len(self):,} total with {nfeats} feature type(s))"

    def __len__(self):
        return self.ranges.shape[0]

    @property
    def lengths(self):
        return np.diff(self.ranges, axis=1).squeeze()

    @property
    def chroms(self):
        return list(self.index.keys())

    @property
    def L(self):
        "Return a dict of segment lengths (L) per chrom"
        return {c: self.lengths[idx] for c, idx in self.index.items()}

    @property
    def rbp(self):
        "Return a dict of segment rates (rbp) per chrom"
        return {c: self.rates[idx] for c, idx in self.index.items()}

    @property
    def mpos(self):
        "Return a dict of segment start/end map positions per chrom"
        return {c: self.map_pos[idx] for c, idx in self.index.items()}

    @property
    def nsegs(self):
        "Return a dict of number of segments per chrom"
        return {c: len(self.index[c]) for c in self.chroms}

    def _calc_segparts(self, w, t, N=None):
        """
        Calculate the fixed components of the classic BGS theory
        equation for each segment, to avoid unnecessary repeated calcs.
        """
        L = self.lengths
        rbp = self.rates
        if t.ndim == 1:
            t = t[:, None]
        self._segment_parts = B_segment_lazy(rbp, L, t)
        if N is not None:
            print(f"calculating SC16 components...\t", end='')
            self._segment_parts_sc16 = BSC16_segment_lazy(w, t, self, N)
            print("done.")

    def _calc_features(self):
        """
        Create a dummy matrix of features for each segment.
        """
        nfeats = len(self.feature_map)
        nsegs = len(self.features)
        F = np.zeros(shape=(nsegs, nfeats), dtype='bool')
        # build a one-hot matrix of features
        np.put_along_axis(F, self.features[:, None], 1, axis=1)
        self.F = F

def process_annotation(features, recmap, split_length=None):
    """
    Split the annotation dictionary (values are a list of range tuples and feature
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


class Genome(object):
    """
    A description of the Genome for B calculations and inference.
    """
    def __init__(self, name, seqlens_file, chroms=None):
        self.name = name
        self.seqlens = None
        self.load_seqlens(seqlens_file, chroms)
        self._seqlens_file = seqlens_file
        self._recmap_file = None
        self._annot_file = None
        self._neutral_file = None
        self.neutral = None
        self.annot = None
        self.recmap = None
        self.all_features = None
        self.segments = None
        self.split_length = None
        self._idx2map = None
        self._map2idx = None

    def load_seqlens(self, file, chroms):
        if isinstance(chroms, str):
            chroms = set([chroms])
        self._seqlens_file = file
        self._loaded_chroms = chroms
        seqlens = load_seqlens(file)
        if chroms is not None:
            seqlens = {c: l for c, l in seqlens.items() if c in chroms}
        self.seqlens = seqlens

    def load_recmap(self, file, **kwargs):
        self._recmap_file = file
        self.recmap = RecMap(file, self.seqlens, **kwargs)

    def create_segments(self, split_length=None):
        """
        Create the conserved segments.
        """
        msg = "annotation needs to be loaded with Genome.load_annot()"
        assert self.annot is not None, msg
        msg = "recombination map needs to me loaded with Genome.load_recmap()"
        assert self.recmap is not None, msg
        self.split_length = split_length
        self.segments = process_annotation(self.annot, self.recmap, split_length)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def __repr__(self):
        rows = [f"Genome '{self.name}' with {len(self.seqlens)} chromosome(s)"]
        for attr in ('recmap', 'annot', 'neutral'):
            attrname = getattr(self, f"_{attr}_file")
            rows.append(f" {attr}: {attrname}")

        seg = f"{len(self.segments):,}" if self.segments is not None else None
        rows.append(f" segments: {seg}")
        return "\n".join(rows)

    def load_annot(self, file):
        self._annot_file = file
        self.annot, self.all_features = load_bed_annotation(file, chroms=set(self.seqlens.keys()))

    def is_complete(self):
        """
        Is the genome data necessary to calculate B complete?

        Note that Genome.neutral, the neutral ranges is not necessary
        to return True (since it's not needed to calculate B).
        """
        complete = (self._annot_file is not None and
                    self._recmap_file is not None)
        return complete

    def _build_segment_idx_interpol(self, verbose=True, **kwargs):
        """
        Build interpolators for mapping segment indices to map positions.
        """
        self._idx2map = dict()
        self._map2idx = dict()
        # self._pos2idx = dict()
        if verbose:
            print("building segment index interpolators... ", end='')
        for chrom in self.seqlens:
            # we don't want indices of total matrix (which inclues other
            # chroms), just this chrom
            indices = np.arange(len(self.segments.index[chrom]))
            # doesn't matter for our purposes difference between segment
            # start/end here... we use start
            map_pos = self.segments.mpos[chrom][:, 0]
            map_ends = map_pos[0], map_pos[-1]
            idx2map = interpolate.interp1d(indices, map_pos, assume_sorted=False,
                                           fill_value=map_ends, bounds_error=False)
            idx_ends = 0, len(indices)
            map2idx = interpolate.interp1d(map_pos, indices, assume_sorted=False,
                                           fill_value=idx_ends, bounds_error=False)
            # pos_ends = pos[0], pos[1]
            # pos = self.segments.ranges[:, 0]
            # pos2idx = interpolate.interp1d(pos, indices,
            #                                fill_value=pos_ends, **kwargs)

            self._idx2map[chrom] = idx2map
            self._map2idx[chrom] = map2idx
            # self._pos2idx[chrom] = pos2idx
        if verbose:
            print("done.")

    def get_segment_slice(self, chrom, mpos=None, pos=None, map_dist=0.1):
        """
        For a given physical position, find the map position, then get
        approximately map_dist in either direction (using a very rough
        linear approximation, which is fine for getting windows).
        For calculating B within a region where it matters.
        """
        if mpos is None and pos is not None:
            mpos = self.recmap.cumm_interpol[chrom](pos)
        elif pos is None and mpos is not None:
            pass
        else:
            raise ValueError("mpos and pos cannot both be None or both be set")
        lower = max(mpos - map_dist, self.recmap.cumm_rates[chrom].rate[0])
        upper = min(mpos + map_dist, self.recmap.cumm_rates[chrom].rate[-1])
        # now use inverse to get the indices
        lower_idx, upper_idx = self._map2idx[chrom](lower), self._map2idx[chrom](upper)
        return int(lower_idx), int(upper_idx)

