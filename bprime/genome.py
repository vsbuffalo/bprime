from dataclasses import dataclass
import numpy as np
from collections import defaultdict, namedtuple
from bprime.utils import load_seqlens, load_bed_annotation
from bprime.recmap import RecMap

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
        - index, a dict of chrom->indices of the ranges array
    """
    ranges: np.ndarray
    rates: np.ndarray
    map_pos: np.ndarray
    features: np.ndarray
    feature_map: dict
    index: defaultdict

    def __repr__(self):
        nfeats = len(self.feature_map)
        return f"Segments ({len(self):,} total with {nfeats} feature type(s))"

    def __len__(self):
        return self.ranges.shape[0]

    @property
    def lengths(self):
        return np.diff(self.ranges, axis=1).squeeze()

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

    def load_seqlens(self, file, chroms):
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
        self.segments = process_annotation(self.annot, self.recmap, split_length)

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
        self.annot, self.all_features = load_bed_annotation(file)


    def is_complete(self):
        """
        Is the genome data necessary to calculate B complete?

        Note that Genome.neutral, the neutral ranges is not necessary
        to return True (since it's not needed to calculate B).
        """
        complete = (self._annot_file is not None and
                    self._recmap_file is not None)
        return complete








