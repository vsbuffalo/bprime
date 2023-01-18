import pickle
import warnings
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from scipy import interpolate
from collections import defaultdict, namedtuple, deque
from bgspy.utils import load_seqlens, load_bed_annotation, BScores
from bgspy.recmap import RecMap
from bgspy.theory2 import B_segment_lazy, BSC16_segment_lazy_parallel

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
        self.inverse_feature_map = {i: k for k, i in self.feature_map.items()}
        self.index = index
        self._segment_parts = None
        self._segment_parts_sc16 = None
        self.F = None
        self.rescaling = None
        # model fits often exclude chrX, etc so this is a hack
        # so when rescaling is set, BChunkIterator, etc can know
        # to exclude certain chroms
        #self._rescaling_excluded = set()
        self._calc_features()

    def __repr__(self):
        nfeats = len(self.feature_map)
        return f"Segments ({len(self):,} total with {nfeats} feature type(s))"

    def __len__(self):
        return self.ranges.shape[0]

    @property
    def midpoints(self):
        mids = np.mean(self.ranges, axis=1).squeeze()
        return {c: mids[idx] for c, idx in self.index.items()}

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

    def _calc_segparts(self, w, t, N=None, ncores=None):
        """
        Calculate the fixed components of the classic BGS theory
        and the new BGS theory (if N is set).
        These components are for for each segment, to avoid unnecessary repeated
        calcs.
        """
        L = self.lengths
        rbp = self.rates
        if t.ndim == 1:
            t = t[:, None]
        print(f"calculating classic B components...\t", end='', flush=True)
        self._segment_parts = B_segment_lazy(rbp, L, t)
        print("done.")
        if N is not None:
            print(f"\ncalculating B' components...\t", end='', flush=True)
            rescaling = self.rescaling
            parts = BSC16_segment_lazy_parallel(w, t, L, rbp, N,
                                                ncores=ncores,
                                                rescaling=rescaling)
            self._segment_parts_sc16 = parts
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

    def _load_rescaling(self, bdict, w, t):
        """
        Common code for both ways of rescaling Ne locally, i.e. from a MLE
        fit and from a set of existing Bs, e.g. grid
        """

        predicted_bscores = BScores(bdict, posdict, w=w, t=t)
        predicted_bscores._build_interpolators()

        rescaling = list()
        for chrom, mids in self.midpoints.items():
            if chrom not in predicted_bscores._interpolators:
                # we don't have an interpolator for this chrom (usually X)
                raise ValueError(f"{chrom} not in interpolators!")
                #warnings.warn(f"skipping chromosome {chrom}!")
                #self._rescaling_excluded.add(chrom)

            b = predicted_bscores.B_at_pos(chrom, mids)
            #rescaling[chrom] = b
            rescaling.extend(b.squeeze())
        self.rescaling = np.array(rescaling)

    def load_rescaling_from_fixed_params(self, bp, w, t):
        """
        A test way to do rescaling is to use the unscaled B' values for
        each mutation/selection coefficient combination. This is
        primarily used for testing/comparison with sims with fixed mu/s
        combinations.

        """
        assert isinstance(bp, BScores), "bp must be a BScores object"

        # slice out this w/t
        wi, ti = bp.indices(w, t)
        # make a BScores object with just this predicted reduction.
        bdict = {c: bp[c, w, t][1][:, None, None] for c in bp.B.keys()}
        posdict = {c: bp[c, w, t][0] for c in bp.B.keys()}
        predicted_bscores = BScores(bdict, posdict, np.array([w]), np.array([t]))

        predicted_bscores._build_interpolators()

        rescaling = list()
        for chrom, mids in self.midpoints.items():
            if chrom not in predicted_bscores._interpolators:
                # we don't have an interpolator for this chrom (usually X)
                raise ValueError(f"{chrom} not in interpolators!")
                #warnings.warn(f"skipping chromosome {chrom}!")
                #self._rescaling_excluded.add(chrom)

            b = predicted_bscores.B_at_pos(chrom, mids)
            #rescaling[chrom] = b
            rescaling.extend(b.squeeze())
        self.rescaling = np.array(rescaling)

    def load_rescaling_from_fit(self, fit):
        """
        Take a fit and run predict_B to get the rescaling factor.
        """
        predicted_B = fit.predict(B=True)
        bins = fit.bins

        bin_mids = fit.bins.midpoints()
        bdict = dict()
        posdict = dict()
        for chrom in fit.bins.keys():
            idx = fit.bins.chrom_indices(chrom)
            bdict[chrom] = predicted_B[idx, None, None]
            posdict[chrom] = np.array(bin_mids[chrom])

        # dummy w/t
        w = np.array([0])
        t = np.array([0])

        predicted_bscores = BScores(bdict, posdict, w=w, t=t)
        self._load_rescaling(predicted_bscores, w, t)

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

        bad = set(features.keys()).difference(set(recmap.rates.keys()))
        raise ValueError(f"features contains sequences not in recmap, {bad}")

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
    def __init__(self, name, seqlens_file=None, seqlens=None, chroms=None):
        self.name = name
        self.seqlens = None
        msg = "set seqlens_file or seqlens, not both!"
        if seqlens is not None:
            assert seqlens_file is None, msg
            if chroms is not None:
                sl = {c: s for c, s in seqlens.items() if c in chroms}
            else:
                sl = {c: s for c, s in seqlens.items()}
            self.seqlens = sl
        if seqlens_file is not None:
            assert seqlens is None, msg
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

    @property
    def chroms(self):
        return list(self.seqlens.keys())

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

    def feature_stats(self, by_chrom=False, exclude_chroms=None):
        """
        Calculate feature statistics for all segments. Returns a tuple
        of chromosome dictionaries of number features, total bp of feature,
        and the fractions of each feature type by total chromsome length.
        """
        exclude_chroms = set() if exclude_chroms is None else set(exclude_chroms)
        nfeats = defaultdict(Counter)
        nbp = defaultdict(Counter)
        frac = defaultdict(dict)

        tot_nfeats = Counter()
        tot_nbp = Counter()
        tot_frac = dict()

        for chrom, idx in self.segments.index.items():
            for l, f in zip(self.segments.lengths[idx], self.segments.features[idx]):
                feat = self.segments.inverse_feature_map[f]
                # chrom-specific
                nfeats[chrom][feat] += 1
                nbp[chrom][feat] += l

                # total features
                tot_nfeats[feat] += 1
                tot_nbp[feat] += l

                for feat, val in nbp[chrom].items():
                    frac[chrom][feat] = val / self.seqlens[chrom]

        totlen = sum(self.seqlens.values())
        for feat in self.segments.feature_map.keys():
            tot_frac[feat] = sum([nbp[c][feat] for c in nbp if c not in exclude_chroms]) / totlen

        if by_chrom:
            return nfeats, nbp, frac
        return tot_nfeats, tot_nbp, tot_frac


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

