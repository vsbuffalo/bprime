"""
annotation.py 

note: this was a late addition to the API, when the 
simple NamedTuple was no longer sufficient.
"""
import warnings
from collections import defaultdict, Counter
import numpy as np

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
    from bgspy.utils import readfile  # prevent circular import
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
    #Annotation = namedtuple('Annotation', ('ranges', 'features'))
    return (ranges, all_features)


class Annotation:
    def __init__(self, ranges, features, seqlens):
        self.ranges = ranges
        self.features = features
        self.seqlens = seqlens

    @staticmethod
    def load_bed(file, seqlens=None):
        ranges, features = load_bed_annotation(file)
        annot = Annotation(ranges, features)
        obj.seqlens = seqlens
        return obj

    def __repr__(self):
        return f"Annotation object with {len(self.features)} features."
    
    def stats(self):
        """
        Return coverage statistics:
        """
        gw = Counter()
        seen_chroms = set()
        for chrom, feature_ranges in self.ranges.items():
            seen_chroms.add(chrom)
            for range, feature in zip(*feature_ranges):
                start, end = range
                gw[feature] += end-start

        percents = None
        if self.seqlens is not None:
            # get percents
            percents = dict()
            total_bp = sum([l for c, l in self.seqlens.items()
                            if c in seen_chroms])
            for feature in gw:
                frac = gw[feature] / total_bp
                percents[feature] = np.round(100*frac, 2)

        return gw, percents
