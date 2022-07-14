import os
import numpy as np
from bgspy.utils import read_bed3, ranges_to_masks, GenomicBins
from bgspy.utils import aggregate_site_array

# error out on overflows
np.seterr(all='raise')

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


def pairwise_summary(ac):
    """
    Compute the number of pairwise same and different differenes per position,
    given an allele count matrix.

    Based on the approach in scikit-allel and Nei.
    """
    ac = ac.astype(np.int32)
    an = np.sum(ac, axis=1)
    n_pairs = an * (an - 1) / 2
    n_same = np.sum(ac * (ac - 1) / 2, axis=1)
    n_diff = n_pairs - n_same
    return np.stack((n_same, n_diff)).T

def pi_from_pairwise_summaries(pair_data):
    """
    Given a summary of pairwise combinations (nsame, ndiff cols) return
    π. This assumes site arrays (e.g. the denominator is handled naturally).
    """
    denom = pair_data.sum(axis=1)
    return np.divide(pair_data[:, 1], denom,
                     out=np.full(denom.shape[0], np.nan),
                     where=denom > 0)

class CountsDirectory:
    """
    Lazy load a directory of .npy files of chromosome count data.
    """
    def __init__(self, dir, chrom_extract, lazy=True):
        self.npy_files = {chrom_extract(f): os.path.join(dir, f) for f in
                          os.listdir(dir) if f.endswith('.npy')}
        self.lazy = True
        if not lazy:
            self.counts = {c: np.load(f) for c, f in self.npy_files.items()}
            self.lazy = False

    def __getitem__(self, key):
        if self.lazy:
            return np.load(self.npy_files[key])
        return self.counts[key]

class GenomeData:
    def __init__(self, genome, neutral_masks=None):
        self.genome = genome
        self.neutral_masks = neutral_masks
        self._npy_files = None
        self.counts = None
        self.dac = None

    def _load_mask(self, file):
        """
        Load a 3-column BED file into a mask objects.
        """
        ranges = read_bed3(file, keep_chroms=set(self.genome.seqlens.keys()))
        return ranges_to_masks(ranges, self.genome.seqlens)

    def load_accessibile_masks(self, file):
        """
        Load a 3-column BED file of accessible ranges into mask objects.
        """
        self.accesssible_masks = self._load_mask(file)

    def load_neutral_masks(self, file):
        """
        Load a 3-column BED file of neutral ranges into mask objects.
        """
        self.neutral_masks = self._load_mask(file)

    def load_counts_dir(self, dir, lazy=True):
        """
        Load a directory of allele count data in .npy format. Each file
        should begin with the chromosome, e.g. chr10_counts.npy.
        """
        chrom_extract = lambda x: x.split('_')[0]
        self.counts = CountsDirectory(dir, chrom_extract, lazy)
        self.dac = False

    def load_dac_file(self, filename):
        """
        Load a file of only polymorphic sites into site arrays.
        """
        pass

    def bin_reduce(self, width, filter_neutral=True, filter_accessible=True):
        """
        Given a genomic window width, bin the data and compute bin-level
        summaries for the likelihood.

        Returns: Y matrix (nsame, ndiff cols) and a chrom dict of bin ends.
        """
        assert self.neutral_masks is not None, "GenomeData.neutral_masks not set!"
        assert self.counts is not None, "GenomeData.counts is not set!"

        bins = GenomicBins(self.genome.seqlens, width)
        reduced = dict()
        for chrom in self.genome.chroms:
            site_ac = self.counts[chrom]
            if filter_neutral:
                msg = "GenomeData.neutral_masks not set!"
                assert self.neutral_masks is not None, msg
                site_ac = site_ac * self.neutral_masks[:, None]

            if filter_accessible:
                msg = "GenomeData.accesssible_masks not set!"
                assert self.accesssible_masks is not None, msg
                site_ac = site_ac * self.accesssible_masks[:, None]

            site_ac[site_ac == -1] = 0 # TODO remove
            d = pairwise_summary(site_ac)
            reduced[chrom] = aggregate_site_array(d, bins[chrom], np.nansum, axis=0)

        return bins, reduced

    def bin_pi(self, **kwargs):
        """
        Calculate π from the binned summaries.
        """
        bins, reduced = self.bin_reduce(**kwargs)
        pi = dict()
        for chrom in reduced.keys():
            pi[chrom] = pi_from_pairwise_summaries(reduced[chrom])
        return bins, pi







