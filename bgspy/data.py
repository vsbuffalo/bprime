import os
import warnings
from collections import defaultdict
import itertools
import gzip
import numpy as np
from tqdm import tqdm
import tskit
from tabulate import tabulate
from bgspy.utils import read_bed3, ranges_to_masks
from bgspy.utils import aggregate_site_array, BinnedStat
from bgspy.utils import readfile, parse_param_str
from bgspy.utils import readfq, pretty_percent, bin_chrom, mean_ratio
from bgspy.genome import Genome

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
    return positions, indices, ac, position_map, parse_param_str(params[0])

def serialize_metadata(md):
    return ';'.join([f"{k}={v[0]}" for k, v in md.items()])


def load_fasta(fasta_file, chroms=None):
    """
    Create a accesssible mask file from the reference FASTA file.
    """
    f = readfile(fasta_file)
    seqs = dict()
    ignored = list()
    for chrom, seq, _ in readfq(f):
        if chroms is not None:
            if chrom not in chroms:
                ignored.append(chrom)
                continue
        seqs[chrom] = np.fromiter(map(ord, seq), dtype=np.ubyte)
    #if len(ignored):
    #    warnings.warn(f"{len(ignored)} chromosomes ignored during loading.")
    return seqs

def get_accessible_from_seqs(seqs, mask_chars='Nnatcg'):
    """
    """
    mask_chars = [ord(x) for x in mask_chars]
    masks = {c: np.full(x.size, 1, dtype='bool') for c, x in seqs.items()}
    for chrom, seq in seqs.items():
        masks[chrom][np.isin(seq, mask_chars)] = 0
    return masks


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

def pi_from_pairwise_summaries(x):
    """
    Given a summary of pairwise combinations (nsame, ndiff cols) return
    π. This assumes site arrays (e.g. the denominator is handled naturally).
    If x is a BinnedStat, this does the right thing too.
    """
    is_binstat = isinstance(x, BinnedStat)
    if is_binstat:
        # extract the data
        pair_data = x.stat
    else:
        pair_data = x
    denom = pair_data.sum(axis=1)
    out = np.divide(pair_data[:, 1], denom,
                    out=np.full(denom.shape[0], np.nan),
                    where=denom > 0)
    if not is_binstat:
        return out
    n = x.n
    n[denom == 0] = 0
    return BinnedStat(out, x.bins, n)


def trimmed_pi_bounds(Y, alpha):
    """
    """
    if isinstance(alpha, float):
        lower, upper = alpha/2, 1-alpha/2
    else:
        lower, upper = alpha
    pi = pi_from_pairwise_summaries(Y)
    q1, q2 = np.nanquantile(pi, lower), np.nanquantile(pi, upper)
    return q1, q2


def trimmed_pi_mask(Y, bounds):
    """
    From histograms of binned π along the genome some windows have outlier
    diversity (usually in the upper tail). Likelihood-based methods may be
    biased by these outlier windows, so a robust approach is to trim these
    tails to the α specified here. If alpha is a tuple, it's for (lower, upper)
    thresholds, e.g. trim the Y matrix by π to α/2, 1-α/2
    """
    q1, q2 = bounds
    pi = pi_from_pairwise_summaries(Y)
    keep_idx = (q1 < pi) & (q2 > pi)
    return keep_idx

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

def filter_sites(counts, chrom, filter_neutral, filter_accessible,
                 neutral_masks=None, accesssible_masks=None):
    ac = counts[chrom]
    if filter_neutral or neutral_masks is not None:
        msg = "GenomeData.neutral_masks not set!"
        assert neutral_masks is not None, msg
        #print("using neutral masks...")
        ac = ac * neutral_masks[chrom][:, None]

    if filter_accessible or accesssible_masks is not None:
        msg = "GenomeData.accesssible_masks not set!"
        assert accesssible_masks is not None, msg
        # print("using accessibility masks...")
        ac = ac * accesssible_masks[chrom][:, None]
    return ac

def trim_map_ends(recmap, thresh_cM):
    """
    Find the positions for each chromosome that with thresh_cM cM
    cut off each chromosome end. Returns chrom dict of lower, upper positions.
    """
    ends = dict()
    for chrom in recmap.rates:
        pos = recmap.cumm_rates[chrom].end
        cumm_rates = recmap.cumm_rates[chrom].rate

        start = thresh_cM / 100

        # note: we use pos[1] because first is NaN
        lower, upper = pos[1], pos[-1]
        if any(cumm_rates < start):
            lower = np.min(pos[cumm_rates < start])
        end = cumm_rates[-1] - thresh_cM / 100
        upper = np.max(pos[cumm_rates < end])
        ends[chrom] = lower, upper
    return ends

class GenomeData:
    def __init__(self, genome=None):
        self.genome = genome
        self.neutral_masks = None
        self.accesssible_masks = None
        self._npy_files = None
        self.counts = None
        self.thresh_cM = None

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

    def trim_ends(self, thresh_cM=0.1):
        """
        Trim off thresh_cM cM of ends.
        """
        assert self.genome.recmap is not None, "GenomeData.genome.recmap is not set!"
        ends = trim_map_ends(self.genome.recmap, thresh_cM)
        for chrom, (lower, upper) in ends.items():
            self.accesssible_masks[chrom][0:(lower+1)] = 0
            self.accesssible_masks[chrom][upper:] = 0
        self.thresh_cM = thresh_cM

    def load_fasta(self, fasta_file, soft_mask=True):
        """
        Load a FASTA genome reference file, which is used to
        determine which sites are accessible (e.g. Ns and soft-masked
        are set as inaccessible).
        """
        self.seqs = load_fasta(fasta_file, self.genome.chroms)
        if soft_mask:
            acc = get_accessible_from_seqs(self.seqs)
            if self.accesssible_masks is None:
                self.accesssible_masks = acc
            else:
                self.combine_accesssible(acc)

    def combine_accesssible(self, masks):
        for chrom in masks:
            self.accesssible_masks[chrom] = masks[chrom] * self.accesssible_masks[chrom]

    def load_counts_from_ts(self, ts=None, file=None, chrom=None):
        """
        Count the number of derived alleles in a TreeSequence -- assumes
        BinaryMutationModel. This is primarily for loading simulation data
        into for testing.

        If chrom is not set a new dummy Genome object is created.
        """
        if chrom is not None:
            assert chrom in self.genome.chroms, f"{chrom} not in GenomeData.genome"
        msg = "set either ts or file, not both"
        if file is not None:
            assert ts is None, msg
            ts = tskit.load(file)
        else:
            assert file is None, msg
        sl = int(ts.sequence_length)
        num_deriv = np.zeros(sl)
        for var in ts.variants():
            nd = (var.genotypes > 0).sum()
            num_deriv[int(var.site.position)] = nd
        assert np.sum(np.isnan(num_deriv)) == 0, "remaining nans -- num mut/num allele mismatch"
        ntotal = np.repeat(ts.num_samples, sl)
        if self.genome is None:
            chrom = 'chrom'
            g = Genome('dummy genome', seqlens={chrom: sl})
        g = self.genome
        self.genome = g
        nanc = ntotal - num_deriv
        self.counts = {chrom: np.stack((nanc, num_deriv)).T}

    def counts_to_tsv(self, outfile):
        """
        Write the counts to a TSV file,
        """
        self._assert_is_consistent()
        ntot = self.counts.sum(axis=1)
        md = self.metadata
        with gzip.open(outfile, 'wt') as f:
            if md is not None:
                f.write("#"+serialize_metadata(md)+"\n")
            for chrom in self.genome.chroms:
                for i, pos in enumerate(self.counts[chrom].shape[0]):
                    row = [chrom, pos, ntot[i], self.counts[chrom][i, 1]]
                    f.write("\t".join(map(str, row)) + "\n")

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

    def load_dac_file(self, filename):
        """
        Load a file of only polymorphic sites into site arrays.
        """
        ac = {c: np.zeros((sl, 2), dtype=np.int32) for c, sl in
              self.genome.seqlens.items()}
        with readfile(filename) as f:
            for line in f:
                if line.startswith('#'):
                    self._metadata = parse_param_str(line)
                    continue
                chrom, pos, ntotal, nderiv = line.strip().split('\t')
                pos, ntotal, nderiv = int(pos), int(ntotal), int(nderiv)
                ac[chrom][pos, 0] = ntotal - nderiv
                ac[chrom][pos, 1] = nderiv
        self.counts = ac

    @property
    def chroms(self):
        return self.genome.chroms

    def bin_pairwise_summaries(self, width, filter_neutral=None,
                               filter_accessible=None, progress=True):
        """
        Given a genomic window width, compute the pairwise statistics (the Y
        matrix of same and different pairwise combinations) and bin reduce them
        by summing counts within a bin.

        Returns: a GenomicBins object with the resulting data in GenomicBins.data.
        """
        bins = GenomicBinnedData(self.genome.seqlens, width)

        chroms = self.chroms
        chroms = chroms if not progress else tqdm(chroms)
        for chrom in chroms:
            site_ac = filter_sites(self.counts, chrom, filter_neutral,
                                   filter_accessible,
                                   self.neutral_masks, self.accesssible_masks)
            # the combinatoric step -- turn site allele counts into
            # same/diff comparisons
            Y = pairwise_summary(site_ac)
            # number of combinations summed into bins per chrom
            Y_binstat = aggregate_site_array(Y, bins[chrom], np.sum, axis=0)
            bins.data_[chrom] = Y_binstat.stat
            bins.n_[chrom] = Y_binstat.n
        return bins

    def npoly(self, filter_neutral=None, filter_accessible=None):
        out = dict()
        for chrom in self.genome.chroms:
            site_ac = filter_sites(self.counts, chrom,
                                   filter_neutral, filter_accessible,
                                   self.neut_masks, self.accesssible_masks)
            out[chrom] = (site_ac > 0).sum(axis=1).sum()
        return out

    def pi(self):
        pi = dict()
        n = dict()
        for chrom in self.genome.chroms:
            ac = self.counts[chrom]
            pi[chrom] = np.nanmean(pi_from_pairwise_summaries(pairwise_summary(ac)))
            n[chrom] = np.sum(ac.sum(axis=1) > 0)
        return pi, n

    def gwpi(self):
        pis, ns = self.pi()
        pis = np.fromiter(pis.values(), dtype=float)
        ns = np.fromiter(ns.values(), dtype=float)
        return np.average(pis, weights=ns)

    def mask_stats(self):
        stats = {}
        for chrom in self.genome.chroms:
            seq_len = self.genome.seqlens[chrom]
            acces = np.nan
            neut = np.nan
            both = np.ones(seq_len, dtype='bool')
            if self.accesssible_masks is not None:
                access_mask = self.accesssible_masks[chrom]
                acces = access_mask.mean()
                both = both & access_mask
            if self.neutral_masks is not None:
                neut_mask = self.neutral_masks[chrom]
                neut = neut_mask.mean()
                both = both & neut_mask
            stats[chrom] = (np.round(acces, 2),
                            np.round(neut, 2),
                            np.round(both.mean(), 2))
        return stats

    def __repr__(self):
        nchroms = len(self.genome.chroms)
        msg = [f"GenomeData ({nchroms} chromosomes)\nMasks:"]
        tab = ([('chrom', 'acc', 'neut', 'both')] +
                [(chrom, a, n, b) for chrom, (a, n, b) in self.mask_stats().items()])
        msg += [tabulate(tab, headers="firstrow")]
        trim = self.thresh_cM is not None
        trim_end = "" if not trim else f"{self.thresh_cM} cM"
        msg += [f"chromosome ends trimmed? {trim} {trim_end}"]
        return "\n".join(msg)

class GenomicBins:
    """
    A dictionary-like class for GenomicBins. The bin ends span the entire
    chromosome; so to indicate when bins are dropped, e.g. for analysis, we use
    a mask (true indicates valid, zero not).

    Attributes ending in an underscore (_) store the full data (e.g.
    unmasked. Note that the masks dictionary (GenomicBins.masks_) uses
    True to denote "keep" or pass".
    """
    def __init__(self, seqlens, width, dtype='uint32'):
        width = int(width)
        self.width = width
        # since the bins span the entire chromosome, we use masks
        # to indicate bins to drop
        self.dtype = dtype
        self.seqlens = seqlens
        self._bin_chroms(seqlens, width, dtype)

    def __repr__(self):
        width = self.width
        nseqs = len(self.seqlens)
        msg = f"GenomicBins: {width:,}bp windows on {nseqs} chromosomes\n"
        msg = f"  "
        for chrom in self.seqlens:
            x = list(map(str, self.bins_[chrom][:2].tolist()))
            y = list(map(str, self.bins_[chrom][-2:].tolist()))
            mperc = np.round((~self.masks_[chrom]).mean() * 100, 2)
            msg += f"  {chrom}: [" + ', '.join(x + ["..."] + y) + f"] (masked {mperc}%)\n"
        return msg

    def iter_(self):
        for chrom, bins in self.bins.items():
            for i, end in enumerate(bins):
                if i == 0:
                    continue
                start = bins[i-1]
                yield chrom, start, end

    @property
    def chroms_(self):
        return np.array([c for c, _, _ in self.iter_()])

    @property
    def starts_(self):
        return np.array([s for _, s, _ in self.iter_()])

    @property
    def ends_(self):
        return np.array([e for _, _, e in self.iter_()])

    def _bin_chroms(self, seqlens, width, dtype='uint32'):
        "Bin all chromosomes and put the results in a dictionary."
        self.bins_ = {c: bin_chrom(seqlens[c], width, dtype) for c in seqlens}
        # the number of data points is one less than the number of bins (e.g.
        # because of 0 rightmost bin
        self.clear_masks()

    def nbins_(self):
        n = {c: len(self.bins_[c])-1 for c in self.bins_.keys()}

    def nbins(self):
        return sum(self.mask_array)

    def bins(self, filter_masked=True):
        """
        Return a dictionary of the mask-filtered bins.
        """
        out = defaultdict(list)
        for chrom, bins in self.bins_.items():
            for i, end in enumerate(bins):
                if i == 0:
                    continue
                start = bins[i-1]
                if not filter_masked or self.masks_[chrom][i-1]:
                    out[chrom].append((start, end))
        return out

    def flat_bins(self, filter_masked=True):
        out = []
        for chrom, bins in self.bins(filter_masked).items():
            for start, end in bins:
                out.append((chrom, start, end))
        return out

    def keys(self):
        return self.bins_.keys()

    def __getitem__(self, chrom):
        """
        Return the raw bin boundaries for the specified chromosome 'chrom'.
        No mask-based filtering, etc. -- e.g. for use with aggregating statistics.
        """
        return self.bins_[chrom]

    def midpoints(self, filter_masked=True):
        """
        Calculate midpoints of mask-filtered bins.
        """
        out = defaultdict(list)
        for chrom, bins in self.bins(filter_masked=filter_masked).items():
            for start, end in bins:
                out[chrom].append((start+end)/2)
        return out

    @property
    def mask_array(self):
        mask = []
        for chrom, masks in self.masks_.items():
            for m in masks:
                mask.append(m)
        return np.array(mask, dtype='bool')

    def clear_masks(self):
        self.masks_ = {c: np.ones(len(b)-1, dtype='bool') for c, b in self.bins_.items()}


class GenomicBinnedData(GenomicBins):
    """
    A class for chromosome genomic bins and data necessary to fit BGS likelihood
    models (the pairwise summaries matrix Y). The B arrays are loaded
    separately, and the BScores.bin_means() is called for these bin boundaries.
    Outlier π bins can be masked using GenomicBins.mask_outliers.

    Note that this supports multiple B' and B objects, each named and
    stored in a dictionary, e.g. for multiple model runs and comparison.
    """
    def __init__(self, seqlens, width, dtype='uint32'):
        super().__init__(seqlens, width, dtype=dtype)
        self.n_ = dict()
        self.data_ = dict()
        self.B_ = None
        self.Bp_ = None

    def data(self, filter_masked=True):
        """
        Return the data, filtering by masks if filter_masked is True.
        """
        out = dict()
        for chrom in self.data_:
            dat = self.data_[chrom]
            if filter_masked:
                mask = self.masks_[chrom]
                dat = dat[mask, ...]
            out[chrom] = dat
        return out

    def pi_pairs(self, chrom, ratio=False, filter_masked=True):
        """
        Return pair of midpoints, π for the specified chromosome.
        Assumed self.data[chrom] is a Y matrix.

        The behavior of mask filtering here is a bit different, since
        the intended usage of this is for plots -- removed π values
        are marked NaN since these are plotter (lines are disconnected
        with plt.plot())
        """
        pi = pi_from_pairwise_summaries(self.data(filter_masked=False)[chrom])
        if filter_masked:
            pi[~self.masks_[chrom]] = np.nan
        mp = self.midpoints(filter_masked=False)[chrom]
        if ratio:
            pi = mean_ratio(pi)
        return mp, pi

    def mask_outliers(self, alpha):
        """
        Mark all bins that are within the quantile bounds defined by
        alpha (can be single float or lower, upper bounds).
        """
        # get the bounds for the trimmed means -- these are chromosome-wide
        # so on the full Y matrix
        Y = self.Y(filter_masked=False)
        cutoff = trimmed_pi_bounds(Y, alpha)
        for chrom, Y_chrom in self.data_.items():
            trim_mask = trimmed_pi_mask(Y_chrom, cutoff)
            self.masks_[chrom] = trim_mask & self.masks_[chrom]
        self.outlier_quantiles = cutoff

    def Y(self, filter_masked=True):
        """
        Return the pairwise summary data matrix across all chromosomes,
        concatenated along the first axis, e.g. for statistical analyses.
        """
        Y = []
        for chrom, data in self.data_.items():
            for j in range(data.shape[0]):
                if filter_masked is True:
                    if self.masks_[chrom][j]:
                        Y.append(data[j, :])
                else:
                    Y.append(data[j, :])
        return np.stack(Y)

    def nbins(self, filter_masked=True):
        if filter_masked:
            return self.mask_array.sum()
        else:
            return self.mask_array.size

    def bin_Bs(self, bscores, filter_masked=True):
        """
        Bin the BScores object using these genomic bins.
        Bins corresponding to masked windows are removed.
        """
        b = bscores.bin_means(self, merge=True)
        if not filter_masked:
            return b
        return b[self.mask_array, ...]

    def merge_filtered_data(self, data):
        """
        If we have data that's been mask-filtered, merge it back into an array
        with NaNs for masked sites.
        """
        mask_array = self.mask_array
        n = mask_array.size
        nkeep = mask_array.sum()
        nrows = data.shape[0]
        assert nkeep == nrows, "data is not the size of masked filtered elements"
        X = np.full((n, *data.shape[1:]), np.nan)
        j = 0
        for i in range(n):
            if mask_array[i]:
                X[i, ...] = data[j, ...]
                j += 1
        return X


