import os
import gzip
import numpy as np
import tskit
from bgspy.utils import read_bed3, ranges_to_masks, GenomicBins
from bgspy.utils import aggregate_site_array, BinnedStat
from bgspy.utils import readfile, parse_param_str
from bgspy.utils import readfq, pretty_percent
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
    for chrom, seq, _ in readfq(f):
        if chroms is not None:
            if chrom not in chroms:
                continue
        seqs[chrom] = np.fromiter(map(ord, seq), dtype=np.ubyte)
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
    if filter_neutral and neutral_masks is not None:
        msg = "GenomeData.neutral_masks not set!"
        assert neutral_masks is not None, msg
        print("using neutral masks...")
        ac = ac * neutral_masks[chrom][:, None]

    if filter_accessible and accesssible_masks is not None:
        msg = "GenomeData.accesssible_masks not set!"
        assert accesssible_masks is not None, msg
        print("using accessibility masks...")
        ac = ac * accesssible_masks[chrom][:, None]
    return ac

class GenomeData:
    def __init__(self, genome=None):
        self.genome = genome
        self.neutral_masks = None
        self.accesssible_masks = None
        self._npy_files = None
        self.counts = None

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

    def load_fasta(self, fasta_file, soft_mask=True):
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
        Count the number of derived alleles -- assumes BinaryMutationModel.

        If chrom is not set a new dummy Genome object is created.
        """
        if chrom is not None:
            assert chrom in self.genome.chroms, "chrom not in GenomeData.genome"
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
        assert np.sum(np.isnan(num_deriv)) == 0, "remauning nans -- num mut/num allele mismatch"
        ntotal = np.repeat(ts.num_samples, sl)
        if chrom is None:
            chrom = 'chrom'
            g = Genome('dummy genome', seqlens={chrom: sl})
        else:
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

    def stats(self):
        out = dict()
        for chrom in self.genome.chroms:
            nneut = self.neutral_masks[chrom].mean()
            nacc = self.accesssible_masks[chrom].mean()
            out[chrom] = nneut, nacc
        return out

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


    def bin_reduce(self, width, filter_neutral=None, filter_accessible=None):
        """
        Given a genomic window width, bin the data and compute bin-level
        summaries for the likelihood.

        Returns: Y matrix (nsame, ndiff cols) and a chrom dict of bin ends.
        """
        bins = GenomicBins(self.genome.seqlens, width)
        reduced = dict()
        for chrom in self.genome.chroms:
            site_ac = filter_sites(self.counts, chrom, filter_neutral,
                                   filter_accessible,
                                   self.neutral_masks, self.accesssible_masks)

            # the combinatoric step -- turn site allele counts into
            # same/diff comparisons
            d = pairwise_summary(site_ac)
            reduced[chrom] = aggregate_site_array(d, bins[chrom], np.sum, axis=0)

        return bins, reduced

    def npoly(self, filter_neutral=None, filter_accessible=None):
        out = dict()
        for chrom in self.genome.chroms:
            site_ac = filter_sites(self.counts, chrom,
                                   filter_neutral, filter_accessible,
                                   self.neut_masks, self.accesssible_masks)
            out[chrom] = (site_ac > 0).sum(axis=1).sum()
        return out

    def bin_pi(self, width, filter_neutral=None, filter_accessible=None):
        """
        Calculate π from the binned summaries.
        """
        bins, reduced = self.bin_reduce(width, filter_neutral, filter_accessible)
        pi = dict()
        for chrom in reduced.keys():
            pi[chrom] = pi_from_pairwise_summaries(reduced[chrom])
        return bins, pi

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
            if self.accesssible_masks is not None:
                acces = self.accesssible_masks[chrom].sum() / seq_len
            if self.neutral_masks is not None:
                neut = self.neutral_masks[chrom].sum() / seq_len
            stats[chrom] = (acces, neut)
        return stats

    def __repr__(self):
        nchroms = len(self.genome.chroms)
        msg = [f"GenomicData ({nchroms} chromosomes)\nMasks:",
               f"         accessible    neutral"]
        for chrom, (acces, neut) in self.mask_stats().items():
            msg += [f"  {chrom}     {pretty_percent(acces)}%        {pretty_percent(neut)}%"]
        return "\n".join(msg)



