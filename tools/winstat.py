import re
import sys
from collections import deque, Counter, defaultdict, namedtuple
from multiprocessing import Pool
from functools import partial
import numpy as np
from cyvcf2 import VCF, Writer
import allel as al
import gzip
import click

WindowStats = namedtuple('WindowStats', ('start', 'end', 'pi',
                                         'n', 'depth', 'ncalled',
                                         'nskipped', 'duplicates'))

SEQLEN_RE = re.compile(r'##contig=<ID=(?P<seqname>\w+),length=(?P<length>\d+)>')
REGION_RE = re.compile(r'(?P<seqname>\w+)(:(?P<start>\d+)-(?P<end>\d+))?')

HEADER = "chrom\t" + "\t".join(WindowStats._fields) + "\n"

def region_width(region, seqlens):
    mtch = REGION_RE.match(region)
    groups = mtch.groupdict()
    start, end = groups.get('start', None), groups.get('end', None)
    seqname = groups['seqname']
    if start is None or end is None:
        return seqlens[seqname]
    msg = (f"only whole sequence (i.e. '{seqname}') regions are supported, " +
           f"regions like '{region}' are not yet implemented.")
    raise NotImplementedError(msg)
    return int(end) - int(start)

def get_seqlens(header_str):
    seqlens = dict()
    for line in header_str.split('\n'):
        mtch = SEQLEN_RE.match(line)
        if mtch is None:
            continue
        groups = mtch.groupdict()
        seqlens[groups['seqname']] = int(groups['length'])
    return seqlens

def window_genome(seqlens, width):
    """Window the genome using the sequence lengths.
    """
    bins = dict()
    for chrom, length in seqlens.items():
        last = width * (length // width + 1) + 1
        cuts = np.arange(0, last, width)
        bins[chrom] = (cuts[:-1], cuts[1:])
    return bins

class AlleleCounter(object):
    """
    """

    def __init__(self, width, start=None, end=None):
        """
        Initialize the genotype counter.

        This is a (width x four-column)-shaped numpy array
        of genotype counts. The last column is unknown (e.g. due
        to no coverage).
        """
        self.width = width
        self.start = start
        self.end = end
        self._counter = np.zeros(shape=(width, 4), dtype='int64')
        # both of the following are conditional on not all individuals
        # having missing genotype depth data
        self._depth_sum = 0  # the total genotype depth for sites
        self._depth_n = 0  # the total number of individuals with genotype depth
        self._num_called_sum = 0  #
        self._num_called_n = 0
        self._num_called_skipped = 0
        self._pos = list()
        self._complete_pos = list()
        self._last_pos = -1
        self._pos_set = set()
        self._debug = defaultdict(list)
        self._idx_set = set()
        self.duplicates = 0

    def _check_consistency(self):
        """
        This checks this consistency of the underlying matrix.
        The number of non-zero row genotype entries must equal 
        the number of unique positions.
        """
        non_empty = (self._counter.sum(axis=1) > 0).sum()
        assert(non_empty == len(self._pos_set))
        nacc = (self._counter[:, :-1].sum(axis=1) > 0).sum()
        assert(nacc == len(set(self._complete_pos)))

    @property
    def naccessible(self):
        """
        Get the bases that have non-missing genotypes (the first
        three columns). Note that this does *not* mean that all
        samples were genotyped.
        """
        self._check_consistency()
        nacc = (self._counter[:, :-1].sum(axis=1) > 0).sum()
        return nacc

    @property
    def accessible_regions(self):
        self._check_consistency()
        last_pos = None
        start = None
        regions = []
        # ignore cases with all missing genotypes
        positions = sorted(list(set(self._complete_pos)))
        if not len(positions):
            return []
        for pos in positions:
            if start is None:
                start = pos
                last_pos = pos
                continue
            if pos > last_pos + 1:
                # disconnect in region, push the current region
                # add one, since we do zero-based indexing and
                # the its [start, end)
                regions.append((start, last_pos+1))
                start = pos
                last_pos = pos
            else:
                last_pos = pos
        regions.append((start, max(self.end, last_pos+1)))
        return regions


    def add_variant(self, gtypes, pos, gdepths=None, num_called=None):
        """
        Note: cyvcf2 has issues handling GVCFs correctly. If a locus isn't
        polymorphic and the DP genotype field isn't diploid (e.g. 0/0:20,0)
        then gt_depths isn't propogated.

	Note: I use DP rather than AD even though this is filtered because
        DP is more common, and mixing them would be bad.
        """
        # check everything sorted, but we allow duplicates
        # which can mess up order
        #assert(pos >= self.start and pos < self.end)
        assert((pos >= self._last_pos) or pos in self._pos_set)
        self._last_pos = pos

        # convert position to index in window
        idx = pos % self.width
        self._debug[idx].append(pos)

        empty_row = np.all(self._counter[idx, :] == 0)
        if not empty_row:
            self.duplicates += 1 
            # make sure there's not an indexing error:
            # if the counts row is full, we should have seen it...
            assert(pos in self._pos_set)
            assert(idx in self._idx_set)
            # we do *not* add this. First occurrence takes precedence
            # this is because we don't want duplicates in any output VCF
            return False
        # add this to the positions list and set, and index set
        self._pos.append(pos)
        self._pos_set.add(pos)
        self._idx_set.add(idx)

        # add the genotypes to the counter, ensuring we don't overrun shape
        assert(np.all(gtypes >= 0))
        assert(np.all(np.logical_and(gtypes >= 0, gtypes <= 3)))
        np.add.at(self._counter, (idx, gtypes), 1)

        # add this to the positions list and set, and index set
        # if it's not all empty data
        all_empty = (self._counter[idx, :-1] == 0).all()
        if not all_empty:
            self._complete_pos.append(pos)

        if num_called is not None:
            self._num_called_sum += num_called
            self._num_called_n += 1
        if gdepths is not None:
            complete = gdepths > 0
            if not np.any(complete):
                self._num_called_skipped += 1
            else:
                x = np.nansum(gdepths[complete])
                if not np.isnan(x):
                    self._depth_sum += x
                self._depth_n += np.nansum(complete)
        return True

    def _allel_diversity(self):
        """
        For debugging.
        """
        self._allele_counts = al.AlleleCountsArray(self._counter[:, :-1])
        ave_diffs = al.mean_pairwise_difference(self._allele_counts)
        return np.nansum(ave_diffs) / self.naccessible

    def diversity(self):
        #np.sum(self._counter.sum(axis=1) > 0) - len(self._pos)
        ac = self._counter[:, :-1]
        an = ac.sum(axis=1)
        n_pairs = an * (an - 1) / 2
        n_same = np.sum(ac * (ac - 1) / 2, axis=1)
        n_diff = n_pairs - n_same
        with np.errstate(invalid='ignore'):
            mpd = np.where(n_pairs > 0, n_diff / n_pairs, np.nan)
        n = self.naccessible
        return np.nansum(mpd) / n, n

    @property
    def mean_depth(self):
        if self._depth_n == 0:
            return np.nan
        return self._depth_sum / self._depth_n

    @property
    def mean_num_called(self):
        if self._num_called_n == 0:
            return np.nan
        return self._num_called_sum / self._num_called_n

    def stats(self):
        pi, n = self.diversity()
        depth, ncalled = self.mean_depth, self.mean_num_called
        nskipped = self._num_called_skipped
        return WindowStats(start=self.start, end=self.end,
                           pi=pi, n=n, depth=depth, ncalled=ncalled,
                           nskipped=nskipped, duplicates=self.duplicates)


def calc_stats(region, vcf_file, width, seqlens, pass_only=True,
               outvcf=None, outbed=None):
    """
    The main loop, which if region is None, this is run across
    the entire VCF. If region is a chromosome, this can be
    the function called by multiprocessing.Pool().
    """
    vcf = VCF(vcf_file, gts012=True, strict_gt=True)
    windows = window_genome(seqlens, width)

    row, kept = 0, 0
    chrom, start, end = None, 0, width
    var_types = Counter()
    window_counts = defaultdict(list)
    counter = None
    accepted_var_types = set(('unknown', 'snp'))
    stats = Counter()
    accessible_regions = defaultdict(list)
    filters = Counter()
    last_pos = None
    # gts = [] # for debugging
    for variant in vcf(region):
        row += 1
        var_types[variant.var_type] += 1
        pos = variant.POS - 1 # 0-indexed, VCFs are 1-indexed

        if chrom != variant.CHROM:
            # process and output results from last chrom
            if counter is not None:
                # add to the window counts (if not the first, non-initialized
                # counter
                window_counts[chrom].append(counter.stats())
                accessible_regions[chrom].extend(counter.accessible_regions)

            # switch up new chromosome
            chrom = variant.CHROM
            chrom_windows = windows[chrom]
            starts = deque(chrom_windows[0])
            ends = deque(chrom_windows[1])

            start = -1
            while not (pos >= start and pos < end):
                # shimmy up to the first variant by finding
                # enclosing window
                start = starts.popleft()
                end = ends.popleft()
                #print(f"shimmy 1, window {start}-{end}, pos {variant.POS-1}, {pos >= start and pos < end}")
            counter = AlleleCounter(width, start=start, end=end)

        ## the current position is past the current window
        if pos >= end:
            # process and output results from last window
            window_counts[chrom].append(counter.stats())
            accessible_regions[chrom].extend(counter.accessible_regions)
            # switch up new window
            while not (pos >= start and pos < end):
                # shimmy up to the first variant
                end = ends.popleft()
                start = starts.popleft()
            counter = AlleleCounter(width, start=start, end=end)

            # TODO: if we need to do region process at the below-chrom
            # level, we need to handle flushing windows before the region
            # while start < variant.POS:
            #     end = ends.popleft()
            #     start = starts.popleft()
            # # genotypes = []

        # status couniting
        if chrom is not None and row % 10000 == 0:
            if region is not None:
                total = region_width(region, seqlens)
            else:
                total = seqlens[chrom]
            progress = np.round(100 * row / total, 1)
            sys.stderr.write(f"{chrom} progress: {progress}%\r")

        multiallelic = len([x for x in variant.ALT if x != "<NON_REF>"]) > 1
        if multiallelic:
            stats['multiallelic'] += 1
            continue
        if variant.var_type not in accepted_var_types and variant.ALT != ['<NON_REF>']:
            stats[variant.var_type] += 1
            continue
        if pass_only and variant.FILTER is not None:
            # cyvcf uses None to indicate pass (i.e. no filter)
            stats['not_pass'] += 1
            filters[variant.FILTER] += 1
            continue
        # if not np.all(variant.gt_depths == -1) and variant.gt_depths.sum() >0:
            # __import__('pdb').set_trace()
            # print(variant.gt_depths)

        # See note in AlleleCounter.add_variant for why we don't use
        # variant.gt_types. We need to try to get the genotype data
        # in a few ways...
        DP = variant.format('DP')
        if DP is None:
            stats['no_DP'] += 1
            continue
        DP[DP == -1] = 0
        gdepths = np.nansum(DP, axis=1)
        added = counter.add_variant(variant.gt_types, pos,
                            gdepths=gdepths,
                            num_called=variant.num_called)
        if outvcf is not None and len(variant.ALT) > 0 and added:
            # not we only output the VCF rows that aren't duplicates
            # that have been added to the statistics
            outvcf.write_record(variant)
        stats['pass'] += 1
        kept += 1

    # output any last entries
    if counter is not None:
        window_counts[chrom].append(counter.stats())
        accessible_regions[chrom].extend(counter.accessible_regions)

    # output a track of accessible regions
    if outbed is not None:
        for chrom, regions in accessible_regions.items():
            for start, end in regions:
                outbed.write(f"{chrom}\t{start}\t{end}\n")

    if outvcf is not None:
        outvcf.close()
    return window_counts, stats, filters

def write_table(outfile, results):
    outfile.write(HEADER)
    for chrom, windows in results.items():
        for stats in windows:
            if stats is None:
                continue
            outfile.write(f"{chrom}\t" + "\t".join(map(str, stats)) + "\n")

@click.command()
@click.option('--cores', default=None, type=int, help="the number of cores to use in processing")
@click.option('--width', default=int(1e6), help="The window width in basepairs")
@click.option('--outfile', default=sys.stdout, type=click.File("w"), help="output file")
@click.option('--outvcf', default=None, type=str, help="VCF of sites included in statistics")
@click.option('--outbed', default=None, type=str, help="BED of all regions included")
@click.option('--region', default=None, help="region to process")
@click.option('--pass-only', default=True, is_flag=True, help="only keep variants with the PASS in the filter column")
@click.argument('vcf_file')
def main(vcf_file, outfile, region, width=int(1e6), cores=None, pass_only=True,
         outvcf=None, outbed=None):
    vcf = VCF(vcf_file, gts012=True)
    seqlens = get_seqlens(vcf.raw_header)
    vcf.close()

    if cores is not None:
        with Pool(cores) as p:
            if outvcf is not None or outbed is not None:
                raise ValueError("--cores cannot be used with --outvcf or --outbed")
            runner = partial(calc_stats, vcf_file=vcf_file,
                             width=width, seqlens=seqlens, pass_only=pass_only)
            if region is None:
                chroms = seqlens.keys()
            else:
                # only process this chromosome
                # region coords will be implemented (TODO)
                chroms = [region]
            res = p.map(runner, chroms)

        # out out of map(), results are a list, so we merge them all
        all_res = {k: v for d[0] in res for k, v in d.items()}
        all_stats = {k: v for d[1] in res for k, v in d.items()}
        all_filters = {k: v for d[2] in res for k, v in d.items()}
    else:
        if outbed is not None:
            if outbed.endswith('gz'):
                outbed = gzip.open(outbed, 'wt')
            else:
                outbed = open(outbed, 'w')
        if outvcf is not None:
            # mode is inferred from filename
            outvcf = Writer(outvcf, VCF(vcf_file))
        all_res, all_stats, all_filters = calc_stats(region=None, vcf_file=vcf_file, width=width,
                                                     seqlens=seqlens, pass_only=pass_only,
                                                     outvcf=outvcf, outbed=outbed)
    write_table(outfile, all_res)
    return all_res, all_stats, all_filters


if __name__ == "__main__":
    all_res, all_stats, all_filters = main()
