import os
import re
import sys
import gzip
from collections import deque, Counter, defaultdict, namedtuple
from functools import partial
import numpy as np
from cyvcf2 import VCF
import allel as al
import tqdm
import click

SEQLEN_RE = re.compile(r'##contig=<ID=(?P<seqname>\w+),length=(?P<length>\d+)>')
REGION_RE = re.compile(r'(?P<seqname>\w+)(:(?P<start>\d+)-(?P<end>\d+))?')

COUNT_DTYPE = np.short
COUNT_MAX = np.iinfo(COUNT_DTYPE).max

def get_seqlens(header_str):
    seqlens = dict()
    for line in header_str.split('\n'):
        mtch = SEQLEN_RE.match(line)
        if mtch is None:
            continue
        groups = mtch.groupdict()
        seqlens[groups['seqname']] = int(groups['length'])
    return seqlens

@click.command()
@click.argument('vcf_file')
@click.option('--outdir', type=click.Path(exists=True), required=True, help="output directory")
@click.option('--samples', default=None,
              help="comma separated list of samples to include")
@click.option('--pass-only', default=True, is_flag=True, help="only keep variants with the PASS in the filter column")
@click.option('--min-qual', default=50, help="QUAL threshold (< min-qual are removed)")
@click.option('--min-gq', default=30, help="genotype quality (GQ) threshold (< min-qual are removed)")
def main(vcf_file, outdir, samples=None, pass_only=True, min_qual=50, min_gq=30):
    """
    Convert a GVCF into a numpy file of ref/alt counts.

    Note: the default QUAL and GQ filtering values are based on this study:
    https://www.nature.com/articles/nature13673.
    """
    vcf = VCF(vcf_file, gts012=True, samples=samples)
    seqlens = get_seqlens(vcf.raw_header)
    vcf_samples = vcf.samples

    stats = np.zeros(2)
    FILTER, QUAL = 0, 1

    last_chrom = None
    for var in vcf:
        if var.CHROM != last_chrom:
            if last_chrom is not None:
                # write the current chromosome counts table
                print(f"writing {last_chrom}...")
                filepath = os.path.join(outdir, f"{last_chrom}_counts.npy")
                np.save(filepath, counts)
            # create a new counts table
            pbar = tqdm.tqdm(total = seqlens[var.CHROM], unit_scale=True)
            counts = np.full((seqlens[var.CHROM], 2), -1, dtype=np.short)
            ncomplete = 0
            last_chrom = var.CHROM

        # site-specific filtering
        if pass_only and var.FILTER is not None:
            # cyvcf uses None to indicate pass (i.e. no filter)<Paste>
            stats[FILTER] += 1
            continue

        # for invariant sites, RGQ is used instead of GQ
        if "RGQ" in var.FORMAT:
            gt_quals = var.format('RGQ').squeeze()
        else:
            gt_quals = var.gt_quals
        pass_gq = gt_quals >= min_gq
        gts = var.gt_types[pass_gq]
        #gt_counts = Counter()
        #gt_counts.update(zip(*[y.tolist() for y in np.unique(gts, return_counts=True)]))
        nhom_ref = np.sum(gts == 0)
        nhom_alt = np.sum(gts == 2)
        nhet = np.sum(gts == 1)
        # this is slower: 7k it/sec vs 9k for np
        #gt_counts = Counter(gts)
        #nhom_ref, nhom_alt, nhet = [gt_counts[k] for k in (0, 2, 1)]
        qual_pass = var.QUAL is not None and var.QUAL >= min_qual
        nref = 2*nhom_ref + nhet*qual_pass
        nalt = (2*nhom_alt + nhet)*qual_pass
        stats[QUAL] += ~ qual_pass
        assert nref + 1 < COUNT_MAX, "counts dtype overflow!"
        assert nalt + 1 < COUNT_MAX, "counts dtype overflow!"
        i = var.POS
        counts[i, 0] = nref
        counts[i, 1] = nalt
        pbar.update(i - pbar.n)
        ncomplete += (counts[i, :] > -1).sum()

    # save the last chromosome
    filepath = os.path.join(outdir, f"{last_chrom}_counts.npy")
    np.save(filepath, counts)

    print(stats)
    vcf.close()


if __name__ == "__main__":
    main()
