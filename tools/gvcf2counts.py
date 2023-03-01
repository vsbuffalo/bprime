"""
GVCF (per-basepair VCF) to ref/alt counts. The goal of
using a GVCF is that the denominator is computed alongside
the allele count data, since all bases undergo the same 
filtering steps.

This tool reads a GVCF line by line, filtering out sites that 
don't match simple criteria, and outputs a numpy .npz file 
full of ref/alt allele counts per chromosome. This uses cyvcf2, 
which marks all genotypes as 0/1/2/3 for hom ref / het / hom alt / 
unknown.

Variant sites that are not biallelic or SNP variation (e.g. 
indels) are excluded, leaving zeros at ref/alt.
"""

import os
import re
import warnings
import sys
import gzip
from collections import deque, Counter, defaultdict, namedtuple
from functools import partial
import numpy as np
from cyvcf2 import VCF
import allel as al
import tqdm
import defopt

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


def main(gvcf: str, *, outdir: str, samples: str=None, 
         pass_only: bool=True, min_qual: int=50, min_gq: int=30,
         snps_only: bool=True, statsfile: str=None):
    """
    Convert a GVCF into a numpy file of ref/alt counts.

    :param gvcf: GVCF file.
    :param outdir: the output directory for chromosome ref/alt allele count data in .npz.
    :param samples: a list of samples to include genotypes from.
    :param pass_only: only keep variants with PASS in the filter column.
    :param min_qual: minimum QUAL threshold (< min_qual are removed).
    :param min_gq: minimum genotype quality threshold (< min-gq are removed).
    :param statsfile: output file for statistics (default: to standard out).


    This will a multiple-chromosome GVCF into separate .npz files in outdir/.
    Note that it may be faster to run chromosomes in parallel.

    Note: the default QUAL and GQ filtering values are based on this study:
    https://www.nature.com/articles/nature13673.
    """
    if samples is not None:
        samples = [l.strip() for l in open(samples)]
        print(f"subsetting to only include {len(samples)} samples...")
    vcf = VCF(gvcf, gts012=True, samples=samples)
    seqlens = get_seqlens(vcf.raw_header)
    vcf_samples = vcf.samples
    if samples is not None:
        if set(vcf_samples) != set(samples):
            diff = set(samples) - set(vcf_samples)
            warnings.warn(f"requested samples not in the VCF: {diff}! total: {len(vcf_samples)}")

    ntot = 0
    stats = np.zeros(4)
    FILTER, QUAL, NOT_SNP, NOT_BI = 0, 1, 2, 3
    stats_names = {0: 'failed FILTER', 
                   1: f'failed QUAL (<{min_qual})',
                   2: f'not SNP', 
                   3: f'not biallelic'}
    trns, trvs = 0, 0

    last_chrom = None
    for var in vcf:
        ntot += 1
        if var.CHROM != last_chrom:
            if last_chrom is not None:
                # write the current chromosome counts table
                print(f"writing {last_chrom}...")
                filepath = os.path.join(outdir, f"{last_chrom}_counts.npy")
                np.save(filepath, counts)
            # create a new counts table
            pbar = tqdm.tqdm(total = seqlens[var.CHROM], unit_scale=True)
            counts = np.zeros((seqlens[var.CHROM], 2), dtype=np.short)
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
        is_poly = nhom_alt + nhet > 0
        if is_poly and not var.is_snp:
            # drop polysites that are not SNPs
            stats[NOT_SNP] += 1
            continue
        if is_poly and len(var.ALT) > 1:
            stats[NOT_BI] += 1
            continue
        # this is slower: 7k it/sec vs 9k for np
        #gt_counts = Counter(gts)
        #nhom_ref, nhom_alt, nhet = [gt_counts[k] for k in (0, 2, 1)]
        qual_pass = int(var.QUAL is not None and var.QUAL >= min_qual)
        # NOTE: QUAL tells us the confidence that there is some variation at a site.
        # If we observe some homozygous ref and alt homs/hets, but we *fail* at 
        # QUAL, we *only count* the homozygous genotypes to the nref.
        nref = 2*nhom_ref + nhet*qual_pass
        nalt = (2*nhom_alt + nhet)*qual_pass
        # we only keep stats for the sites that have a ALT-including genotype
        # that we are rejecting based on QUAL
        stats[QUAL] += not qual_pass and is_poly > 0
        assert nref + 1 < COUNT_MAX, "counts dtype overflow!"
        assert nalt + 1 < COUNT_MAX, "counts dtype overflow!"
        i = var.POS
        counts[i, 0] = nref
        counts[i, 1] = nalt
        if is_poly:
            trns += var.is_transition
            trvs += not var.is_transition
        pbar.update(i - pbar.n)
        ncomplete += (counts[i, :] > 0).sum()
        #if ncomplete % 1000000 == 0 and trvs > 0:
        #    print(f"transition/transversion = {trns/trvs}")

    # save the last chromosome
    filepath = os.path.join(outdir, f"{last_chrom}_counts.npy")
    np.save(filepath, counts)
    vcf.close()

    statsfile = sys.stdout if statsfile is None else open(statsfile, 'w') 
    frac = lambda x: np.round(100 * x/ntot, 2)
    statsfile.write(f"total\t{ntot}\t{frac(ntot)}\n")
    for i, msg in stats_names.items():
        statsfile.write(f"{msg}\t{stats[i]}\t{frac(stats[i])}\n")
    statsfile.write(f"tn/tv\t{trns}/{trvs}\t{trns/trvs}\n")


if __name__ == "__main__":
    defopt.run(main)
