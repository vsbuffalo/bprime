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
def main(vcf_file, outdir, samples=None):
    vcf = VCF(vcf_file, gts012=True)
    seqlens = get_seqlens(vcf.raw_header)
    vcf_samples = vcf.samples

    subset_samples = samples is not None
    if subset_samples:
        keep_sample_idx = [i for i, sample in enumerate(vcf_samples) if sample in samples]

    last_chrom = None
    for var in vcf:
        if var.CHROM != last_chrom:
            if last_chrom is not None:
                # write the current chromosome counts table
                print(f"writing {last_chrom}...")
                filepath = os.path.join(outdir, f"{last_chrom}_counts.npy")
                np.save(filepath, counts)
            # create a new counts table
            pbar = tqdm.tqdm(total = seqlens[var.CHROM])
            counts = np.full((seqlens[var.CHROM], 2), -1, dtype=np.short)
            ncomplete = 0
            last_chrom = var.CHROM

        gts = var.gt_types
        if subset_samples:
            gts = gts[keep_sample_idx]
        #gt_counts = Counter()
        #gt_counts.update(zip(*[y.tolist() for y in np.unique(gts, return_counts=True)]))
        nhom_ref = np.sum(gts == 0)
        nhom_alt = np.sum(gts == 2)
        nhet = np.sum(gts == 1)
        # this is slower: 7k it/sec vs 9k for np
        #gt_counts = Counter(gts)
        #nhom_ref, nhom_alt, nhet = [gt_counts[k] for k in (0, 2, 1)]
        assert 2*nhom_ref + nhet + 1 < COUNT_MAX, "counts dtype overflow!"
        assert 2*nhom_alt + nhet + 1 < COUNT_MAX, "counts dtype overflow!"
        i = var.POS
        pbar.update(1)
        counts[i, 0] = 2*nhom_ref + nhet
        counts[i, 1] = 2*nhom_alt + nhet
        ncomplete += counts[i, :].sum()

    vcf.close()


if __name__ == "__main__":
    main()
