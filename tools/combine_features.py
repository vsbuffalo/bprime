import sys
import click
from collections import defaultdict
from bgspy.utils import combine_features, read_bed3, load_seqlens
from bgspy.utils import masks_to_ranges, readfile

# PRIORITY = ['phastcons', 'cds', 'intron', 'utr', 'promoter']

# lower priority labels are filled first
# stuff like phastcons is sort of a catch-all - we care about
# finer-grained labels
# NOTE: a lot commented out; these can be tried again later, but likely 
# have identifiability issues due to collinearity
PRIORITY = [#'CTCF_binding_site', 'TF_binding_site',
            #'binding_site',
            # 'promoter', 'enhancer', 
            'utr',
            'cds', 'gene', 'intron',
            'phastcons',
            #'open_chromatin_region'
            ]

@click.command()
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option("--bed", type=(str, str), multiple=True, help="feature type, bed file pair")
@click.option("--bedfile", help="single bed file with multiple features", type=str)
def main(seqlens, bed, bedfile=None):
    bed = dict(bed)
    assert all([k in PRIORITY for k in bed.keys()]), f"keys ({list(bed.keys())}) must be in {PRIORITY}"
    beds = {c: read_bed3(f) for c, f in bed.items()}

    # load other bed with many features (e.g. for encode)
    if bedfile is not None:
        for line in readfile(bedfile):
            cols = line.strip().split('\t')
            assert(len(cols) >= 3)
            chrom, start, end, feature = cols[:4]
            assert feature in PRIORITY, f"feature '{feature}' not in PRIORITY!"
            if feature not in beds:
                beds[feature] = defaultdict(list)
            beds[feature][chrom].append((int(start), int(end)))
    masks = combine_features(beds, PRIORITY, load_seqlens(seqlens))
    res = masks_to_ranges(masks, labels=PRIORITY)
    # import pdb;pdb.set_trace()
    for chrom, ranges in res.items():
        for range in ranges:
            print(f"{chrom}\t{range[0]}\t{range[1]}\t{range[2]}")

if __name__ == "__main__":
    main()


