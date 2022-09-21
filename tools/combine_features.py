import sys
import click
from bgspy.utils import combine_features, read_bed3, load_seqlens
from bgspy.utils import masks_to_ranges

PRIORITY = ('phastcons', 'cds', 'intron', 'utr', 'promoter')

@click.command()
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option("--bed", type=(str, str), multiple=True)
def main(seqlens, bed):
    bed = dict(bed)
    assert all([k in PRIORITY for k in bed.keys()]), f"keys must be in {PRIORITY}"
    beds = {c: read_bed3(f) for c, f in bed.items()}
    masks = combine_features(beds, PRIORITY, load_seqlens(seqlens))
    res = masks_to_ranges(masks, PRIORITY)
    for chrom, ranges in res.items():
        for range in ranges:
            print(f"{chrom}\t{range[0]}\t{range[1]}\t{range[2]}")

if __name__ == "__main__":
    main()

