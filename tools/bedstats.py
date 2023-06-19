import csv
import defopt
from collections import Counter
from bgspy.utils import load_seqlens, readfile


def count_features(bedfile, seqlens):
    basepair_counts = {}
    seen_chroms = set()
    for line in readfile(bedfile):
        row = line.strip().split('\t')
        seen_chroms.add(row[0])
        feature = row[3] if len(row) == 4 else '.'
        if feature not in basepair_counts:
            basepair_counts[feature] = 0
        basepair_counts[feature] += int(row[2]) - int(row[1])
    return basepair_counts, seen_chroms


def main(bedfile: str, genomefile: str):
    """
    :param bedfile: BED file
    :param genomefile: TSV of chromosomes and their lengths
    """
    seqlens = load_seqlens(genomefile)
    basepair_counts, seen_chroms = count_features(bedfile, genomefile)
    genome_size = sum([l for c, l in seqlens.items() if c in seen_chroms])
    percentages = {}
    for feature, count in basepair_counts.items():
        percentages[feature] = count / genome_size * 100
    for feature, percent in percentages.items():
        print(f"{feature}\t{round(percent, 4)}%")


if __name__ == "__main__":
    defopt.run(main)



