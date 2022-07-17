import sys
import gzip
import os
import click
import numpy as np
import collections
import tskit
import msprime
import pyslim
from bgspy.recmap import RecMap
from bgspy.utils import load_seqlens

def load_neutregions(file, rate, seqlen):
    chroms = set()
    positions, rates = [], []
    last_end = None
    first = True
    with open(file) as f:
        for line in f:
            chrom, start, end = line.strip().split('\t')
            chroms.add(chrom)
            start, end = int(start), int(end)
            if first:
                positions.extend((start, end))
                rates.append(rate)
                first = False
                continue
            positions.append(start)
            rates.append(0)
            positions.append(end)
            rates.append(rate)
            last_end = end

    assert(len(chroms) == 1)
    if end < seqlen:
        positions.append(seqlen)
        rates.append(0)
    if positions[0] != 0:
        positions.insert(0, 0)
        rates.insert(0, 0)
    ratemap =  msprime.RateMap(position=positions, rate=rates)
    ratemap.chrom = list(chroms)[0]
    return ratemap

def write_rate_map(ratemap):
    """
    Pretty print the RateMap as ranges.
    """
    pos, rate = ratemap.position, ratemap.rate
    ranges = np.stack((pos[:-1], pos[1:])).T.astype('int')
    with open('ratemap.tsv', 'w') as f:
        for i in range(ranges.shape[0]):
            f.write(f"{ranges[i, 0]}\t{ranges[i, 1]}\t{rate[i]}\n")


# the following two functions are from:
# https://github.com/tskit-dev/tskit/issues/504
def count_site_alleles(ts, tree, site):
    counts = collections.Counter({site.ancestral_state: ts.num_samples})
    for m in site.mutations:
        current_state = site.ancestral_state
        if m.parent != tskit.NULL:
            current_state = ts.mutation(m.parent).derived_state
        # Silent mutations do nothing
        if current_state != m.derived_state:
            num_samples = tree.num_samples(m.node)
            counts[m.derived_state] += num_samples
            counts[current_state] -= num_samples
    return counts

def count_ancestral(ts):
    num_ancestral = np.zeros(ts.num_sites, dtype=int)
    positions = []
    for tree in ts.trees():
        for site in tree.sites():
            positions.append(int(site.position))
            counts = count_site_alleles(ts, tree, site)
            num_ancestral[site.id] = counts[site.ancestral_state]
    return positions, num_ancestral

def serialize_metadata(md):
    return ';'.join([f"{k}={v[0]}" for k, v in md.items()])


@click.command()
@click.argument('treefile')
@click.option('--chrom', required=True, help="output DAC file")
@click.option('--outfile', default=None, help="output DAC file")
@click.option('--regions', required=True,
              help="BED track of regions to drop mutations onto")
@click.option('--recmap', required=True, help="BED recombination map")
@click.option('--mu', default=1.5e-8, help="mutation rate")
@click.option('--seed', default=None, help="random seed")
def treeseq2dac(treefile, chrom, outfile, regions, recmap, mu, seed=None):
    """
    Take a tree seequence file, recapitate, and overlay mutations.
    """
    ts = pyslim.load(treefile)
    rm = RecMap(recmap, seqlens={chrom: ts.sequence_length})
    rp = rm.rates[chrom]
    ends, rates = rp.end, rp.rate
    rates[0] = 0 # change the nan
    recmap = msprime.RecombinationMap(ends, rates)
    md = ts.metadata['SLiM']['user_metadata']
    N = md['N'][0]
    rts = ts.recapitate(recombination_map=recmap, Ne=N, random_seed=seed)

    region_length = ts.sequence_length
    ratemap = load_neutregions(regions, mu, region_length)
    # for debugging; TODO comment out
    #write_rate_map(ratemap)
    rts = rts.delete_sites([m.site for m in rts.mutations()])
    ts = msprime.sim_mutations(rts, rate=ratemap, discrete_genome=True)
    #__import__('pdb').set_trace()
    if outfile is None:
        outfile = treefile.replace('_treeseq.tree', '_dac.tsv.gz')

    chrom = ratemap.chrom
    pos, nanc = count_ancestral(ts)
    nderv = 2*N - nanc
    with gzip.open(outfile, 'wt') as f:
        f.write("#"+serialize_metadata(md)+"\n")
        for i in range(len(pos)):
            if nderv[i] == 2*N:
                continue
            row = [chrom, pos[i], 2*N, nderv[i]]
            f.write("\t".join(map(str, row)) + "\n")




if __name__ == "__main__":
    treeseq2dac()
