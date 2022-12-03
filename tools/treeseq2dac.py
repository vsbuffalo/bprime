# DEPRECATED: use GenomeData.load_counts_from_ts()
import sys
import gzip
import os
import click
import numpy as np
import tskit
import msprime
import pyslim
from bgspy.recmap import RecMap
from bgspy.utils import load_seqlens
from bgspy.sim_utils import count_ancestral

def write_rate_map(ratemap):
    """
    Pretty print the RateMap as ranges.
    """
    pos, rate = ratemap.position, ratemap.rate
    ranges = np.stack((pos[:-1], pos[1:])).T.astype('int')
    with open('ratemap.tsv', 'w') as f:
        for i in range(ranges.shape[0]):
            f.write(f"{ranges[i, 0]}\t{ranges[i, 1]}\t{rate[i]}\n")


@click.command()
@click.argument('treefile')
@click.option('--chrom', required=True, help="output DAC file")
@click.option('--outfile', default=None, help="output DAC file")
@click.option('--recmap', required=True, help="BED recombination map")
@click.option('--mu', default=1.5e-8, help="mutation rate")
@click.option('--seed', default=None, help="random seed")
def treeseq2dac(treefile, chrom, outfile, recmap, mu, seed=None):
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
    # for debugging; TODO comment out
    #write_rate_map(ratemap)
    # remove the selected sites -- we might at one point want to see
    # whether including these messes up inference
    rts = rts.delete_sites([m.site for m in rts.mutations()])
    ts = msprime.sim_mutations(rts, rate=mu, discrete_genome=True)
    #__import__('pdb').set_trace()
    if outfile is None:
        outfile = treefile.replace('_treeseq.tree', '_dac.tsv.gz')

    g = GenomeData()
    g.load_counts_from_ts(ts)
    g.metadata = md
    g.counts_to_tsv(outfile)


if __name__ == "__main__":
    treeseq2dac()
