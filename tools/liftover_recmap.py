"""
Liftover the recombination maps from Hinch et al. (2011).

These maps only have cumulative positions. From looking 
at the data, the format is:

    chrom    physical position (inclusive)     cumulative map distance from end

This differs from the more standard HapMap recombination map format,
in that there are no per-basepair rates.

It's terribly easy to have off-by-one errors here, so I've 
included my notes. 


   c0'     c1        c2       c3        c4'
    |       |         |        |.........|
   p0      p1        p2       p2         p4

The apostrophes are auxiliary data points added so the 
positions go to the full chromosome (0 indexed, right exclusive).

These are added in when the cumulative map is read in. 
For these two auxiliary points, the first is given a cumulative map
position of 0 (correct), and the last of c_i where i is the number of 
rows (e.g. the last cumulative map position in the data).

If the number of marginal rates is L (e.g. L = 4 above) then 
the number of positions (including 0 and the end) is L+1. The 
number of spans is L.


For C = [0, c_1, c_2, ..., c_L] (L+1 items)

If Δx = x_{i+1} - x_i, then

ΔC = [c_1, c_2 - c_1, c_3 - c_2, ..., c_L-c_{L-1}]

is the L-item marginal rate vector, and

ΔP = [p_1, p_2 - p_1, ..., p_L - p_{L-1}]

is the L-item span vector. Then, 

the rate vector C and P is, 

r = ΔC/ΔP

Note that this requires that all coordinates are 0-indexed, and
have [start, end) ranges.

"""


import defopt
import numpy as np
import tempfile
import subprocess
import pandas as pd


def read_cumulative(file, seqlens, end_inclusive=True):
    """
    """
    d = pd.read_table(file, names=('chrom', 'pos', 'cumrate'))
    new_rates = dict()
    for chrom, df in d.groupby('chrom'):
        pos = df['pos'].tolist()
        cumrates = df['cumrate'].tolist()
        if pos[0] != 0:
            assert pos[0] > 0  # something is horribly wrong
            pos.insert(0, 0)
            # with cumulative data this must be the case
            cumrates.insert(0, 0)
        if pos[-1] != seqlens[chrom]:
            # we need to add the last row
            pos.append(seqlens[chrom])
            # we add in the last rate to the end, which is 
            # assuming there is no recombination between the 
            # last marker and the end of the chromosome. It
            # is technically missing data, but adding an nan
            # just causes more hassle downstream
            cumrates.append(cumrates[-1])

        starts = np.array(pos[:-1])
        # note: some maps could have the end be inclusive
        ends = np.array(pos[1:]) + int(end_inclusive)
        # since we added on the start and end, start=0, 
        # number of cum. rates is L+1
        cumrates = np.array(cumrates)
        pos = np.array(pos)
        L = len(cumrates)-1
        assert L == len(starts)
        assert L == len(ends)
        assert len(cumrates) == len(pos)
        spans = np.diff(pos)
        rates = np.diff(cumrates) / spans
        new_rates[chrom] = (pos, rates, cumrates)
    return new_rates


def cumulative_to_rates(cumulative, pos):
    """
    Given end-to-end positions and cumulative 
    rates, compute the marginal per-basepair rates.
    """
    assert len(cumulative) == len(pos)
    spans = np.diff(pos)
    rates = np.diff(cumulative)
    return rates / spans


def rates_to_cumulative(rates, pos):
    """
    Given end-to-end positions are marginal (per-basepair rates)
    compute the cumulative map distance at each position.
    """
    spans = np.diff(pos)
    return np.array([0] + np.cumsum(rates * spans).tolist())


def write_bed(file, rates_dict):
    with open(file, 'w') as f:
        for chrom, data in rates_dict.items():
            pos, rates = data
            starts, ends = pos[:-1], pos[1:]
            rates = rates[:-1]
            for s, e, r in zip(starts, ends, rates):
                f.write(f"{chrom}\t{s}\t{e}\t{r}\n")


def read_bed(file):
    d = pd.read_table(file, names=('chrom', 'start', 'end', 'rate'))
    out = dict()
    for chrom, df in d.groupby('chrom'):
        starts, ends, rates = df.start.values, df.end.values, df.rate.values
        out[chrom] = (starts, ends, rates)
    return out


def run_liftover(oldmap, chain, minmatch=0.99):
    newmap = tempfile.NamedTemporaryFile().name
    unmap = oldmap.replace('.bed', '') + "_unmapped.bed"
    cmd = ["liftOver", f"-minMatch={minmatch}", oldmap, chain, newmap, unmap]
    subprocess.run(cmd)
    liftover = read_bed(newmap)
    return liftover


def sort_bed(rate_dict):    
    for chrom, data in rate_dict.items():
        starts, ends, rates = data
        idx = np.lexsort((starts, ends))
        rate_dict[chrom] = (starts[idx], ends[idx], rates[idx])
    return rate_dict



