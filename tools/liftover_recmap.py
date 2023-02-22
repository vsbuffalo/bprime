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
from scipy.interpolate import interp1d
import os
from pathlib import Path
import tempfile
import subprocess
import pandas as pd

# next two funcs are lifted from bgspy
# so this is stand alone
def readfile(filename):
    is_gzip = filename.endswith('.gz')
    if is_gzip:
        return gzip.open(filename, mode='rt')
    return open(filename, mode='r')


def load_seqlens(file):
    seqlens = dict()
    params = []
    with readfile(file) as f:
        for line in f:
            if line.startswith('#'):
                params.append(line.strip().lstrip('#'))
                continue
            chrom, end = line.strip().split('\t')
            seqlens[chrom] = int(end)
    return seqlens


def read_cumulative(file, seqlens, is_hapmap=False,
                    end_inclusive=True):
    """
    """
    if is_hapmap:
        hapmap_cols = ('chrom', 'pos', 'rate', 'cumrate')
        d = pd.read_table(file, names=hapmap_cols)
        end_inclusive = False
    else:
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
        msg = "rates are not cumulative for {chrom}"
        assert np.all(np.diff(cumrates) >= 0), msg
        pos = np.array(pos)
        L = len(cumrates)-1
        assert L == len(starts)
        assert L == len(ends)
        assert len(cumrates) == len(pos)
        spans = np.diff(pos)
        rates = np.diff(cumrates) / spans
        #new_rates[chrom] = (pos, rates, cumrates)
        new_rates[chrom] = (pos, cumrates)
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


def read_cumulative_bed(file):
    """
    Read a BED written by write_bed, full of cumulative rates.
    This assumes this goes to the end of the chromosome.
    """
    d = pd.read_table(file, names=('chrom', 'start', 'end', 'rate'))
    rates_dict = dict()
    for chrom, df in d.groupby('chrom'):
        starts, ends = df.start.tolist(), df.end.tolist()
        rates = df.rate.tolist()
        pos = [starts[0]] + ends
        # append on final cumulative rate
        rates = rates + [rates[-1]]
        rates_dict[chrom] = (pos, rates)
    return rates_dict


def run_liftover(oldmap, chain, minmatch=0.99):
    with tempfile.NamedTemporaryFile() as fp:
        newmap = fp.name
        unmap = oldmap.replace('.bed', '') + "_unmapped.bed"
        cmd = ["liftOver", f"-minMatch={minmatch}", oldmap,
               chain, newmap, unmap]
        subprocess.run(cmd)
        liftover = read_cumulative_bed(newmap)
    return liftover


def sort_rates(rate_dict):
    for chrom, data in rate_dict.items():
        pos, rates = data
        pos, rates = np.array(pos), np.array(rates)
        idx = np.argsort(pos)
        rate_dict[chrom] = (pos[idx], rates[idx])
    return rate_dict


def write_hapmap(rates_dict, file, is_cumulative=True):
    """
    Write a HapMap-formatted file.
    """
    assert is_cumulative, "not implemented"
    with open(file, 'w') as f:
        for chrom, data in rates_dict.items():
            pos, cumrates = data
            rates = cumulative_to_rates(cumrates, pos)
            for e, r, cr in zip(pos, rates, cumrates):
                f.write(f"{chrom}\t{e}\t{r}\t{cr}\n")


def remove_nonmontonic(rate_dict):
    """
    Clean a recombination map (cumulative) by removing
    non-monotonic points.

    Example: 
      0 1 1 3 4 1 5 4 7
                x   x
    """
    for chrom, data in rate_dict.items():
        pos, rates = data

        assert np.all(np.diff(pos) >= 0), "positions are not sorted!"

        keep_idx = np.where(np.diff(rates) >= 0)[0] + 1
        pos, rates = pos[keep_idx], rates[keep_idx]

    return rate_dict


def generate_interpolators(rates_dict):
    interps = dict()
    for chrom, data in rates_dict.items():
        pos, rates = data
        interps[chrom] = interp1d(pos, rates)
    return interps

def validation_plots(liftback_map, old_map, dir):
    """
    """
    # we interpolate the new map lifted back to old
    # coordinates
    interps = generate_interpolators(liftback_map)

    if not os.path.exists(dir):
        os.markdirs(dir)

    fig, ax = plt.subplots()
    for i, (chrom, data) in enumerate(old_map.items()):
        pos, rates = data
        pred = interps[chrom](pos)
        ax.scatter(rates, pred)

    fig.savefig(os.path.join(dir, 'diagnostic.pdf'))


def read_hapmap(file, seqlens):
    """
    Read a hapmap file into a 
    """


def main(*, mapfile: str, genome: str, 
         outfile: str,
         chain_to: str, chain_from: str):
    """
    Take a cumulative recombination map file and lift it over
    using the --chain-to. The --chain-from option is for 
    validation.

    NOTE: this will overwrite <mapfile>.bed if it exists!

    :param mapfile: the input TSV file of chrom, position, 
                     and cumulative map position.
    :param chain_to: chain file from the genome of the mapfile
                      to the new genome.
    :param chain_from: chain file from the new genome back to the
                       initial genome, for validation.
    """
    sl = load_seqlens(genome)
    oldmap = read_cumulative(mapfile, sl)
    path = mapfile.replace('.tsv', '').replace('.txt', '')
    #with tempfile.NamedTemporaryFile() as fp:
    with open(path + '.bed', 'w') as fp:
        mapfile_bed = fp.name

        # write the cumulative map to a temporary BED file
        write_bed(mapfile_bed, oldmap)

        # run liftover
        new_map = run_liftover(mapfile_bed, chain_to)

        # drop all markers that mapped to chromosomes not 
        # in the original map
        chroms = list(new_map.keys())
        for chrom in chroms:
            if chrom not in oldmap:
                del new_map[chrom]

        # clean up the lifted over rates, first sorting
        new_map_sorted = sort_rates(new_map)

        # then removing non-monotonic
        clean_new_map = remove_nonmontonic(new_map)

        # save to hapmap -- this is the final outfile
        write_hapmap(clean_new_map, outfile)

    # validation, by lifting back
    new_map = read_cumulative(outfile, sl, is_hapmap=True)
    #with tempfile.NamedTemporaryFile() as fp:
    with open(path + '_tmp.bed', 'w') as fp:
        new_mapfile_bed = fp.name

        # output the BED version
        write_bed(new_mapfile_bed, new_map)

        # lift back
        liftback_map = run_liftover(new_mapfile_bed, chain_from)

        # create validation plots
        valid_dir = os.path.join(path, "validation_plots")
        validation_plots(liftback_map, oldmap, valid_dir)


if __name__ == "__main__":
    defopt.run(main)
