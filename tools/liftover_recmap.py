"""
liftover_recmap

This was made particularly for lifting over the Hinch et al. 
recombination map to hg38. This map and others do not follow
the HapMap recombination map format, so there is a subcommand to
convert these. 

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

NOTE: throughout, this version uses 'rates' to mean both cumulative
and per-basepair rate. This is unclear and should be corrected in 
future versions (TODO).
"""

import logging
import pickle
import defopt
import itertools
from math import ceil
import numpy as np
from scipy.interpolate import interp1d
import os
import gzip
import matplotlib.pyplot as plt
import tempfile
import subprocess
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# next two funcs are lifted from bgspy
# so this is stand alone
def readfile(filename):
    is_gzip = filename.endswith('.gz')
    if is_gzip:
        return gzip.open(filename, mode='rt')
    return open(filename, mode='r')


def check_file(file):
    assert os.path.isfile(file), f"{file} does not exist"


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


def assert_cumulative(cumrates):
    msg = "rates are not cumulative for {chrom}"
    assert np.all(np.diff(cumrates) >= 0), msg


def read_cumulative(file, seqlens, end_inclusive=True):
    """
    Certain recombination maps are in a tab-delimited
    cumulative format, with rows like

        chrom	basepair	cumulative_rate

    where `cumulative_rate` is the rate in centiMorgans
    from position = 0 to `basepair` (which is inclusive
    if `end_inclusive=True`.

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
        assert_cumulative(cumrates)
        pos = np.array(pos)
        L = len(cumrates)-1
        assert L == len(starts)
        assert L == len(ends)
        assert len(cumrates) == len(pos)
        spans = np.diff(pos)
        # these would be in centiMorgans per basepair
        # so we convert them to cM/Mb
        rates = np.diff(cumrates) / (spans / 1e6)
        #new_rates[chrom] = (pos, rates, cumrates)
        new_rates[chrom] = (pos, cumrates)
    return new_rates


def cumulative_to_rates(cumulative, pos):
    """
    Given end-to-end positions and cumulative 
    rates, compute the marginal per-basepair rates.

    NOTE: assumes cumulative in cM.
    """
    assert len(cumulative) == len(pos)
    spans = np.diff(pos)
    rates = np.diff(cumulative)
    return rates / (spans / 1e6)


def rates_to_cumulative(rates, pos):
    """
    Given end-to-end positions are marginal (per-basepair rates)
    compute the cumulative map distance at each position.

    NOTE: assumes rates in cM/Mb.
    """
    assert len(rates) == len(pos)-1
    spans = np.diff(pos) / 1e6  # convert to Mb
    return np.array([0] + np.cumsum(rates * spans).tolist())


def write_bed(file, rates_dict):
    with open(file, 'w') as f:
        for chrom, data in rates_dict.items():
            pos, rates = data
            #starts, ends = pos[:-1], pos[1:]
            starts, ends = pos[1:], pos[1:]+1
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
        logging.info(f"running liftover command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        liftover = read_cumulative_bed(newmap)
        # now we quantify the unmapped
        unmapped = 0
        if os.path.exists(unmap):
            with open(unmap) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    else:
                        unmapped += 1
        else:
            logging.info("no liftOver unmapped file")
        total = len([l for l in open(oldmap)])
        up = np.round(unmapped / total, 2)
        msg = f"{unmapped} ({up}% BED entries unmapped during liftover"
        logging.info(msg)
    return liftover


def sort_chroms(chromosomes):
    chromosomes = list(chromosomes)
    END = 1e6
    def chromosome_key(chrom):
        chrom_number = chrom.replace('chr', '')
        if chrom_number.isdigit():
            assert int(chrom_number) < END
            return int(chrom_number)
        else:
            return {'X': END, 'Y': END+1, 'Mt': END+1}[chrom_number]
    chromosomes.sort(key=chromosome_key)
    return chromosomes

 
def sort_rates(rate_dict):
    for chrom, data in rate_dict.items():
        pos, rates = data
        pos, rates = np.array(pos), np.array(rates)
        idx = np.argsort(pos)
        rate_dict[chrom] = (pos[idx], rates[idx])
    return rate_dict


def write_hapmap(rates_dict, file, header=True):
    """
    Write a HapMap-formatted file.

    By default, the rate column is in cM/Mb and the map 
    column is in cM.
    """
    with open(file, 'w') as f:
        if header:
            f.write("chrom\tpos\trate_cM_Mb\tmap_cM\n")
        chroms = sort_chroms(rates_dict.keys())
        for chrom in chroms:
            data = rates_dict[chrom]
            pos, cumrates = data
            check_positions_sorted(pos)
            rates = cumulative_to_rates(cumrates, pos)
            # these are in Morgans/basepair
            for e, r, cr in zip(pos, rates, cumrates):
                # this is in cM/bp so covert to cM/Mb
                f.write(f"{chrom}\t{e}\t{r}\t{cr}\n")


def check_positions_sorted(pos):
    msg = "positions are not sorted!"
    assert np.all(np.diff(pos) >= 0), msg


def find_monotonic(pos, rates):
    """
    Filter out the indices of non-monotonically increasing
    cumulative map positions. This is a two-round process.
    First, extreme outliers are removed. Then non-monotonic
    positons are removed.
    """
    check_positions_sorted(pos)

    # First we do isotonic regression. Because 
    # the data are sparse, this amounts to interpolation;
    # any error are mis-mapped markers
    iso = IsotonicRegression().fit(pos, rates)
    err = (iso.predict(pos) - rates)
    bad_markers = (err != 0)
    nb = bad_markers.sum()
    nbp = np.round(100 * nb / len(err), 2)
    return np.where(~bad_markers)[0], np.where(bad_markers)[0]


def rate_plot(rates_dict, alt_rates_dict=None, removed_positions=None,
              outfile=None, marginal_rates=False, sharey=True):
    chroms = sorted(rates_dict.keys(), key=lambda x: x.replace('chr', ''))
    nchroms = len(rates_dict)
    nc, nr = 4, ceil(nchroms / 4)
    fig, ax = plt.subplots(ncols=nc, nrows=nr,
                           figsize=(12, 5), sharex=True, sharey=sharey)

    entries = list(itertools.product(list(range(nc)), list(range(nr))))
    for i, chrom in enumerate(chroms):
        row, col = entries[i]
        fax = ax[col, row]
        pos, rates = rates_dict[chrom]
        if marginal_rates:
            rates = cumulative_to_rates(rates, pos)
            pos = pos[1:]
        fax.plot(pos, rates, linewidth=0.5, zorder=10)

        if alt_rates_dict is not None:
            pos, rates = alt_rates_dict[chrom]
            if marginal_rates:
                rates = cumulative_to_rates(rates, pos)
                pos = pos[1:]
            fax.plot(pos, rates, linewidth=0.5)

        #rm = removed_positions[chrom]
        #fax.scatter(rm, [0]*len(rm), c='r', s=0.1)

        fax.text(0.5, 0.8, chrom, fontsize=6, 
                 horizontalalignment='center',
                 transform=fax.transAxes)

    if outfile is not None:
        plt.tight_layout()
        fig.savefig(outfile)


def clean_map(rate_dict, thresh=None):
    """
    Clean a recombination map (cumulative) by removing
    non-monotonic points.

    Example:
      0 1 1 3 4 1 5 4 7
                x   x
    """
    out_dict = dict()
    removed_positions = dict()
    for chrom, data in rate_dict.items():
        pos, rates = data

        check_positions_sorted(pos)

        keep_idx, remove_idx = find_monotonic(pos, rates)
        remove_pos = pos[remove_idx]
        nrm = len(remove_idx)
        rmp = np.round(100 * nrm / len(rates), 2)
        msg = f"{nrm} ({rmp}%) non-monotonic sites removed from {chrom}"
        logging.info(msg)
        pos, rates = pos[keep_idx], rates[keep_idx]
        if thresh is not None:
            # get the marginal rates and remove outliers
            marginal_rates = cumulative_to_rates(rates, pos)
            # note: we lose the first position here
            outlier_thresh = np.std(marginal_rates) * thresh
            keep_idx = np.where(marginal_rates < outlier_thresh)[0] + 1
            nrm = len(pos) - len(keep_idx)
            rmp = np.round(100 * nrm / len(pos), 2)
            pos, rates = pos[keep_idx], rates[keep_idx]
            msg = (f"{nrm} ({rmp}%) markers removed on {chrom} because per-basepair "
                   f"rates > {thresh}σ (={np.round(outlier_thresh, 5)})")
            logging.info(msg)
        assert_cumulative(rates)
        out_dict[chrom] = (pos, rates)
        removed_positions[chrom] = remove_pos
    return out_dict, removed_positions


def generate_interpolators(rates_dict):
    interps = dict()
    for chrom, data in rates_dict.items():
        pos, rates = data
        interps[chrom] = interp1d(pos, rates, bounds_error=False)
    return interps


def validation_plots(liftback_map, old_map, dir):
    """
    """
    # we interpolate the new map lifted back to old
    # coordinates
    interps = generate_interpolators(liftback_map)

    fig, ax = plt.subplots()
    for i, (chrom, data) in enumerate(old_map.items()):
        pos, rates = data
        pred = interps[chrom](pos)
        ax.scatter(rates, pred)

    fig.savefig(os.path.join(dir, 'diagnostic.pdf'))


def read_hapmap(file, seqlens):
    """
    Read in a standard HapMap recombination file. The format
    looks like,

        chr1    5    1.1
        chr1    6    2.3
        chr1    10   3.2

    with units in cM. Optionally, there is a total map distance
    from position 0 fourth column in cM. The rate column
    in row i gives the rate between the rows i (inclusive)
    and i+1 (exclusive).

    The above example defines the following rate map:

         1.1       2.3         3.2
       [5, 6)    [6, 10)     [10, end)

    Note that unless the last row is the to the end of the 
    chromosome, the end is implied (and thus needs need 
    supplementary chromosome length data. Note that
    the rate at the end is not informative unless the file
    is not to the end but the end is specified elsewhere.

    If the end is specified as 12 in the example above,
    then the ratemap is:

         1.1       2.3         3.2
       [5, 6)    [6, 10)     [10, 12)

    Through this code, we rely on a full rate map like tskit
    which includes zero and the end point.

    Notes: tskit.RateMap.read_hapmap() reads in a four-column
    HapMap-formatted recombination map, and seems to only use
    the total map distance in fourth column.
    """
    is_four_col = len(open(file).readline().split('\t')) == 4
    three_col = ('chrom', 'pos', 'rate')
    four_col = ('chrom', 'pos', 'rate', 'cumrate')
    header = four_col if is_four_col else three_col
    has_header = open(file).readline().lower().startswith('chrom')
    d = pd.read_table(file, skiprows=int(has_header), names=header)
    new_rates = dict()
    new_cumulative = dict()
    for chrom, df in d.groupby('chrom'):
        pos = df['pos'].tolist()
        rates = df['rate'].tolist()
        cumrates = df['cumrate'].tolist()
        if pos[0] != 0:
            assert pos[0] > 0  # something is horribly wrong
            pos.insert(0, 0)
            # with cumulative data this must be the case
            rates.insert(0, 0)
        if pos[-1] != seqlens[chrom]:
            # let's add the end in -- not this already has a rate!
            pos.append(seqlens[chrom])

        starts = np.array(pos[:-1])
        # note: some maps could have the end be inclusive
        ends = np.array(pos[1:])
        # since we added on the start and end, start=0, 
        # number of cumulative rates is L+1
        rates = np.array(rates)
        pos = np.array(pos)
        check_positions_sorted(starts)
        check_positions_sorted(ends)
        L = len(rates)
        assert L == len(starts)
        assert L == len(ends)

        # now we need to convert these to cumulative
        cumulative = rates_to_cumulative(rates, pos)

        # TODO: check that rates and cumulative 
        # map positions all match
        #spans = np.diff(pos)

        new_rates[chrom] = (pos, cumulative)
    return new_rates


def liftover(*, mapfile: str, genome: str,
             outfile: str, chain_to: str,
             thresh: float=None,
             chain_from: str=None):
    """
    Take a cumulative recombination map file and lift it over
    using the --chain-to. The --chain-from option is for 
    validation.

    NOTE: this will overwrite <mapfile>.bed if it exists!

    :param mapfile: the input TSV file of chrom, position, 
                     and cumulative map position.
    :param genome: a TSV of chromosome lengths.
    :param chain_to: chain file from the genome of the mapfile
                      to the new genome.
    :param thresh: markers with a per-basepair rate above
                   thresh x (stderr of chrom) are removed.
    :param chain_from: chain file from the new genome back to the
                       initial genome, for validation. 
    """
    sl = load_seqlens(genome)
    logging.info("reading map")
    oldmap = read_hapmap(mapfile, sl)
    path = mapfile.replace('.tsv', '').replace('.txt', '')

    valid_dir = path + "_validation_plots"
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    #with tempfile.NamedTemporaryFile() as fp:
    with open(path + '.bed', 'w') as fp:
        mapfile_bed = fp.name

        # write the cumulative map to a temporary BED file
        write_bed(mapfile_bed, oldmap)

        # run liftover
        logging.info("lifting over map")
        check_file(chain_to)
        new_map = run_liftover(mapfile_bed, chain_to)

        # drop all markers that mapped to chromosomes not 
        # in the original map
        logging.info("removing markers to other chromosomes")
        chroms = list(new_map.keys())
        for chrom in chroms:
            if chrom not in oldmap:
                del new_map[chrom]

        # clean up the lifted over rates, first sorting
        logging.info("sorting liftover markers")
        chroms = list(new_map.keys())
        new_map_sorted = sort_rates(new_map)

        # then removing non-monotonic
        logging.info("removing non-monotonic cumulative rates")
        clean_new_map, removed_positions = clean_map(new_map_sorted, thresh=thresh)

        # save to hapmap -- this is the final outfile
        logging.info("writing new map")
        write_hapmap(clean_new_map, outfile)

        # output the rate plots for validation
        plot_file = os.path.join(valid_dir, "liftover_cumulative.pdf")
        rate_plot(clean_new_map, oldmap, outfile=plot_file)

        plot_file = os.path.join(valid_dir, "liftover_rates.pdf")
        rate_plot(clean_new_map, oldmap, outfile=plot_file, marginal_rates=True, sharey=False)

    if not chain_from:
        # no liftback validation
        return

    check_file(chain_from)

    # validation, by lifting back
    logging.info("starting validation")
    new_map = read_cumulative(outfile, sl, is_hapmap=True)
    #with open(path + '_tmp.bed', 'w') as fp:
    with tempfile.NamedTemporaryFile() as fp:
        new_mapfile_bed = fp.name

        # output the BED version
        write_bed(new_mapfile_bed, new_map)

        # lift back
        logging.info("lifting recombinaton back to original coordinates")
        liftback_map = run_liftover(new_mapfile_bed, chain_from)

        # create validation plots
        logging.info("generating validation plots")
        validation_plots(liftback_map, oldmap, valid_dir)

    with open('debug.pkl', 'wb') as f:
        pickle.dump(dict(liftback_map=liftback_map, oldmap=oldmap, clean_new_map=clean_new_map), f)


def cumulative_to_hapmap(file: str, *, outfile: str, 
                         genome: str, inclusive: bool=True):
    """
    Read in a three-column cumulative map and convert it to
    a HapMap format.

    :param file: input three-column TSV of cumulative map position.
    :param genome: a TSV of chromosome lengths.
    :inclusive: whether to treat the physical position as inclusive.
    """
    sl = load_seqlens(genome)
    recmap = read_cumulative(file, sl, end_inclusive=inclusive)
    write_hapmap(recmap, outfile)

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)
    defopt.run({'liftover': liftover, 'convert': cumulative_to_hapmap})

