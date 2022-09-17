import click
import numpy as np
import tqdm
import pickle
import os
from collections import namedtuple
from matplotlib.backends.backend_pdf import PdfPages
from bgspy.models import BGSModel
from bgspy.recmap import RecMap
from bgspy.utils import sum_logliks_over_chroms
from bgspy.plots import chrom_plot, ll_grid
from bgspy.genome import Genome
from bgspy.utils import Grid
from bgspy.likelihood import fit_likelihood

SPLIT_LENGTH_DEFAULT = 10_000
STEP_DEFAULT = 10_000
NCHUNKS_DEFAULT = 200

LogLiks = namedtuple('LogLiks', ('pi0', 'pi0_ll', 'w', 't', 'll'))


# human grid defaults
# sorry non-human researchers, I like other organisms too, just for
# my sims :)
MIN_W = np.sqrt(1e-8 * 1e-7) # midpooint between 1e-7 and 1e-8 on a log10 scale
#HUMAN_W = (-10, np.log10(MIN_W))
HUMAN_W = (-11, -7)
HUMAN_T = (-5, -1)
def grid_maker(nw, nt, w_range=HUMAN_W, t_range=HUMAN_T):
  return Grid(w=np.logspace(*w_range, nw), t=np.logspace(*t_range, nt))

def grid_maker_from_str(x):
  nw, nt = tuple(map(int, x.split('x')))
  return grid_maker(nw, nt)


def make_bgs_model(seqlens, annot, recmap, conv_factor, w, t, g=None,
                   chroms=None, name=None, split_length=SPLIT_LENGTH_DEFAULT):
    """
    Build the BGSModel and the Genome object it uses.
    """
    if g is not None:
        # g takes priority
        w, t = grid_maker_from_str(g)
    else:
        w, t = parse_gridstr(w), parse_gridstr(t),
    if name is None:
        # infer the name
        bn = os.path.splitext(os.path.basename(seqlens))[0]
        name = bn.replace('_seqlens', '')
    g = Genome(name, seqlens_file=seqlens, chroms=chroms)
    g.load_annot(annot)
    g.load_recmap(recmap, conversion_factor=conv_factor)
    g.create_segments(split_length=split_length)
    m = BGSModel(g, w_grid=w, t_grid=t, split_length=split_length)
    return m

def parse_gridstr(x):
    if ',' not in x and ':' not in x:
        # fixed value
        return np.array([float(x)])
    if ',' in x:
        try:
            return np.array(list(map(float, x.split(','))))
        except:
            raise click.BadParameter("misformated grid string, needs to be 'x,y'")
    lower, upper, ngrid = list(map(float, x.split(':')))
    if int(ngrid) != ngrid:
        msg = f"the grid string '{x}' does not have an integer number of samples"
        raise click.BadParameter(msg)
    return 10**np.linspace(lower, upper, int(ngrid))

def calc_stats(x, stats={'mean': np.mean, 'median': np.median, 'var': np.var,
                        'min': np.min, 'max': np.max,
                         'non-zero-min': lambda x: np.min(x[x > 0]),
                         'lower-decile': lambda x: np.quantile(x, 0.10),
                         'lower-quartile': lambda x: np.quantile(x, 0.25),
                         'upper-decile': lambda x: np.quantile(x, 0.9),
                        'total': lambda y: y.shape[0]}):
    return {s: func(x) for s, func in stats.items()}

@click.group()
def cli():
    pass

@cli.command()
@click.option('--recmap', type=str, required=True,
              help="HapMap or BED-like TSV of recombination rates")
@click.option('--annot', required=True, type=click.Path(exists=True),
              help='BED file with conserved regions, fourth column is optional feature class')
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option('--name', type=str, help="genome name (otherwise inferred from seqlens file)")
@click.option('--conv-factor', default=1e-8,
                help="Conversation factor of recmap rates to M (for "
                     "cM/Mb rates, use 1e-8)")
@click.option('--t', help="string of lower:upper:grid_size or comma-separated "
                   "list for log10 heterozygous selection coefficient",
                    default='-6:-1:50')
@click.option('--w', help="string of lower:upper:grid_size or comma-separated "
                   "list for log10 mutation rates", default='-10:-7:50' )
@click.option('--g', help="a grid string for human defaults, e.g. 6x8 "
              "(--t/--w ignored if this is set)", default=None)
@click.option('--chrom', help="process a single chromosome", default=None)
@click.option('--popsize', help="population size for B'", type=int, default=None)
@click.option('--split-length', default=SPLIT_LENGTH_DEFAULT, help='conserved segments larger than split-length will be broken into chunks')
@click.option('--step', help='step size for B in basepairs (default: 1kb)',
              default=STEP_DEFAULT)
@click.option('--only-Bp', default=False, is_flag=True, help="only calculate B'")
@click.option('--only-B', default=False, is_flag=True, help="only calculate B")
@click.option('--nchunks', default=NCHUNKS_DEFAULT, help='number of chunks to break the genome up into (for parallelization)')
@click.option('--ncores', help='number of cores to use for calculating B', type=int, default=None)
@click.option('--ncores-Bp', help="number of cores to use for calculating B' (more memory intensive)",
              type=int, default=None)
@click.option('--fill-nan', default=True, is_flag=True,
              help="fill NANs from B' with B values")
@click.option('--output', required=True, help='output file',
              type=click.Path(exists=False, writable=True))
def calcb(recmap, annot, seqlens, name, conv_factor, t, w, g,
          chrom, popsize, split_length, step, only_bp, only_b, nchunks,
          ncores, ncores_bp, fill_nan, output):

    if ncores_bp is None and ncores is not None:
        ncores_bp = ncores
    N = popsize
    chrom = [chrom] if chrom is not None else None
    m = make_bgs_model(seqlens, annot, recmap, conv_factor,
                       w, t, g, chroms=chrom, name=name,
                       split_length=split_length)
    if not only_bp:
        m.calc_B(step=step, ncores=ncores, nchunks=nchunks)
    if not only_b:
        assert N is not None, "--popsize is not set and B' calculated!"
        m.calc_Bp(N=N, step=step, ncores=ncores_bp, nchunks=nchunks)
    if not only_b and fill_nan:
        assert m.Bps is not None, "B' not set!"
        print(f"filling in B' NaNs with B...\t", end='', flush=True)
        m.fill_Bp_nan()
        print(f"done.")
    m.save(output)

@cli.command()
@click.option('--recmap', required=True, type=click.Path(exists=True),
              help='BED file with rec rates per window in the 4th column')
@click.option('--annot', required=True, type=click.Path(exists=True),
              help='BED file with conserved regions, fourth column is optional feature class')
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option('--conv-factor', default=1e-8,
                help="Conversation factor of recmap rates to M (for cM/Mb rates, use 1e-8)")
@click.option('--split-length', default=SPLIT_LENGTH_DEFAULT, help='conserved segments larger than split-length will be broken into chunks')
@click.option('--output', required=False,
              help='output file for a pickle of the segments',
              type=click.Path(exists=False, writable=True))
def stats(recmap, annot, seqlens, conv_factor, split_length, output=None):
    m = make_bgs_model(seqlens, annot, recmap, conv_factor,
                       w=None, t=None, chroms=None, name=None,
                       split_length=split_length)
    segments = m.segments
    if output:
        with(outfile, 'wb') as f:
            pickle.dump(segments, f)
        return
    ranges = segments.ranges
    lens = np.diff(ranges, axis=1).flatten()
    ranges_stats = calc_stats(lens)
    rec = segments.rates
    rec_stats = calc_stats(rec)
    print(f"range stats: {ranges_stats}")
    print(f"rec stats: {rec_stats}")

@cli.command()
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option('--recmap', required=True, type=click.Path(exists=True),
              help='HapMap formatted recombination map')
@click.option('--counts-dir', required=True, type=click.Path(exists=True),
              help='directory to Numpy .npy per-basepair counts')
@click.option('--neutral', required=True, type=click.Path(exists=True),
              help='neutral region BED file')
@click.option('--access', required=True, type=click.Path(exists=True),
              help='accessible regions BED file (e.g. no centromeres)')
@click.option('--fasta', required=True, type=click.Path(exists=True),
              help='FASTA reference file (e.g. to mask Ns and lowercase'+
                   '/soft-masked bases')
@click.option('--bs-file', required=True, type=click.Path(exists=True),
              help="BGSModel genome model pickle file (contains B' and B)")
@click.option('--outfile', required=True,
              type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=None)
@click.option('--nstarts',
              help='number of starts for multi-start optimization',
              type=int, default=None)
@click.option('--window',
              help='size (in basepairs) of the window',
              type=int, default=1_000_000)
@click.option('--outliers',
              help='quantiles for trimming bin π',
              type=str, default='0.0,0.995')
def loglik(seqlens, recmap, counts_dir, neutral, access, fasta,
           bs_file, outfile, ncores, nstarts, window, outliers):
    outliers = tuple([float(x) for x in outliers.split(',')])
    fit_likelihood(seqlens_file=seqlens, recmap_file=recmap,
                   counts_dir=counts_dir, neut_file=neutral,
                   access_file=access, fasta_file=fasta,
                   bs_file=bs_file, outfile=outfile, ncores=ncores,
                   nstarts=nstarts, window=window, outliers=outliers)

@cli.command()
@click.option('--fit', required=True, type=click.Path(exists=True),
              help='pickle file of fitted results')
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option('--recmap', required=True, type=click.Path(exists=True),
              help='HapMap formatted recombination map')
@click.option('--counts-dir', required=True, type=click.Path(exists=True),
              help='directory to Numpy .npy per-basepair counts')
@click.option('--neutral', required=True, type=click.Path(exists=True),
              help='neutral region BED file')
@click.option('--access', required=True, type=click.Path(exists=True),
              help='accessible regions BED file (e.g. no centromeres)')
@click.option('--fasta', required=True, type=click.Path(exists=True),
              help='FASTA reference file (e.g. to mask Ns and lowercase'+
                   '/soft-masked bases')
@click.option('--bs-file', required=True, type=click.Path(exists=True),
              help="BGSModel genome model pickle file (contains B' and B)")
@click.option('--outfile', required=True,
              type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=None)
@click.option('--nstarts',
              help='number of starts for multi-start optimization',
              type=int, default=None)
@click.option('--window',
              help='size (in basepairs) of the window',
              type=int, default=1_000_000)
@click.option('--outliers',
              help='quantiles for trimming bin π',
              type=str, default='0.0,0.995')
@click.option('--B', type=int, help='number of bootstrap replicates')
@click.option('--blocksize', type=int,
              help='number of basepairs for block size for bootstrap')
def bootstrap(fit, seqlens, recmap, counts_dir, neutral, access, fasta,
              bs_file, outfile, ncores, nstarts, window, outliers,
              b, blocksize):
    outliers = tuple([float(x) for x in outliers.split(',')])
    # internally we use blocksize to represent the number of adjacent windows
    blocksize  = int(window / blocksize)
    fit_likelihood(fit_file=fit,
                   seqlens_file=seqlens, recmap_file=recmap,
                   counts_dir=counts_dir, neut_file=neutral,
                   access_file=access, fasta_file=fasta,
                   bs_file=bs_file, boots_outfile=outfile, ncores=ncores,
                   nstarts=nstarts, window=window, outliers=outliers,
                   B=b, blocksize=blocksize)


if __name__ == "__main__":
    res = cli()
