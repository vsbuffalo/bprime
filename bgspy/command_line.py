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

SPLIT_LENGTH_DEFAULT = 10_000
STEP_DEFAULT = 10_000
NCHUNKS_DEFAULT = 200

LogLiks = namedtuple('LogLiks', ('pi0', 'pi0_ll', 'w', 't', 'll'))


# human grid defaults
# sorry non-human researchers, I like other organisms too, just for
# my sims :)
MIN_W = np.sqrt(1e-8 * 1e-7) # midpooint between 1e-7 and 1e-8 on a log10 scale
#HUMAN_W = (-10, np.log10(MIN_W))
HUMAN_W = (-10, -7)
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
    if fill_nan:
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
@click.option('--b', required=True, type=click.Path(exists=True),
              help='B file from calcb')
@click.option('--dac', required=True, type=click.Path(exists=True),
              help='three column TSV of chrom, pos, nchroms, derived allele counts')
@click.option('--regions', required=True, type=click.Path(exists=True),
              help='BED file of neutral regions')
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option('--pi0', help='manual π0', default=None, required=False, type=float)
#@click.option('--pi0-lower', default=0, required=False, type=float)
#@click.option('--pi0-upper', default=0.01, required=False, type=float)
@click.option('--rmin', default=0.1, required=False, type=float)
@click.option('--pi0-ngrid', default=30, required=False, type=int)
@click.option('--width', default=1_000_000, required=False, type=int,
              help='width for π and B')
@click.option('--output', required=True,
              type=click.Path(exists=False, writable=True))
@click.option('-progress/--no-progress', default=True, help="display the progress bar")
def loglik(b, dac, regions, seqlens, pi0, rmin, pi0_ngrid,
           width, output, progress):
    assert False, "all out of date -- eg DAC file"
    sl = load_seqlens(seqlens)
    nr = read_bed(regions)
    m = BGSModel(seqlens=sl)
    m.load_B(b)
    m.load_dacfile(dac, nr)
    #pi0_bounds = None
    pi_bar = m.gwpi()
    pi0_grid = None
    if pi0 is None:
        #pi0_bounds = (pi0_lower, pi0_upper)
        pi0_max = pi_bar/rmin
        pi0_grid = np.linspace(rmin*pi_bar, pi0_max, pi0_ngrid)
    if pi0_grid is not None and pi0 is not None:
        print("warning: both fixed π0 and π0 grid are set; using fixed π0!")
        pi0_grid = None
    ll, pi0 = m.loglikelihood(pi0=pi0, pi0_grid=pi0_grid)
    lls = ll.sum(axis=0) # sum over sites
    binned_B = m.bin_B(width)
    binned_pi = m.pi(width)
    wi_mle, ti_mle = np.where(lls == np.nanmax(lls))
    if m.pi0i_mle is not None:
        pi0_mle = pi0_grid[m.pi0i_mle]
    else:
        pi0_mle = None
    w_mle, t_mle = m.w[wi_mle], m.t[ti_mle]
    llo = (lls, pi0, m.pi0_ll, pi0_grid, m.w, m.t, binned_B, binned_pi,
           pi_bar, pi0_mle, w_mle, t_mle, wi_mle, ti_mle, m.metadata)
    with open(output, 'wb') as f:
        pickle.dump(llo, f)


@cli.command()
@click.option('--lik', required=True, type=click.Path(exists=True),
              help='likelihood results in tuple')
@click.option('--width', required=False, type=float, default='10')
@click.option('--height', required=False, type=float, default='7.5')
@click.option('--output', required=True,
              type=click.Path(exists=False, writable=True))
def llfig(lik, width, height, output):
    with open(lik, 'rb') as f:
        ll_tuple = pickle.load(f)
        ll, pi0, pi0_ll, pi0_grid, ws, ts, binned_B, binned_pi, gwpi, pi0_mle, w_mle, t_mle, wi_mle, ti_mle, md = ll_tuple
        pi0_neut  = 4 * 2.5e-7*1000
        w_sim, t_sim = float(md['mu']), -float(md['s'])*float(md['h'])
        ti_true = np.argmin(np.abs(ts-t_sim))
        wi_true = np.argmin(np.abs(ws-w_sim))
        chroms = list(ll_tuple[6].keys())
        with PdfPages(output) as pdf:
            fig, ax = ll_grid(ll, ws, ts,
                              xlabel='$t$', ylabel='$\mu$',
                              true=(t_sim, w_sim),
                              mle=(t_mle, w_mle),
                              ncontour=10,
                              mid_quant=0.50)
            pdf.savefig(fig)
            if ll_tuple[2] is not None:
                # pi0 grid MLE is set
                if pi0_mle is None:
                    pi0_mle = pi0_neut
                fig, ax = ll_grid(pi0_ll[:, :, ti_true], pi0_grid, ws,
                                  xlabel='$\mu$', ylabel='$\pi_0$',
                                  true=(w_sim, pi0_neut),
                                  mle=(w_mle, pi0_mle),
                                  mid_quant=0.50)
                pdf.savefig(fig)
                fig, ax = ll_grid(pi0_ll[:, wi_true, :], pi0_grid, ts,
                                  xlabel='$t$', ylabel='$\pi_0$',
                                  true=(t_sim, pi0_neut),
                                  mle=(t_mle, pi0_mle),
                                  mid_quant=0.50)
                pdf.savefig(fig)
            for chrom in chroms:
                fig, ax = chrom_plot(ll_tuple, chrom, figsize=(width, height))
                pdf.savefig(fig)
    return ll_tuple

if __name__ == "__main__":
    res = cli()
