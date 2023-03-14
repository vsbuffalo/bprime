import click
import logging
import numpy as np
import pickle
import os
from collections import namedtuple
from bgspy.models import BGSModel
from bgspy.genome import Genome
from bgspy.utils import Grid, load_pickle
from bgspy.pipeline import summarize_data, mle_fit, summarize_sim_data
from bgspy.bootstrap import block_bins

SPLIT_LENGTH_DEFAULT = 10_000
STEP_DEFAULT = 10_000
NCHUNKS_DEFAULT = 200

LogLiks = namedtuple('LogLiks', ('pi0', 'pi0_ll', 'w', 't', 'll'))


# human grid defaults
# sorry non-human researchers, I like other organisms too, just for
# my sims :)
# midpooint between 1e-7 and 1e-8 on a log10 scale
MIN_W = np.sqrt(1e-8 * 1e-7)
#HUMAN_W = (-10, np.log10(MIN_W))
HUMAN_W = (-11, -7)
HUMAN_T = (-8, -1)


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
    """
    Grid strings are in the format v1,v2,v3, etc or lower:upper:ngrid
    """
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
@click.option('--chrom', help="process specified chromosome(s)", default=None, multiple=True)
@click.option('--popsize', help="population size for B'", type=int, default=None)
@click.option('--split-length', default=SPLIT_LENGTH_DEFAULT,
              help='conserved segments larger than split-length will be broken into chunks')
@click.option('--step', help='step size for B in basepairs (default: 1kb)',
              default=STEP_DEFAULT)
@click.option('--only-Bp', default=False, is_flag=True, help="only calculate B'")
@click.option('--only-B', default=False, is_flag=True, help="only calculate B")
@click.option('--rescale-fit', default=None, help="pickle of B and B' fits for locally-rescaling genome-wide N")
@click.option('--rescale', default=None, help="manual rescaling of a 'μ,s' pair, set --rescale-Bp-file too")
@click.option('--rescale-Bp-file', default=None, help="previous B' calculations file")
@click.option('--nchunks', default=NCHUNKS_DEFAULT, help='number of chunks to break the genome up into (for parallelization)')
@click.option('--ncores', help='number of cores to use for calculating B', type=int, default=None)
@click.option('--ncores-Bp', help="number of cores to use for calculating B' (more memory intensive)",
              type=int, default=None)
@click.option('--output', required=True, help='output file',
              type=click.Path(exists=False, writable=True))
def calcb(recmap, annot, seqlens, name, conv_factor, t, w, g,
          chrom, popsize, split_length, step, only_bp, only_b,
          rescale_fit, rescale, rescale_bp_file, nchunks, ncores,
          ncores_bp, output):
    """
    Calculate B and/or B'.
    """

    N = popsize
    if ncores_bp is None and ncores is not None:
        ncores_bp = ncores

    bpfit = None

    ### rescaling stuff: this creates a tuple passed to calc_Bp,
    ### of (Bp, w, t, fit) (some can be None)
    # load the fits if they exist -- used for rescaling
    if rescale_fit is not None:
        assert rescale is None, "--fit and --rescale cannot both be set!"
        assert rescale_bp_file is not None, "--rescale-Bp-file must also be set!"
        gm = BGSModel.load(rescale_bp_file)
        fits = pickle.load(open(rescale_fit, 'rb'))
        bfit, bpfit = fits['mb'], fits['mbp']
        rescale = (gm.BpScores, None, None, bpfit)

    # manual rescaling from a single fixed set of parameters is set.
    if rescale_fit is None and rescale is not None:
        assert rescale_bp_file is not None, "specify --rescale-Bp-file too!"
        rs_w, rs_t = list(map(float, rescale.split(',')))
        gm = BGSModel.load(rescale_bp_file)
        assert rs_w in gm.w, "μ not in ΒGSModel.w!"
        assert rs_t in gm.t, "s not in ΒGSModel.t!"
        rescale = (gm.BpScores, rs_w, rs_t, None)

    if rescale_fit is None:
        ## Make sure chromosomes match
        # use specified chromosome or set to None to use all in seqlens file
        if isinstance(chrom, str):
            chrom = [chrom]
    else:
        # match what's in the fit object.
        chrom = list(bpfit.bins.keys())

    m = make_bgs_model(seqlens, annot, recmap, conv_factor,
                       w, t, g, chroms=chrom, name=name,
                       split_length=split_length)

    if not only_bp:
        m.calc_B(step=step, ncores=ncores, nchunks=nchunks)
    if not only_b:
        assert N is not None, "--popsize is not set and B' calculated!"
        m.calc_Bp(N=N, step=step, ncores=ncores_bp, nchunks=nchunks,
                  rescale=rescale)
    #if not only_b and fill_nan:
    #    assert m.Bps is not None, "B' not set!"
    #    print(f"filling in B' NaNs with B...\t", end='', flush=True)
    #    m.fill_Bp_nan()
    #    print(f"done.")
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
    """
    Calculate segment stats.
    """
    #TODO check this, as it's not been used in age.
    m = make_bgs_model(seqlens, annot, recmap, conv_factor,
                       w=None, t=None, chroms=None, name=None,
                       split_length=split_length)
    segments = m.segments
    if output:
        with(output, 'wb') as f:
            pickle.dump(segments, f)
        return
    ranges = segments.ranges
    lens = np.diff(ranges, axis=1).flatten()
    ranges_stats = calc_stats(lens)
    rec = segments.rates
    rec_stats = calc_stats(rec)
    print(f"range stats: {ranges_stats}")
    print(f"rec stats: {rec_stats}")

# @click.option('--sim-tree-file', required=False, type=click.Path(exists=True),
#               help="a tree sequence file from a simulation")
# @click.option('--sim-mu', required=False, type=float, help="simulation neutral mutation rate (to bring treeseqs to counts matrices)")


@cli.command()
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
@click.option('--recmap', required=True, type=click.Path(exists=True),
              help='HapMap formatted recombination map')
@click.option('--neutral', required=True, type=click.Path(exists=True),
              help='neutral region BED file')
@click.option('--access', required=True, type=click.Path(exists=True),
              help='accessible regions BED file (e.g. no centromeres)')
@click.option('--fasta', required=True, type=click.Path(exists=True),
              help='FASTA reference file (e.g. to mask Ns and lowercase'+
                   '/soft-masked bases')
@click.option('--bs-file', required=True, type=click.Path(exists=True),
              help="BGSModel genome model pickle file (contains B' and B)")
@click.option('--output', default=None, type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--window',
              help='size (in basepairs) of the window',
              type=int, default=1_000_000)
@click.option('--counts-dir', required=False, type=click.Path(exists=True),
              help='directory to Numpy .npy per-basepair counts')
@click.option('--outliers',
              help='quantiles for trimming bin π',
              type=str, default='0.0,0.995')
@click.option('--soft-mask', is_flag=True, default=True, 
              help="whether to exclude soft-masked regions in the reference")
@click.option('--Bp-only', default=False, is_flag=True, help="only calculate B'")
def data(seqlens, recmap, neutral, access, fasta, 
         bs_file, counts_dir, output, window, outliers, 
         soft_mask, bp_only):
    """
    Pre-process genomic count data for MLE fit.


    Load genomic data from numpy .npz ref/alt counts, read in accessible and
    neutral regions, summarize into windows, filter outliers, and bin B/B'
    to prepare for MLE fit. This outputs a pickle file of the windowed
    genomic and B/B' data.
    """
    outliers = tuple([float(x) for x in outliers.split(',')])
    summarize_data(seqlens_file=seqlens, recmap_file=recmap,
                   neut_file=neutral, access_file=access, fasta_file=fasta,
                   bs_file=bs_file,
                   counts_dir=counts_dir,
                   window=window,
                   outliers=outliers,
                   soft_mask=soft_mask,
                   output_file=output, bp_only=bp_only)


@cli.command()
@click.argument('sim-tree', required=True, nargs=-1)
@click.option('--bs-file', required=True, type=click.Path(exists=True),
              help="BGSModel genome model pickle file (contains B' and B)")
@click.option('--sim-mu', required=True, type=float,
              help="simulation neutral mutation rate (to bring treeseqs to counts matrices)")
@click.option('--neutral', required=True, type=click.Path(exists=True),
              help='neutral region BED file')
@click.option('--access', required=True, type=click.Path(exists=True),
              help='accessible regions BED file (e.g. no centromeres)')
@click.option('--output', default=None, required=True,
              type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--window', help='size (in basepairs) of the window',
              type=int, required=True)
@click.option('--Bp-only', default=False, is_flag=True, help="only calculate B'")
def simdata(sim_tree, bs_file, sim_mu, neutral, access, 
            output, window, bp_only):
    """
    Pre-process a tskit.TreeSequence simulated tree.
    """
    summarize_sim_data(sim_tree, bs_file, 
                       neut_file=neutral, access_file=access, 
                       output_file=output,
                       window=window, sim_mu=sim_mu, bp_only=bp_only)


@cli.command()
@click.option('--data', required=True, default=None,
              type=click.Path(exists=True),
              help="pickle of pre-computed summary statistics")
@click.option('--output',
              required=True, type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--mu', help='fixed mutation rate (by default, free)', 
              default=None)
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=None)
@click.option('--nstarts',
              help='number of starts for multi-start optimization',
              type=int, default=500)
@click.option('--chrom', default=None,
              help="only fit on using this chromosome (default: genome-wide)")
@click.option('--only-Bp', default=False, is_flag=True, help="only calculate B'")
def fit(data, output, mu, ncores, nstarts, chrom, only_bp):
    """
    Run the MLE fit on pre-processed data.
    """
    # for fixed mu
    mu = None if mu in (None, 'None') else float(mu) # sterialize CL input
    mle_fit(data=data,
            output_file=output,
            ncores=ncores,
            nstarts=nstarts,
            mu=mu, chrom=chrom, bp_only=only_bp)


@cli.command()
@click.option('--bs-file', required=True, type=click.Path(exists=True),
              help="BGSModel genome model pickle file (contains B' and B)")
@click.option('--fit', required=True, type=click.Path(exists=True),
              help='pickle file of fitted results')
@click.option('--force-feature', default=None,
              help='force all predictions using DFE estimates of this feature (experimental)')
@click.option('--output', required=True,
              type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--split', default=False, is_flag=True,
              help="split into different files for each feature "
              "(uses '<output>_<feature_a>.bed', etc)")
def subrate(bs_file, fit, force_feature, output, split):
    """
    Take a fit and predict the substitution rates.
    """
    m = BGSModel.load(bs_file)
    bfit, bpfit = pickle.load(open(fit, 'rb'))

    feature_idx = None
    if force_feature is not None:
        avail_feats = [x.lower() for x in bpfit.features]
        force_feature = force_feature.lower()
        assert force_feature in avail_feats
        feature_idx = avail_feats.index(force_feature)

    rdf = m.ratchet_df(bpfit, predict_under_feature=feature_idx)
    msg = "feature mismatch between BGSModel and fit!"
    assert bpfit.features == list(m.genome.segments.feature_map.keys()), msg
    rdf = rdf.sort_values(['chrom', 'start', 'end'])
    if not split:
        rdf.to_csv(output, sep='\t', header=False, index=False)
        return
    for feature in m.genome.segments.feature_map:
        if output.endswith('.bed'):
            filename = output.replace('.bed', f'_{feature}.bed')
        else:
            filename = f'{output}_{feature}.bed'
        rdfx = rdf.loc[rdf['feature'] == feature]
        rdfx.to_csv(filename, sep='\t', header=False, index=False)


@cli.command()
@click.option('--data', required=True, default=None, type=click.Path(exists=True),
              help="pickle of pre-computed summary statistics")
@click.option('--fit', default=None, type=click.Path(exists=True),
              help=('pickle file of fitted results, for starting at MLE '
                    '(ignore for random starts)'))
@click.option('--blocksize', type=int,
              help='the blocksize, in number of consecutive windows')
@click.option('--blockwidth', type=int, help="the blockwidth in basepairs")
@click.option('--blocknum', default=None, type=int,
              help='which block to run the fit on (e.g. for cluster use)')
@click.option('--blockfrac', default=None, type=float,
              help='for not a full jackknife, drop the block at this fraction '
                   "(this approach doesn't require a prior knowledge of number of blocks")
@click.option('--output', required=True, type=click.Path(writable=True),
              help='an .npz output file')
@click.option('--fit-dir', default=None, help="fit directory for saving whole fits")
@click.option('--chrom', default=None,
              help="leave-one-out chromosome, e.g. for parallel processing across a cluster")
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=1)
@click.option('--nstarts',
              help='number of starts for multi-start optimization',
              type=int, default=1)
@click.option('--include-Bs', default=False, is_flag=True, help="whether to include classic Bs too")
def jackblock(data, fit, blocksize, blockwidth, blocknum, blockfrac, output, 
              fit_dir, chrom, ncores, nstarts, include_bs):
    """
    Run the block-jackknifing routine. 

    There are two modes:
        1. Run with --blocksize set only, and all blocks will be 
            jackknifed.
        2. Run with --blocksize and --blocknum, and only that 
            block number is left out (e.g. for cluster use).
    """
    if blockfrac is not None or blockwidth is not None:
        # we need to figure out the block width or num blocks
        bins = load_pickle(data)['bins']
        if blockwidth is not None:
            blocksize = int(blockwidth / bins.width)
        blocks = block_bins(bins, blocksize)
        nblocks = len(blocks)
        if blockfrac is not None:
            blocknum = int(blockfrac * nblocks)
    logging.info("blockwidth={blockwidth}bp, blockfrac={blockfrac}, blocknum={blocknum}")
    assert blocksize > 1, "blocksize must be > 1"
    mle_fit(data=data,
            output_file=output,
            ncores=ncores,
            mu=mu,
            nstarts=nstarts,
            loo_chrom=chrom,
            blocksize=blocksize, blocknum=blocknum,
            bp_only=True)


@cli.command()
@click.option('--data', required=True, default=None, type=click.Path(exists=True),
              help="pickle of pre-computed summary statistics")
@click.option('--fit', default=None, type=click.Path(exists=True),
              help=('pickle file of fitted results, for starting at MLE '
                    '(ignore for random starts)'))
@click.option('--mu', help='fixed mutation rate (by default, free)', 
              default=None)
@click.option('--output', required=True, type=click.Path(writable=True),
              help='an .npz output file')
@click.option('--fit-dir', default=None, help="fit directory for saving whole fits")
@click.option('--chrom', default=None,
              help="leave-one-out chromosome, e.g. for parallel processing across a cluster")
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=1)
@click.option('--nstarts',
              help='number of starts for multi-start optimization',
              type=int, default=1)
@click.option('--include-Bs', default=False, is_flag=True, help="whether to include classic Bs too")
def loo(data, fit, mu, output, fit_dir, chrom,
              ncores, nstarts, include_bs):
    """
    Leave-one-out chromosome fit. 
    
    This is used for estimating out-sample R2 by leaving out a chromosome,
    fitting to the rest of the genome, and predicting the observed diversity on
    the excluded chromosome.
    """
    mle_fit(data=data,
            output_file=output,
            ncores=ncores,
            mu=mu,
            nstarts=nstarts,
            loo_chrom=chrom,
            bp_only=True)


@click.option('--fit-dir', required=True, 
              help="fit directory for reading whole fits"
                   " (reads all .pkl files)")
@click.option('--output', required=True, type=click.Path(writable=True),
              help='an .npz output file')
@click.option('--fit', default=None, type=click.Path(exists=True),
              help='pickle file of main MLE fit')
def collect(fit_dir, output, fit):
    """
    TODO:
        - for B' only
    """
    main_fits = load_pickle(fit)
    mbp = main_fits['mbp']
    mbp.load_jackknives(fit_dir)
    mbp.loo_stderr()
    mbp.loo_R2()
    # TODO


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    res = cli()
