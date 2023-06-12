import click
import logging
import gc
import yaml
import numpy as np
import pickle
from collections import defaultdict
from si_prefix import si_format
import os
from collections import namedtuple
from bgspy.models import BGSModel
from bgspy.genome import Genome
from bgspy.data import GenomeData
from bgspy.utils import Grid, load_pickle, bin_chroms
from bgspy.utils import read_bed3, combine_features, masks_to_ranges
from bgspy.utils import load_seqlens
from bgspy.pipeline import summarize_data, mle_fit, summarize_sim_data
from bgspy.pipeline import ModelDir
from bgspy.bootstrap import block_bins
from bgspy.likelihood import PI0_BOUNDS, MU_BOUNDS

SPLIT_LENGTH_DEFAULT = 10_000
STEP_DEFAULT = 10_000
NCHUNKS_DEFAULT = 200
MU_NOT_SET = (None, 'None', 'free', 'Free')

# CL bounds are given in linear scale
# TODO change name  to include log in likelihood.py
PI0_BOUNDS = f"{10**PI0_BOUNDS[0]},{10**PI0_BOUNDS[1]}"
MU_BOUNDS = f"{10**MU_BOUNDS[0]},{10**MU_BOUNDS[1]}"

LogLiks = namedtuple('LogLiks', ('pi0', 'pi0_ll', 'w', 't', 'll'))


# human grid defaults
# sorry non-human researchers, I like other organisms too, just for
# my sims :)
# midpooint between 1e-7 and 1e-8 on a log10 scale
MIN_W = np.sqrt(1e-8 * 1e-7)
#HUMAN_W = (-10, np.log10(MIN_W))
HUMAN_W = (-11, -7)
HUMAN_T = (-8, -1)

HUMAN_W_STR = f"{HUMAN_W[0]}:{HUMAN_W[1]}:6"
HUMAN_T_STR = f"{HUMAN_T[0]}:{HUMAN_T[1]}:8"


def sterialize_mu(mu):
    try:
        mu = None if mu in MU_NOT_SET else float(mu)
    except ValueError:
        raise ValueError("mu must be 'free'/None, or a float")
    return mu


def grid_maker(nw, nt, w_range=HUMAN_W, t_range=HUMAN_T):
    return Grid(w=np.logspace(*w_range, nw), t=np.logspace(*t_range, nt))


def grid_maker_from_str(x):
    nw, nt = tuple(map(int, x.split('x')))
    return grid_maker(nw, nt)


def make_bgs_model(seqlens, annot, recmap, conv_factor, w, t, g=None,
                   chroms=None, name=None, genome_only=False,
                   split_length=SPLIT_LENGTH_DEFAULT):
    """
    Build the BGSModel and the Genome object it uses.
    """
    if name is None:
        # infer the name
        bn = os.path.splitext(os.path.basename(seqlens))[0]
        name = bn.replace('_seqlens', '')
    gn = Genome(name, seqlens_file=seqlens, chroms=chroms)
    gn.load_annot(annot)
    gn.load_recmap(recmap, conversion_factor=conv_factor)
    gn.create_segments(split_length=split_length)
    # segments are sufficient for recmap, so we deleter
    #delattr(gn, 'recmap')
    #gn.recmap = None
    #gc.collect()
    if genome_only:
        return gn
    
    # get the grid 
    if g is not None:
        # g takes priority
        w, t = grid_maker_from_str(g)
    else:
        w, t = parse_gridstr(w), parse_gridstr(t),

    m = BGSModel(gn, w_grid=w, t_grid=t, split_length=split_length)
    return m


def parse_gridstr(x):
    """
    Grid strings are in the format v1,v2,v3, etc or lower:upper:ngrid (log10, 
    e.g. -8:-1,8.  The comma-separated values are *not* in log-space.
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


DEFAULT_STATS = {'mean': np.mean, 'median': np.median, 'var': np.var,
                        'min': np.min, 'max': np.max,
                         'non-zero-min': lambda x: np.min(x[x > 0]),
                         'lower-decile': lambda x: np.quantile(x, 0.10),
                         'lower-quartile': lambda x: np.quantile(x, 0.25),
                         'upper-decile': lambda x: np.quantile(x, 0.9),
                        'total': lambda y: y.shape[0]}
def calc_stats(x, stats=DEFAULT_STATS):
    """
    calc a bunch of statistics, unweighted
    """
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
@click.option('--t', help="string of log10 lower:upper:grid_size (e.g. -7,-1,7) or comma-separated "
                   "list (e.g. 0.01,0.001) for heterozygous selection coefficient",
                    default=HUMAN_T_STR)
@click.option('--w', help="string of lower:upper:grid_size or comma-separated "
                   "list for log10 mutation rates", default=HUMAN_W_STR)
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
        del gm
        gc.collect()

    # manual rescaling from a single fixed set of parameters is set.
    if rescale_fit is None and rescale is not None:
        assert rescale_bp_file is not None, "specify --rescale-Bp-file too!"
        rs_w, rs_t = list(map(float, rescale.split(',')))
        gm = BGSModel.load(rescale_bp_file)
        assert rs_w in gm.w, "μ not in ΒGSModel.w!"
        assert rs_t in gm.t, "s not in ΒGSModel.t!"
        rescale = (gm.BpScores, rs_w, rs_t, None)
        del gm
        gc.collect()

    if rescale_fit is None:
        ## Make sure chromosomes match
        # use specified chromosome or set to None to use all in seqlens file
        if isinstance(chrom, str):
            chrom = [chrom]
    else:
        # make sure that all chroms are in the keys of the original B'
        assert all([chr in bpfit.bins.keys() for chr in chrom])
        
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
@click.argument('models', required=True, nargs=-1)
@click.option('--output', required=True,
              help="output file for merged B' results",
              type=click.Path(exists=False, writable=True))
def merge(models, output):
    """
    Merge B' maps computed separately.
    """ 
    obj = BGSModel.from_chroms(models)
    obj.save(output)
   

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
    m = make_bgs_model(seqlens, annot, recmap, conv_factor,
                       w=None, t=None, chroms=None, name=None,
                       genome_only=True,
                       split_length=split_length)
    segments = m.segments
    if output:
        with open(output, 'wb') as f:
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
@click.option('--recmap', required=True, type=click.Path(exists=True),
              help='BED file with rec rates per window in the 4th column')
@click.option('--annot', required=True, type=click.Path(exists=True),
              help='BED file with conserved regions, fourth column is optional feature class')
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help="tab-delimited file of chromosome names and their length (name should be '<genome>_seqlens.tsv')")
@click.option('--counts-dir', required=False, type=click.Path(exists=True),
              help='directory to Numpy .npy per-basepair counts')
@click.option('--neutral', required=True, type=click.Path(exists=True),
              help='neutral region BED file')
@click.option('--access', required=True, type=click.Path(exists=True),
              help='accessible regions BED file (e.g. no centromeres)')
@click.option('--fasta', required=True, type=click.Path(exists=True),
              help='FASTA reference file (e.g. to mask Ns and lowercase'+
                   '/soft-masked bases')
@click.option('--outliers',
              help='quantiles for trimming bin π',
              type=str, default='0.0,0.995')
@click.option('--conv-factor', default=1e-8,
                help="Conversation factor of recmap rates to M (for cM/Mb rates, use 1e-8)")
@click.option('--window', required=True, help="window size")
@click.option('--output', required=True,
              help='output filename for a TSV of the window data',
              type=click.Path(exists=False, writable=True))
def windowstats(recmap, annot, seqlens, counts_dir,
                neutral, access, fasta, outliers,
                conv_factor, window, output=None):
    """
    Calculate window statistics.
    """
    # get the genome name
    bn = os.path.splitext(os.path.basename(seqlens))[0]
    name = bn.replace('_seqlens', '')
    gn = Genome(name, seqlens_file=seqlens)
    gn.load_annot(annot)
    gn.load_recmap(recmap, conversion_factor=conv_factor)

    gd = GenomeData(gn)
    gd.load_counts_dir(counts_dir)
    gd.load_neutral_masks(neutral)
    gd.load_accessibile_masks(access)
    gd.load_fasta(fasta, soft_mask=True)
    gd.trim_ends(1)  # removes 1cM off ends

    # bin the diversity data
    logging.info("binning pairwise diversity") 
    bgs_bins = gd.bin_pairwise_summaries(window,
                                         filter_accessible=True,
                                         filter_neutral=True)

    outliers = tuple(map(float, outliers.split(',')))
    bgs_bins.mask_outliers(outliers)

    # get the features per window
    bins = bin_chroms(gn.seqlens, int(window))
    feature_window_counts = defaultdict(lambda: defaultdict(dict))
    recrates_window = defaultdict(list)
    for chrom, (ranges, features) in gn.annot.items():
        mask = dict() 
        for feature, range in zip(features, ranges):
            if feature not in mask:
                mask[feature] = np.zeros(gn.seqlens[chrom], dtype='bool')
            mask[feature][slice(*range)] = 1
        
        # now window
        windows = bins[chrom]
        window_ranges = [(s, e) for s, e in zip(windows[:-1], windows[1:])]
        for feature in mask.keys():
            counts = np.zeros(len(window_ranges))
            for i, range in enumerate(window_ranges):
                counts[i] = sum(mask[feature][slice(*range)])
            feature_window_counts[feature][chrom] = (window_ranges, counts)

        for i, range in enumerate(window_ranges):
            rates = gn.recmap.lookup(chrom, np.array(range), cumulative=True)
            rate = (rates[1] - rates[0]) / (range[1] - range[0])
            recrates_window[chrom].append(rate)

    with open(output, 'w') as f:
        for feature in feature_window_counts.keys():
            for chrom in feature_window_counts[feature]:
                ranges, counts = feature_window_counts[feature][chrom]
                recrates = recrates_window[chrom]
                pis = bgs_bins.pi_pairs(chrom)[1]
                assert len(pis) == len(recrates)
                for range, counts, recrates, pi in zip(ranges, counts, recrates, pis):
                    row = [chrom, range[0], range[1], feature, counts, recrates, pi]
                    f.write('\t'.join(map(str, row)) + '\n')



        




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
@click.option('--n', default=None, type=int, help="number of samples to draw")
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
def simdata(sim_tree, bs_file, sim_mu, n, neutral, access, 
            output, window, bp_only):
    """
    Pre-process a tskit.TreeSequence simulated tree.
    """
    summarize_sim_data(sim_tree, bs_file, 
                       neut_file=neutral, access_file=access, 
                       output_file=output,
                       n=n,
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
@click.option('--mu-bounds', help='bounds for μ', default=MU_BOUNDS)
@click.option('--pi0-bounds', help='bounds for π0', default=PI0_BOUNDS)
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=None)
@click.option('--nstarts',
              help='number of starts for multi-start optimization',
              type=int, default=500)
@click.option('--chrom', default=None,
              help="only fit on using this chromosome (default: genome-wide)")
@click.option('--only-Bp', default=False, is_flag=True, help="only calculate B'")
def fit(data, output, mu, mu_bounds, pi0_bounds, ncores, nstarts, chrom, only_bp):
    """
    Run the MLE fit on pre-processed data.
    """
    # for fixed mu
    mu = sterialize_mu(mu)
    pi0_bounds = tuple(map(float, pi0_bounds.split(',')))
    mu_bounds = tuple(map(float, mu_bounds.split(',')))
    mle_fit(data=data,
            output_file=output,
            ncores=ncores,
            nstarts=nstarts,
            pi0_bounds=pi0_bounds,
            mu_bounds=mu_bounds,
            mu=mu, chrom=chrom, bp_only=only_bp)


@cli.command()
@click.option('--bs-file', required=True, type=click.Path(exists=True),
              help="BGSModel genome model pickle file (contains B' and B)")
@click.option('--fit', required=True, type=click.Path(exists=True),
              help='pickle file of fitted results')
#@click.option('--force-feature', default=None,
#              help='force all predictions using DFE estimates of this feature (experimental)')
@click.option('--ncores',
              help='number of cores to use for multi-start optimization',
              type=int, default=1)
@click.option('--mu', default=None, type=float, help="optional mutation rate to predict under (MLE used otherwise)")
@click.option('--output', required=True,
              type=click.Path(dir_okay=False, writable=True),
              help="pickle file for results")
@click.option('--split', default=False, is_flag=True,
              help="split into different files for each feature "
              "(uses '<output>_<feature_a>.bed', etc)")
def subrate(bs_file, fit, 
            #force_feature,
            ncores,
            mu,
            output, split):
    """
    Take a fit and predict the substitution rates.
    """
    m = BGSModel.load(bs_file)
    fits = pickle.load(open(fit, 'rb'))
    bpfit = fits['mbp']

    rdf = m.ratchet_df(bpfit, mu=mu, ncores=ncores)
    msg = "feature mismatch between BGSModel and fit!"
    assert bpfit.features == list(m.genome.segments.feature_map.keys()), msg
    rdf = rdf.sort_values(['chrom', 'start', 'end'])
    if not split:
        rdf.to_csv(output, sep='\t', header=True, index=False)
        return
    for feature in m.genome.segments.feature_map:
        if output.endswith('.bed'):
            filename = output.replace('.bed', f'_{feature}.bed')
        else:
            filename = f'{output}_{feature}.bed'
        rdfx = rdf.loc[rdf['feature'] == feature]
        rdfx.to_csv(filename, sep='\t', header=True, index=False)


@cli.command()
@click.option('--data', required=True, default=None, type=click.Path(exists=True),
              help="pickle of pre-computed summary statistics")
@click.option('--fit', default=None, type=click.Path(exists=True),
              help=('pickle file of fitted results, for starting at MLE '
                    '(ignore for random starts)'))
@click.option('--blocksize', type=int,
              help='the blocksize, in consecutive number of windows')
@click.option('--blockwidth', type=int, help="the blockwidth in basepairs")
@click.option('--blocknum', default=None, type=int,
              help='which block to run the fit on (e.g. for cluster use)')
@click.option('--blockfrac', default=None, type=float,
              help='for not a full jackknife, drop the block at this fraction position of the genome'
                   "(this approach doesn't require a prior knowledge of number of blocks")
@click.option('--mu', help='fixed mutation rate (by default, free)', 
              default=None)
@click.option('--mu-bounds', help='bounds for μ', default=MU_BOUNDS)
@click.option('--pi0-bounds', help='bounds for π0', default=PI0_BOUNDS)
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
def jackblock(data, fit, blocksize, blockwidth, blocknum, blockfrac, mu, 
              mu_bounds, pi0_bounds, output, fit_dir, chrom, ncores, 
              nstarts, include_bs):
    """
    Run the block-jackknifing routine. 

    There are three modes:
        1. Run with --blocksize set only, and all blocks will be 
            jackknifed.
        2. Run with --blocksize and --blocknum, and only that 
            block number is left out (e.g. for cluster use).
        3. --blockfrac and --blocksize, exclude blocks at the fraction 
            specified (which is the fraction of total genome); this is for 
            cluster use when we need to parallelize things (approximate jackknife)

    <!> blocksize is in NUMBER of consecutive blocks to jackknife over (e.g. blockwidth = window width x blocksize)
    """
    if blockfrac is not None or blockwidth is not None:
        # we need to figure out the block width or num blocks
        bins = load_pickle(data)['bins']
        if blockwidth is not None:
            # get the the number of window this blocksize in bp is
            blocksize = int(blockwidth / bins.width)
        blocks = block_bins(bins, blocksize)
        nblocks = len(blocks)
        if blockfrac is not None:
            blocknum = int(blockfrac * nblocks)
    logging.info("blockwidth={blockwidth}bp, blockfrac={blockfrac}, blocknum={blocknum}")
    assert blocksize > 1, "blocksize must be > 1"
    mu = sterialize_mu(mu)
    pi0_bounds = tuple(map(float, pi0_bounds.split(',')))
    mu_bounds = tuple(map(float, mu_bounds.split(',')))
    start = None
    if fit is not None:
        # recycle the MLE for a start (saves time)
        start = fit.theta_
        nstarts = None
    mle_fit(data=data,
            output_file=output,
            ncores=ncores,
            mu=mu,
            pi0_bounds=pi0_bounds,
            mu_bounds=mu_bounds,
            nstarts=nstarts,
            start=start,
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
@click.option('--mu-bounds', help='bounds for μ', default=MU_BOUNDS)
@click.option('--pi0-bounds', help='bounds for π0', default=PI0_BOUNDS)
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
def loo(data, fit, mu, mu_bounds, pi0_bounds, output, fit_dir, chrom,
              ncores, nstarts, include_bs):
    """
    Leave-one-out chromosome fit. 
    
    This is used for estimating out-sample R2 by leaving out a chromosome,
    fitting to the rest of the genome, and predicting the observed diversity on
    the excluded chromosome.
    """
    mu = sterialize_mu(mu)
    pi0_bounds = tuple(map(float, pi0_bounds.split(',')))
    mu_bounds = tuple(map(float, mu_bounds.split(',')))
    mle_fit(data=data,
            output_file=output,
            ncores=ncores,
            mu=mu,
            pi0_bounds=pi0_bounds,
            mu_bounds=mu_bounds,
            nstarts=nstarts,
            loo_chrom=chrom,
            bp_only=True)

@cli.command()
@click.argument("config", required=True)
def newfit(config):
    """
    Set up the directory structure for a fit for an MLE fit, leave-one-out 
    chromosome fits, and jackknifes.
    """
    yaml_file = config
    dir = os.path.realpath(os.path.basename(yaml_file).replace(".yml", ""))
    curr_dir = os.path.realpath(".")
    
    os.mkdir(dir)
    
    os.symlink(os.path.join(curr_dir, "Snakefile"), os.path.join(dir, "Snakefile"))
    os.symlink(os.path.join(curr_dir, yaml_file), os.path.join(dir, yaml_file))
    
    os.makedirs(os.path.join(dir, "logs", "out"))
    os.makedirs(os.path.join(dir, "logs", "error"))


@cli.command()
@click.argument("config", required=False, default=None)
@click.option("--full", is_flag=True, help="generate full-coverage tracks too")
@click.option("--bed", type=(str, str), multiple=True, help="feature type, bed file pair")
@click.option('--seqlens', required=True, type=click.Path(exists=True),
              help='tab-delimited file of chromosome names and their length')
def tracks(config, full, bed, seqlens):
    """
    Build the tracks for a YAML fit config file.
    """
    if config is not None:
        assert len(bed) == 0, "config argument and --bed options cannot be combined"
        with open(config) as f:
            features = yaml.safe_load(f)['features']
        features = features['features']
    else:
        assert len(bed) > 0, "if config argument not set, --bed must be set"
        features = dict(bed)

    priority = list(features.keys())
    beds = {c: read_bed3(f) for c, f in features.items()}
    masks = combine_features(beds, priority, load_seqlens(seqlens))
    res = masks_to_ranges(masks, labels=priority)
    for chrom, ranges in res.items():
        for range in ranges:
            start, end, label = range
            if not full:
                if label is None:
                    continue
            else:
                label = 'other' if label is None else label
            print(f"{chrom}\t{start}\t{end}\t{label}")


@cli.command()
@click.argument('fitdir', required=True)
@click.option('--jackwidth', default=10_000_000,
              help="the jackknife blocksize to use")
@click.option('--output', required=False, default=None,
              type=click.Path(writable=True),
              help='a pickle file of the entire fit object')
def collect(fitdir, jackwidth, output):
    """
    Load all the objects in the fit directory, run from a
    snakemake pipeline.
    """
    res = ModelDir(fitdir, jackwidth=jackwidth)
    if output is not None:
        assert output.endswith('.pkl'), "output filename should end in .pkl"
        res.save(output)
    for key, fit in res.fits.items():
        print('------ ' + ' '.join(key) + ' ------')
        print(fit)

    #for key, fit in res.fits.items():
    #    #print('-'*10 + f" window size: {si_format(window)}bp " + '-'*10)
    #    print(key)
    #    print(fit)


@cli.command()
@click.option('--bfile', required=True, help="the pickled B file")
@click.option('--simfile', required=True, help="simulation .npz")
@click.option('--output', required=True,
              type=click.Path(writable=True),
              help='a pickle file of the entire fit object')
@click.option('--t', required=True, help="list of selection coefficients to load in")
@click.option('--w', required=True, help="list of mutation rates to load in")
def bfix(bfile, simfile, output):
    """
    Given a set of empirically-estimated B values from simulations,
    load them into an existing B file, to fix issues with strong 
    selection.
    """
    sims = np.load(simfile)
    pass # TODO



@cli.command()
@click.argument('file', required=True)
@click.option('--only-Bp', default=False, is_flag=True, help="only calculate B'")
@click.option('--only-B', default=False, is_flag=True, help="only calculate B")
def inspect(file, only_bp, only_b):
    """
    Load a pickle model fit file and print to screen.
    """
    fit = load_pickle(file)
    if not only_bp and not only_b:
        print(fit)
    if only_bp:
        print(fit['mbp'])
    if only_b:
        print(fit['mb'])
 
if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s',
                        level=logging.INFO)
    res = cli()
