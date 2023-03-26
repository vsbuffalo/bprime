import warnings
import logging
import os
from os.path import basename, join
import pickle
import tskit as tsk
import numpy as np
from bgspy.models import BGSModel
from bgspy.utils import load_seqlens, load_pickle, save_pickle
from bgspy.sim_utils import mutate_simulated_tree
from bgspy.genome import Genome
from bgspy.data import GenomeData
from bgspy.likelihood import SimplexModel

AVOID_CHRS = set(('M', 'chrM', 'chrX', 'chrY', 'Y', 'X'))


class ModelDir:
    """ 
    This class is to manage large-scale fits based on the snakemake pipeline.
    """
    def __init__(self, dir, jackknife_blocksize=25_000_000):
        """
        """
        self.dir = dir 
        self._get_fits()
        
    def _get_fits(self, jackknife_blocksize=25_000_000):
         dirs = {int(f.replace('fit_', '')): join(self.dir, f) for 
                 f in os.listdir(self.dir) if f.startswith('fit_')}
         logging.info("loading fits...")
         fits = {w: (f, load_pickle(join(f, 'mle.pkl'))['mbp']) for w, f 
                 in dirs.items() if os.path.exists(join(f, 'mle.pkl'))}
         self._fits = fits
         out_fits = {}
         r2s = {}
         for window, (fit_dir, fit) in self._fits.items(): 
             fit.std_error_type = None  # hack for back compat; remove
             fit.load_jackknives(join(fit_dir, f'jackknife/{jackknife_blocksize}/'))
             fit.load_loo(join(fit_dir, 'loo_chrom/'))
             out_fits[window] = fit
         self.fits = out_fits
         self.r2s = r2s

    def save(self, output):
        save_pickle(self, output)


def summarize_data(# annotation
         seqlens_file, recmap_file, neut_file, access_file, fasta_file,
         bs_file, output_file,
         # window size
         window,
         # data
         counts_dir,
         # filters
         thresh_cM=1, outliers=(0.0, 0.995), soft_mask=True,
         # other
         bp_only=False, name=None, only_autos=True, verbose=True):
    """
    Take all the genome data, the allele counts data (or simulated trees),
    and the B/B' maps and combine, computing and outputting to pickle:

        1. Y, the pairwise summaries matrix (with filtering, etc)
        2. The binned B/B' values at the scale matching Y.
        3. The bins object corresponding to the binned values.
        4. The loaded BGSModel containing the B/B's.

    """
    # infer the genome name if not supplies
    name = seqlens_file.replace('_seqlens.tsv', '') if name is None else name
    seqlens = load_seqlens(seqlens_file)
    if only_autos:
        seqlens = {c: l for c, l in seqlens.items() if c not in AVOID_CHRS}
    logging.info("loading genome")
    g = Genome(name, seqlens=seqlens)
    # the recombination map is used for the cM end filtering
    g.load_recmap(recmap_file)
    gd = GenomeData(g)
    gd.load_counts_dir(counts_dir)
    gd.load_neutral_masks(neut_file)
    gd.load_accessibile_masks(access_file)
    gd.load_fasta(fasta_file, soft_mask=soft_mask)
    gd.trim_ends(thresh_cM=thresh_cM)

    # bin the diversity data
    logging.info("binning pairwise diversity") 
    bgs_bins = gd.bin_pairwise_summaries(window,
                                         filter_accessible=True,
                                         filter_neutral=True)

    del gd  # we don't need it after this, it's summarized
    # mask bins that are outliers
    bgs_bins.mask_outliers(outliers)

    # genome models
    logging.info("loading Bs")
    gm = BGSModel.load(bs_file)

    # features -- load for labels
    features = list(gm.segments.feature_map.keys())

    # handle if we're doing B too or just B'
    m_b = None
    if gm.Bs is None:
        warnings.warn(f"BGSModel.Bs is not set, so not fitting classic Bs (this is likely okay.)")
        bp_only = True  # this has to be true now, no B

    # bin Bs
    if not bp_only:
        logging.info("binning B")
        b = bgs_bins.bin_Bs(gm.BScores)

    # B' is always fit
    logging.info("binning B'")
    bp = bgs_bins.bin_Bs(gm.BpScores)

    # get the diversity data
    logging.info("making diversity matrix Y")
    Y = bgs_bins.Y()

    logging.info("saving pre-fit model data")
    with open(output_file, 'wb') as f:
        dat = {'bins': bgs_bins, 'Y': Y, 
               'bp': bp, 'features': features,
               't': gm.t, 'w': gm.w}
        if not bp_only:
            dat['b'] = b
        pickle.dump(dat, f)


def summarize_sim_data(sim_tree_files,
                       bs_file, 
                       output_file,
                       # window size
                       window,
                       n=None, # number of samples
                       neut_file=None, access_file=None,
                       sim_mu=None,
                       # other
                       bp_only=False, verbose=True):

    """
    """
    # because we're likely processing a lot of sim
    # trees, we store metadata for downstream stuff
    #md = dict(sim_tree_file = sim_tree_file, sim_mu=sim_mu)
    logging.info("loading tree(s)")
    trees = dict()
    metadata = dict()
    for sim_tree_file in sim_tree_files:
        md = dict(sim_tree_file = sim_tree_file,
                  sim_mu=sim_mu)
        tree = tsk.load(sim_tree_file)
        # the metadata from the sims
        tmd = tree.metadata['SLiM']['user_metadata']
        # add in metadata from SLiM user md
        md = md | tmd

        # get the chromosome from the SLiM user metadata
        chrom = tmd['chrom']
        # everything's a list in slim metadata, but double check
        assert isinstance(chrom, list)
        assert len(chrom) == 1
        chrom = chrom[0]  # unpack it
        metadata[chrom] = md

        # subsample if we need do
        if n is not None:
            inds = np.random.choice(tree.samples(), n, replace=False)
            tree = tree.simplify(samples=inds)

        # add mutations to the tree
        ts = mutate_simulated_tree(tree, rate=sim_mu)
        trees[chrom] = ts

    gd = GenomeData.from_ts_dict(trees)
    gd.load_neutral_masks(neut_file)
    gd.load_accessibile_masks(access_file)
 
    # bin the diversity data
    logging.info("binning pairwise diversity") 
    bgs_bins = gd.bin_pairwise_summaries(window, 
                                         filter_accessible=True,
                                         filter_neutral=True)

    # genome models
    logging.info("loading Bs")
    gm = BGSModel.load(bs_file)
    metadata['bs_file'] = basename(bs_file)

    # features -- load for labels
    features = list(gm.segments.feature_map.keys())

    # handle if we're doing B too or just B'
    m_b = None
    if gm.Bs is None:
        warnings.warn(f"BGSModel.Bs is not set, so not fitting classic Bs (this is likely okay.)")
        bp_only = True  # this has to be true now, no B

    # bin Bs
    if not bp_only:
        logging.info("binning B")
        b = bgs_bins.bin_Bs(gm.BScores)

    # B' is always fit
    logging.info("binning B'")
    bp = bgs_bins.bin_Bs(gm.BpScores)

    # get the diversity data
    logging.info("making diversity matrix Y")
    Y = bgs_bins.Y()

    logging.info("saving pre-fit model data")
    with open(output_file, 'wb') as f:
        dat = {'bins': bgs_bins, 'Y': Y,
               'bp': bp, 'features': features,
               't': gm.t, 'w': gm.w, 'md': metadata}
        if not bp_only:
            dat['b'] = b
        pickle.dump(dat, f)


def mle_fit(data, output_file, ncores=70, nstarts=200,
            mu=None, 
            pi0_bounds=None, mu_bounds=None,
            verbose=True, 
            loo_chrom=None, chrom=None,  # for LOO on only one chrom
            blocksize=None, blocknum=None,  # for block-jackknifing
            start=None, bp_only=False):
    """
    The general fitting pipeline, for MLE fits, chromosome-specific 
    fits, leave-one-out chromosome fits, and block-jackknifing.

    Basic pipeline:
      - Load the binned data
      - Fit B' (and optionally B) model, possibly LOO or single
        chromosome.
      - Save the results.
    save the results.
    """
    # deal with start
    if start is not None:
        starts = [start]
    else:
        starts = nstarts

    msg = "fitting B' model"
    if chrom is not None:
        msg += f" (leaving out {chrom})"
    logging.info(msg)
    m_bp = SimplexModel.from_data(data,
                        log10_pi0_bounds=np.log10(pi0_bounds),
                        log10_mu_bounds=np.log10(mu_bounds))

    ## fitting (main MLE, loo-chrom, block jackknife)
    # NOTE: for loo/block JK we need to manually load the 
    # optim results.
    if loo_chrom is not None:
        m_bp = m_bp.jackknife_chrom(starts=starts, ncores=ncores,
                                       chrom=loo_chrom, mu=mu)
    elif blocksize is not None:
        assert blocksize > 1, "blocksize must be > 1"
        m_bp = m_bp.jackknife_block(starts=starts, ncores=ncores,
                                    blocksize=blocksize, 
                                    blocknum=blocknum, mu=mu)
    else:
        m_bp.fit(starts=starts, ncores=ncores,
                 mu=mu, chrom=chrom)

    # save the B' results
    if bp_only:
        msg = "saving B' model results"
        logging.info(msg)
        obj = {'mbp': m_bp}
        with open(output_file, 'wb') as f:
            pickle.dump(obj, f)
        return

    # we also need to fit the classic B
    logging.info("fitting B model")
    m_b = SimplexModel.from_data(data, use_classic_B=True)
    if blocksize is not None:
        raise NotImplementedError("B' block jackknifing not implemented")
    if loo_chrom is not None:
        m_b = m_b.jackknife_chrom(starts=nstarts, ncores=ncores, 
                                     chrom=loo_chrom, mu=mu)
    else:
        m_b.fit(starts=nstarts, ncores=ncores, chrom=chrom, mu=mu)

    # save the B results
    logging.info("adding B to model results")
    obj = {'mb': m_b, 'mbp': m_bp}
    with open(output_file, 'wb') as f:
        pickle.dump(obj, f)



