import warnings
import logging
import pickle
import numpy as np
from bgspy.models import BGSModel
from bgspy.utils import load_seqlens, load_pickle
from bgspy.sim_utils import mutate_simulated_tree
from bgspy.genome import Genome
from bgspy.data import GenomeData
from bgspy.likelihood import SimplexModel

AVOID_CHRS = set(('M', 'chrM', 'chrX', 'chrY', 'Y', 'X'))


def summarize_data(# annotation
         seqlens_file, recmap_file, neut_file, access_file, fasta_file,
         bs_file, output_file,
         # window size
         window,
         # data
         counts_dir=None, 
         # optional sim data for sim run
         sim_tree_file=None, sim_chrom=None, sim_mu=None,
         # filters
         thresh_cM=0.3, outliers=(0.0, 0.995),
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
    g.load_recmap(recmap_file)
    gd = GenomeData(g)
    is_sim = sim_tree_file is not None
    if counts_dir is not None:
        assert not is_sim, "set either counts directory or sim tree file"
        gd.load_counts_dir(counts_dir)
    else:
        assert counts_dir is None, "set either counts directory or sim tree file"
        assert sim_chrom is not None, "sim_chrom needs to be specified when loading from a treeseq"
        assert sim_mu is not None, "set a mutation rate to turn treeseqs to counts"
        ts = mutate_simulated_tree(sim_tree_file, rate=sim_mu)
        gd.load_counts_from_ts(ts, chrom=sim_chrom)

    gd.load_neutral_masks(neut_file)
    gd.load_accessibile_masks(access_file)
    gd.load_fasta(fasta_file, soft_mask=True)
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


def mle_fit(data, output_file, ncores=70, nstarts=200, 
            verbose=True, chrom=None, 
            start=None, ignore_B=False):
    """
    Load the binned data, fit B' (and optionally B) models, and
    save the results.
    """
    dat = load_pickle(data)
    bins, Y, bp = dat['bins'], dat['Y'], dat['bp']
    w, t = dat['w'], dat['t']
    features = dat['features']
    b = dat.get('b', None)

    # deal with start
    if start is not None:
        starts = [start]
    else:
        starts = nstarts

    msg = "fitting B' model"
    if chrom is not None:
        msg += f" (leaving out {chrom})"
    logging.info(msg)
    m_bp = SimplexModel(w=w, t=t, logB=bp, Y=Y,
                        bins=bins, features=features)
    if chrom is not None:
        jk_opt = m_bp.jackknife_chrom(starts=starts, ncores=ncores, 
                                       chrom=chrom)
        # we need to load manually...
        m_bp._load_optim(jk_opt)
    else:
        m_bp.fit(starts=starts, ncores=ncores)

    msg = "saving B' model results"
    logging.info(msg)
    obj = {'mbp': m_bp}
    with open(output_file, 'wb') as f:
        pickle.dump(obj, f)

    if b is not None and not ignore_B:
        logging.info("fitting B model")
        m_b = SimplexModel(w=w, t=t, logB=b, Y=Y,
                           bins=bins, features=features)
        if chrom is not None:
            jk_opt = m_b.jackknife_chrom(starts=nstarts, ncores=ncores, chrom=chrom)
            m_b._load_optim(jk_opt)
        else:
            m_b.fit(starts=nstarts, ncores=ncores)
    else:
        return # we're done, so leave

    logging.info("adding B to model results")
    obj = {'mb': m_b, 'mbp': m_bp}
    with open(output_file, 'wb') as f:
        pickle.dump(obj, f)


