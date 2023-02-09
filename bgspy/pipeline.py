import warnings
import pickle
import numpy as np
from bgspy.models import BGSModel
from bgspy.utils import load_seqlens 
from bgspy.sim_utils import mutate_simulated_tree
from bgspy.genome import Genome
from bgspy.data import GenomeData

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
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # infer the genome name if not supplies
    name = seqlens_file.replace('_seqlens.tsv', '') if name is None else name
    seqlens = load_seqlens(seqlens_file)
    if only_autos:
        seqlens = {c: l for c, l in seqlens.items() if c not in AVOID_CHRS}
    vprint("-- loading genome --")
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
    vprint("-- binning pairwise diversity --")
    bgs_bins = gd.bin_pairwise_summaries(window,
                                         filter_accessible=True,
                                         filter_neutral=True)

    del gd  # we don't need it after this, it's summarized
    # mask bins that are outliers
    bgs_bins.mask_outliers(outliers)

    # genome models
    vprint("-- loading Bs --")
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
        vprint("-- binning B --")
        b = bgs_bins.bin_Bs(gm.BScores)

    # B' is always fit
    vprint("-- binning B' --")
    bp = bgs_bins.bin_Bs(gm.BpScores)

    # get the diversity data
    vprint("-- making diversity matrix Y --")
    Y = bgs_bins.Y()

    vprint("-- saving pre-fit model data --")
    with open(output_file, 'wb') as f:
        dat = {'bins': bgs_bins, 'Y': Y, 
               'bp': bp, 'gm': gm, 'features': features}
        if not bp_only:
            dat['b'] = b
        pickle.dump(dat, f)



def fit(data, output_file, ncores=70, nstarts=200):
    dat = load_pickle(data)
    bins, Y, bp, gm = (dat['bins'], dat['Y'], 
                       dat['bp'], dat['gm'])
    features = dat['features']
    b = dat.get('b', None)

    # TODO FIT

    obj = {'mb': m_b, 'mbp': m_bp}
    with open(fit_outfile, 'wb') as f:
        pickle.dump(obj, f)
    vprint("-- model saved --")


def boostrap():
    # TODO
    #  --------- bootstrap ----------
    bootstrap = B is not None
    msg = "cannot do both bootstrap and jackknife"
    if bootstrap:
        assert fit_chrom is None, "cannot set fit_chrom"
        assert J is None, msg
        vprint('note: recycling MLE for bootstrap')
        starts_b = nstarts if not recycle_mle else [m_b.theta_] * nstarts
        starts_bp = nstarts if not recycle_mle else [m_bp.theta_] * nstarts

        nlls_b, thetas_b = None, None # in case only-Bp
        if not bp_only:
            print("-- bootstrapping B --")
            nlls_b, thetas_b = m_b.bootstrap(nboot=B, blocksize=blocksize,
                                             starts=starts_b, ncores=ncores)
        print("-- bootstrapping B' --")
        nlls_bp, thetas_bp = m_bp.bootstrap(nboot=B, blocksize=blocksize,
                                            starts=starts_bp, ncores=ncores)
        if bootjack_outfile is not None:
            np.savez(bootjack_outfile, 
                     nlls_b=nlls_b, thetas_b=thetas_b,
                     nlls_bp=nlls_bp, thetas_bp=thetas_bp)
            print("-- bootstrapping results saved --")
            return

    #  --------- jackknife ----------
    jackknife = J is not None
    if jackknife:
        assert fit_chrom is None, "cannot set fit_chrom"
        assert B is None, msg
        vprint('note: recycling MLE for jackknife')
        starts_b = nstarts if not recycle_mle else [m_b.theta_] * nstarts
        starts_bp = nstarts if not recycle_mle else [m_bp.theta_] * nstarts

        nlls_b, thetas_b = None, None  # in case only-Bp
        if not bp_only:
            print("-- jackknife B --")
            nlls_b, thetas_b = m_b.jackknife(njack=J, 
                                             starts=starts_b, ncores=ncores)
        print("-- jackknife B' --")
        nlls_bp, thetas_bp = m_bp.jackknife(njack=J,
                                            starts=starts_bp, ncores=ncores)
        if bootjack_outfile is not None:
            np.savez(bootjack_outfile, 
                     nlls_b=nlls_b, thetas_b=thetas_b,
                     nlls_bp=nlls_bp, thetas_bp=thetas_bp)
            print("-- jackknife results saved --")
            return

    #  --------- leave-one-out ----------
    if loo_chrom is not False:
        assert fit_chrom is None, "cannot set fit_chrom"
        # can be False (don't do LOO), True (do LOO for all chroms), or string
        # chromosome name (do LOO, exluding chromosome)
        starts_b = nstarts if not recycle_mle else [m_b.theta_] * nstarts
        starts_bp = nstarts if not recycle_mle else [m_bp.theta_] * nstarts
        b_r2 = None
        if loo_chrom is True:
           # iterate through everything
           out_sample_chrom = None
        else:
           # single chrom specified...
           out_sample_chrom = loo_chrom
        if not bp_only:
            print("-- leave-one-out R2 estimation for B --")
            b_r2 = m_b.loo_chrom_R2(starts=loo_nstarts,
                                    out_sample_chrom=out_sample_chrom,
                                    loo_fits_dir=loo_fits_dir,
                                    ncores=ncores)
        print("-- leave-one-out R2 estimation for B' --")
        bp_r2 = m_bp.loo_chrom_R2(starts=loo_nstarts,
                                  out_sample_chrom=out_sample_chrom,
                                  loo_fits_dir=loo_fits_dir,
                                  ncores=ncores)

        np.savez(r2_file, b_r2=b_r2, bp_r2=bp_r2)


