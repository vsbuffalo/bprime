import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from bgspy.utils import midpoint_linear_interp, signif

def ratchet_df(model, fit, predict_under_feature=None):
    """
    DEPRECATED
    Predict the ratchet given a model and fits.

    Note that this depends on *existing* rescaling in the
    Segments object.
    """
    m = model
    mus = fit.mle_mu
    W = fit.mle_W

    from bgspy.likelihood import SimplexModel # to prevent circular import
    if isinstance(fit, SimplexModel):
        # repeat one for each feature (for simplex, these are all same)
        mus = np.repeat(mus, fit.nf)

    F = m.genome.segments.F
    nf = F.shape[1]
    assert fit.nf == nf

    segments = m.genome.segments
    ranges = segments.ranges
    seglens = np.diff(segments.ranges, axis=1).squeeze()
    fmaps = segments.inverse_feature_map

    # get chroms
    chroms = []
    for chrom, indices in segments.index.items():
        chroms.extend([chrom] * len(indices))
    chroms = np.array(chroms)

    T = segments._segment_parts_sc16[2]
    R = 1/T
    with np.errstate(under='ignore'):
        r = R/seglens

    pred_rs = []
    pred_Rs = []
    pred_load = []
    chrom_col = []
    start_col, end_col = [], []
    seglen_col = []
    feature_col = []
    for i in range(nf):
        # get the substiution rates and other info *for this feature type*
        fidx = F[:, i]
        rs = r[..., fidx]
        Rs = R[..., fidx]
        nsegs = fidx.sum()
        mu = mus[i]

        r_interp = np.zeros(nsegs)
        # r_interp2 = np.zeros(nsegs)
        R_interp = np.zeros(nsegs)
        load_interp = np.zeros(nsegs)

        # iterate over each selection coefficient, adding it's contribution
        # to this feature type's ratchet rate for each segment.
        for k in range(len(m.t)):
            if predict_under_feature is None:
                # use the corresponding DFE estimate for this feature
                muw = mu*W[k, i]
            else:
                # force the DFE to follow one feature
                muw = mu*W[k, predict_under_feature]


            r_this_selcoef = midpoint_linear_interp(rs[:, k, :], m.w, muw)
            R_this_selcoef = midpoint_linear_interp(Rs[:, k, :], m.w, muw)

            try:
                assert np.all(r_this_selcoef <= muw)
            except:
                n_over = (r_this_selcoef > muw).sum()
                frac = signif(100*(r_this_selcoef > muw).mean(), 2)
                msg = (f"{n_over} segments ({frac}%) had estimated substitution rates "
                        "greater than the DFE-weighted mutation rate for the"
                        f" class s={m.t[k]}, feature '{fmaps[i]}'. This "
                        "occurrs most likely due to numeric error in solving "
                        "the nonlinear system of equations. These are set to zero.")
                warnings.warn(msg)

                # this is due to numeric issues I believe in interpolation,
                # so we bound the ratchet rate to the mutation rate for this
                # class since r < mu under neutrality and selection.
                # r_this_selcoef2[r_this_selcoef > muw] = muw
                r_this_selcoef[r_this_selcoef > muw] = muw
            r_interp += r_this_selcoef
            # r_interp2 += r_this_selcoef2
            #__import__('pdb').set_trace()
            R_interp += R_this_selcoef
            with np.errstate(under='ignore'):
                load_interp += np.log((1-2*m.t[k]))*R_this_selcoef

        pred_rs.extend(r_interp)
        pred_Rs.extend(R_interp)
        pred_load.extend(load_interp)

        chrom_col.extend(chroms[fidx])
        start_col.extend(ranges[fidx, 0])
        end_col.extend(ranges[fidx, 1])
        seglen_col.extend(seglens[fidx])
        fs = [fmaps[i]] * fidx.sum()
        feature_col.extend(fs)

    d = pd.DataFrame({'chrom': chrom_col,
                        'start': start_col,
                        'end': end_col,
                        'feature': feature_col,
                        'R': pred_Rs,
                        'r': pred_rs,
                        'seglen': seglen_col,
                        'load': pred_load})
    return d


def ratchet_df2(model, fit, mu=None, bootstrap=False, ncores=None,
                B_parts=False):
    """
    A better version of the function above.
    """
    m = model

    if bootstrap:
        mux, pi0, W = fit.normal_draw()
        if mu is not None:
            # not fixed mu, use the sample
            mux = mu
    else:
        # use the MLE
        if mu is None:
            mu = fit.mle_mu
        else:
            mu = mu
        W = fit.mle_W

    F = m.genome.segments.F
    ns, nf = F.shape
    nt = fit.nt
    assert fit.nf == nf

    segments = m.genome.segments
    ranges = segments.ranges
    seglens = np.diff(segments.ranges, axis=1).squeeze()
    fmaps = segments.inverse_feature_map

    # get chroms
    chroms = []
    for chrom, indices in segments.index.items():
        chroms.extend([chrom] * len(indices))
    chroms = np.array(chroms)

    # predict the Vs, Vms, and Ts for all segments
    Vs, Vms, Ts = segments._predict_segparts(fit, model.N, W=W, mu=mu, 
                                             ncores=ncores)
    if B_parts:
        return Vs, Vms, Ts

    R = 1/Ts
    with np.errstate(under='ignore'):
        # per bp
        r = R/seglens
 
    pred_rs = []
    pred_Rs = []
    pred_loads = []
    pred_Vs = []
    pred_Vms = []
    chrom_col = []
    start_col, end_col = [], []
    seglen_col = []
    feature_col = []
    # iterate over features
    for i in range(nf):
        fidx = F[:, i]
        pred_rs.extend(r[:, fidx].sum(axis=0))
        pred_Rs.extend(R[:, fidx].sum(axis=0))
        pred_Vs.extend(Vs[:, fidx].sum(axis=0))
        pred_Vms.extend(Vms[:, fidx].sum(axis=0))
        # the indexed r below is nt, nl long
        # the m.t is nt so we expand dim to sweep down
        # note: load has to be whole region!
        load = (np.log((1-2*m.t[:, None]))*R[:, fidx]).sum(axis=0)
        pred_loads.extend(load)
        chrom_col.extend(chroms[fidx])
        start_col.extend(ranges[fidx, 0])
        end_col.extend(ranges[fidx, 1])
        seglen_col.extend(seglens[fidx])
        fs = [fmaps[i]] * fidx.sum()
        feature_col.extend(fs)

    d = pd.DataFrame({'chrom': chrom_col,
                      'start': start_col,
                      'end': end_col,
                      'feature': feature_col,
                      'R': pred_Rs,
                      'r': pred_rs,
                      'V': pred_Vs,
                      'Vm': pred_Vms,
                      'load': pred_loads,
                      'seglen': seglen_col})
    return d


