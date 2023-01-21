import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from bgspy.likelihood import SimplexModel
from bgspy.utils import midpoint_linear_interp

def ratchet_df(model, fit):
    """
    Predict the ratchet given a model and fits.

    Note that this depends on *existing* rescaling in the
    Segments object.
    """
    m = model
    if isinstance(fit, SimplexModel):
        # repeat one for each feature (for simplex, these are all same)
        mus = np.repeat(mus, fit.nf)

    nf = F.shape[1]
    assert fit.nf == nf

    F = m.genome.segments.F
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
            muw = mu*W[k, i]

            r_this_selcoef = midpoint_linear_interp(r_interp[:, k, :], m.w, muw)
            R_this_selcoef = midpoint_linear_interp(R_interp[:, k, :], m.w, muw)

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

