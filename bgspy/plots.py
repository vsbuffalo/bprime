import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import statsmodels.api as sm
import statsmodels.formula.api as smf
from bgspy.utils import mean_ratio, argsort_chroms, center_and_scale
import matplotlib as mpl
lowess = sm.nonparametric.lowess

#  ---- some figure size preset for the paper


def mm_to_inches(x):
    # from https://writing.stackexchange.com/questions/21658/what-is-the-image-size-in-scientific-paper-if-indicated-as-a-single-1-5-or-2-c 
    return 0.0393701 * x


img_size = dict(one=mm_to_inches(90),
                onehalf=mm_to_inches(140),
                two=mm_to_inches(190))

asp_ratio = dict(golden=(1 + np.sqrt(5))/2, one=1, two=2)
sizes = {(k, ar): np.round((v, v/asp_ratio[ar]), 4) for
         k, v in img_size.items() for ar in asp_ratio}


def center_scale(x):
    return (x-x.mean())/x.std()


def get_figax(figax, **kwargs):
    if figax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig, ax = figax
    return fig, ax


def smooth(x, y, frac=None):
    """
    Smooth a line for visual clarity.
    """
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    if frac is None:
        return x, y
    sx, sy = lowess(y, x, frac=frac).T
    return sx, sy


def chromosome_ticks_formatter(scale):
    """
    Return a ticker formatter function to scale the
    x-axis by some physical length.

    Use like:
      ticks_x = chromosome_ticks_formatter(1e6) # for Mb
      ax.xaxis.set_major_formatter(ticks_x)
    """

    def formatter(x, pos):
        return '{0:g}'.format(x/scale)

    return mpl.ticker.FuncFormatter(formatter)


def resid_fitted_plot(model, color=True, figax=None):
    """
    Residuals-fitted plot, with lowess.
    """
    m = model
    fig, ax = get_figax(figax)
    lowess = sm.nonparametric.lowess
    fitted, resid = m.predict(), m.resid()
    bins = m.bins.flat_bins()
    if color != 'chrom':
        if color is True:
            col = np.arange(len(bins))
        else:
            col = 'k'
        ax.scatter(fitted, resid, c=col, linewidth=0, s=4, alpha=1)
    else:
        chroms = set([c for c, _, _ in bins])
        for chrom in chroms:
            chrom_idx = np.array([i for i, (c, s, e) in enumerate(bins) if c == chrom])
            ax.scatter(fitted[chrom_idx], resid[chrom_idx], label=chrom, linewidth=0, s=4, alpha=1)
        ax.legend()
    lw = lowess(resid, fitted)
    ax.plot(*lw.T, c='r')
    ax.axhline(0, linestyle='dashed', c='cornflowerblue', zorder=-2)
    ax.set_ylabel('residuals')
    ax.set_xlabel('predicted $\\hat{\pi}$')
    return fig, ax


def model_diagnostic_plots(model, figax=None):
    fig, ax = get_figax(figax, ncols=2, nrows=2)
    resid_fitted_plot(model, figax=(fig, ax[0, 0]))
    return fig, ax


def predict_sparkplot(model,
                      label='prediction',
                      ratio=True, figax=None):
    """
    In progress TODO
    """
    m = model
    fig, ax = get_figax(figax)
    bins = m.bins.flat_bins(filter_masked=False)
    midpoints = m.bins.cum_midpoints(filter_masked=False)

    predicts = m.predict()
    # fill the predicted values into the full unmask-filtered matrix
    predicts_full = model.bins.merge_filtered_data(predicts)

    pi = []
    for chrom in m.bins.seqlens:
        _, pis = model.bins.pi_pairs(chrom)
        pi.append(pis)

    y = predicts_full
    if ratio:
        y = mean_ratio(y)

    ax.plot(midpoints, y, label=label)
    if ratio:
        pi = mean_ratio(pi)
    ax.plot(midpoints, pi, c='g', alpha=0.4, label='data')
    return fig, ax


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def predict_chrom_plot(model, chrom, ratio=True, center_scale=False,
                       label='prediction', lw=2,
                       pred_col='cornflowerblue', pi_col='0.22',
                       alpha_predict=1, alpha_pi=1,
                       xlab='position',
                       add_r2=False, figax=None, predict_kwargs=None):
    m = model
    fig, ax = get_figax(figax)
    midpoints, pi = model.bins.pi_pairs(chrom)
    midpoints = to_mb(midpoints)
    bins = m.bins.flat_bins(filter_masked=False)
    chrom_idx = np.array([i for i, (c, s, e) in enumerate(bins) if c == chrom])
    
    if predict_kwargs is not None:
        predicts = m.predict(**predict_kwargs)
    else:
        predicts = m.predict()
    # fill the predicted values into the full unmask-filtered matrix
    predicts_full = model.bins.merge_filtered_data(predicts)

    y = predicts_full[chrom_idx]
    ylab = "$\\pi$"
    msg = "center_scale and ratio cannot be set"
    if ratio:
        assert not center_scale, msg
        y = mean_ratio(y)
        ylab = "$\\pi/\\bar{\\pi}$"
    if center_scale:
        assert not ratio, msg
        y = center_and_scale(y)
    #pi = savitzky_golay(pi, window_size=5, order=1)
    ax.plot(midpoints, y, label=label, linewidth=lw, c=pred_col,
            alpha=alpha_predict, zorder=3)
    if ratio:
        assert not center_scale, msg
        pi = mean_ratio(pi)
    if center_scale:
        assert not ratio, msg
        pi = center_and_scale(pi)
    ax.plot(midpoints, pi, label='data', linewidth=lw, c=pi_col,
            alpha=alpha_pi)
    if add_r2:
        ax.set_title(f"$R^2 = {np.round(model.R2(), 2)}$")
    ax.set_ylabel(ylab)
    if xlab is not None:
        ax.set_xlabel(xlab)
    return fig, ax


def surface_plot(x, y, z, xlabel=None, ylabel=None,
                 scale=None, ncontour=None, contour_ndigits=2,
                 figax=None, **kwargs):
    """
    Create a surface plot (e.g. for visualizing likelihood surfaces, functions
    with a 2D domain, etc.). This is mostly for convenience; it has a sensible
    set of visual defaults.

    This wraps plt.pcolormesh().
    """
    fig, ax = get_figax(figax)
    ax.pcolormesh(x, y, z, **kwargs)
    if scale is not None and scale != 'linear':
        avail_scales = ('loglog', 'semilogx', 'semilogy')
        if not scale in avail_scales:
            raise ValueError(f"scale must be one of {', '.join(avail_scales)}")
        getattr(ax, scale)()
    if ncontour is not None:
        cs = ax.contour(x, y, z, ncontour, colors='0.44',
                        linestyles='dashed', linewidths=0.8, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def predicted_observed(fit, n_bins=40, smooth_col = 'orange',
                       c='0.22', s=15, use_lowess=True, frac=0.1,
                       alpha=0.6,
                       use_B=False, figax=None):
    equal_num = False
    fig, ax = get_figax(figax)
    x, y = fit.predict(B=use_B), fit.pi()
    ax.scatter(x, y, s=s, c=c, alpha=alpha, linewidth=0)
    if equal_num:
        bins = [np.percentile(x, 100 * i / n_bins) for i in range(n_bins + 1)]
    else:
        bins = n_bins
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    bin_counts, *_ = binned_statistic(x, y, statistic=lambda x: len(x), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    keep_idx = bin_counts >= 10
    ax.scatter(bin_centers[keep_idx], bin_means[keep_idx], marker='o', color=smooth_col, s=2, 
               zorder=2)
    if use_lowess:
        sx, sy = lowess(y, x, frac=frac).T
        ax.plot(sx, sy, color=smooth_col, zorder=2)
    else:
        ax.plot(bin_centers[keep_idx], bin_means[keep_idx],color=smooth_col, zorder=2)
    m = ax.get_xlim()[0]
    #ax.axline((m, m), slope=1, zorder=1)
    return fig, ax

def binned_means_plot(df, min_n=None, gen=None,
                      stat='mean', s=5,
                      c='0.22', linewidth=0.5,
                      thin=None, zorder=None,
                      label=None, figax=None):
    fig, ax = get_figax(figax)

    if min_n is not None:
        df = df.loc[df['n'] > min_n]
    x = df['midpoint'].values
    if gen is not None:
        x = x / gen
    mean, sd = df[stat].values, df['sd'].values
    n = df['n'].values
    if thin is None:
        points = ax.scatter(x, mean, c=c, s=s, alpha=1, zorder=10, label=label)
        ax.errorbar(x, mean, 2*sd/np.sqrt(n), fmt='none',
                    c=points.get_edgecolor(),
                    elinewidth=linewidth, zorder=zorder)
    else:
        x, mean = x, mean
        se = (2*sd/np.sqrt(n))
        x, mean = x[::thin], mean[::thin]
        se = se[::thin]
        points = ax.scatter(x, mean, c=c, s=s, alpha=1,
                            zorder=10, label=label)
        ax.errorbar(x, mean, se,
                    fmt='none', c=points.get_edgecolor(), elinewidth=linewidth, zorder=zorder)
    return fig, ax


def chrom_resid_plot(obj, figax=None):
    fig, ax = get_figax(figax)
    df = pd.DataFrame(dict(chrom=obj.bins.chroms(), resid=obj.resid()))

    # clean up the chrom order
    df = df.iloc[argsort_chroms(df['chrom'])]
    sd = df['resid'].std()

    grpd = {c: d['resid'].values for c, d in df.groupby('chrom')}
    chroms = df['chrom'].unique()
    resids = [grpd[chrom] / sd for chrom in chroms]
    _ = ax.boxplot(resids, labels=[x.replace('chr', '') for x in chroms])
    ax.axhline(0, c='0.66')
    return fig, ax


def rec_resid(fit, annot_df, alpha=None, figax=None, scatter_kwargs={}):
    """
    Look at the residuals by recombination rate in a window.

    fit: the SimplexModel fit
    annot_df: a dataframe of windows with recombination rate column
              (can be produced by bgspy windowstats).
    """
    fig, ax = get_figax(figax)
    resid = fit.resid()
    bins = fit.bins.flat_bins()
    assert(len(resid) == len(bins))

    d = pd.DataFrame(bins)
    #d['resid'] = resid
    #d.columns = ('chrom', 'start', 'end', 'resid')
    d.columns = ('chrom', 'start', 'end')
    d = d.merge(annot_df, on=['chrom', 'start', 'end'])
    assert d.shape[0] > 0, "merge failed -- are windows same?"

    ax.scatter(d['rec'], d['resid'], **scatter_kwargs)
    ax.axhline(0, c='0.5', linestyle='dashed')
    #mod = smf.ols(formula='resid ~ rec', data=d).fit()
    #ax.axline((0, mod.params[0]), slope=mod.params[1])
    ax.set_ylabel('residual')
    ax.set_xlabel('recombination rate')
    return fig, ax
    #return mod


def annot_resid(fit, annot_df, figax=None, scatter_kwargs={}):
    """
    Look at the residuals by number of annotated basepairs 
    in a window. These basepairs should be putatively selected,
    e.g. coding.

    fit: the SimplexModel fit
    annot_df: a dataframe of windows with a counts column
              containing the counts of the annotated basepairs
              that are likely under selection
              (can be produced by bgspy windowstats).
    """
    fig, ax = get_figax(figax)
    resid = fit.resid()
    bins = fit.bins.flat_bins()
    assert(len(resid) == len(bins))

    d = pd.DataFrame(bins)
    d['resid'] = resid
    d.columns = ('chrom', 'start', 'end', 'resid')
    d = d.merge(annot_df)

    ax.scatter(d['prop'], d['resid'],  **scatter_kwargs)
    ax.axhline(0, c='0.5', linestyle='dashed')
    mod = smf.ols(formula='resid ~ log10_prop', data=d).fit()
    x = np.linspace(d['prop'].min(), d['prop'].max(), 100)
    print(x)
    ax.axline((0, mod.params[0]), slope=mod.params[1])
    ax.set_xlabel('percentage window overlapping feature')
    ax.set_ylabel('residual')
    ax.plot(mod.params[0] * np.log10(x)*mod.params[1])
    ax.semilogx()
    return mod

def to_mb(x):
    return np.array(x) / 1e6
