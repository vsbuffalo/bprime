import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from bgspy.utils import mean_ratio, argsort_chroms
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


def predict_chrom_plot(model, chrom, ratio=True,
                       label='prediction',
                       add_r2=False, figax=None):
    m = model
    fig, ax = get_figax(figax)
    midpoints, pi = model.bins.pi_pairs(chrom)
    bins = m.bins.flat_bins(filter_masked=False)
    chrom_idx = np.array([i for i, (c, s, e) in enumerate(bins) if c == chrom])

    predicts = m.predict()
    # fill the predicted values into the full unmask-filtered matrix
    predicts_full = model.bins.merge_filtered_data(predicts)

    y = predicts_full[chrom_idx]
    ylab = "$\\pi$"
    if ratio:
        y = mean_ratio(y)
        ylab = "$\\pi/\\bar{\\pi}$"

    ax.plot(midpoints, y, label=label)
    if ratio:
        pi = mean_ratio(pi)
    ax.plot(midpoints, pi, c='g', alpha=0.4, label='data')
    if add_r2:
        ax.set_title(f"$R^2 = {np.round(model.R2(), 2)}$")
    ax.set_ylabel(ylab)
    ax.set_xlabel("position")
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


def binned_means_plot(df, min_n=None, gen=None,
                      stat='mean', s=5,
                      c='0.22', linewidth=0.5,
                      thin=None,
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
                    elinewidth=linewidth)
    else:
        x, mean = x, mean
        se = (2*sd/np.sqrt(n))
        x, mean = x[::thin], mean[::thin]
        se = se[::thin]
        points = ax.scatter(x, mean, c=c, s=s, alpha=1,
                            zorder=10, label=label)
        ax.errorbar(x, mean, se,
                    fmt='none', c=points.get_edgecolor(), elinewidth=linewidth)
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

