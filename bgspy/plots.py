import numpy as np
from math import ceil
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import statsmodels.api as sm
import scipy.stats as stats
from bgspy.theory import B_var_limit
from bgspy.utils import signif, mean_ratio

def center_scale(x):
    return (x-x.mean())/x.std()


lowess = sm.nonparametric.lowess

def get_figax(figax, **kwargs):
    if figax is None:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig, ax = figax
    return fig, ax

def resid_fitted_plot(model, color=True, figax=None):
    """
    Residuals-fitted plot, with lowess.
    """
    m = model
    fig, ax = get_figax(figax)
    lowess = sm.nonparametric.lowess
    fitted, resid = m.predict(), m.resid()
    if color != 'chrom':
        bins = m.bins.flat_bins()
        if color is True:
            col = np.arange(len(bins))
        else:
            col = 'k'
        ax.scatter(fitted, resid, c=col, linewidth=0, s=4, alpha=1)
    else:
        chroms = set([c for c, _, _ in m.bins])
        for chrom in chroms:
            chrom_idx = np.array([i for i, (c, s, e) in enumerate(m.bins) if c == chrom])
            ax.scatter(fitted[chrom_idx], resid[chrom_idx], label=chrom, linewidth=0, s=4, alpha=1)
        ax.legend()
    lw = lowess(resid, fitted)
    ax.plot(*lw.T, c='r')
    ax.axhline(0, linestyle='dashed', c='cornflowerblue', zorder=-2)
    return fig, ax


def model_diagnostic_plots(model, figax=None):
    m = model
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
    midpoints = m.bins.cumm_midpoints(filter_masked=False)

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
    if ratio:
        y = mean_ratio(y)

    ax.plot(midpoints, y, label=label)
    if ratio:
        pi = mean_ratio(pi)
    ax.plot(midpoints, pi, c='g', alpha=0.4, label='data')
    if add_r2:
        ax.set_title(f"$R^2 = {np.round(model.R2(), 2)}$")
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
