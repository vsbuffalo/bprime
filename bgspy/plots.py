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


def predict_chrom_plot(model, chrom, ratio=True,
                       label='prediction',
                       add_r2=False, figax=None):
    m = model
    fig, ax = get_figax(figax)
    pi_midpoints, pi = model.bins.pi_pairs(chrom)
    bins = m.bins.flat_bins()
    chrom_idx = np.array([i for i, (c, s, e) in enumerate(bins) if c == chrom])
    y = m.predict()[chrom_idx]
    if ratio:
        y = mean_ratio(y)
    midpoints = model.bins.midpoints()[chrom]
    ax.plot(midpoints, y, label=label)
    if ratio:
        pi = mean_ratio(pi)
    ax.plot(pi_midpoints, pi, c='g', alpha=0.4, label='data')
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

def chrom_plot(ll_tuple, chrom, figsize=(10, 5)):
    ll, pi0, pi0_ll, pi0_grid, ws, ts, binned_B, binned_pi, gwpi, pi0_mle, w_mle, t_mle, wi_mle, ti_mle, md = ll_tuple
    binned_B = binned_B[chrom]
    binned_pi = binned_pi[chrom]
    midpoints = binned_pi[1].mean(axis=1)
    pi = binned_pi[0]
    # sum over the features -- TODO check
    R = binned_B[0][:, wi_mle, ti_mle]
    fig, ax = plt.subplots(figsize=figsize)
    #ax.set_title(latex_label(ws[wi_mle], 'w', True, True) + ", " + latex_label(ts[ti_mle], 't', True, True))
    ax.plot(midpoints, pi)
    ax.plot(midpoints, pi0*np.exp(R), color='r')
    #ax.axhline(pi0, c='k')
    ax.axhline(gwpi, c='g')
    ax.set_ylabel('$\pi$')
    ax.set_xlabel('position')
    return fig, ax

def ll_grid(m, row_vals, col_vals,
            xlog10_format=True, ylog10_format=True,
            xlabel=None, ylabel=None,
            true=None, mle=None,
            ncontour=7,
            colorbar=True,
            decimals=2, mid_quant=0.5, interpolation='quadric'):
    assert(m.shape == (len(row_vals), len(col_vals)))
    vmin, vcenter, vmax = np.nanmin(m), np.quantile(m, mid_quant), np.nanmax(m)
    divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, ax = plt.subplots()
    nx, ny = len(col_vals), len(row_vals)
    im = ax.imshow(m, norm=divnorm, extent=[0, nx, 0, ny], origin='lower', interpolation=interpolation)
    #ax.imshow(m, norm=divnorm, interpolation=interpolation)
    xticks_idx = np.array(get_visible_ticks(ax, 'x'), dtype=int)
    yticks_idx = np.array(get_visible_ticks(ax, 'y'), dtype=int)
    #print(ax.get_yticks())
    ytick_interpol = tick_interpolator(row_vals[yticks_idx[:-1]])
    xtick_interpol = tick_interpolator(col_vals[xticks_idx[:-1]])

    xaxis_type = get_axis_type(col_vals)
    yaxis_type = get_axis_type(row_vals)
    x_labeller = latex_labeller(xlog10_format, xaxis_type, decimals)
    y_labeller = latex_labeller(ylog10_format, yaxis_type, decimals)

    ax.set_xticklabels(x_labeller(xtick_interpol(xticks_idx)))
    ax.set_yticklabels(y_labeller(ytick_interpol(yticks_idx)))

    if ncontour > 0:
        cs = ax.contour(m, ncontour, colors='0.55', linewidths=0.8, alpha=0.4)
        ax.clabel(cs, inline=True, fontsize=10,
                  fmt=lambda x: f"$-10^{{{np.round(np.log10(-x), 2)}}}$")

    if true is not None:
        true_x, true_y = nearest_indices(true, row_vals, col_vals)
        ax.scatter(true_x, true_y, c='r')
    if mle is not None:
        mle_x, mle_y = nearest_indices(mle, row_vals, col_vals)
        ax.scatter(mle_x, mle_y, c='0.44')
    if colorbar:
        fig.colorbar(im)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    return fig, ax


def marginal_plot(bfs, var, nbins,
                  add_interval=True, nstd=1,
                  fix={'mu': 1e-8, 'sh': 1e-2, 'rf': 1e-6, 'L': 1000, 'rbp': 1e-7},
                  figax=None, log=False):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()
    bf = bfs[0] # all use same training usually
    cols = ('mu', 'sh', 'L', 'rbp', 'rf')
    X = pd.DataFrame(bf.func.X, columns=cols)
    y = bf.func.y
    # bins
    x = X[var]
    if log:
        x = np.log10(x)
    bins = pd.cut(x, nbins).values

    X['y'] = y
    bin_col = f'{var}_bin'
    X[bin_col] = bins
    grp = X.groupby([bin_col]).mean().reset_index()
    grp_std = np.sqrt(X.groupby([bin_col]).var().reset_index()['y'].values)
    mids = np.array([(x.left + x.right)/2 for x in grp[bin_col].values])
    ybin = grp['y'].values

    if log:
        mids = 10**mids
    ax.plot(mids, ybin, c='0.22')
    if add_interval:
        ax.fill_between(mids, ybin - nstd*grp_std, ybin + nstd*grp_std, linewidth=0, color='0.22', alpha=0.2)

    if fix is not None:
        # exclude the variable
        fix = {k: v for k, v in fix.items() if k != var}
    for bf in bfs:
        if fix is not None:
            lower, upper = bf.func.bounds[var]
            log_it = log
            if not log:
                # if we want something on the linear scale, be sure to
                # get the linear bounds
                if bf.func.logscale[var]:
                    lower, upper = 10**lower, 10**upper
                # and don't log scale the grid!
                log_it = False

            # we need this exception for linear scale variables so we don't try
            # to log them
            if log and not bf.func.logscale[var]:
                log_it = False
            manual = {var: (lower, upper, nbins, log_it)}
            (mu, sh, L, rbp, rf), X_mesh_raw, _, predict = bf.func.predict_grid(#{var: nbins},
                                                                                None,
                                                                                manual_domains=manual,
                                                                                fix_X=fix, verbose=False)
            var_val = X_mesh_raw[:, cols.index(var)]
        else:
            var_val = mids
            # we take a random sample of the whole dataset
            # and inject the mid Ls in to estimate predictions
            X = bf.func.X
            predict = bf.func.predict(X)
            predict = pd.DataFrame(dict(y=predict, bin=bins)).groupby(['bin']).mean().reset_index()['y'].values
        ax.plot(mids, predict.squeeze())
    if log:
        ax.semilogx()
    ax.set_title(var)
    return fig, ax


def marginal_plot2(bfs, var, nbins, add_interval=False, nstd=1, figax=None):
    if figax is not None:
        fig, ax = figax
    else:
        fig, ax = plt.subplots()
    bf = bfs[0] # all use same training usually
    cols = ('mu', 'sh', 'L', 'rbp', 'rf')
    X = pd.DataFrame(bf.func.X, columns=cols)
    y = bf.func.y
    # bins
    x = X[var]
    log = bf.func.logscale[var]
    if log:
        x = np.log10(x)
    bins = pd.cut(x, nbins).values

    X['y'] = y
    bin_col = f'{var}_bin'
    X[bin_col] = bins
    grp = X.groupby([bin_col]).mean().reset_index()
    grp_std = np.sqrt(X.groupby([bin_col]).var().reset_index()['y'].values)
    mids = np.array([(x.left + x.right)/2 for x in grp[bin_col].values])
    ybin = grp['y'].values

    if log:
        mids = 10**mids
    ax.plot(mids, ybin, c='0.22')
    if add_interval:
        ax.fill_between(mids, ybin - nstd*grp_std, ybin + nstd*grp_std, linewidth=0, color='0.22', alpha=0.2)

    for bf in bfs:
        var_val = mids
        # we take a random sample of the whole dataset
        # and inject the mid Ls in to estimate predictions
        X = bf.func.X
        predict = bf.func.predict(X)
        predict = pd.DataFrame(dict(y=predict, bin=bins)).groupby(['bin']).mean().reset_index()['y'].values
        ax.plot(mids, predict.squeeze())
    if log:
        ax.semilogx()
    ax.set_title(var)
    return fig, ax

def marginal_plots(bfs, nbins, add_interval=False, nstd=1):
    fig, ax = plt.subplots(ncols=3, nrows=2)
    marginal_plot2(bfs, 'mu', nbins, add_interval=add_interval, nstd=nstd, figax=(fig, ax[0, 0]))
    marginal_plot2(bfs, 'sh', nbins, add_interval=add_interval, nstd=nstd, figax=(fig, ax[0, 1]))
    marginal_plot2(bfs, 'L', nbins,  add_interval=add_interval, nstd=nstd, figax=(fig, ax[0, 2]))
    marginal_plot2(bfs, 'rbp', nbins, add_interval=add_interval, nstd=nstd,  figax=(fig, ax[1, 0]))
    marginal_plot2(bfs, 'rf', nbins, add_interval=add_interval, nstd=nstd,  figax=(fig, ax[1, 1]))
    ax[1, 2].axis('off')
    plt.tight_layout()
    return fig, ax
