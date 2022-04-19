import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import statsmodels.api as sm
from bprime.theory import B_var_limit

lowess = sm.nonparametric.lowess

def get_figax(figax):
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
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
        ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

def bhat_plot(bfunc, bins, figax=None):
    fig, ax = get_figax(figax)
    _, bin_mid, ytest = bfunc.binned_Bhats(bins=bins)
    ax.scatter(bin_mid, ytest, c='0.22')
    ax.set_ylabel('$\hat{B}$')
    ax.set_xlabel('binned $B_\mathrm{ML}$')
    ax.axline((0.4, 0.4), slope=1, c='r')
    return fig, ax

def loss_plot(bfunc, figax=None):
    fig, ax = get_figax(figax)
    history = bfunc.func.history
    ax.plot(history['loss'][1:], label='training loss')
    ax.plot(history['val_loss'][1:], label='validation loss')
    ax.set_ylabel("MSE")
    ax.set_xlabel("epoch")

def loss_limits_plot(bfunc, N=None, mu=1, add_lowess=True, figax=None):
    fig, ax = get_figax(figax)
    predict = bfunc.predict_test()
    y_test = bfunc.func.y_test_orig
    xnew = np.linspace(predict.min(), predict.max(), 100)
    y = (predict - y_test.squeeze())**2
    x = predict
    ax.scatter(x, y, color='0.44', linewidth=0, edgecolor='black', alpha=0.2)
    if add_lowess:
        z = lowess(y, x, frac=1/10, it=0)
        ax.plot(z[:, 0], z[:, 1], c='r', linewidth=2)
    if N is not None:
        ax.axhline(B_var_limit(1, N, mu), c='0.22', linewidth=1.6, linestyle='dashed') # Note the 1/2 factor — see sim_power.ipynb! TODO
        # ax.plot(xnew, B_var_limit(xnew, N, mu), c='cornflowerblue', linewidth=1.6, linestyle='dashed') # Note the 1/2 factor — see sim_power.ipynb! TODO
    #ax.text(0.03, 0.88, "$\sigma^2 = \\frac{3 \mu + 8 B N \mu}{36 B N}$", size=13, rotation=-1.5, transform=ax4.transAxes)
    #ax.text(0.03, 0.88, "$\sigma^2 \\approx \\frac{2}{9}$", size=13, rotation=-1.5, transform=ax.transAxes)
    ax.semilogy()
    ax.set_ylabel('validation loss')
    ax.set_xlabel('predicted')

def rate_plot(bfunc, c=None, figax=None, **kwargs):
    fig, ax = get_figax(figax)
    Xcols = bfunc.func.col_indexer()
    Xgrids, Xmesh, Xmeshcols, predict_grid = bfunc.func.predict_grid(**kwargs)
    rate = (Xmesh[:, Xcols('mu')]/Xmesh[:, Xcols('sh')]).squeeze()
    predict = bfunc.theory_B(Xmesh)
    if c is not None:
        c = Xmesh[:, Xcols(c)]
    ax.scatter(rate, predict, c=c)
    ax.semilogx()


def b_learn_diagnostic_plot(func, figsize=(10, 7)):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

