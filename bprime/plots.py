import numpy as np
from math import ceil
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import statsmodels.api as sm
import scipy.stats as stats
from bprime.theory import B_var_limit
from bprime.utils import signif
from bprime.learn import LearnedB
from bprime.loss import get_loss_func

def center_scale(x):
    return (x-x.mean())/x.std()


lowess = sm.nonparametric.lowess

def get_figax(figax, **kwargs):
    if figax is None:
        fig, ax = plt.subplots(**kwargs)
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
    mse = signif(bfunc.Bhat_mse(bins), 4)
    ax.text(0.4, 0.05, f"$MSE(\hat{{B}}, B_\mathrm{{ML}})$ = {mse}")
    return fig, ax

def loss_plot(bfunc, figax=None):
    fig, ax = get_figax(figax)
    history = bfunc.func.history
    ax.plot(history['loss'][1:], label='training loss')
    ax.plot(history['val_loss'][1:], label='validation loss')
    ax.set_ylabel("MSE")
    ax.set_xlabel("epoch")
    ax.legend()
    return fig, ax

def rate_density_plot(bfunc, figax=None):
    fig, ax = get_figax(figax, ncols=2)
    Xcols = bfunc.func.col_indexer()
    X_test = bfunc.func.X_test_raw
    B_theory = bfunc.theory_B(X_test)
    ax[0].hist(np.log10(X_test[:, Xcols('mu')]/X_test[:, Xcols('sh')]))
    ax[0].set_xlabel("$\mu/s$")
    ax[0].set_ylabel("density")
    ax[1].hist(B_theory)
    ax[1].set_xlabel("$B_\mathrm{theory}$")
    ax[1].set_ylabel("density")
    ax[1].set_yscale('log')
    return fig, ax

def theory_loss_plot(bfunc, X=None, title="", figax=None):
    fig, ax = get_figax(figax)
    x, y = bfunc.theory_B(X), bfunc.func.predict(X)
    ax.scatter(x, y, alpha=0.1, s=2)
    ax.axline((0, 0), slope=1, c='r', linestyle='dashed')
    ax.set_ylabel("predicted")
    ax.set_ylabel("theory")
    mae = bfunc.theory_loss(X)
    mse = bfunc.theory_loss(X, loss='mse')
    title += f"\ntheory MAE={signif(mae, 4)}, MSE={signif(mse, 3)}"
    ax.set_title(title, fontsize=8)
    return fig, ax

def loss_limits_plot(bfunc, add_lowess=True, figax=None):
    fig, ax = get_figax(figax)
    predict = bfunc.predict_test()
    y_test = bfunc.func.y_test
    xnew = np.linspace(predict.min(), predict.max(), 100)
    y = (predict - y_test.squeeze())**2
    x = predict
    ax.scatter(x, y, color='0.44', linewidth=0, edgecolor='black', alpha=0.2)
    if add_lowess:
        z = lowess(y, x, frac=1/10, it=0)
        ax.plot(z[:, 0], z[:, 1], c='r', linewidth=2)
    b = np.linspace(x.min(), x.max(), 100)
    ax.plot(b, B_var_limit(b), c='0.22', linewidth=1.6, linestyle='dashed') # Note the 1/2 factor — see sim_power.ipynb! TODO
        # ax.plot(xnew, B_var_limit(xnew, N, mu), c='cornflowerblue', linewidth=1.6, linestyle='dashed') # Note the 1/2 factor — see sim_power.ipynb! TODO
    #ax.text(0.03, 0.88, "$\sigma^2 = \\frac{3 \mu + 8 B N \mu}{36 B N}$", size=13, rotation=-1.5, transform=ax4.transAxes)
    #ax.text(0.03, 0.88, "$\sigma^2 \\approx \\frac{2}{9}$", size=13, rotation=-1.5, transform=ax.transAxes)
    ax.semilogy()
    ax.set_ylabel('validation loss')
    ax.set_xlabel('predicted')
    return fig, ax

def rate_plot(bfunc, c=None, figax=None, add_theory=True, **predict_grid_kwargs):
    fig, ax = get_figax(figax)
    Xcols = bfunc.func.col_indexer()

    X_test = bfunc.func.X_test_raw
    test_rate = (X_test[:, Xcols('mu')]/X_test[:, Xcols('sh')]).squeeze()
    if c is not None:
        if c == 'rate':
            c = np.log10(X_test[:, Xcols('mu')]/X_test[:, Xcols('sh')])
        elif c == 'theory':
            c = bfunc.theory_B(X_test)
        else:
            c = X_test[:, Xcols(c)]
    test_predict = bfunc.predict_test()
    ax.scatter(test_rate, test_predict, c=c, s=3)

    if add_theory:
        Xgrids, Xmesh, Xmeshcols, predict_grid = bfunc.func.predict_grid(**predict_grid_kwargs)
        rate = (Xmesh[:, Xcols('mu')]/Xmesh[:, Xcols('sh')]).squeeze()
        theory = bfunc.theory_B(Xmesh)
        idx = np.argsort(rate)
        ax.plot(rate[idx], theory[idx], c='r', linestyle='dashed')
    ax.set_ylabel("predicted")
    ax.set_xlabel("$\mu/s$")
    ax.semilogx()
    return fig, ax

def sorted_arch(results):
    def capacity(item):
        key, _ = item
        a, b, c, d = key[:4]
        return 128**a + 64**b + 32**c + 8**d
    return sorted(results.items(), key=capacity)

def arch_loss_plot(results, ncols=3):
    """
    Visualize all the losses on separate panels for each architecture.
    Different replicates are shown as different colored lines.
    """
    nrows = len(results) // ncols
    figsize = (ncols*3, nrows*2.5)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharey=True, figsize=figsize)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    panels = itertools.product(range(nrows), range(ncols))
    labs = set()
    for i, (arch, funcs) in enumerate(results.items()):
        input_size = funcs[0].func.X_train.shape[1]
        lays = [128, 64, 32, 8, input_size]
        arch_lab = ", ".join(f"n{lays[i]}={n}" for i, n in enumerate(arch))
        panel = next(panels)
        mses = []
        if nrows > 1:
            ax = axs[panel[0], panel[1]]
        else:
            ax = axs[panel[1]]
        for j, func in enumerate(funcs):
            func = func.func
            history = func.history
            line, = ax.plot(history['loss'][1:], label=j, linestyle='solid')
            ax.plot(history['val_loss'][1:], c=line.get_color(), label=None, linestyle='dashed')
            mses.append(signif(func.test_mse(), 6))
            if panel[1] == 0:
                ax.set_ylabel("loss")
            if panel[0] == 1:
                ax.set_xlabel("epoch")
            ax.set_title(arch_lab, fontsize=8)
        mse_text = '\n'.join([f"MSE rep {i} = {mse}" for i, mse in enumerate(mses)])
        ax.text(0.6, 0.9, mse_text, size=5, transform=ax.transAxes)
    #ax.legend()
    plt.tight_layout()
    return fig, ax

def arch_loss_activs_plot(results, ncols=3, add_mse=False, sharey=True):
    """
    Visualize all the losses on separate panels for each architecture and
    activation.
    Different activations are shown as different colored lines.
    """
    nrows = ceil(len(results) / ncols)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharey=sharey)
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    panels = itertools.product(range(nrows), range(ncols))
    labs = set()
    n = len(results)
    for i, (arch, activs) in enumerate(results.items()):
        panel = next(panels)
        mses = []
        if nrows > 1:
            ax = axs[panel[0], panel[1]]
        else:
            ax = axs[panel[1]]
        for j, (activ, funcs) in enumerate(activs.items()):
            input_size = funcs[0].func.X_train.shape[1]
            lays = [128, 64, 32, 8, input_size]
            arch_lab = ", ".join(f"n{lays[i]}={n}" for i, n in enumerate(arch))
            for k, func in enumerate(funcs):
                func = func.func
                history = func.history
                line, = ax.plot(history['loss'][1:], label=activ, linestyle='solid')
                ax.plot(history['val_loss'][1:], c=line.get_color(), label=None, linestyle='dashed')
                mses.append(signif(func.test_mse(), 6))
                if panel[1] == 0:
                    ax.set_ylabel("loss")
                if panel[0] == 1:
                    ax.set_xlabel("epoch")
                ax.set_title(arch_lab, fontsize=8)
        if add_mse:
            mse_text = '\n'.join([f"MSE rep {i} = {mse}" for i, mse in enumerate(mses)])
            ax.text(0.6, 0.9, mse_text, size=5, transform=ax.transAxes)
    #ax.legend()
    plt.tight_layout()
    return fig, ax

def feature_loss_plot(bfunc, feature, bins, log10=True, loss='mae',
                      logx=True, logy=False, figax=None):
    lossfunc = get_loss_func(loss)
    fig, ax = get_figax(figax)
    Xcols = bfunc.func.col_indexer()
    X_test = bfunc.func.X_test_raw
    test_theory = bfunc.theory_B(X_test)
    rate = X_test[:, Xcols(feature)].squeeze()
    if log10:
        rate = np.log10(rate)
    if isinstance(bins, int):
        bins = np.linspace(rate.min(), rate.max(), bins)
    predict = bfunc.predict_test()
    lossvals = lossfunc(test_theory, predict)
    binned_loss = stats.binned_statistic(rate, lossvals, bins=bins)
    binned_counts = stats.binned_statistic(rate, lossvals, statistic=len, bins=bins)
    mids = 0.5*(binned_loss.bin_edges[1:] + binned_loss.bin_edges[:-1])
    if log10:
        mids = 10**mids
    ax.scatter(mids, binned_loss.statistic, c=binned_counts.statistic)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    loss_lab = loss.upper() if loss != 'bias' else loss
    ax.set_ylabel(f"binned {loss_lab}")
    ax.set_xlabel(feature)
    return fig, ax


def rate_loss_plot(bfunc, bins, loss='mae', logx=True, logy=True, figax=None):
    lossfunc = get_loss_func(loss)
    fig, ax = get_figax(figax)
    Xcols = bfunc.func.col_indexer()
    X_test = bfunc.func.X_test_raw
    test_theory = bfunc.theory_B(X_test)
    rate = np.log10(X_test[:, Xcols('mu')] / X_test[:, Xcols('sh')]).squeeze()
    if isinstance(bins, int):
        bins = np.linspace(rate.min(), rate.max(), bins)
    predict = bfunc.predict_test()
    lossvals = lossfunc(test_theory, predict)
    binned_loss = stats.binned_statistic(rate, lossvals, bins=bins)
    binned_counts = stats.binned_statistic(rate, lossvals, statistic=len, bins=bins)
    mids = 0.5*(binned_loss.bin_edges[1:] + binned_loss.bin_edges[:-1])
    ax.scatter(10**mids, binned_loss.statistic, c=binned_counts.statistic)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    loss_lab = loss.upper() if loss != 'bias' else loss
    ax.set_ylabel(f"binned {loss_lab}")
    ax.set_xlabel("$\mu/s$")
    return fig, ax

def B_loss_plot(bfunc, bins, loss='mae', relative=False, figax=None):
    lossfunc = get_loss_func(loss)
    fig, ax = get_figax(figax)
    Xcols = bfunc.func.col_indexer()
    B = bfunc.theory_B()
    if isinstance(bins, int):
        bins = np.linspace(B.min(), B.max(), bins)
    predict = bfunc.predict_test()
    lossvals = lossfunc(B, predict)
    binned_loss = stats.binned_statistic(B, lossvals, bins=bins)
    binned_counts = stats.binned_statistic(B, lossvals, statistic=len, bins=bins)
    mids = 0.5*(binned_loss.bin_edges[1:] + binned_loss.bin_edges[:-1])
    ax.scatter(mids, binned_loss.statistic, c=binned_counts.statistic)
    loss_lab = loss.upper() if loss != 'bias' else loss
    ax.set_ylabel(f"binned {loss_lab}")
    ax.set_xlabel("$B$")
    #ax.semilogy()
    return fig, ax


def feature_loss_plots(bfunc, bins, loss='mae', figax=None):
    ncols = 2
    features = ['rate', 'B'] + list(bfunc.func.features.keys())
    nrows = ceil(len(features) / ncols)
    fig, axs = get_figax(figax, ncols=ncols, nrows=nrows, figsize=(7, 8))
    panels = list(itertools.product(range(nrows), range(ncols)))
    for i, feature in enumerate(features):
        panel = panels[i]
        ax = axs[panel[0], panel[1]]
        figax = (fig, ax)
        if feature == 'B':
            B_loss_plot(bfunc, bins, loss, figax=figax)
        elif feature == 'rate':
            rate_loss_plot(bfunc, bins, loss, figax=figax)
        else:
            feature_loss_plot(bfunc, feature, bins=bins, loss=loss, figax=figax)
        if panel[1] > 0:
            ax.set_ylabel("")
        ax.axhline(0, c='0.55', linestyle='dashed')
        ax.axhline(bfunc.theory_loss(loss=loss), c='cornflowerblue', linestyle='dashed')

    plt.tight_layout()
    return fig, axs

def rf_plot(bfunc, figax=None):
    fig, ax = get_figax(figax)
    # Note that the training data does not log10 mu, but here we do (otherwise the grid looks
    # like chunky peanut butter).
    bs = []
    ys = []
    func = bfunc.func
    for sh in np.logspace(-5, -1,  20):
        (mu_grid_rbp, s_grid_rbp, a, b, c), X_mesh_orig_rbp, X_mesh_rbp, predict_grid_rbp = func.predict_grid({'rf': 100},
                                                                                                              fix_X={'mu': 1e-5,
                                                                                                                     'sh': sh,
                                                                                                                     'rbp': 1e-8,
                                                                                                                     'L': 1_000})
        bs.append(a)
        ys.append(predict_grid_rbp)
    cmap = cm.viridis(np.linspace(0, 1, 20))
    for i, y in enumerate(ys):
        ax.plot(np.log10(bs[0]), y.squeeze(), c=cmap[i])
    return fig, ax



def b_learn_diagnostic_plot(bfunc, bins=50, figsize=(10, 7), bhat=False, **rate_kwargs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    rate_plot(bfunc, **rate_kwargs, figax=(fig, ax1))
    loss_plot(bfunc, figax=(fig, ax2))
    if bhat:
        bhat_plot(bfunc, bins=bins, figax=(fig, ax3))
    else:
        theory_loss_plot(bfunc, figax=(fig, ax3))
    loss_limits_plot(bfunc, figax=(fig, ax4))
    plt.tight_layout()
    return fig, ((ax1, ax2), (ax3, ax4))


