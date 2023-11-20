import numpy as np
from sklearn.utils.validation import check_X_y
import warnings
from scipy import stats
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import defaultdict
from bgspy.utils import binned_statistic, cutbins
from bgspy.plots import get_figax

def gaussian_kernel(x):
    # for underflow
    w = np.zeros_like(x)
    v = x**2/2
    np.exp(-v, out=w, where=v < 0.95*np.log(np.finfo('float64').max))
    return 1/np.sqrt(2*np.pi) * w

class KernelRegression(object):
    def __init__(self, h=1, kernel=gaussian_kernel):
        self.h = h
        self.kernel = kernel

    def fit(self, X, y):
        X, y = check_X_y(X, y, ensure_2d=False)
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        K = self.kernel(np.subtract.outer(X, self.X) / self.h)
        Ky = K * self.y
        return Ky.sum(axis=1) / K.sum(axis=1)

    def get_params(self, deep=False):
        return {'h': self.h}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def score(self, y):
        return np.mean((self.predict(y) - y)**2)

def bin_kfolds(x, y, bins=np.arange(5, 100, 2), n_splits=100, cut_tails=None, **cutbin_args):
    """
    Use cross-validation to find the best number of bins, using
    bgspy.utils.binned_statistic.
    """
    kf = KFold(n_splits=n_splits)
    mses = defaultdict(list)
    cv_mses = []
    nbs = []

    bin_range = (x.min(), 1.001*x.max()) # to include the right side

    def safe_mean(x):
        if not len(x) or np.all(~np.isfinite(x)):
            return np.nan
        return np.nanmean(x)

    for nb in bins.astype(int):
        for train, test in kf.split(x):
            # cut bins
            bins = cutbins(x[train], nb, xrange=bin_range, **cutbin_args)

            # train: bin based on training data
            bin_means = binned_statistic(x[train], y[train], statistic=safe_mean, bins=bins)

            # get the number of elements per bin
            bin_ns = binned_statistic(x[train], y[train], statistic=lambda x: np.sum(np.isfinite(x)), bins=bins)

            # take the test data and bin it
            idx = np.digitize(x[test], bins)

            # estimate the squared error of predictions
            se = (bin_means.statistic[idx-1] - y[test])**2

            # we weight the MSE by the bin sizes
            weights = bin_ns.statistic[idx-1]
            keep = np.isfinite(se)
            mse = np.average(se[keep], weights=weights[keep])

            mses[nb].append(mse)
            cv_mses.append(mse)
            nbs.append(nb)

    msedat = {nb: np.mean(v) for nb, v in mses.items()}
    return list(msedat.keys()), list(msedat.values())

def kfolds_results(kf_res, figax=None):
    fig, ax = get_figax(figax)
    ax.plot(*kf_res)
    b, mse = kf_res
    best_idx = np.argmin(mse)
    best = b[best_idx]
    ax.scatter(b[best_idx], mse[best_idx])
    return b[best_idx]


class Lowess(BaseEstimator):
    def __init__(self, frac):
        self.frac = frac
    def fit(self, x, y):
        self.x_ = np.array(x)
        idx = np.argsort(self.x_)
        self.x_ = self.x_[idx]
        self.y_ = np.array(y)[idx]
        return self
    def predict(self, x=None):
        if x is None:
            x = self.x_
        return lowess(self.y_, self.x_, xvals=x, frac=self.frac)
    def pairs(self):
        preds = self.predict(self.x_)
        return self.x_, preds
    def bootstrap(self, *args, **kwargs):
        straps, lb, ub = lowess_bootstrap(self.x_, self.y_, self.frac, *args, **kwargs)
        return lb, ub

def lowess_cv_frac(x, y, cv=20, fracs=np.linspace(0.05, 1, 60),
                   min_n=10):
    """
    Find the lowess frac with CV.

    min_n is the minimum number of data points to consider for a 
    fraction, e.g. if frac * len(x) < min_n the score is set to NaN
    """
    n = len(x[~np.isnan(x)])
    res = []
    for f in fracs:
        if f * n < min_n:
            res.append(np.nan)
            continue
        clf = Lowess(f)
        scores = cross_validate(clf, x, y, scoring='neg_mean_squared_error', cv=cv)
        res.append(scores['test_score'].mean())
    res = np.array(res)
    idx = np.nanargmax(res)
    return (fracs[idx], res[idx]), fracs, res

def lowess_bootstrap(x, y, frac, nboot=200, alpha=0.05):
    x = np.array(x)
    y = np.array(y)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    n = len(x)
    res = []
    for i in range(nboot):
        idx = np.random.choice(range(n), size=n, replace=True)
        x_rs = x[idx]
        y_rs = y[idx]
        smoothed_bootstrap = Lowess(frac).fit(x_rs, y_rs).predict(np.sort(x_rs))
        res.append(smoothed_bootstrap)
    res = np.array(res)
    lb = np.percentile(res, 100 * alpha/2, axis=0)
    ub = np.percentile(res, 100 * (1-alpha/2), axis=0)
    return np.array(res), lb, ub


def linregress_bootstrap(x, y, nboot=200, alpha=0.05):
    n = len(x)
    res = []
    for i in range(nboot):
        idx = np.random.choice(range(n), size=n, replace=True)
        x_rs = x[idx]
        y_rs = y[idx]
        xs = np.sort(x_rs)
        slope, yint, *lrres = stats.linregress(x_rs, y_rs)
        pred = xs * slope + yint 
        res.append(pred)
    res = np.array(res)
    lb = np.percentile(res, 100 * alpha/2, axis=0)
    ub = np.percentile(res, 100 * (1-alpha/2), axis=0)
    lb = np.percentile(res, 100 * alpha/2, axis=0)
    ub = np.percentile(res, 100 * (1-alpha/2), axis=0)
    return np.array(res), lb, ub



