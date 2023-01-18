import numpy as np
from sklearn.utils.validation import check_X_y
import warnings
from sklearn.model_selection import KFold, LeaveOneOut
from collections import defaultdict
from bgspy.utils import binned_statistic, cutbins

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

def bin_kfolds(x, y, bins=np.arange(5, 100, 2), n_splits=100, **bin_args):
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
            bins = cutbins(x[train], nb, xrange=bin_range, **bin_args)

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

