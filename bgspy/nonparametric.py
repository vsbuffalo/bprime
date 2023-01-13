import numpy as np
from sklearn.utils.validation import check_X_y

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


