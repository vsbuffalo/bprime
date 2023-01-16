import numpy as np
from scipy.optimize import curve_fit

def grid_interp_weights(w, mu):
    j = np.searchsorted(w, mu)
    l, u = w[j-1], w[j]
    assert l < u
    weight = ((mu-l)/(u - l))
    assert 0 <= weight <= 1
    w = 1-weight # so weighting is w*lower + (1-w)*upper
    return j, w

## an experiment -- deprecated
#def JC69(dist):
#    return 0.75 - 0.75*np.exp(-4/3 * dist)

#def JC69_parametric(lambda_d, T):
#    """
#    """
#    return JC69(T*lambda_d)

#def fit_JC69(predicted, subrate, gen_time=30, p0=(1, 12e6, 0)):
#    """
#    free parameters: T
#    """
#    lambda_d, y = predicted/gen_time, subrate
#    keep = np.isfinite(lambda_d) & np.isfinite(y)
#    lambda_d, y = lambda_d[keep], y[keep]
#    fit, cv = curve_fit(JC69_parametric, lambda_d, y, p0=p0)
#    pred = JC69_parametric(lambda_d, fit[0], fit[1], fit[2])
#    return lambda_d, y, (fit, np.mean((pred - y)**2))

#class SubstitutionPredictionModel:
#    def __init__(self):
#        self.fit_ = None
#        pass

#    def __repr__(self):
#        out = [f"SubstitutionPredictionModel\n"]
#        if self.fit_ is not None:
#            T = self.fit_[0]
#            info = (
#                   # f"intercept (λ_B T), a = {a:.3g}\n"
#                    #f"  λ_B = {a / T:.3g} \n"
#                    f"T = {T/1e6:.4g} (Mya)\n"
#                    #f"β = {beta:.3g}\n"
#                    #f"{(a / T) / (a + T*beta*1e-9)}")
#                    )
#            out.append(info)
#        return "\n".join(out)

#    def fit(self, predicted, subrate, start, gen_time=30):
#        lambda_d, y = predicted/gen_time, subrate
#        keep = np.isfinite(lambda_d) & np.isfinite(y)
#        lambda_d, y = lambda_d[keep], y[keep]
#        fit, cv = curve_fit(JC69_parametric, lambda_d, y, p0=start)
#        self.X_ = predicted
#        self.y_ = subrate
#        self.fit_ = fit
#        self.cov_ = cv
#        return self

#    def predict(self, predicted=None):
#        if predicted is None:
#            predicted = self.X_
#        return JC69_parametric(predicted, *self.fit_)

#    def R2(self):
#        return np.mean((self.predict() - self.y_)**2)
