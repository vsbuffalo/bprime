import warnings
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp2d


BGS_MODEL_PARAMS = {'bgs_rec': ('mu', 'sh', 'L', 'rbp'),
                    'bgs_segment': ('mu', 'sh', 'L', 'rbp', 'rf')}

@np.vectorize
def bgs_rec(mu, sh, L, rbp, log=False):
    """
    The BGS function of McVicker et al (2009) and Elyashiv et al. (2016).
    """
    val = -L * mu/(sh*(1+(1-sh)*rbp/sh)**2)
    if log:
        return val
    return np.exp(val)

@np.vectorize
def bgs_segment(mu, sh, L, rbp, rf, log=False):
    """
    Return reduction factor B of a segment L basepairs long with recombination
    rate rbp, with deleterious mutation rate with selection coefficient s. This
    segment is rf recombination fraction away. This is the result of integrating
    over the BGS formula after dividing up the recombination distance between
    the focal neutral site and each basepair in the segment into rf and rbp.
    """
    r = rbp*L
    a = -sh*mu*L
    b = (1-sh)**2 # rf^2 terms
    c = 2*sh*(1-sh)+r*(1-sh)**2 # rf terms
    d = sh**2 + r*sh*(1-sh) # constant terms
    val = a / (b*rf**2 + c*rf + d)
    if log:
        return val
    return np.exp(val)

@np.vectorize
def bgs_segment_sc16(mu, sh, L, r, N, full_output=False, return_both=False):
    U = L*mu
    #G = L*r
    Vm = U*sh**2
    #print(U, G, np.exp((-2*U / (2*sh + G))))
    start_T = (np.exp(2*sh*N) - 1)/(2*U*sh*N)
    def func(x):
        T, Ne = x
        V = U*sh - sh/T
        #Q2 = 1/((Vm/V) * (Vm/V + G/2))
        Q2 = (1/((Vm/V) + r))**2
        return [np.log((np.exp(2*sh*Ne) - 1)/(2*U*sh*Ne)) - np.log(T),
                 np.log(N * np.exp(-V*Q2)) - np.log(Ne)]
    out = fsolve(func, [start_T, N], full_output=True)
    Ne = out[0][1]
    T =  out[0][0]
    V = U*sh - sh/T
    Q2 = (1/((Vm/V) + r))**2
    if full_output:
        return out
    if out[2] != 1:
        warnings.warn("no solution found!")
        return np.nan
    if return_both:
        return T, Ne, Q2, V
    return Ne

def bgs_segment_sc16_manual_vec(mu, sh, L, r, N):
    """
    This is a manually vectorized version of
    bgs_segment_sc16 for comparison/testing.
    """
    x = np.empty((mu.size, sh.size, L.size))
    mu = mu.squeeze()
    sh = sh.squeeze()
    for i, m in enumerate(np.nditer(mu)):
        for j, s in enumerate(np.nditer(sh)):
            # if np.allclose(m, 1e-8) and np.allclose(s, 1e-3):
                # __import__('pdb').set_trace()
            x[i, j, :] = bgs_segment_sc16(m, s, L, r, N)
            #for l in range(len(L)):
    return x


def interpolate_bgs_sc16(mu, sh, L, rf, N, verbose=True):
    interpols = dict()
    nmu, nsh = len(mu), len(sh)
    LL, rr = np.meshgrid(L, rf)
    print("building theory interpolators...\t", end='')
    for i, s in enumerate(np.nditer(sh)):
        for j, m in enumerate(np.nditer(mu)):
            z = bgs_segment_sc16(m, s, LL, rr, N)
            key = (float(m), float(s))
            interpols[key] = interp2d(LL, rr, z, kind='cubic')
    print("done.")
    def func(L, rf):
        assert len(L) == len(rf)
        return np.array([interpol(L, rf) for key, interpol in interpols.items()]).reshape(nsh, nmu, len(L)).T
    return func


@np.vectorize
def B_var_limit(B, R=1, N=None, n=None):
    if N is None or n is None:
        # the case where n = N, and N --> inf
        return 2/9 * B**2 / R
    VarBhat = (n+1)*B / (12*N*(n-1)) + 2*(n**2 + n + 3)*B**2 / (9*n*(n-1))
    return VarBhat/R



BGS_MODEL_FUNCS = {'bgs_rec': bgs_rec,
                   'bgs_segment': bgs_segment}



