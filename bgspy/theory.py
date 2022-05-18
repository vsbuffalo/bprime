import numpy as np

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
def B_var_limit(B, R=1, N=None, mu=1):
    if N is None:
        return 2/9 * B**2 / R
    return B/(12 * N * mu) + 2/9 * B**2 + B**2/N


BGS_MODEL_FUNCS = {'bgs_rec': bgs_rec,
                   'bgs_segment': bgs_segment}



