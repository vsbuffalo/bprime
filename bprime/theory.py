import numpy as np

BGS_MODEL_PARAMS = {'bgs_rec': ('mu', 's', 'L', 'rbp', 'h'),
                    'bgs_segment': ('mu', 's', 'L', 'rbp', 'rf', 'h')}


@np.vectorize
def bgs_rec(mu, s, L, rbp, h=1/2, log=False):
    """
    The BGS function of McVicker et al (2009) and Elyashiv et al. (2016).
    """
    val = -L * mu/(s*(1+(1-s)*rbp/s)**2)
    if log:
        return val
    return np.exp(val)

@np.vectorize
def bgs_segment(mu, s, L, rbp, rf, h=1/2, log=False):
    """
    Return reduction factor B of a segment L basepairs long with recombination
    rate rbp, with deleterious mutation rate with selection coefficient s. This
    segment is rf recombination fraction away. This is the result of integrating
    over the BGS formula after dividing up the recombination distance between
    the focal neutral site and each basepair in the segment into rf and rbp.
    """
    r = rbp*L
    a = -s*mu*L
    b = (1-s)**2 # rf^2 terms
    c = 2*s*(1-s)+r*(1-s)**2 # rf terms
    d = s**2 + r*s*(1-s) # constant terms
    val = a / (b*rf**2 + c*rf + d)
    if log:
        return val
    return np.exp(val)


