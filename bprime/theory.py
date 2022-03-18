import numpy as np

@np.vectorize
def bgs_rec(mu, s, r, L):
    s = 0.5*s # to mirror slim sims, where s is homozygous effect
    return np.exp(-L * mu/(s*(1+(1-s)*r/s)**2))

@np.vectorize
def bgs_segment(mu, s, rf, rbp, L):
    s = 0.5*s # to mirror slim sims, where s is homozygous effect
    r = rbp*L
    a = -s*mu*L
    b = (1-s)**2 # rf^2 terms
    c = 2*s*(1-s)+r*(1-s)**2 # rf terms
    d = s**2 + r*s*(1-s) # constant terms
    return np.exp(a / (b*rf**2 + c*rf + d))


