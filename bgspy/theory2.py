import warnings
import numpy as np
from scipy.optimize import fsolve
from scipy.special import expi, gammaincc, gamma
from scipy.integrate import quad

def Q2_asymptotic(Z, M):
	"""
	This is the asymptotic Q² term — up to the factor of two
	that *cancels* with the V/2 in a diploid model.
	"""
	return -2/((-1 + Z)*(2 + (-2 + M)*Z))

def ratchet_time(sh, Ne, U):
	return (np.exp(4*sh*Ne) - 1)/(2*U*sh*Ne)

def Q2_sum_integral(Z, M, tmax=1000, thresh=0.01, use_sum=False):
    """

    This calculates the series 2/M ∫ (∑_t^T f(t, r))²dr for each T, over T = [0,
    tmax]. M is map length is Morgans. Q(t, r) = ∑_i^t Q(i, r)) can be
    calculated from either a closed-form solution or an explicit sum. The ∑Q
    value asymptotes; if the relative change is less than thresh, the last value
    is just repeated and not computed to save computation time.

    """
    assert Z < 1
    end_ts = np.arange(tmax)
    last = None
    asymptoted = False
    vals = []
    # loop over the maximum term in the sum, since we want the series over this
    # max
    Power = lambda x, y: x**y
    for T in end_ts:
        if not asymptoted:
            if use_sum:
                # one offset is because Python isn't right-inclusive, so T=1 is up to 0
                t = np.arange(T+1)
                assert len(t), T
                integrand = quad(lambda r: np.sum((1-r)**t * Z**t)**2, 0, M/2)[0]
            else:
                integrand = quad(lambda r: ((1-(1-r)**(T+1) * Z**(T+1))/(1 - (1-r)*Z))**2, 0, M/2)[0]
            if last is not None and last > 0:
                if (integrand-last)/last < thresh:
                    asymptoted = True
            else:
                last = integrand
        else:
            integrand = last
        vals.append(integrand)
    return (2/M)*np.array(vals)

def Gamma(a, z):
    """
    An alias of Mathematica's Gamma function.
    """
    if a == 0:
        return -expi(-z)
    return gamma(a)*gammaincc(a, z)

def Q2_sum_integral2(Z, M, tmax=1000, thresh=0.01):
    """
    Like Q2_sum_integral() but approximates the geometric decay with an
    exponential. This allows for an easier closed-form analytic solution to the
    sum (approximated by an integral) and the outer integral. This voids the
    need for numeric integration used in Q2_sum_integral().

    This is imported directly from Mathematica (hence the function aliasing).
    """
    assert Z < 1
    end_ts = np.arange(1, tmax+1)
    Log = np.log
    Power = lambda x, y: x**y
    E = np.exp(1)
    ExpIntegralEi = expi
    vals = []
    k = 1-Z
    for T in end_ts:
        # From Mathematica:
        integrand = (-((Power(-1 + Power(E,k*T),2)/(Power(E,2*k*T)*k) +
                        2*T*ExpIntegralEi(-2*k*T) -
                        2*T*ExpIntegralEi(-(k*T)))/(-1 + k)) +
                     ((-2*Power(E,k*(-2 + M)*T - M*T) *
                       Power(-1 + Power(E,((-(k*(-2 + M)) + M)*T)/2.),2))/
                      (k*(-2 + M) - M) -
                      2*T*ExpIntegralEi(((k*(-2 + M) - M)*T)/2.) +
                      2*T*ExpIntegralEi((k*(-2 + M) - M)*T))/(-1 + k))
        vals.append(integrand)
    #vals[0] = 0  # normally it's NaN, but the series value is 0, so we set it to that
    return (2/M)*np.array(vals)

def ave_het(Ne_t, return_parts=False):
    """
    Het x 2N. If return_parts is True, the series is returned, not the sum.
    """
    x = np.array([np.prod(1 - 1/(2*Ne_t[:i])) for i in np.arange(len(Ne_t))])
    if return_parts:
        return x
    return x.sum()

def bgs_segment_sc16(mu, sh, L, rbp, N, asymptotic=True, sum_n=10,
                     dont_fallback=False, return_parts=False):
    """
    Using a non-linear solver to solve the pair of S&C '16 equations and
    return the B values.

    asymptotic: Return the B from the asymptotic Ne (which determines
                 selection processes).
    sum_n: How many N generations to calculate the Q(t) series over, for the
            non-asymptotic Ne(t)
    """
    U = 2*L*mu
    Vm = U*sh**2
    M = L*rbp
    start_T = ratchet_time(sh, N, U)
    def func(x):
        T, Ne = x
        V = U*sh - 2*sh/T
        VmV = Vm/V
        Z = 1 - VmV
        Q2 = Q2_asymptotic(Z, M)
        new_T = ratchet_time(sh, Ne, U)  # NOTE: this must depend on Ne, not N!
        return [np.log(new_T) - np.log(T),
                 np.log(N * np.exp(-V/2 * Q2)) - np.log(Ne)]
    # try to solve the non-linear system of equations
    out = fsolve(func, [start_T, N], full_output=True)
    Ne = out[0][1]
    T = out[0][0]

    if out[2] != 1:
        # don't use BGS, just return NaNs
        if dont_fallback:
            if return_parts:
                return np.nan, np.nan, np.nan, np.nan, np.nan
            return np.nan
        # A solution could not be found for the non-linear system of equations.
        # In this case, this is due to numeric overflow in T, the time
        # between ratchet clicks. The solution is to fall back to the R = 0
        # case, which is the classic BGS solution.
        #warnings.warn(f"no solution found!? mu={mu}, sh={sh}, L={L}, rbp={rbp}; T={T}, Ne={Ne}")
        V = U*sh  # no ratchet term; this falls back to the BGS model
    else:
        # use the no-linear solution with a ratchet time
        assert np.isfinite(T), "T is not finite!"
        V = U*sh - 2*sh/T

    # the new V and Z are:
    VmV = Vm/V
    Z = 1-VmV
    # use the single asymptotic Q2
    Q2_asymp = Q2_asymptotic(Z, M)
    Ne_asymp = N*np.exp(-V/2 * Q2_asymp)
    B_asymp = Ne_asymp/N
    if not return_parts and asymptotic:
        return B_asymp

    # the full Q² series going back sum_n*N generations
    Q2 = Q2_sum_integral2(Z, M, tmax=sum_n*N)
    Ne_t = N*np.exp(-V/2 * Q2)
    assert len(Ne_t) > 1, "Ne_t's length <= 1"
    B = ave_het(Ne_t)/(2*N)
    if not return_parts:
        return B

    return B, B_asymp, T, V, Vm, Q2


@np.vectorize
def bgs_segment_sc16_vec(*args, **kwargs):
    return bgs_segment_sc16(*args, **kwargs)

def bgs_segment_sc16_parts(*args, **kwargs):
    """
    Vectorized version to return the components of the BGS calculation.
    """
    return np.vectorize(bgs_segment_sc16, otypes=(tuple,))(*args, **kwargs, return_parts=True)


@np.vectorize
def bgs_rec(mu, sh, L, rbp, log=False):
    """
    The BGS function of McVicker et al (2009) and Elyashiv et al. (2016).
    """
    val = -L * mu/(sh*(1+(1-sh)*rbp/sh)**2)
    if log:
        return val
    return np.exp(val)

