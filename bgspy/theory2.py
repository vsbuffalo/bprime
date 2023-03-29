import os
import warnings
import tqdm
import functools
import multiprocessing
import numpy as np
from scipy.optimize import fsolve
from scipy.special import expi, gammaincc, gamma
from scipy.integrate import quad
from ctypes import c_double, c_int, c_ssize_t

from bgspy.utils import dist_to_segment

# load the library (relative to this file in src/)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
Bclib = np.ctypeslib.load_library("Bclib", LIBRARY_PATH)

# define the types of the dynamic library functions
Bclib.B_BK2022.argtypes = (np.ctypeslib.ndpointer(np.double, ndim=1, flags=('C')),
                           np.ctypeslib.ndpointer(np.double, ndim=1, flags=('C')),
                           np.ctypeslib.ndpointer(np.double, ndim=1, flags=('C','W')),
                           c_ssize_t, c_int, c_double)
Bclib.B_BK2022.restype = None

Bclib.Ne_t.argtypes = (c_double, c_double, c_int);
Bclib.Ne_t.restype = c_double

Bclib.Ne_t_rescaled.argtypes = (c_double, c_double, c_int, c_double);
Bclib.Ne_t_rescaled.restype = c_double


## Classic BGS
def B_segment_lazy(rbp, L, t):
    """
    rt/ (rf*(-1 + t) - t) * (rf*(-1 + t) + r*(-1 + t) - t)

    together:
      x = a/(b*rf**2 + c*rf + d)
    """
    r = rbp*L
    a = -t*L # numerator -- ignores u
    b = (1-t)**2  # rf^2 terms
    c = 2*t*(1-t)+r*(1-t)**2 # rf terms
    d = t**2 + r*t*(1-t) # constant terms
    return a, b, c, d

@np.vectorize
def bgs_rec(mu, sh, L, rbp, log=False):
    """
    The BGS function of McVicker et al (2009) and Elyashiv et al. (2016).
    """
    val = -L * mu/(sh*(1+(1-sh)*rbp/sh)**2)
    if log:
        return val
    return np.exp(val)

# parallel stuff
def calc_B_chunk_worker(args):
    map_positions, chrom_seg_mpos, F, segment_parts, mut_grid, N, _ = args
    a, b, c, d = segment_parts
    Bs = []
    # F is a features matrix -- eventually, we'll add support for
    # different feature annotation class, but for now we just fix this
    #F = np.ones(len(chrom_seg_mpos))[:, None]
    for f in map_positions:
        rf = dist_to_segment(f, chrom_seg_mpos)
        #rf = np.abs(f - chrom_seg_mpos[:, 0])
        if np.any(b + rf*(rf*c + d) == 0):
            raise ValueError("divide by zero in calc_B_chunk_worker")
        x = a/(b*rf**2 + c*rf + d)
        assert(not np.any(np.isnan(x))), f"x={x}, a={a}, b={b}, c={c}, d={d}, rf={rf}"
        B = np.einsum('ts,w,sf->wtf', x, mut_grid, F)
        # the einsum below is for when a features dimension exists, e.g.
        # there are feature-specific μ's and t's -- commented out now...
        #B = np.einsum('ts,w,sf->wtf', x, mut_grid,
        #              F, optimize=BCALC_EINSUM_PATH)
        Bs.append(B)
    return Bs

###### New Theory

## Map Version of Functions

def Q2_asymptotic(Z, M):
    """
    This is the map-version asymptotic Q² term — including the factor of two
    that *cancels* with the V/2 in a diploid model.
    """
    return -2/((-1 + Z)*(2 + (-2 + M)*Z))

def ratchet_time(sh, Ne, U):
    out = (np.exp(4*sh*Ne) - 1)/(2*U*sh*Ne)
    assert out > 0, "negative ratchet waiting time"
    return out

def Q2_sum_integral(Z, M, tmax=1000, thresh=0.01, use_sum=False):
    """

    This calculates the series 2/M ∫ (∑_t^T f(t, r))²dr for each T, over T = [0,
    tmax]. M is map length is Morgans. Q(t, r) = ∑_i^t Q(i, r)) can be
    calculated from either a closed-form solution or an explicit sum. The ∑Q
    value asymptotes; if the relative change is less than thresh, the last value
    is just repeated and not computed to save computation time.

    """
    assert np.all(Z <= 1)
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


def Q2_sum_integral2(Z, M, tmax=1000, thresh=1e-5):
    """
    Like Q2_sum_integral() but approximates the geometric decay with an
    exponential. This allows for an easier closed-form analytic solution to the
    sum (approximated by an integral) and the outer integral. This voids the
    need for numeric integration used in Q2_sum_integral().

    This is imported directly from Mathematica (hence the function aliasing).

    WARNING: this has a tendency to have NaN values in the strong BGS domain
    """
    assert np.all(Z < 1)
    end_ts = np.arange(1, tmax+1)
    # Log = np.log
    # Power = lambda x, y: x**y
    # E = np.exp(1)
    # ExpIntegralEi = expi
    vals = []
    k = 1-Z
    for T in end_ts:
        # From Mathematica:
        # integrand = (-((Power(-1 + Power(E,k*T),2)/(Power(E,2*k*T)*k) +
        #                 2*T*ExpIntegralEi(-2*k*T) -
        #                 2*T*ExpIntegralEi(-(k*T)))/(-1 + k)) +
        #              ((-2*Power(E,k*(-2 + M)*T - M*T) *
        #                Power(-1 + Power(E,((-(k*(-2 + M)) + M)*T)/2.),2))/
        #               (k*(-2 + M) - M) -
        #               2*T*ExpIntegralEi(((k*(-2 + M) - M)*T)/2.) +
        #               2*T*ExpIntegralEi((k*(-2 + M) - M)*T))/(-1 + k))
        integrand = (-(((np.expm1(k*T)**2)/(np.exp(2*k*T)*k) +
                2*T*expi(-2*k*T) -
                2*T*expi(-(k*T)))/(k-1)) +
             ((-2*np.exp(k*(M-2)*T - M*T) *
               (-1 + np.exp(((-(k*(M-2)) + M)*T)/2.))**2)/
              (k*(M-2) - M) -
              2*T*expi(((k*(M-2) - M)*T)/2.) +
              2*T*expi((k*(M-2) - M)*T)) / (k-1))
        vals.append(integrand)
    #vals[0] = 0  # normally it's NaN, but the series value is 0, so we set it to that
    return (2/M)*np.array(vals)


def ave_het(Ne_t, return_parts=False):
    """
    DEPRECATED: Slow test function.
    Het x 2N. If return_parts is True, the series is returned, not the sum.
    """
    x = np.array([np.prod(1 - 1/(2*Ne_t[:i])) for i in np.arange(len(Ne_t))])
    if return_parts:
        return x
    return x.sum()

### Recombination Functions
# Functions that don't integrate over a map length


@np.vectorize
def Ne_asymp(V, Vm, rf, N):
    return N * np.exp(-V/2 * Qr_asymp(V, Vm, rf)**2)


@np.vectorize
def Ne_asymp2(a, V, N):
    x = V/2 * Qr_asymp2(a)**2
    if x >= 0.95*np.log(np.finfo(np.float64).max):
        # prevent underflow
        return 0
    return N * np.exp(-x)


def Qr_asymp(V, Vm, rf):
    a = (1-Vm/V)*(1-rf)
    return 1/(1-a)


def Qr_asymp2(a):
    return 1/(1-a)


def Qr_fixed(t, V, Vm, rf):
    a = (1-Vm/V)*(1-rf)
    return (1-a**(t+1)) / (1-a)


def Qr_fixed2(t, a):
    return (1-a**(t+1)) / (1-a)


def pfix(N, s, p0=None):
    """
    probability of fixation of an allele at frequency
    p0 (set to 1/(2N) if not specified. This is a useful
    way to check the scale of the selection coefficient grid;
    effectively neutral sites have pfix ~ 1/2N (the neutral rate)
    """
    p0 = 0.5/N if p0 is None else p0
    return (1-np.exp(-2*N*s*p0))/(1-np.exp(-2*N*s))


@np.vectorize
def Ne_t_full(T, N, V, Vm, rf, as_B=False, return_parts=False):
    Q_sum = 0
    Ne_sum = 0
    prod = 1
    prods = []
    Ne_sums = []
    for t in np.arange(T):
        Q_sum += (1-rf)**t * (1-Vm/V)**t
        Q2 = Q_sum**2
        Ne = N*np.exp(-V/2 * Q2)
        prod *= 1-0.5/Ne
        prods.append(prod)
        Ne_sum += prod
        Ne_sums.append(Ne_sum)
    #print(f"old product: {prods[-1]}")
    if return_parts:
        return prods, Ne_sums
    if as_B:
        return Ne_sum/(2*N)
    return Ne_sum/2


@np.vectorize
def Ne_t_full2(N, V, a, as_B=False, return_parts=False):
    Q_sum = 0
    Ne_sum = 0
    prod = 1
    prods = []
    Ne_sums = []
    T = 50*N
    for t in np.arange(T):
        Q_sum = Qr_fixed2(t, a)
        Q2 = Q_sum**2
        Ne = N*np.exp(-V/2 * Q2)
        prod *= 1-0.5/Ne
        prods.append(prod)
        Ne_sum += prod
        Ne_sums.append(Ne_sum)
    #print(f"old product: {prods[-1]}")
    if return_parts:
        return prods, Ne_sums
    if as_B:
        return Ne_sum/(2*N)
    return Ne_sum/2

#### Main compututional functions
# stuff used for the business end of things


def bgs_segment_sc16(mu, sh, L, rbp, N, asymptotic=True, T_factor=10,
                     dont_fallback=False, return_parts=False):
    """
    Using a non-linear solver to solve the pair of S&C '16 equations and
    return the B values.

    asymptotic: Return the B from the asymptotic Ne (which determines
                 selection processes).
    T_factor: How many N generations to calculate the Q(t) series over, for the
            non-asymptotic Ne(t)
    """
    U = 2*L*mu
    Vm = U*sh**2
    M = L*rbp
    try:
        start_T = ratchet_time(sh, N, U)
    except FloatingPointError:
        # the sh is too high for there to be a possibility of fixation, hence
        # the overflow
        start_T = None

    def func(x):
        T, Ne = x
        V = U*sh - 2*sh/T
        VmV = Vm/V
        Z = 1 - VmV
        Q2 = Q2_asymptotic(Z, M)
        try:
            assert T > 0
            assert Ne > 0
        except:
            return (np.inf, np.inf)
        new_T = ratchet_time(sh, Ne, U)  # NOTE: this must depend on Ne, not N!
        #new_logNe = np.log(N * np.exp(-V/2 * Q2))
        new_logNe = np.log(N) - V*Q2/2
        return [np.log(new_T) - np.log(T),
                 new_logNe - np.log(Ne)]

    if start_T is not None:
        # try to solve the non-linear system of equations
        try:
            out = fsolve(func, [start_T, N], full_output=True)
            Ne = out[0][1]
            T = out[0][0]
        except:
            # mimic a failed convergence to solution
            out = (None, None, -1)
            Ne = None
            T = np.inf
    else:
        # mimic a failed convergence to solution
        out = (None, None, -1)
        Ne = None
        T = np.inf

    classic_bgs = False
    if out[2] != 1:
        # don't use BGS, just return NaNs
        if dont_fallback:
            if return_parts:
                return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None
            return np.nan
        # A solution could not be found for the non-linear system of equations.
        # In this case, this is due to numeric overflow in T, the time
        # between ratchet clicks. The solution is to fall back to the R = 0
        # case, which is the classic BGS solution.
        #warnings.warn(f"no solution found!? mu={mu}, sh={sh}, L={L}, rbp={rbp}; T={T}, Ne={Ne}")
        V = U*sh # no ratchet term; this falls back to the BGS model
        classic_bgs = True
    else:
        # use the no-linear solution with a ratchet time
        assert np.isfinite(T), "T is not finite!"
        V = U*sh - 2*sh/T

    # the new V and Z are:
    VmV = Vm/V
    Z = 1-VmV
    # use the single asymptotic Q2
    Q2_asymp = Q2_asymptotic(Z, M)
    try:
        Ne_asymp = max(N*np.exp(-V/2 * Q2_asymp), 1)
    except FloatingPointError:
        Ne_asymp = 1
    B_asymp = Ne_asymp/N
    if not return_parts and asymptotic:
        return B_asymp
    if return_parts and asymptotic:
        return np.nan, B_asymp, T, V, Vm, Q2_asymp, classic_bgs

    # the full Q² series going back T_factor*N generations
    Q2 = Q2_sum_integral(Z, M, tmax=T_factor*N)
    Ne_t = N*np.exp(-V/2 * Q2)
    assert len(Ne_t) > 1, "Ne_t's length <= 1"
    B = ave_het(Ne_t)/(2*N)
    if not return_parts:
        return B

    return B, B_asymp, T, V, Vm, Q2, classic_bgs

@np.vectorize
def bgs_segment_sc16_vec(*args, **kwargs):
    return bgs_segment_sc16(*args, **kwargs)

def bgs_segment_sc16_parts(*args, **kwargs):
    """
    Vectorized version to return the components of the BGS calculation.
    """
    return np.vectorize(bgs_segment_sc16, otypes=(tuple,))(*args, **kwargs, return_parts=True)


def bgs_segment_sc16_components(L_rbp_rescaling, mu, sh, N, 
                                return_all=False, pairwise=False):
    """
    A manually-vectorized version of bgs_segment_sc16_parts().
    This mimics np.vectorize() but the vectorization is done manually.
    This is because np.vectorize() creates a closure that cannot be
    pickled for multiprocessing work, meaning the np.vectorize()
    version is not parallelizable.

    return_all: whether to return all the various parts calculated from
    bgs_segment_sc16, which is mostly for debugging purposes.

    pairwise: if True, this function does not calculate for every 
               grid combination, but only pairwise mu/sh combinations.
    """
    L, rbp, rescaling = L_rbp_rescaling
    assert isinstance(L, (int, float))
    assert isinstance(rbp, float)
    if rescaling is not None:
        assert isinstance(rescaling, float)
    else:
        rescaling = 1.
    assert isinstance(mu, (float, np.ndarray))
    assert isinstance(sh, (float, np.ndarray))
    shape = mu.size, sh.size

    Ts, Vs, Vms = np.empty(shape), np.empty(shape), np.empty(shape)
    if return_all:
        Bs, Bas, Q2s, cbs = np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape)
    for i, m in enumerate(mu.flat):
        for j, s in enumerate(sh.flat):
            if pairwise and i != j:
                continue
            res = bgs_segment_sc16(m, s, L, rbp, rescaling*N, return_parts=True)
            B, B_asymp, T, V, Vm, Q2, classic_bgs = res
            Ts[i, j] = T
            Vs[i, j] = V
            Vms[i, j] = Vm
            
            if return_all:
                Bs[i, j] = B
                Bas[i, j] = B_asymp
                Q2s[i, j] = Q2
                cbs[i, j] = classic_bgs

    if return_all:
        return Bs, Bas, Ts, Vs, Vms, Q2s, cbs
    if pairwise:
        return np.diag(Ts), np.diag(Vs), np.diag(Vms)
    return Ts, Vs, Vms


def calc_BSC16_chunk_worker(args):
    # ignore rbp, no need here yet under this approximation
    map_positions, chrom_seg_mpos, F, segment_parts, mut_grid, N, interp_parts = args


    # NOTE: this is slow, and turned out to not change much.
    # build the bias-correction interplator
    # a_grid, V_grid, interp_bias = interp_parts
    # bias = RegularGridInterpolator((a_grid, V_grid),
    #                                 interp_bias.T,
    #                                 method='linear', bounds_error=True,
    #                                 fill_value=0)

    Bs = []
    # F is a features matrix -- eventually, we'll add support for
    # different feature annotation class, but for now we just fix this
    #F = np.ones(len(chrom_seg_mpos))[:, None]
    #mu = w_grid[:, None, None]
    #sh = t_grid[None, :, None]
    #max_dist = 0.1
    V, Vm = segment_parts
    one_minus_k = 1 - Vm/V

    for f in map_positions:
        rf = dist_to_segment(f, chrom_seg_mpos)[None, None, :]
        # NOTE: using haldane's function always leads to bias for unknown
        # reasons. So instead, we take the rate to be linear with the map
        # distance, which fits simulations much more closely. But this is odd
        # and not in accordance with the probability of recombination in
        # the model.
        rf[rf > 1] = 1.
        a = one_minus_k*(1-rf)
        x = -V/2 * (1/(1-a))**2

        # # bias correct based on the interpolator
        # # x = np.log(zbias((a, V)))
        # a[a == 0] = 1e-14
        # x = np.log(np.exp(x) - bias((a, V)))

        # print(f"max diff: {np.abs(x_asymp - x).mean()}")
        # # we allow Nans because the can be back filled later
        try:
            assert(not np.any(np.isnan(x)))
        except AssertionError:
            warnings.warn("NaNs encountered in calc_BSC16_chunk_worker!")
        #B = np.sum(x, axis=2)
        B = np.einsum('wts,sf->wtf', x, F)
        # the einsum below is for when a features dimension exists, e.g.
        # there are feature-specific μ's and t's -- commented out now...
        #B = np.einsum('ts,w,sf->wtf', x, mut_grid,
        #              F, optimize=BCALC_EINSUM_PATH)
        Bs.append(B)
    return Bs

#@np.vectorize
#def bgs_segment_from_parts_sc16(V, Vm, rf, N, T_factor=5,
#                                log=True, min_rec=1e-12):
#    """
#    Take the pre-computed components of the S&C '16 equation
#    and use them to compute Ne.
#    """
#    #B, Ba, T, V, Vm, Q2, classic_bgs = parts
#    #assert T.shape[2] == rf.shape[2]
#    #Q2 = (1/(Vm/V + rf))**2
#    VmV = Vm/V
#    Z = 1-VmV
#    # The Q2 sequence for rf
#    # for closely linked stuff, the recombination fraction must be > 0
#    rf = max(min_rec, rf)
#    # t0 = time.time()
#    Q2 = Q2_sum_integral(Z, rf, tmax=T_factor*N)
#    # t1 = time.time()
#    #Q2 = Q2_sum_integral(Z, rf, tmax=sum_n*N)
#    # relerr = (np.abs(Q2 - Q2a)/Q2).mean()
#    # print(f"Q2 time: {t1-t0}, rel error: {relerr}")
#    Ne_t = N*np.exp(-V/2 * Q2)
#    # t0 = time.time()
#    B = ave_het(Ne_t)/(2*N)
#    # t1 = time.time()
#    # print(f"ave het time: {t1-t0}")
#    if log:
#        return np.log(B)
#    return B

#### C Wrappers


def B_BK2022(a, V, N, scaling=0):
    """
    """
    B = np.empty(V.size, dtype=np.double)
    Bclib.B_BK2022(a.reshape(-1), V.reshape(-1), B, a.size, N, scaling);
    return B.reshape(V.shape)

@np.vectorize
def Ne_t(a, V, N):
    """
    """
    np.seterr(under='ignore')
    return Bclib.Ne_t(a, V, N);

@np.vectorize
def Ne_t_rescaled(a, V, N, scaling=0.1):
    """
    """
    return Bclib.Ne_t_rescaled(a, V, N, scaling);


