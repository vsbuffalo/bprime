import time
import os
import warnings
import tqdm
import functools
import multiprocessing
import numpy as np
from scipy.optimize import fsolve
from scipy.special import expi, gammaincc, gamma
from scipy.integrate import quad
from numba import jit
from ctypes import c_double, c_int, POINTER, c_ssize_t, c_void_p

from bgspy.utils import dist_to_segment
from bgspy.parallel import BChunkIterator, MapPosChunkIterator

# load the library (relative to this file in src/)
LIBRARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'src')
Bclib = np.ctypeslib.load_library("Bclib", LIBRARY_PATH)

# define the types of the dynamic library functions
Bclib.B_BK2022.argtypes = (np.ctypeslib.ndpointer(np.double, ndim=1, flags=('C')),
                            np.ctypeslib.ndpointer(np.double, ndim=1, flags=('C')),
                            np.ctypeslib.ndpointer(np.double, ndim=1, flags=('C','W')),
                            c_ssize_t, c_int, c_double)
Bclib.B_BK2022.restype = None



## Classic BGS
def B_segment_lazy(rbp, L, t):
    """
    TODO check rbp = 0 case
    rt/ (b*(-1 + t) - t) * (b*(-1 + t) + r*(-1 + t) - t)
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
    map_positions, chrom_seg_mpos, F, segment_parts, mut_grid = args
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
        assert(not np.any(np.isnan(x)))
        B = np.einsum('ts,w,sf->wtf', x, mut_grid, F)
        # the einsum below is for when a features dimension exists, e.g.
        # there are feature-specific μ's and t's -- commented out now...
        #B = np.einsum('ts,w,sf->wtf', x, mut_grid,
        #              F, optimize=BCALC_EINSUM_PATH)
        Bs.append(B)
    return Bs

def calc_B_parallel(genome, mut_grid, step, nchunks=1000, ncores=2):
    chunks = BChunkIterator(genome,  mut_grid, step, nchunks)
    print(f"Genome divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    not_parallel = ncores is None or ncores <= 1
    if not_parallel:
        res = []
        for chunk in tqdm.tqdm(chunks, total=chunks.total):
            res.append(calc_B_chunk_worker(chunk))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(calc_B_chunk_worker, chunks),
                                 total=chunks.total))
    return chunks.collate(res)


## New Theory

def Q2_asymptotic(Z, M):
    """
    This is the asymptotic Q² term — up to the factor of two
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
    Het x 2N. If return_parts is True, the series is returned, not the sum.
    """
    x = np.array([np.prod(1 - 1/(2*Ne_t[:i])) for i in np.arange(len(Ne_t))])
    if return_parts:
        return x
    return x.sum()


@jit
def ave_het2(Ne_t):
    """
    Het x 2N. If return_parts is True, the series is returned, not the sum.
    """
    x = np.array([np.prod(1 - 1/(2*Ne_t[:i])) for i in np.arange(len(Ne_t))])
    return x.sum()


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
        out = fsolve(func, [start_T, N], full_output=True)
        Ne = out[0][1]
        T = out[0][0]
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
    Ne_asymp = N*np.exp(-V/2 * Q2_asymp)
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


def bgs_segment_sc16_components(L_rbp, mu, sh, N):
    """
    A manually-vectorized version of bgs_segment_sc16_parts().
    This mimics np.vectorize() but the vectorization is done manually.
    This is because np.vectorize() creates a closure that cannot be
    pickled for multiprocessing work, meaning the np.vectorize()
    version is not parallelizable.
    """
    L, rbp = L_rbp
    assert isinstance(L, (int, float))
    assert isinstance(rbp, float)
    assert isinstance(mu, np.ndarray)
    assert isinstance(sh, np.ndarray)
    mug, shg = np.meshgrid(mu, sh)
    Bs, Bas, Ts, Vs, Vms, Q2s, cbs = [], [], [], [], [], [], []
    for m, s in zip(mug.flat, shg.flat):
        res = bgs_segment_sc16(m, s, L, rbp, N, return_parts=True)
        B, B_asymp, T, V, Vm, Q2, classic_bgs = res
        Bs.append(B)
        Bas.append(B_asymp)
        Ts.append(T)
        Q2s.append(Q2)
        Vs.append(V)
        Vms.append(Vm)
        cbs.append(classic_bgs)

    shape = sh.size, mu.size
    Bs = np.array(Bs).reshape(shape).T
    Bas = np.array(Bas).reshape(shape).T
    Ts = np.array(Ts).reshape(shape).T
    Vs = np.array(Vs).reshape(shape).T
    Vms = np.array(Vms).reshape(shape).T
    Q2s = np.array(Q2s).reshape(shape).T
    cbs = np.array(cbs).reshape(shape).T

    return Bs, Bas, Ts, Vs, Vms, Q2s, cbs

def BSC16_segment_lazy_parallel(mu, sh, L, rbp, N, ncores):
    """
    Compute the fixation time, B, etc for each segment, using the
    equation that integrates over the entire segment *in parallel*.

    Note: N is diploid N but bgs_segment_sc16() takes haploid_N, hence
    the factor of two.
    """
    # stuff that's run on each core
    mu = mu.squeeze()[:, None]
    sh = sh.squeeze()[None, :]

    # stuff that's shipped off to cores
    rbp = rbp.squeeze().tolist()
    L = L.squeeze().tolist()

    # iterate over the segments, but each segments gets the full μ x sh.
    func = functools.partial(bgs_segment_sc16_components, mu=mu, sh=sh, N=N)

    if ncores is None or ncores == 1:
        res = list(tqdm.tqdm(map(func, zip(L, rbp)), total=len(L)))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(func, zip(L, rbp)), total=len(L)))


    # the current function spits out everything (for debugging and validating
    # against the region sims
    #Bs, Bas, Ts, Vs, Vms, Q2s, cbs = zip(*res)
    _, _, _, Vs, Vms, _, _ = zip(*res)
    # we only need to store V and Vm for each mu/sh (this is
    # the mapping of parameters; this determines Z with rf)

    # let's turn each of these into an array, nw x nt x nloci
    V = np.moveaxis(np.stack(Vs), 0, 2)
    Vm = np.moveaxis(np.stack(Vms), 0, 2)
    return V, Vm


def calc_BSC16_chunk_worker(args):
    # ignore rbp, no need here yet under this approximation
    map_positions, chrom_seg_mpos, F, segment_parts, mut_grid, N = args
    Bs = []
    # F is a features matrix -- eventually, we'll add support for
    # different feature annotation class, but for now we just fix this
    #F = np.ones(len(chrom_seg_mpos))[:, None]
    #mu = w_grid[:, None, None]
    #sh = t_grid[None, :, None]
    #max_dist = 0.1
    V, Vm = segment_parts
    one_minus_k = 1 - Vm/V
    B = np.empty(V.size, dtype=np.double)

    for f in map_positions:
        rf = dist_to_segment(f, chrom_seg_mpos)[None, None, :]
        a = one_minus_k*(1-rf)

        # interface directly with the c library function
        t0 = time.time()
        Bclib.B_BK2022(a.reshape(-1), V.reshape(-1), B, a.size, N, 0.1);
        B = B.reshape(V.size)
        t1 = time.time()
        print(f"B_BK2022 time: {t1-t0}")

        t0 = time.time()
        x2 = np.exp(-V/2 * (1/(1-a))**2)
        t1 = time.time()
        print(f"numpy time: {t1-t0}")

        __import__('pdb').set_trace()
        # we allow Nans because the can be back filled later
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

def calc_BSC16_parallel(genome, step, N, nchunks=1000, ncores=2):
    # the None argument is because we do not need to pass in the mutations
    # for the S&C calculation
    chunks = BChunkIterator(genome, None, step, nchunks, N, use_SC16=True)
    s = len(genome.segments)
    print(f"Genome ({s:,} segments) divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    not_parallel = ncores is None or ncores <= 1
    if not_parallel:
        res = []
        for chunk in tqdm.tqdm(chunks):
            res.append(calc_BSC16_chunk_worker(chunk))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(calc_BSC16_chunk_worker, chunks),
                                 total=chunks.total))
    return chunks.collate(res)

@np.vectorize
def bgs_segment_from_parts_sc16(V, Vm, rf, N, T_factor=5,
                                log=True, min_rec=1e-12):
    """
    Take the pre-computed components of the S&C '16 equation
    and use them to compute Ne.
    """
    #B, Ba, T, V, Vm, Q2, classic_bgs = parts
    #assert T.shape[2] == rf.shape[2]
    #Q2 = (1/(Vm/V + rf))**2
    VmV = Vm/V
    Z = 1-VmV
    # The Q2 sequence for rf
    # for closely linked stuff, the recombination fraction must be > 0
    rf = max(min_rec, rf)
    # t0 = time.time()
    Q2 = Q2_sum_integral(Z, rf, tmax=T_factor*N)
    # t1 = time.time()
    #Q2 = Q2_sum_integral(Z, rf, tmax=sum_n*N)
    # relerr = (np.abs(Q2 - Q2a)/Q2).mean()
    # print(f"Q2 time: {t1-t0}, rel error: {relerr}")
    Ne_t = N*np.exp(-V/2 * Q2)
    # t0 = time.time()
    B = ave_het2(Ne_t)/(2*N)
    # t1 = time.time()
    # print(f"ave het time: {t1-t0}")
    if log:
        return np.log(B)
    return B

def B_BK2022(a, V, N, scaling=0):
    """
    """
    B = np.empty(V.size, dtype=np.double)
    Bclib.B_BK2022(a.reshape(-1), V.reshape(-1), B, a.size, N, scaling);
    return B.reshape(V.shape)

