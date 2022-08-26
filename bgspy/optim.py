import multiprocessing
import numpy as np
import tqdm
from tabulate import tabulate
from scipy.optimize import minimize

# nlopt has different error messages; we use this for scipy to
# but just force to 'success' (1) or 'failure' (-1)
NL_OPT_CODES = {1:'success', 2:'stopval reached', 3:'ftol reached',
                4:'xtol reached', 5:'max eval', 6:'maxtime reached', -1:'failure',
                -2:'invalid args', -3:'out of memory', -4:'forced stop'}


def extract_opt_info(x):
    is_scipy = hasattr(x, 'fun')
    if is_scipy:
        nll, res, success = x.fun, x.x, x.success
        success = -1 if not success else 1
    else:
        nll, res, success = x
        assert success in NL_OPT_CODES.keys()
    return nll, res, success


def run_nloptims(workerfunc, startfunc, nstarts, ncores = 70):
    starts = [startfunc() for _ in range(nstarts)]
    with multiprocessing.Pool(ncores) as p:
        res = list(tqdm.tqdm(p.imap(workerfunc, starts), total=nstarts))
    nlls = np.array([x[0] for x in res])
    thetas = np.array([x[1] for x in res])
    success = np.array([x[2] for x in res])
    return nlls, thetas, success

def array_all(x):
    return tuple([np.array(a) for a in x])


def run_optims(workerfunc, startfunc, nstarts, ncores=50):
    """
    """
    starts = [startfunc() for _ in range(nstarts)]
    with multiprocessing.Pool(ncores) as p:
        res = list(tqdm.tqdm(p.imap(workerfunc, starts), total=nstarts))
    nlls, thetas, success = array_all(zip(*map(extract_opt_info, res)))
    return nlls, thetas, success


class Likelihood:
    def __init__(self, engine, algo):
        pass

    def run_optims(self, workerfunc, startfunc
