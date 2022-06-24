## classic.py -- funcs for calculating B using McVicker approach and parallelizing stuff
from collections import defaultdict
import itertools
import tqdm
import time
import numpy as np
import multiprocessing
from bgspy.utils import bin_chrom, chain_dictlist, dist_to_segment
from bgspy.utils import haldanes_mapfun
from bgspy.parallel import BChunkIterator, MapPosChunkIterator
from bgspy.theory import bgs_segment_sc16, bgs_rec, bgs_segment_sc16_manual_vec

# pre-computed optimal einsum_path
BCALC_EINSUM_PATH = ['einsum_path', (0, 2), (0, 1)]

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

def T_interpolator(mu, sh, L, rbp, N):
    pass

def BSC16_segment_lazy(mu, sh, rbp, L, t):
    pass


def calc_B(genome, mut_grid, step):
    """
    A Non-parallel version of calc_B_chunk_worker. For the most part,
    this is for debugging.

    Note, features_matrix is not yet implemented - but we can eventually. This
    would be a matrix of which segments belong to what annotation class.
    """
    # alias some stuff for convenience
    segments = genome.segments
    seqlens = genome.seqlens
    recmap = genome.recmap
    segment_parts = genome.segments._segment_parts

    Bs = defaultdict(list)
    Bpos = defaultdict(list)
    chroms = seqlens.keys()
    for chrom in chroms:
        optimal_einsum_path = None
        end = seqlens[chrom]
        print(f"calculating B for {chrom}:", flush=True)
        positions = bin_chrom(end, step)
        Bpos[chrom] = positions
        # pre-compute the map positions for each physical position
        map_pos = recmap.lookup(chrom, positions, cummulative=True)
        assert(len(positions) == len(map_pos))
        # get the segments on this chromosome
        idx = segments.index[chrom]
        nsegs = len(idx)
        chrom_seg_mpos = segments.map_pos[idx, :]

        # alias the segments parts to shorter names
        a, b, c, d = (segment_parts[0][:, idx], segment_parts[1],
                      segment_parts[2][:, idx], segment_parts[3][:, idx])
        total_sites = len(positions)
        t0 = time.time()
        #F = features_matrix[idx, :]
        #F = np.ones(len(idx))[:, None]

        # pre-compute the optimal path -- this shouldn't vary accros positions
        if optimal_einsum_path is None:
            # TODO FIX
            print(f"computing optimal contraction with np.einsum_path()")
            focal_map_pos = np.random.choice(map_pos)
            # as an approx, this only considers dist to start of segement
            #rf = -0.5*np.expm1(-np.abs(chrom_seg_mpos[:, 0] - focal_map_pos))[None, :]
            rf = np.abs(chrom_seg_mpos[:, 0] - focal_map_pos)[None, :]
            x = a/(b*rf**2 + c*rf + d)
            #__import__('pdb').set_trace()
            B = np.einsum('ts,w->wt', x, mut_grid)
            # the einsum below is for when a features dimension exists, e.g.
            # there are feature-specific μ's and t's -- commented out now...
            #optimal_einsum_path = np.einsum_path('ts,w,sf->wtf', x,
            #                                     mut_grid, F, optimize='optimal')
            print(optimal_einsum_path[0])
            print(optimal_einsum_path[1])

        i = 0
        t0, t5, t10, t50 = 0, 0, 0, 0
        tq = tqdm.tqdm(zip(map_pos, positions), total=total_sites)
        #xs = list()
        for f, pos in tq:
            rf = dist_to_segment(f, chrom_seg_mpos)
            assert(not np.any(np.isnan(rf)))
            #hrf = haldanes_mapfun(np.abs(segment_posts - f))
            #assert(np.allclose(rf, hrf))
            x = a/(b*rf**2 + c*rf + d)
            #xs.append(x)
            B = np.einsum('ts,w,sf->wtf', x, mut_grid,
                          F, optimize=optimal_einsum_path[0])
            # TOOD einsum seems to flip these -- need to be flipped back
            #import pdb;pdb.set_trace()
            #B = np.flip(B, axis=1)
            #B = np.outer(x.sum(axis=1), mut_grid[:, None])
            #__import__('pdb').set_trace()
            if i > 1:
                frac_chng = 1-np.mean(np.isclose(B, Bs[chrom][-1]))
                rel_diff = np.median((B - Bs[chrom][-1])/Bs[chrom][-1])
                # how big is the diff?
                t0 += int(rel_diff < 0.05)
                t5 += int(0.05 < rel_diff < 0.1)
                t10 += int(0.1 <= rel_diff < 0.5)
                t50 += int(rel_diff >= 0.5)
                # percent values changed: {np.round(100*frac_chng, 2)}%,
                med_chng = np.abs(np.round(100*rel_diff, 2))
                msg = f"{chrom}: step {int(step/1e3)}kb, median change {med_chng:6.2f}%, ndiffs {t0} (x < 5%) {t5} (5% < x < 10%), {t10} (10% < x < 50%), {t50} (x > 50%))"
                tq.set_description(msg)
            i += 1
            Bs[chrom].append(B)
            #if i > 10:
            #    import pdb;pdb.set_trace()

        t1 = time.time()
        #print(f"chromosome {chrom} complete.", flush=True)
        diff = t1 - t0
        sites_per_hour = total_sites / (diff / (60**2))
        #print(f"{total_sites} sites took {np.round(diff/(60), 4)} minutes.\nA 3Gbp genome would take ~{np.round((3e9/step) / sites_per_hour,2)} hours.")
    return Bs, Bpos#, xs


def calc_B_chunk_worker(args):
    map_positions, chrom_seg_mpos, segment_parts, mut_grid = args
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
        B = np.einsum('ts,w->wt', x, mut_grid)
        # the einsum below is for when a features dimension exists, e.g.
        # there are feature-specific μ's and t's -- commented out now...
        #B = np.einsum('ts,w,sf->wtf', x, mut_grid,
        #              F, optimize=BCALC_EINSUM_PATH)
        Bs.append(B)
    return Bs

def calc_B_parallel(genome, mut_grid, step, nchunks=1000, ncores=2):
    chunks = BChunkIterator(genome,  mut_grid, step, nchunks)
    print(f"Genome divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    not_parallel = ncores <= 1 or ncores is None
    if not_parallel:
        res = []
        for chunk in tqdm.tqdm(chunks):
            res.append(calc_B_chunk_worker(chunk))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(calc_B_chunk_worker, chunks),
                                 total=chunks.total))
    return chunks.collate(res)

def calc_B_SC16_chunk_worker(args):
    # ignore rbp, no need here yet under this approximation
    map_positions, chrom_seg_mpos, seg_L, _, w_grid, t_grid = args
    Bs = []
    # F is a features matrix -- eventually, we'll add support for
    # different feature annotation class, but for now we just fix this
    #F = np.ones(len(chrom_seg_mpos))[:, None]
    mu = w_grid[:, None, None]
    sh = t_grid[None, :, None]
    max_dist = 0.1
    for f in map_positions:
        rf = dist_to_segment(f, chrom_seg_mpos)
        idx = rf <= max_dist
        rf = rf[idx]
        L = seg_L[idx]
        #xm = bgs_segment_sc16_manual_vec(mu, sh, L, rf, N=1000)
        x = bgs_segment_sc16(mu, sh, L, rf, N=1000)
        #__import__('pdb').set_trace()
        assert(not np.any(np.isnan(x)))
        B = np.sum(np.log(x), axis=2)
        # the einsum below is for when a features dimension exists, e.g.
        # there are feature-specific μ's and t's -- commented out now...
        #B = np.einsum('ts,w,sf->wtf', x, mut_grid,
        #              F, optimize=BCALC_EINSUM_PATH)
        Bs.append(B)
    return Bs

def calc_B_SC16_parallel(genome, w_grid, t_grid, step, nchunks=1000, ncores=2):
    chunks = MapPosChunkIterator(genome,  w_grid, t_grid, step, nchunks)
    print(f"Genome divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    not_parallel = ncores is None or ncores <= 1
    if not_parallel:
        res = []
        for chunk in tqdm.tqdm(chunks):
            res.append(calc_B_SC16_chunk_worker(chunk))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(calc_B_SC16_chunk_worker, chunks),
                                 total=chunks.total))
    return chunks.collate(res)


