## classic.py -- funcs for calculating B using McVicker approach and parallelizing stuff
from collections import defaultdict
import itertools
import tqdm
import numpy as np
import multiprocessing
from bprime.utils import bin_chrom, chain_dictlist
from bprime.parallel import ChunkIterator

# The rec fraction between the segment and the focal netural site.
# If this is None, 0 rec fractions (fully linked) are allowed. This
# looks to cause pathologies, where as t -> 0, for reasonable mutation,
# etc parameters, B asymptotes to 1-e ~ 0.37.
MIN_RF = None


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


def calc_B(segments, segment_parts, features_matrix, mut_grid,
           recmap, seqlens, step):
    """
    A Non-parallel version of calc_B_chunk_worker. For the most part,
    this is for debugging.

    segments: A Segments named tuple.
    segment_parts: a tuple of pre-computed segment parts.
    recmap: a RecMap object.
    features_matrix: a matrix of which segments belong to what annotation class.
    """
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
        chrom_segments = segments.map_pos[idx, :]

        # alias the segments parts to shorter names
        a, b, c, d = (segment_parts[0][:, idx], segment_parts[1],
                      segment_parts[2][:, idx], segment_parts[3][:, idx])
        total_sites = len(positions)
        t0 = time.time()
        F = features_matrix[idx, :]

        # pre-compute the optimal path -- this shouldn't vary accros positions
        if optimal_einsum_path is None:
            print(f"computing optimal contraction with np.einsum_path()")
            focal_map_pos = np.random.choice(map_pos)
            # as an approx, this only considers dist to start of segement
            #rf = -0.5*np.expm1(-np.abs(chrom_segments[:, 0] - focal_map_pos))[None, :]
            rf = np.abs(chrom_segments[:, 0] - focal_map_pos)[None, :]
            if MIN_RF is not None:
                rf[rf < MIN_RF] = MIN_RF
            x = a/(b*rf**2 + c*rf + d)
            #__import__('pdb').set_trace()
            optimal_einsum_path = np.einsum_path('ts,w,sf->wtf', x,
                                                 mut_grid, F, optimize='optimal')
            print(optimal_einsum_path[0])
            print(optimal_einsum_path[1])

        i = 0
        t0, t5, t10, t50 = 0, 0, 0, 0
        tq = tqdm.tqdm(zip(map_pos, positions), total=total_sites)
        #xs = list()
        for f, pos in tq:
            is_left = (chrom_segments[:, 0] - f) > 0
            is_right = (f - chrom_segments[:, 1]) > 0
            is_contained = (~is_left) & (~is_right)
            dists = np.zeros(is_left.shape)
            dists[is_left] = np.abs(f-chrom_segments[is_left, 0])
            dists[is_right] = np.abs(f-chrom_segments[is_right, 1])
            assert(len(dists) == chrom_segments.shape[0])

            #rf = -0.5*np.expm1(-dists)[None, :]
            rf = dists
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
    map_positions, chrom_segments, _, mut_grid, features_matrix, segment_parts = args
    a, b, c, d = segment_parts
    Bs = []
    for f in map_positions:
        # first, figure out for each segment if this map position is left, right
        # or within the segment. This matters for calculating the distance to
        # the segment
        # ---f-----L______R----------
        # ---------L______R-------f--
        is_left = (chrom_segments[:, 0] - f) > 0
        is_right = (f - chrom_segments[:, 1]) > 0
        is_contained = (~is_left) & (~is_right)
        dists = np.zeros(is_left.shape)
        dists[is_left] = np.abs(f-chrom_segments[is_left, 0])
        dists[is_right] = np.abs(f-chrom_segments[is_right, 1])
        assert len(dists) == chrom_segments.shape[0], (len(dists), chrom_segments.shape)

        #rf = -0.5*np.expm1(-dists)[None, :]
        rf = dists
        #rf[dists > 0.5] = 0.5
        if MIN_RF is not None:
            rf[rf < MIN_RF] = MIN_RF
        if np.any(b + rf*(rf*c + d) == 0):
            raise ValueError("divide by zero in calc_B_chunk_worker")
        x = a/(b*rf**2 + c*rf + d)
        assert(not np.any(np.isnan(x)))
        B = np.einsum('ts,w,sf->wtf', x, mut_grid,
                      features_matrix, optimize=BCALC_EINSUM_PATH)
        #B = np.flip(np.flip(B, axis=0), axis=1)
        Bs.append(B)
    return Bs


def calc_B_parallel(segments, segment_parts, features_matrix, mut_grid,
                    recmap, seqlens, step, nchunks=1000, ncores=2):
    t_grid = None # selection coefs are already in pre-calc'd segment_parts
    chunks = ChunkIterator(seqlens, recmap, segments, features_matrix,
                           segment_parts, t_grid, mut_grid, step, nchunks)
    print(f"Genome divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    debug = False
    if debug:
        res = []
        for chunk in tqdm.tqdm(chunks):
            res.append(calc_B_chunk_worker(chunk))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(tqdm.tqdm(p.imap(calc_B_chunk_worker, chunks),
                                 total=chunks.total))
    return chunks.collate(res)


