## parallel.py -- functions to doing stuff in parallel
from collections import defaultdict
import itertools
import numpy as np
import tqdm
import multiprocessing
from bprime.utils import bin_chrom, chain_dictlist

# Some parallelized code lives in classic.py; this is stuff that's common to
# both the classic and DNN code

def share_array(x):
    """
    Convert a numpy array to a multiprocessing.Array
    """
    n = x.size
    cdtype = np.ctypeslib.as_ctypes_type(x.dtype)
    shared_array_base = multiprocessing.Array(cdtype, n)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj()).reshape(x.shape)
    np.copyto(shared_array, x)
    return shared_array


class ChunkIterator(object):
    def __init__(self, seqlens, recmap, segments, features_matrix,
                 segment_parts, t_grid, mut_grid, step, nchunks):

        """
        An iterator for calculating B along the genome in parallel in 'step'
        steps, by splitting the main objects necessary to calculate B (classic)
        B' (DNN) into 'nchunks'. This class also has a method to collate the
        results computed in parallel back into dicts.

        Note that segment_parts is necessary only when calculating B, which are
        pre-computed segment components that are used by the
        calc_B_chunk_worker() to calc B efficiently. t_grid is not necessary
        when classic B is being calculated.
        """

        # pre-compute the positions to calculate B at for all chromosomes
        # and the segment parts

        # this is the main dict of chroms, positions to calc B at
        chrom_pos = {c: bin_chrom(l, step) for c, l in seqlens.items()}
        # get the map positions of the spots to calc B at
        chrom_mpos = {c: recmap.lookup(c, p, cummulative=True) for c, p
                      in chrom_pos.items()}
        # chunk the physical and map positions at which we calculate B
        # and group by chromosomes into a dict
        self.chrom_pos_chunks = {c: np.array_split(p, nchunks) for c, p
                                 in chrom_pos.items()}
        self.chrom_mpos_chunks = {c: np.array_split(m, nchunks) for c, m
                                  in chrom_mpos.items()}


        # extract out the indices for each chromosome -- we're going to group
        # stuff by chromosome with these, since segments, etc are all just one
        # giant array
        chrom_idx = {c: segments.index[c] for c in seqlens}
        # share the segment and feature arrays
        chrom_segments = {c: share_array(segments.map_pos[idx, :]) for c, idx
                          in chrom_idx.items()}
        chrom_features = {c: share_array(features_matrix[idx, :]) for c, idx
                          in chrom_idx.items()}


        if segment_parts is not None:
            # Group the segement parts (these are the parts of the pre-computed
            # equation for calculating B quickly) into chromosomes by the indices
            chrom_segparts = {}
            for chrom in seqlens:
                idx = chrom_idx[chrom]
                # share the following arrays across processes, to save memory
                # (these should not be changed!)
                chrom_segparts[chrom] = tuple(map(share_array,
                                                (segment_parts[0][:, idx], segment_parts[1],
                                                segment_parts[2][:, idx], segment_parts[3][:, idx])))

        self.mpos_iter = chain_dictlist(self.chrom_mpos_chunks)
        self.chrom_segments = chrom_segments
        self.chrom_features = chrom_features
        self.chrom_segparts = chrom_segparts
        self.t_grid = t_grid
        self.mut_grid = mut_grid
        self.nchunks = nchunks

    def __iter__(self):
        return self

    @property
    def total(self):
        return sum(map(len, list(itertools.chain(self.chrom_mpos_chunks.values()))))

    def __next__(self):
        """
        Iterate over the map positions of each site to calculate B at, with the
        requisite data for that chromosome to calculate B.

        Each iteration yields:
            - A chunk of many map positions to calculate B at
            - The conserved segments on this chromosome.
            - The grid of mutation rates.
            - The features matrix of classes of conserved segments
            - The pre-computed segment parts (for efficiently calculating B
              under the classic approach).
        """
        next_chunk = next(self.mpos_iter)
        chrom, mpos_chunk = next_chunk
        chrom_segparts = self.chrom_segparts.get(chrom, None)
        return (mpos_chunk, self.chrom_segments[chrom], self.t_grid,
                self.mut_grid, self.chrom_features[chrom], chrom_segparts)

    def collate(self, results):
        """
        Take the B calculations from the parallel operation and collate
        them back together.
        """
        Bs = defaultdict(list)
        B_pos = defaultdict(list)
        pos_iter = chain_dictlist(self.chrom_pos_chunks)
        for res in results:
            try:
                chunk = next(pos_iter)
            except StopIteration:
                break
            chrom, pos = chunk
            Bs[chrom].extend(res)
            B_pos[chrom].extend(pos)
        for chrom in Bs:
            Bs[chrom] = np.array(Bs[chrom], dtype='float64')
            assert(B_pos[chrom] == sorted(B_pos[chrom]))
            B_pos[chrom] = np.array(B_pos[chrom], dtype='uint32')
        return Bs, B_pos


