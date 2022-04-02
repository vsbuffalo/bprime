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


class MapPosChunkIterator(object):
    """
    An iterator base class for calculating B along the genome in parallel in
    'step' steps, by splitting the main objects necessary to calculate B into
    'nchunks'. This is the base, which is modified by subclasses for calculating
    classic B and the new B'. The relevant statistics for the calculation of B
    and B', e.g. segment lengths and recombination rates, mutation and sel coef
    grids, etc are all grouped and iterated over.

    This class also has a method to collate the results computed in
    parallel back into dicts.
    """
    def __init__(self, seqlens, recmap, segments, features_matrix, w_grid=None,
                 t_grid=None, step=100, nchunks=100):
        self.seqlens = seqlens
        self.recmap = recmap
        self.segments = segments
        self.features_matrix = features_matrix
        self.step = step
        self.nchunks = nchunks

        # fixed vectors, i.e. don't change across genome
        self.w_grid = w_grid
        self.t_grid = t_grid

        ## Map position stuff
        # Pre-compute the physical positions to calculate B at for all
        # chromosomes
        chrom_pos = {c: bin_chrom(l, step) for c, l in seqlens.items()}
        # Get the map positions of the spots to calc B at
        chrom_mpos = {c: recmap.lookup(c, p, cummulative=True) for c, p
                      in chrom_pos.items()}
        self._chrom_pos = chrom_pos
        self._chrom_mpos = chrom_mpos
        # chunk the physical and map positions at which we calculate B
        # and group by chromosomes into a dict
        self.chrom_pos_chunks = {c: np.array_split(p, nchunks) for c, p
                                 in chrom_pos.items()}
        self.chrom_mpos_chunks = {c: np.array_split(m, nchunks) for c, m
                                  in chrom_mpos.items()}

        ## Segment stuff
        # extract out the indices for each chromosome -- we're going to group
        # stuff by chromosome with these, since segments, etc are all just one
        # giant array
        chrom_idx = {c: segments.index[c] for c in seqlens}
        # group by chrom and share the segment and feature arrays
        chrom_seg_mpos = {c: share_array(segments.map_pos[idx, :]) for c, idx
                          in chrom_idx.items()}
        chrom_features = {c: share_array(features_matrix[idx, :]) for c, idx
                          in chrom_idx.items()}
        # group by chrom and share segment rec rates
        chrom_seg_rbp = {c: share_array(segments.rates[idx]) for c, idx
                          in chrom_idx.items()}
        # group by chrom and share segment lengths
        L = np.diff(segments.ranges, axis=1).squeeze()
        chrom_seg_L = {c: share_array(L[idx]) for c, idx in chrom_idx.items()}
        # group by chrom and share features matrix
        chrom_features = {c: share_array(features_matrix[idx, :]) for c, idx
                          in chrom_idx.items()}
        self.chrom_idx = chrom_idx
        self.mpos_iter = chain_dictlist(self.chrom_mpos_chunks)
        self.chrom_seg_mpos = chrom_seg_mpos
        self.chrom_features = chrom_features
        self.chrom_seg_rbp = chrom_seg_rbp
        self.chrom_seg_L = chrom_seg_L

    def __iter__(self):
        return self

    @property
    def total(self):
        return sum(map(len, list(itertools.chain(self.chrom_mpos_chunks.values()))))

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


class BChunkIterator(MapPosChunkIterator):
    def __init__(self, seqlens, recmap, segments, features_matrix,
                 segment_parts, w_grid, step, nchunks):

        """
        An iterator for chunk the components to calculate B' along the genome in
        parallel. This is a a sublass of MapPosChunkIterator for calculating
        classic B, which relies on pre-computed segment_parts. These
        segment_parts have the selection coefficient grid in them (for
        efficiency), which is why this is not necessary for this class.
        """
        super().__init__(seqlens, recmap, segments, features_matrix, w_grid,
                         None, step, nchunks)
        # Group the segement parts (these are the parts of the pre-computed
        # equation for calculating B quickly) into chromosomes by the indices
        chrom_segparts = {}
        for chrom in seqlens:
            idx = self.chrom_idx[chrom]
            # share the following arrays across processes, to save memory
            # (these should not be changed!)
            chrom_segparts[chrom] = tuple(map(share_array,
                                            (segment_parts[0][:, idx], segment_parts[1],
                                            segment_parts[2][:, idx], segment_parts[3][:, idx])))

        # custom stuff for the classic B calculation
        self.chrom_segparts = chrom_segparts

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
        return (mpos_chunk,
                self.chrom_seg_mpos[chrom],
                self.chrom_features[chrom],
                chrom_segparts,
                self.w_grid)


class BpChunkIterator(MapPosChunkIterator):
    def __init__(self, seqlens, recmap, segments, features_matrix,
                 w_grid, t_grid, step, nchunks):

        """

        An iterator for chunk the components to calculate B' along the genome in
        parallel.

        """
        super().__init__(seqlens, recmap, segments, features_matrix, w_grid,
                         t_grid, step, nchunks)
        self.tw_mesh = np.array(list(itertools.product(t_grid, w_grid)))
        self.X = self._build_pred_matrix('chr10')
        # self.Y = _build_pred_tensor('chr10')

    def _build_pred_matrix(self, chrom):
        n = self.tw_mesh.shape[0]
        nsegs = len(self.chrom_idx[chrom])
        X = np.zeros((nsegs*n, 5))
        # repeat the w/t element for each tile
        X[:, 0:2] = np.repeat(self.tw_mesh, nsegs, axis=0)
        # tile the annotation data
        X[:, 3] = np.tile(self.chrom_seg_rbp[chrom], n)
        X[:, 4] = np.tile(self.chrom_seg_L[chrom], n)
        return X

    def _build_pred_tensor(self, chrom):
        n = self.tw_mesh.shape[0]
        nsegs = len(self.chrom_idx[chrom])
        X = np.zeros((n, nsegs, 5))
        # repeat the w/t element for each tile
        X[:, 0:2] = np.repeat(self.tw_mesh, nsegs, axis=0)
        X[:, 0] = np.log10(X[:, ]) # sh is log'd
        # tile the annotation data
        X[:, 3] = np.tile(self.chrom_seg_rbp[chrom], n)
        X[:, 4] = np.tile(self.chrom_seg_L[chrom], n)
        return X




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
        return (mpos_chunk, self.chrom_seg_mpos[chrom],
                self.w_grid, self.t_grid, self.tw_mesh, self.X)


