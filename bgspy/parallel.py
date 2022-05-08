## parallel.py -- functions to doing stuff in parallel
from collections import defaultdict
import itertools
import numpy as np
import tqdm
import multiprocessing
from bgspy.utils import bin_chrom, chain_dictlist, dist_to_segment

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
    classic B (the new B' uses this directly). The relevant statistics for the
    calculation of B and B', e.g. segment lengths and recombination rates,
    mutation and sel coef grids, etc are all grouped and iterated over.

    This class also has a method to collate the results computed in
    parallel back into dicts.
    """
    def __init__(self, genome, w_grid=None, t_grid=None,
                 step=100, nchunks=100):
        self.genome = genome
        self.step = step
        self.nchunks = nchunks

        # fixed vectors, i.e. don't change across genome
        self.w_grid = w_grid
        self.t_grid = t_grid

        ## Map position stuff
        # Pre-compute the physical positions to calculate B at for all
        # chromosomes
        chrom_pos = {c: bin_chrom(l, step) for c, l in genome.seqlens.items()}
        # Get the map positions of the spots to calc B at
        chrom_mpos = {c: genome.recmap.lookup(c, p, cummulative=True) for c, p
                      in chrom_pos.items()}

        self.chrom_pos = chrom_pos
        self.chrom_mpos = chrom_mpos
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
        segments = genome.segments
        chrom_idx = {c: segments.index[c] for c in genome.seqlens}
        # group by chrom and share the segment and feature arrays
        chrom_seg_mpos = {c: share_array(segments.map_pos[idx, :]) for c, idx
                          in chrom_idx.items()}
        # group by chrom and share segment rec rates
        chrom_seg_rbp = {c: share_array(segments.rates[idx]) for c, idx
                          in chrom_idx.items()}
        # group by chrom and share segment lengths
        L = segments.lengths
        chrom_seg_L = {c: share_array(L[idx]) for c, idx in chrom_idx.items()}
        # group by chrom and share features matrix
        chrom_features = {c: share_array(segments.features[idx]) for c, idx
                          in chrom_idx.items()}
        self.chrom_idx = chrom_idx
        # this is the main thing iterated over -- a generator over the
        # chromosome, map positions chunks across the genome
        self.mpos_iter = chain_dictlist(self.chrom_mpos_chunks)
        self.pos_iter = chain_dictlist(self.chrom_pos_chunks)
        self.chrom_seg_mpos = chrom_seg_mpos
        self.chrom_features = chrom_features
        self.chrom_seg_rbp = chrom_seg_rbp
        self.chrom_seg_L = chrom_seg_L

    @property
    def seqlens(self):
        return self.genome.seqlens

    @property
    def recmap(self):
        return self.genome.recmap

    @property
    def segments(self):
        return self.genome.segments

    def __iter__(self):
        return self

    @property
    def total(self):
        return sum(map(len, list(itertools.chain(self.chrom_mpos_chunks.values()))))

    def collate(self, results):
        """
        Take the B calculations from the parallel operation and collate
        them back together. This assumes the results are in the same order
        as initially set, e.g. this is not appropriate for distributed computing
        on a cluster (but is for multiprocessing module's map).
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
    def __init__(self, genome, w_grid, step, nchunks):

        """
        An iterator for chunk the components to calculate B' along the genome in
        parallel. This is a a sublass of MapPosChunkIterator for calculating
        classic B, which relies on pre-computed segment_parts. These
        segment_parts have the selection coefficient grid in them (for
        efficiency), which is why this is not necessary for this class.
        """
        super().__init__(genome, w_grid, None, step, nchunks)
        assert genome.segments._segment_parts is not None, "Genome.segments does not have segment parts"
        segment_parts = genome.segments._segment_parts
        seqlens = genome.seqlens
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
                chrom_segparts,
                self.w_grid)

