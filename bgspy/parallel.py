## parallel.py -- functions to doing stuff in parallel
from collections import defaultdict, deque
import itertools
import numpy as np
from tqdm import tqdm
import multiprocessing
import functools
from bgspy.utils import bin_chrom, chain_dictlist
from bgspy.theory2 import calc_B_chunk_worker, calc_BSC16_chunk_worker
from bgspy.theory2 import bgs_segment_sc16_components

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
        chrom_mpos = {c: genome.recmap.lookup(c, p, cumulative=True) for c, p
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
        chrom_features = {c: share_array(segments.F[idx]) for c, idx
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

    def __next__(self):
        """
        Iterate over the map positions of each site to calculate B at, with the
        requisite data for that chromosome to calculate B.

        Each iteration yields the standard components to calculate B
            - A chunk of many map positions to calculate B at
            - The conserved segments on this chromosome.
            - The grid of mutation rates.
            - The features matrix of classes of conserved segments
            - The pre-computed segment parts (for efficiently calculating B
              under the classic approach).
        """
        next_chunk = next(self.mpos_iter)
        chrom, mpos_chunk = next_chunk
        chrom_seg_L = self.chrom_seg_L.get(chrom, None)
        chrom_seg_rbp = self.chrom_seg_rbp.get(chrom, None)

        return (mpos_chunk,
                self.chrom_seg_mpos[chrom],
                chrom_seg_L,
                chrom_seg_rbp,
                self.w_grid, self.t_grid)

    def collate(self, results):
        """
        Take the B calculations from the parallel operation and collate
        them back together. This assumes the results are in the same order
        as initially set, e.g. this is not appropriate for distributed computing
        on a cluster (but is for multiprocessing module's map).

        Notes: this uses a deque, for O(1) pops of the first element. 
        The popleft is used to avoid doubling the total results size in 
        memory during collation.
        """
        assert isinstance(results, deque), "results must be a deque"
        Bs = defaultdict(list)
        B_pos = defaultdict(list)
        pos_iter = chain_dictlist(self.chrom_pos_chunks)
        while True:
            try:
                chunk = next(pos_iter)
                res = results.popleft()
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
    def __init__(self, genome, w_grid, step, nchunks, N=None, use_SC16=False):

        """
        An iterator for chunk the components to calculate B' along the genome in
        parallel. This is a a sublass of MapPosChunkIterator for calculating
        classic B, which relies on pre-computed segment_parts. These
        segment_parts have the selection coefficient grid in them (for
        efficiency), which is why this is not necessary for this class.
        """
        super().__init__(genome, w_grid, None, step, nchunks)
        self.N = N
        if not use_SC16:
            assert genome.segments._segment_parts is not None, "Genome.segments does not have segment parts"
            segment_parts = genome.segments._segment_parts
        else:
            # we're using the S&C '16 approach
            assert genome.segments._segment_parts_sc16 is not None, "Genome.segments does not have S&C '16 segment parts"
            segment_parts = genome.segments._segment_parts_sc16

        self.interp_parts = None
        # NOTE: this is slow and turned out to not matter much
        #if use_SC16:
        #    # we need to build an Ne(t) vs Ne_asymp bias interpolator
        #    # big grid
        #    print(f"building Ne bias correction parts...\t", end="")
        #    a_grid = np.sort(1-np.logspace(-8, np.log(1-1e-20), 50))
        #    V_grid = np.logspace(-15, -6, 50)
        #    interp_grid = np.meshgrid(a_grid, V_grid)
        #    interp_asymp = Ne_asymp2(*interp_grid, N)/N
        #    interp_real = Ne_t(*interp_grid, N)/N
        #    bias = interp_asymp - interp_real
        #    # It should be that Ne(t) > Ne_asymp. If bias > 0
        #    # this is due to a small amount of error in evaluating the Ne(t)
        #    # series so we truncate
        #    bias[bias > 0] = 0
        #    # __import__('pdb').set_trace()
        #    interp_bias = bias
        #    #func = RegularGridInterpolator((a_grid, V_grid), interp_bias.T, method='linear')
        #    self.interp_parts = (a_grid, V_grid, interp_bias)
        #    print(f"done.")

        seqlens = genome.seqlens
        # Group the segement parts (these are the parts of the pre-computed
        # equation for calculating B quickly) into chromosomes by the indices
        chrom_segparts = {}
        for chrom in seqlens:
            #if self.genome.segments.rescaling is not None:
            #    # this is a hack -- fits often exclude chrX, so if rescaling is being
            #    # used, ignore this chrom
            #    if chrom in self.genome.segments._rescaling_excluded:
            #        continue
            idx = self.chrom_idx[chrom]
            # share the following arrays across processes, to save memory
            # (these should not be changed!)
            #F = self.genome.segments.F[idx, :]
            if use_SC16:
                assert N is not None
                # these are all segment-specific
                # segment_parts for SC16 has V, Vm, and T. We only need the first two
                segparts = (segment_parts[0][:, :, idx],
                            segment_parts[1][:, :, idx])
            else:
                # the b element is not segment specific
                segparts = (segment_parts[0][:, idx], segment_parts[1],
                            segment_parts[2][:, idx], segment_parts[3][:, idx])
            chrom_segparts[chrom] = tuple(map(share_array, segparts))

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
                self.w_grid, self.N, self.interp_parts)


def calc_B_parallel(genome, mut_grid, step, nchunks=1000, ncores=2):
    chunks = BChunkIterator(genome,  mut_grid, step, nchunks)
    print(f"Calculating classic B, Genome divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    not_parallel = ncores is None or ncores <= 1
    if not_parallel:
        res = []
        for chunk in tqdm(chunks, total=chunks.total):
            res.append(calc_B_chunk_worker(chunk))
        res = deque(res)
    else:
        with multiprocessing.Pool(ncores) as p:
            res = deque(p.imap(calc_B_chunk_worker, tqdm(chunks, total=chunks.total)))
            #p.close()
            #p.join()
    print("Collating all B results...  ", end='', flush=True)
    return chunks.collate(res)
    print("done.")


def calc_BSC16_parallel(genome, step, N, nchunks=1000, ncores=2):
    # the None argument is because we do not need to pass in the mutations
    # for the S&C calculation
    chunks = BChunkIterator(genome, None, step, nchunks, N, use_SC16=True)
    s = len(genome.segments)
    print(f"Calculating B', genome ({s:,} segments) divided into {chunks.total} chunks to be processed on {ncores} CPUs...")
    not_parallel = ncores is None or ncores <= 1
    if not_parallel:
        res = []
        for chunk in tqdm(chunks):
            res.append(calc_BSC16_chunk_worker(chunk))
        res = deque(res)
    else:
        with multiprocessing.Pool(ncores) as p:
            res = deque(p.imap(calc_BSC16_chunk_worker, tqdm(chunks, total=chunks.total)))
            #p.close()
            #p.join()
    print("Collating all B' results...  ", end='', flush=True)
    return chunks.collate(res)
    print("done.")


def BSC16_segment_lazy_parallel(mu, sh, L, rbp, N, ncores,
                                rescaling=None, pairwise=False):
    """
    Compute the fixation time, B, etc for each segment, using the
    equation that integrates over the entire segment *in parallel*.

    Note: N is diploid N but bgs_segment_sc16() takes haploid_N, hence
    the factor of two.
    """
    if isinstance(mu, np.ndarray):
        assert isinstance(sh, np.ndarray)
        # stuff that's run on each core
        mu = mu.squeeze()[:, None]
        sh = sh.squeeze()[None, :]

    # stuff that's shipped off to cores
    rbp = rbp.squeeze().tolist()
    L = L.squeeze().tolist()

    # iterate over the segments, but each segments gets the full μ x sh.
    func = functools.partial(bgs_segment_sc16_components, mu=mu, sh=sh, N=N,
                             pairwise=pairwise)

    if rescaling is None:
        # create a dummy list to iterate over; internal code handles changing this to 1.
        rescaling = [None] * len(L)

    if ncores is None or ncores == 1:
        res = list(map(func, tqdm(zip(L, rbp, rescaling), total=len(L))))
    else:
        with multiprocessing.Pool(ncores) as p:
            res = list(p.imap(func, tqdm(zip(L, rbp, rescaling), total=len(L))))
            p.close()
            p.join()

    # the current function spits out everything (for debugging and validating
    # against the region sims
    #Bs, Bas, Ts, Vs, Vms, Q2s, cbs = zip(*res)
    Ts, Vs, Vms = zip(*res)
    if pairwise:
        return np.stack(Vs).T, np.stack(Vms).T, np.stack(Ts).T
    # we only need to store V and Vm for each mu/sh (this is
    # the mapping of parameters; this determines Z with rf)

    # let's turn each of these into an array, nw x nt x nloci
    V = np.moveaxis(np.stack(Vs), 0, 2)
    Vm = np.moveaxis(np.stack(Vms), 0, 2)
    Ts = np.moveaxis(np.stack(Ts), 0, 2)
    return V, Vm, Ts


