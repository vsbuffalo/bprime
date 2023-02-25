

## Installation

    mamba env create -f envs/bprime_dev.yml
    conda activate bprime

The `newick` package is not available through conda (grr!) and must be installed
with:

     ~/miniconda3/envs/bprime/bin/pip install newick

Note I use the full path to `pip`; I had some issues with the wrong `pip` being 
used.

## Pipeline 

The fitting process can be done in a notebook for simple models.
But this requires two main pieces of preliminary data:

 1. The B' (and optionally, B) maps for a specific genome, 
     recombination map, and set of annotation features (takes 
     ~2-3 hours on ~40 cores, depends on number of features).
 2. The summarized genomic data, for a specific window size,
     which is the data used for the composite likelihood approach
     (takes ~10-20 minutes).

These then go into the maximum likelihood fits, which can be:

 - Maximum likelihood fits.
 - Chromosome-specific MLE fits
 - Leave-one-out for a single chromosome (e.g. for estimation
    of R2 and model selection).
 - Block-jackknife fits.

These steps can be done through either the API or command line interface. The
former is best for exploratory fits and testing the quality of data; the
latter is the best for doing multiple steps (e.g. main MLE fit, LOO
estimation of R2, and block-jackknife estimation of standard errors).

    bgspy data --seqlens hg38_seqlens.tsv --recmap hapmap.tsv \
      --counts-dir allele_counts/ --neutral neutral.bed \
      --access accessible.bed --bs-file simple_track.pkl \
      --window 1000000  --fasta hg38.fa  --output hg38_simple_track_data.pkl

Then in Python, one could fit the likelihood with,

```python
from bgspy.likelihood import mb = SimplexModel
m = SimplexModel.from_data('hg38_simple_track_data.pkl')
m.fit(starts=1000, ncores=40)
```

Printing the object `m` will show something like,

```
SimplexModel (interpolated w): 6 x 7 x 4
  w grid: [1.000e-11 6.310e-11 3.981e-10 2.512e-09 1.585e-08 1.000e-07] (before interpolation)
  t grid: [1.e-07 1.e-06 1.e-05 1.e-04 1.e-03 1.e-02 1.e-01]

Simplex model ML estimates: (genome-wide)
negative log-likelihood: 26865005525.228027
number of successful starts: 1000 (100.0% total)
π0 = 0.0016855
μ = 1.8e-08 
Ne = 28,091 (implied from π0 and μ)
R² = 57.0176% (in-sample)
W = 
          cds    gene    other    phastcons
------  -----  ------  -------  -----------
1e-07   0.001   0.675        1            0
1e-06   0.001   0            0            0
1e-05   0       0            0            0
0.0001  0       0.003        0            0
0.001   0       0.035        0            0
0.01    0       0            0            0
```

Then other analysis steps can be done, like leave-one-out estimation of R2,
and block-jackknife estimates of the parameter standard errors.

### Rescaled-B' Fits

The new B' map approach solves a system of two equations for each segment
that gives the B' curves for a mutation rate and DFE for each feature type.
These equations, however, assume the dynamics at that segment follow a single
genome-wide $N_e$, when in reality, the reductions due to selection impact
the local $N_e$ and these dynamics. However, this reduction depends on the
estimated parameters; ideally we would be able to *simultaneously* solve the
equation and estimate the parameters in a single fitting process. But this is
computationally unfeasible currently, so we do an iterative process.

 1. First the fits are done with the global $N_e$ set to $N$.

 2. Then, these preliminary model estimates go into recalculating a
    *locally-rescaled* B' map, where these equations are re-solved 
    with the preliminary estimate of the local $N_e$ in its vicinity.

Note that this is not cheating in any sense, since the full likelihood would
just have us compute the locally-rescaled B' during the fitting process, but 
given the time costs of calculating the B' map, this is not something can be 
done during every step of numeric optimization.

### Snakemake-based Pipeline and Running Fits on a Cluster

Additionally, see the `fits/` directory for model fits in human. These are
all done with a snakemake-based pipeline that wraps the command line
interface. This allows for model comparison, simply by specifying a model
file in YAML. 


For steps like the block-jackknife that require finding a numeric
optimization

## The B Map Human Simulations

### Single conservation track

```
Conservation track: data/annotation/conserved_phastcons_thresh0_slop1k_chr10.bed 
Recombination map: annotation/HapMapII_GRCh37_liftedOverTo_Hg38/genetic_map_Hg38_chr10.txt
Mutation rates: 1e-10, 3.16e-10, 1e-9, 3.16e-9, 1e-8, 3.16e-8
Selection coefficient: 0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316, 0.1
```

Assumes fixed selection/mutation rates for one single feature type (phastcons).
Phastcons regions are unfiltered (threshold = 0), where I've merged features
that are within 1kbp of each other.  This perhaps increases selection over what
we'd expect in reality, but this is to reduce the time it takes for each
simulation. 

Overall, this is a very approximate model, used primarily for comparing the
theoretic and simulation B scores.


## TODO

 - in sim validation notebook -- we need to watch haploid N
 - should move 

