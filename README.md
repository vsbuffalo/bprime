# Methods and Analysis for "A Quantitative Genetic Model of Background Selection in Humans"

Below are some brief notes about the code in this repository, which introduces
a new method and Python software implementation `bgspy`. This repository also
contains all Jupyter analysis notebooks in `notebooks/` (see `main_fits.ipynb`
for the main results in the paper).

## Installation

You will need to install conda (and it is highly recommended you install
[mamba](https://mamba.readthedocs.io/en/latest/) too).

    mamba env create -f envs/bprime_dev.yml
    conda activate bprime

The `newick` package is not available through conda (grr!) and must be installed
with:

     ~/miniconda3/envs/bprime/bin/pip install newick

Note I use the full path to `pip`; I had some issues with the wrong `pip` being 
used.

## Command Line Interface

The fitting process can be done in a notebook for simple models. The main fits
for the paper's results are done using the Snakemake-based cluster (see section
below).

Two main pieces of preliminary data are needed before fitting with the MLE
method:

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

Then in Python, one could fit a simple likelihood model with:

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

## Snakemake-based Pipeline and Running Fits on a Cluster

Additionally, see the `fits/` directory for model fits in human based on a
Snakemake pipeline. This approach uses YAML configuration files to fit lots of
models across a cluster; this wraps the command line interface. 
