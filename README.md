

mamba create -n bprime tensorflow-gpu keras numpy scipy statsmodels matplotlib click scikit-learn pandas msprime slim pyslim jupyterlab tqdm scikit-allel bedtools


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

