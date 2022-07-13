

mamba create -n bprime tensorflow-gpu keras numpy scipy statsmodels matplotlib click scikit-learn pandas msprime slim pyslim jupyterlab tqdm scikit-allel bedtools



## Architecture

Simulations can be either msprime (with known classic BGS theory functions,
used as a validation step), or SLiM forward simulations. Simulations are run
using different tools.

 - msprime: `tools/msprime_bgs.py`, which outputs a `.npz` of the features
   matrix targets matrix, and the column names of both.
 - slim: simulations are run using Snakemake, and are then processed with
   `tools/trees2data.py`, which takes a directory of treeseq files, recapitates
   them, extracts features from the simulation metadata, and outputs the
   features and targets matrices with column names of both. 
   
Then we use `tools/fit_sims.py` to further process the `.npz` data. This is
agnostic to what ran the simulation. `fit_sims.py` has two subcommands: `data`
and `fit`. `fit_sims.py data` reads the `.npz` file, takes the product of the
selection and dominance coefficient columns (currently, we treat `h` as fixed),
and extracts all variable columns in the data. The output of this is a
`LearnedFunction` object, which contains the data for model training.

## Thoughts

 - if the ratchet is clicking, can we trust mutational density maps?
 

## TODO

 - feature annotations
 - double check nfixed calc in likelihood.py line 132
 - we're merging all Bs in loglikelihood() is this right?

