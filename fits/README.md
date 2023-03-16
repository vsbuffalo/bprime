## Model fits

I make directories for results so multiple models from 
different JSON files can be fit simultaneously on the 
cluster, all sharing the same Snakefile. The directories
ensure that snakemake locking directories won't prevent
simultaneously running model fits.

Each configuation `.yml` file has information on the 
input data, number of random starts, bootstraps, etc.
The `Snakefile` and `.yml` configuration file are linked
into each run directory, and each `Snakefile` should be
run in the run directory.

Use

    $ bash init_run.sh model_spec.json 

to create directories/links for model fitting. Then,

    $ cd model_spec 
    $ snakemake data -j1 --configfile ./model_spec.json
    $ snakemake mle -j1 --configfile ./model_spec.json

to calculate the data summaries and fit the model.

Note that the run directory will match the `.yml` 
filename.

### Notes

 - The name of the run (at the top of the `.yml` config file, 
   should only reflect the annotation data and recmap, not the 
   samples. This is because this name goes into the B and B' maps

 - Note that the JSON model specification files must specify 
   files with an extra `../` since these are run one 
   directory deeper than the JSON files are in.


