## Model fits

I make directories for results so multiple models from 
different JSON files can be fit simultaneously on the 
cluster, all sharing the same Snakefile. The directories
ensure that snakemake locking directories won't prevent
simultaneously running model fits.

Use

    $ bash init_run.sh model_spec.json 

to create directories/links for model fitting. Then,

    $ cd model_spec 
    $ snakemake data -j1 --configfile ./model_spec.json
    $ snakemake mle -j1 --configfile ./model_spec.json

to calculate the data summaries and fit the model.

Note that the JSON model specification files must specify files with an extra
`../` since these are run one directory deeper than the JSON files are in.
