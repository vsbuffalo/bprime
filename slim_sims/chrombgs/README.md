## BGS Chromosome simulations

Note that the conservation tracks are passed into 
to the SLiM routine via commandline. The chromosome is
also passed, so the conservation track should be formatted
like `conserved_{details}_chr10.bed` and what is passed
in the JSON file is `conserved_{details}` as the rest is filled
in by the SLiM script.

Run with:

    bash snakemake_runner.sh -c chrombgs_chr10.json all

