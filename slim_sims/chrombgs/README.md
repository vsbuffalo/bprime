# Chromosome-wide BGS sims


## Fixed Selection Coefficient Simulations

    snakemake --profile ~/.config/snakemake/talapas/ all --configfile conserved_cds_utrs_phastcons_merged__hapmap__fixed.yml
    python ../../tools/process_sims.py runs/ results.npz --ncores 20


