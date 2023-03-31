# Chromosome-wide BGS sims


## Fixed Selection Coefficient Simulations with Constant N and Expanding N

    snakemake --profile ~/.config/snakemake/talapas/ all --configfile conserved_cds_utrs_phastcons_merged__hapmap__fixed__demography.yml  --dry-run
    python ../../tools/process_sims.py runs/ results.npz --ncores 20


##  Empirical B Map Simulations

    
    snakemake --profile ~/.config/snakemake/talapas/ all --configfile conserved_cds_utrs_phastcons_merged__hapmap__empiricalB.yml
