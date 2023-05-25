# Chromosome-wide BGS sims


## Fixed Selection Coefficient Simulations with Constant N and Expanding N

    snakemake --profile ~/.config/snakemake/talapas/ all --configfile conserved_cds_utrs_phastcons_merged__hapmap__fixed__demography.yml  --dry-run
    python ../../tools/process_sims.py runs/ results.npz --ncores 20


##  Empirical B Map Simulations

    snakemake --profile ~/.config/snakemake/talapas/ all --configfile conserved_cds_utrs_phastcons_merged__hapmap__empiricalB.yml

## Summarize the Chromosome Sims

This summarizes the empirical B maps, including LD stats, for these sims.

     python ../../tools/process_sims.py runs/conserved_cds_utrs_phastcons_merged__hapmap__fixed_empiricalB/sims/expansion__false/h__0.5/chrom__chr10/N__1000/ empiricalB_chr10__expansion_false__h_0.5__results.npz --ncores 40
