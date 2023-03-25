## Region Simulations

These are used to test theory; they look at B in 
a region of a fixed recombination rate and length, 
under different parameters. 

    $ snakemake all --profile ~/.config/snakemake/talapas/ --configfile region.yml

The `Snakefile` doesn't process these (it's a pain...) so 
this is done manually on a larger machine (20 cores are 
requested by default) with:

    $ python ../../tools/process_region_sims.py runs/region/ region_results.pkl
