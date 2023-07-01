## Data Directory

Most files here are generated from scripts outside the directory 
(e.g. `slim_sims/training/`).

 - `annotation/`: human genome annotation data
 - `fit_annotation/`: new cleaner fit annotation for new pipeline
 - `phylo/`: substitution rate estimation from PhyloFit, etc


## Other Random Data

Agarwal et al. (2023) Supplementary File 2 (gene IDs and MAP sh estimates): 

    wget -O agarwal_et_al_2023_supplementary_file_2.txt https://raw.githubusercontent.com/agarwal-i/loss-of-function-fitness-effects/main/out/Supplementary%20File%202.txt

To map these to positions we used Ensembl biomart manually; this is
`ensemble_genes.txt`.
    
