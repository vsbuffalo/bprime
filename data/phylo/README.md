## Substitution rates from 10-primate phylogeny

### Extraction and Concatenation of CDS

`knownCanonical.exonNuc.fa.gz` was downloaded from UCSC (see `Snakefile`) and
processed with a quick script I wrote (`tools/split_multiz.py`). I BLAT'd about
5 random sequences to forward and reverse strands for validation of ranges.

### PhyloFit

   TREE="(ponAbe2,((hg38,(panTro4,panPan1)),gorGor3))"
    ls cds_alns | paste  | xargs -n1 -I {} basename {} .fa | \
      xargs -n1 -I{} -P 60 phyloFit --tree $TREE --subst-mod HKY85 --out-root phylofit_estimates/{} cds_alns/{}.fa

