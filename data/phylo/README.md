## Substitution rates from 10-primate phylogeny


### Other files 

- repeat_masker.bed.gz: from UCSC, just BED of Repeat Masker.

### Gene ID Mapping

BioMart's API sucks so this was done online. The file is checked in.

### Extraction and Concatenation of CDS

`knownCanonical.exonNuc.fa.gz` was downloaded from UCSC (see `Snakefile`) and
processed with a quick script I wrote (`tools/split_multiz.py`). I BLAT'd about
5 random sequences to forward and reverse strands for validation of ranges.

### PhyloFit For CDS

From the raw CDS FASTA alignment data from UCSC.

   TREE="(ponAbe2,((hg38,(panTro4,panPan1)),gorGor3))"
    ls cds_alns | paste  | xargs -n1 -I {} basename {} .fa | \
      xargs -n1 -I{} -P 60 phyloFit --tree $TREE --subst-mod HKY85 --out-root phylofit_estimates/{} cds_alns/{}.fa

### PhyloFit For CDS Regions

From the Ensembl ranges  -- the file size limit is because in earlier processing you end up with empty MAFs

   TREE="(ponAbe2,((hg38,(panTro4,panPan1)),gorGor3))"
    find cds_region_mafs -size +35c | paste  | xargs -n1 -I {} basename {} .fa | \
      xargs -n1 -I{} -P 60 phyloFit --tree $TREE --subst-mod HKY85 --out-root cds_region_phylofits/{} cds_region_mafs/{}.fa


