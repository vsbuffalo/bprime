## Human Annotation Data

Nearly every instruction is in the `Snakefile` in this directory.

### Ensembl Canonical Transcript 

Ensembl's biomart API is a mess so the file `ensembl_cds_canonical_protein_coding.fa.gz`
was downloaded manually from https://www.ensembl.org/biomart/martview/12100db9d580ca294fbcb310aaf018e8

The following filters were used:
 - Transcript type: protein_coding
 - Ensembl Canonical: Only

Also, `gene_tx_ids_canonical.txt` is from BioMart, was downloaded manually.
