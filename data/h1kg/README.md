## Human Genotype Data

This is the dragen 1kg data, merged in a different repository and
symlinked in. The sample files are from,

    awk '$5=="CEU" { print $2 }' ceu_yri_chb_samples.tsv  > ceu_samples.tsv
    awk '$5=="CHB" { print $2 }' ceu_yri_chb_samples.tsv  > chb_samples.tsv
    awk '$5=="YRI" { print $2 }' ceu_yri_chb_samples.tsv  > yri_samples.tsv

and the `ceu_yri_chb_samples.tsv` file is from the 
[1kg data portal site](https://www.internationalgenome.org/data-portal/sample).

