## Substitution rates from 10-primate phylogeny

Species tree is:

    (((((((pan_troglodytes,pan_paniscus),homo_sapiens),gorilla_gorilla),pongo_abelii),nomascus_leucogenys),((macaca_mulatta,macaca_fascicularis),chlorocebus_sabaeus)),microcebus_murinus);

which is from https://ftp.ensembl.org/pub/current_maf/ensembl-compara/multiple_alignments/10_primates.epo/README.10_primates.epo.

### File manipulations

Download with wget:

Then move them:

Then make a list of files to process:

    ls ensembl_10primates | paste | sed 's/^/ensembl_10primates\//' > primates10_maf.txt


    TREE="(((((((pan_troglodytes,pan_paniscus),homo_sapiens),gorilla_gorilla),pongo_abelii),nomascus_leucogenys),((macaca_mulatta,macaca_fascicularis),chlorocebus_sabaeus)),microcebus_murinus);"
    ls fasta_alns | paste  | xargs -n1 -I {} basename {} .fa | \
      xargs -n1 -I{} -P 60 phyloFit --tree $TREE --subst-mod REV --out-root phylofit_estimates/{} fasta_alns/{}.fa
