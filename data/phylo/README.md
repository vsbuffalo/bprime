## Substitution rates from 10-primate phylogeny

Species tree is:

    (((((((pan_troglodytes,pan_paniscus),homo_sapiens),gorilla_gorilla),pongo_abelii),nomascus_leucogenys),((macaca_mulatta,macaca_fascicularis),chlorocebus_sabaeus)),microcebus_murinus);

which is from https://ftp.ensembl.org/pub/current_maf/ensembl-compara/multiple_alignments/10_primates.epo/README.10_primates.epo.

### File manipulations

Downloaded with:

    wget -e robots=off  -r --no-parent https://ftp.ensembl.org/pub/current_maf/ensembl-compara/multiple_alignments/10_primates.epo/

Then moved to `ensembl_10primates/`

Then make a list of files to process:

    ls ensembl_10primates | paste | sed 's/^/ensembl_10primates\//' | grep -v MT_ | grep -v Y_ | grep -v other_ | grep -v scaffold > primates10_maf.txt

Which ignores some sequences (MT, Y, etc).

Then, we use `../../tools/maf_parser.py` to extract out the FASTA alignments
for each region.

    cat primates10_maf.txt | xargs -n1 -P 10 python ../../tools/maf_parser.py

(Beware, this can be very memory intensive! Don't use more than 10 cores.)

Finally, we do the substitution rate estimates under the REV model using
`phyloFit`. The tree is stored in an environmental variable.

    TREE="(((((((pan_troglodytes,pan_paniscus),homo_sapiens),gorilla_gorilla),pongo_abelii),nomascus_leucogenys),((macaca_mulatta,macaca_fascicularis),chlorocebus_sabaeus)),microcebus_murinus);"
    ls fasta_alns | paste  | xargs -n1 -I {} basename {} .fa | \
      xargs -n1 -I{} -P 60 phyloFit --tree $TREE --subst-mod REV --out-root phylofit_estimates/{} fasta_alns/{}.fa


### Indexing

The ranges here are 0-based according to
https://ftp.ensembl.org/pub/current_maf/ensembl-compara/multiple_alignments/10_primates.epo/README.maf
