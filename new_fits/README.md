# Main Model Fits

Our main model runs are Python pickle files on Data Dryad. These are the files
that would be in the model directories below, and are the result of `bgspy
collect`.

 - CADD 6%: `cadd6__decode__altgrid.pkl`
 - CADD 8%: `cadd8__decode__altgrid.pkl`
 - Feature Priority: `CDS_genes_phastcons__decode__altgrid/CDS_genes_phastcons__decode__altgrid.pkl`
 - PhastCons Priority: `phastcons_CDS_genes__decode__altgrid/phastcons_CDS_genes__decode__altgrid.pkl`

Each can be loaded with `bgspy`, with:

    from bgspy.utils import load_pickle
    m = load_pickle('cadd6__decode__altgrid.pkl')
    m.fits # access main fits
    m.rescaled # access main rescaled fits
    m.mu_predict # access predictions under fixed mutation rate
