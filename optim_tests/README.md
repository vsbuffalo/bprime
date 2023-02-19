## Optimization Experiments

In this directory are the results of trying 
different optimization approaches. The `Snakefile`
contains the code that loads in some model data, 
runs a fit for a given engine (e.g. scipy or nlopt)
and method. Note that the methods for nlopt must 
start with `LN_` and `GN_`, as these are accessed 
directly from the `nlopt` module.

Overall the results seem conclusive that nlopt's
BOBYQA is the fastest, most stable optimization 
routine for the softmax parameterization of the 
simplex model.
