#!/bin/bash

find . -name "varying_nsamples*.json"  | xargs -n1 -P10 -I{} bash snakemake_runner.sh -c {} 
