#!/bin/bash

find . -name "msprime*json"  | xargs -n1 -P10 -I{} bash snakemake_runner.sh -c {}
