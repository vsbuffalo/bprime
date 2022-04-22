#!/bin/bash

bash snakemake_runner.sh -c ./simple.json &
bash snakemake_runner.sh -c ./simple_varL_varrbp.json &
bash snakemake_runner.sh -c ./segment_logL.json &
bash snakemake_runner.sh -c ./segment_uniform_mu.json &
bash snakemake_runner.sh -c ./segment_logmu.json &
bash snakemake_runner.sh -c ./segment.json &
bash snakemake_runner.sh -c ./simple_varL.json &
