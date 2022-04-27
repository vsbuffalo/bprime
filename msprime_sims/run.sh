#!/bin/bash

bash snakemake_runner.sh -c ./simple.json &
bash snakemake_runner.sh -c ./simple_varL.json &&
bash snakemake_runner.sh -c ./simple_varlogL.json &&
bash snakemake_runner.sh -c ./simple_varlogL_varrbp.json &&
bash snakemake_runner.sh -c ./segment_logL_logrbp_logrf.json &&
bash snakemake_runner.sh -c ./segment_logL_logrbp_logrf_wide.json &&
bash snakemake_runner.sh -c ./segment_logL_logrbp_logrf_wide_reject.json &&
bash snakemake_runner.sh -c ./segment_logL_logrbp_logrf_wide_reject_replicated.json
