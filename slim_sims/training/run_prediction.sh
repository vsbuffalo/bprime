#!/bin/bash

python  ../../bgspy/command_line.py dnnb-write --w '1e-7,1e-8,1e-9' --t '0.01,0.05,1e-3,5e-3,1e-4,5e-4,1e-5' \
  --dir ../../data/dnnb --recmap ../../data/annotation/hapmap_genetic_map.txt --annot ../../data/annotation/conserved_slop.bed.gz \
  --seqlens ../../data/annotation/hg38_seqlens.tsv --max-map-dist 0.1 \
  ../../data/slim_sims/segment_logL_logrbp_logrf_wide/fits/segment_logL_logrbp_logrf_wide/segment_logL_logrbp_logrf_wide_0n128_0n64_0n32_0n8_2nx_eluactiv_fit_0rep

bash snakemake_runner.sh -c segment_predict.json -l cluster_talapas_predict.json -s predict_snakefile

