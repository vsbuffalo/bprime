## 
 python  ../../tools/slurm_slim_runner2.py --slim '~/src/SLiM_build/slim'  --secs-per-job 10 --batch-size 150 --seed 3 validate.json
./snakemake_runner.sh -c validate.json -l cluster_talapas_validate.json -s validate_snakefile -r all
