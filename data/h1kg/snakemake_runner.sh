#!/bin/bash
DRY=""
MAXJOBS=999 # submit a ton of jobs, let the job scheduler handler it
CLUSTER_CONFIG="./cluster_talapas.json"
SNAKEFILE="./Snakefile"
RULE="all"

usage() { echo "Usage: $0 [-d (dry-run)]" 1>&2; exit 1; }

while getopts l:s:r:d flag
do
    case "${flag}" in
        l) CLUSTER_CONFIG=${OPTARG};;
        s) SNAKEFILE=${OPTARG};;
        d) DRY=" --dry-run ";;
        r) RULE=${OPTARG};;
        *) usage;;
    esac
done

shift $((OPTIND-1))

NAME="gvcf"

snakemake -j $MAXJOBS --rerun-incomplete --use-conda \
  --snakefile $SNAKEFILE \
  --cluster-config $CLUSTER_CONFIG \
  --jobname=$NAME'.{rulename}.{jobid}.sh' --scheduler greedy \
  --cluster 'sbatch --error {cluster.error} --output {cluster.output} -A {cluster.account} -p {cluster.partition} -n {cluster.n} -t {cluster.time} --mem-per-cpu {cluster.mem} --cpus-per-task {cluster.cores}' \
  $DRY $RULE