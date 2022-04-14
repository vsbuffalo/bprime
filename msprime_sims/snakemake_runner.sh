#!/bin/bash
DRY=""
MAXJOBS=999 # submit a ton of jobs, let the job scheduler handler it
CLUSTER_CONFIG="./cluster_talapas_gpu.json"

usage() { echo "Usage: $0 -c <config.json> [-d (dry-run)]" 1>&2; exit 1; }

while getopts c:d flag
do
    case "${flag}" in
        c) CONFIG=${OPTARG};;
        d) DRY=" --dry-run ";;
        *) usage;;
    esac
done

GROUPLINE=""

shift $((OPTIND-1))

if [ -z "${CONFIG}" ]; then
    usage
fi

NAME=$(basename $CONFIG .json)

snakemake -j $MAXJOBS --rerun-incomplete --use-conda \
  --configfile $CONFIG \
  --cluster-config $CLUSTER_CONFIG \
  $GROUPLINE \
  --jobname=$NAME'.{rulename}.{jobid}.sh' --scheduler greedy \
  --cluster 'sbatch --error {cluster.error} --output {cluster.output} -A {cluster.account} -p {cluster.partition} -n {cluster.n} -t {cluster.time} --mem-per-cpu {cluster.mem} -c {cluster.cores}' \
  $DRY train
