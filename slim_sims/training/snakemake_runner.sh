#!/bin/bash
DRY=""
MAXJOBS=999 # submit a ton of jobs, let the job scheduler handler it
CLUSTER_CONFIG="./cluster_talapas.json"
NBATCHES=1

usage() { echo "Usage: $0 -c <config.json> [-n <nbatches>] [-d (dry-run)] [-g (group)]" 1>&2; exit 1; }

while getopts c:n:dg flag
do
    case "${flag}" in
        c) CONFIG=${OPTARG};;
        n) NBATCHES=${OPTARG};;
        g) GROUPS=1;;
        d) DRY=" --dry-run ";;
        *) usage;;
    esac
done

GROUPLINE=""

if [ $GROUPS ]; then
    CLUSTER_CONFIG="./cluster_talapas_group.json"
    GROUPLINE="--groups bgs=group0 recap=group0 --group-components group0=50"
fi

shift $((OPTIND-1))

if [ -z "${CONFIG}" ]; then
    usage
fi

NAME=$(basename $CONFIG .json)

if (( $NBATCHES > 1 )); then
  echo "splitting into $NBATCHES batches"
  for i in $( seq 1 $NBATCHES )
  do
  echo "----- running batch $i/$NBATCHES"
  snakemake -j $MAXJOBS --rerun-incomplete --use-conda \
     --configfile $CONFIG \
     --cluster-config $CLUSTER_CONFIG \
     --jobname=$NAME'.{rulename}.{jobid}.sh' --scheduler greedy \
     $GROUPLINE \
     --cluster 'sbatch --error {cluster.error} --output {cluster.output} -A {cluster.account} -p {cluster.partition} -n {cluster.n} -t {cluster.time} --mem-per-cpu {cluster.mem} -c {cluster.cores}' \
     $DRY \
     --batch all=$i/$NBATCHES all
  done
else
  snakemake -j $MAXJOBS --rerun-incomplete --use-conda \
     --configfile $CONFIG \
     --cluster-config $CLUSTER_CONFIG \
     $GROUPLINE \
     --jobname=$NAME'.{rulename}.{jobid}.sh' --scheduler greedy \
     --cluster 'sbatch --error {cluster.error} --output {cluster.output} -A {cluster.account} -p {cluster.partition} -n {cluster.n} -t {cluster.time} --mem-per-cpu {cluster.mem} -c {cluster.cores}' \
     $DRY all
fi
