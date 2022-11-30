#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "error: specify JSON file"
    exit 1
fi

dir=$(readlink -f $(basename $1 .json))
curr_dir=$(readlink -f .)

mkdir $dir

ln -s $curr_dir/Snakefile $dir/Snakefile
ln -s $curr_dir/cluster_talapas.json $dir/cluster_talapas.json
ln -s $curr_dir/snakemake_runner.sh $dir/snakemake_runner.sh

ln -s $curr_dir/$1 $dir/$1

mkdir -p $dir/logs/out
mkdir -p $dir/logs/error



