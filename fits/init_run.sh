#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "error: specify YAML file"
    exit 1
fi

dir=$(readlink -f $(basename $1 .yml))
curr_dir=$(readlink -f .)

mkdir $dir

ln -s $curr_dir/Snakefile $dir/Snakefile
ln -s $curr_dir/$1 $dir/$1

mkdir -p $dir/logs/out
mkdir -p $dir/logs/error



