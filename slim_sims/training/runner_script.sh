#!/bin/bash

NSCRIPTS=22

i=0
while [ $i -lt $NSCRIPTS ]
do
  sbatch "slurm_$i.sh"
  echo "submitting job $i!"
  i=$((i + 1))
  nrunning=$(squeue -u vsb  --long | grep slurm_sl | wc -l) 
  while [ $nrunning -gt 0 ]
  do
    sleep 5m
    nrunning=$(squeue -u vsb  --long | grep slurm_sl | wc -l) 
    echo "$nrunning jobs still running..."
  done
done

