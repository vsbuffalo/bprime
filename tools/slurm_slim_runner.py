"""
"""
import sys
# todo: maybe remove this middle one -- used for testing in the slim/training
# dir
sys.path.extend(['..', '../../', '../bgspy'])

import json
import __main__
import pickle
import os
from math import ceil
import numpy as np
import sys
import click
from bgspy.slim import SlimRuns, read_params, time_grower
from bgspy.samplers import Sampler

TEMPLATE = """\
#!/bin/bash
#SBATCH --chdir={cwd}
#SBATCH --error=logs/error/slurm_slim_array_%A_%a.err
#SBATCH --output=logs/out/slurm_slim_array_%A_%a.out
#SBATCH --partition=kern,kerngpu,preempt
#SBATCH --job-name=slurm_slim_array_%A_%a
#SBATCH --time={job_time}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH --account=kernlab
#SBATCH --array=1-{max_array}

INDEX=$(({offset} + $SLURM_ARRAY_TASK_ID - 1))

python {this_script} getjob {batches} --num-files {num_files} --index $INDEX | bash

"""


RUNNER = """\
#!/bin/bash

NSCRIPTS={nscripts}

i=0
while [ $i -lt $NSCRIPTS ]
do
  sbatch "slurm_$i.sh"
  echo "submitting job $i!"
  i=$((i + 1))
  nrunning=$(squeue -u vsb  --long | grep slurm_sl | wc -l)
  while [ $nrunning -gt 0 ]
  do
    sleep 2m
    nrunning=$(squeue -u vsb  --long | grep slurm_sl | wc -l)
    echo "$nrunning jobs still running..."
  done
done

"""

def est_time(secs_per_job, batch_size, factor=5):
    tot_secs = secs_per_job * batch_size * factor
    tot_hours = tot_secs / 60 / 60
    days = int(tot_hours // 24)
    time_left = tot_hours % 24
    hours = int(time_left // 1)
    minutes = ceil(60*time_left)
    return f"{days:02d}-{hours}:{minutes}:00"

def make_job_script_lines(jobs):
    rows = []
    dirs = []
    for job in jobs:
        outfile, cmd = job
        dirs.append(os.path.split(outfile)[0])
        rows.append(cmd)
    mkdirs = "mkdir -p " + " ".join(dirs) + "\n"
    return mkdirs + "\n".join(rows) + "\n"

def get_files(dir, suffix):
    results = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(suffix):
                results.append(os.path.join(root, file))
    return results


@click.group()
def cli():
    pass

@cli.command()
@click.argument('batchfile', type=str, required=True)
@click.option('--num-files', default=1, help='how many files to break the pickled batch files into')
@click.option('--index', required=True, type=int, help="batch index number")
def getjob(batchfile, num_files, index):
    if num_files > 1:
        group = index % num_files
        root = batchfile
        batchfile = batchfile + f"_{group}.pkl"
        all_there = all(os.path.exists(f"{root}_{r}.pkl") for r in range(num_files))
        assert all_there, "some grouped batch files do not exist! is --num-files right?"
    assert batchfile.endswith('.pkl')
    with open(batchfile, 'rb') as f:
        job_batches = pickle.load(f)
    sys.stdout.write(make_job_script_lines(job_batches[index]))

@cli.command()
@click.argument('config', type=click.File('r'), required=True)
@click.option('--batch-file', default=None, help="pickle file for serialized SlimRuns, default: <config>[_group].pkl")
@click.option('--secs-per-job', required=True, type=int, help="number of seconds per simulation")
@click.option('--dir', required=True, help="output directory")
@click.option('--seed', required=True, type=int, help='seed to use')
@click.option('--script', type=str, default="slurm_array.sh", help='script file')
@click.option('--split-dirs', default=3, type=int, help="number of seed digits to use as subdirectory")
@click.option('--nreps', default=None, help='number of simulations to average over per parameter set')
@click.option('--slim', default='slim', help='path to SLiM executable')
@click.option('--num-files', default=1, help='how many files to break the pickled batch files into')
@click.option('--max-array', default=None, type=int, help='max number of array jobs')
@click.option('--batch-size', default=None, type=int, help='size of number of sims to run in one job')
def generate(config, batch_file, secs_per_job, dir, seed, script, split_dirs=3,
             nreps=None, slim='slim', max_array=None, num_files=1, batch_size=None):
    suffix = 'treeseq.tree'

    batch_file = os.path.basename(config.name).replace(".json", "_batches.pkl") if batch_file is None else batch_file
    config = json.load(config)
    # note: we package all the sim seed-based subdirs into a sims/ directory
    run = SlimRuns(config, dir=dir, sims_subdir=True, sampler=Sampler, split_dirs=split_dirs, seed=seed)

    # get the existing files
    print("searching for existing simulation results...   ", end='')
    existing = get_files(run.dir, suffix)  # run.dir has the name included
    print("done.")
    print(f"{len(existing):,} result files have been found -- these are ignored.")

    # generate and batch all the sims
    run.generate(suffix=suffix, ignore_files=existing)
    job_batches = run.batch_runs(batch_size=batch_size, slim_cmd=slim)
    if num_files > 1:
        batch_groups = [dict() for _ in range(num_files)]
        for idx, jobs in job_batches.items():
            group = idx % num_files
            batch_groups[group][idx] = jobs

        for i, group_batches in enumerate(batch_groups):
            with open(batch_file.replace('.pkl', f"_{i}.pkl"), 'wb') as f:
                pickle.dump(group_batches, f)
    else:
        with open(batch_file, 'wb') as f:
            pickle.dump(job_batches, f)

    # now write the script
    job_time = est_time(secs_per_job, batch_size)
    batch_file = batch_file if num_files == 1 else batch_file.replace('.pkl', '')

    njobs = len(job_batches) # how many batch there are (so need array indices)
    if max_array is None or njobs < max_array:
        script_handle = open(script, 'w')
        script_handle.write(TEMPLATE.format(this_script=__main__.__file__,
                                            job_time=job_time,
                                            cwd=os.getcwd(),
                                            num_files=num_files,
                                            batches=batch_file,
                                            offset=0,
                                            max_array=njobs))
    else:
        # we need to break the slurm script into smaller batches, grrr
        array_job_ids = np.arange(njobs)
        nscripts = np.split(array_job_ids, np.arange(0, njobs, max_array)[1:])
        for i, script_batch_ids in enumerate(nscripts):
            start, end = script_batch_ids[0], script_batch_ids[-1]
            scriptname = script.replace('.sh', f"_{i}.sh")
            script_handle = open(scriptname, 'w')
            script_handle.write(TEMPLATE.format(this_script=__main__.__file__, job_time=job_time,
                                                cwd=os.getcwd(),
                                                num_files=num_files,
                                                batches=batch_file,
                                                offset=start,
                                                max_array=max_array))


    n = len(run.runs)
    print(f"Script '{script}' written, {n:,} simulation commands "
          f"generated and written to '{batch_file}'.\nBatched into {len(job_batches):,} {batch_size}-size groups\n"
          f"assuming {secs_per_job} seconds per job, each batch should run in "
          f"~{round(secs_per_job * batch_size / 60, 2)} minutes\n")
    ncores = [1, 50, 100, 500, 1000]
    print("est. total time with n cores:")
    for ncore in ncores:
        print(f"  n={ncore} ~{round(n*secs_per_job/ 60 / 60 / 24 / ncore, 2)} days")

    if max_array is not None and njobs >= max_array:
        print("run each slurm script with: runner_script.sh")
        with open('runner_script.sh', 'w') as f:
            f.write(RUNNER.format(nscripts=len(nscripts)))
    else:
        print("run each slurm script with: slurm.sh")


if __name__ == "__main__":
    cli()
