"""
"""
import sys
# todo: maybe remove this middle one -- used for testing in the slim/training
# dir
sys.path.extend(['..', '../../', '../bprime'])

import json
import __main__
import pickle
import os
from math import ceil

TEMPLATE = """
#!/bin/bash
#SBATCH --partition=kern,kerngpu,preempt
#SBATCH --job-name=slurm_slim_array
#SBATCH --output=logs/out/
#SBATCH --error=logs/error/
#SBATCH --time={job_time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=500M
#SBATCH --account=kernlab
#SBATCH --array=0-{nbatches}

python {this_script} getjob {batches} --index $SLURM_ARRAY_TASK_ID | bash

"""

import sys
import click
from bprime.slim import SlimRuns, read_params, time_grower
from bprime.samplers import Sampler

def est_time(secs_per_job, batch_size):
    tot_secs = secs_per_job * batch_size
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
@click.argument('batches', type=click.File('rb'), required=True)
@click.option('--index', required=True, type=int, help="batch index number")
def getjob(batches, index):
    job_batches = pickle.load(batches)
    sys.stdout.write(make_job_script_lines(job_batches[index]))

@cli.command()
@click.argument('config', type=click.File('r'), required=True)
@click.option('--batch-file', default=None, help="pickle file for serialized SlimRuns, default: <config>.pkl")
@click.option('--secs-per-job', required=True, type=int, help="number of seconds per simulation")
@click.option('--dir', required=True, help="output directory")
@click.option('--seed', required=True, type=int, help='seed to use')
@click.option('--script', type=click.File('w'), default="slurm_array.sh", help='script file')
@click.option('--split-dirs', default=3, type=int, help="number of seed digits to use as subdirectory")
@click.option('--nreps', default=None, help='number of simulations to average over per parameter set')
@click.option('--slim', default='slim', help='path to SLiM executable')
@click.option('--batch-size', default=None, type=int, help='seed to use')
def generate(config, batch_file, secs_per_job, dir, seed, script, split_dirs=3,
             nreps=None, slim='slim', batch_size=None):
    suffix = 'treeseq.tree'

    batch_file = os.path.basename(config.name).replace(".json", "_batches.pkl") if batch_file is None else batch_file
    config = json.load(config)
    run = SlimRuns(config, dir=dir, sampler=Sampler, split_dirs=split_dirs, seed=seed)

    # get the existing files
    existing = get_files(run.dir, suffix)  # run.dir has the name included
    print(f"{len(existing)} result files have been found -- these are ignored.")

    # generate and batch all the sims
    run.generate(suffix=suffix, ignore_files=existing)
    job_batches = run.batch_runs(batch_size=batch_size)

    with open(batch_file, 'wb') as f:
        pickle.dump(job_batches, f)

    # now write the script
    job_time = est_time(secs_per_job, batch_size)
    script.write(TEMPLATE.format(this_script=__main__.__file__, job_time=job_time,
                                 batches=batch_file,
                                 nbatches=len(job_batches)-1))
    n = len(run.runs)
    print(f"Script '{script.name}' written, {n:,} simulation commands "
          f"generated and written to '{batch_file}'.\nBatched into {len(job_batches):,} {batch_size}-size groups\n"
          f"assuming {secs_per_job} seconds per job, each batch should run in "
          f"~{round(secs_per_job * batch_size / 60, 2)} minutes\n")
    ncores = [1, 50, 100, 500, 1000]
    print("est. total time with n cores:")
    for ncore in ncores:
        print(f"  n={ncore} ~{round(n*secs_per_job/ 60 / 60 / 24 / ncore, 2)} days")


if __name__ == "__main__":
    cli()
