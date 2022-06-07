"""
"""
import sys
# todo: maybe remove this middle one -- used for testing in the slim/training
# dir
sys.path.extend(['..', '../../', '../bgspy'])

import json
import __main__
import subprocess
import pickle
import time
import os
from math import ceil
import numpy as np
import sys
import click
from bgspy.slim import SlimRuns, read_params, time_grower
from bgspy.samplers import Sampler, ParamGrid
from bgspy.utils import make_dirs

CMD = "squeue --user vsb -r --array-unique -h -t  pending,running --format='%.18i %.30j'"
JOBNAME = "slurm_slim"
DATADIR = '../../data/slim_sims/'

TEMPLATE = f"""\
#!/bin/bash
#SBATCH --chdir={{cwd}}
#SBATCH --error=logs/error/{JOBNAME}_%j.err
#SBATCH --output=logs/out/{JOBNAME}_%j.out
#SBATCH --partition=kern,kerngpu,preempt
#SBATCH --job-name={JOBNAME}_%j
#SBATCH --time={{job_time}}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --account=kernlab

{{cmd}}

"""

def est_time(secs_per_job, batch_size, factor=5):
    tot_secs = secs_per_job * batch_size * factor
    tot_hours = tot_secs / 60 / 60
    days = int(tot_hours // 24)
    time_left = tot_hours % 24
    hours = int(time_left // 1)
    minutes = ceil(60*time_left)
    return f"{days:02d}-{hours}:{minutes}:00"

def get_files(dir, suffix):
    results = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(suffix):
                results.append(os.path.join(root, file))
    return results


def query_jobs():
    "Return lines from squeue containing this jobname."
    run = subprocess.run(CMD, shell=True, capture_output=True)
    stdout = [x.strip() for x in run.stdout.decode().split('\n') if len(x)]
    # note: only look for part of jobname, squeue truncates
    jobs = [x.split()[0] for x in stdout if JOBNAME[:30] in x]
    return jobs

def make_job_script_lines(batch):
    """
    Write a command strig that concatenates all the lines for a job, 
    and make the output directories if they don't exist.
    """
    rows = []
    dirs = []
    for job in batch:
        outfile, cmd = job
        assert not os.path.exists(outfile), f"file {outfile} exists!"
        dirs.append(os.path.split(outfile)[0])
        rows.append(cmd)
    if not len(rows):
        return ""
    mkdirs = "mkdir -p " + " ".join(dirs) + "\n"
    return mkdirs + "\n".join(rows) + "\n"

def make_job(batch, job_time):
    "Take a batch of (outfile, cmd) tuples and make a sbatch script to run the commands"
    cmd = make_job_script_lines(batch)
    sbatch = TEMPLATE.format(job_time=job_time, 
                             cwd=os.getcwd(),
                             cmd=cmd)
    return sbatch
    
    

def job_dispatcher(jobs, max_jobs, batch_size, secs_per_job, sleep=30):
    """
    Submit multiple sbatch scripts through standard in.
    """
    t0 = time.time()
    nbatches = len(jobs)

    live_jobs = dict()
    done_jobs = dict()
    total_time = 0
    total_done = 0

    def update_jobs():
        running_jobs = query_jobs()
        nonlocal total_time
        nonlocal total_done
        for job in list(live_jobs):
            if job not in running_jobs:
                t1 = time.time()
                tdelta = t1 - live_jobs.pop(job)
                done_jobs[jobid] = (t1, tdelta)
                total_time += tdelta 
                total_done += 1
        return running_jobs
 
    while True:
        running_jobs = update_jobs()
        while len(running_jobs) < max_jobs:
            # submit batches until the queue is full
            try:
                this_batch = jobs.pop()
            except IndexError:
                break
            sbatch_cmd = make_job(this_batch, job_time=est_time(secs_per_job, batch_size))
            res = subprocess.run(["sbatch"], input=sbatch_cmd, text=True, capture_output=True)
            assert res.returncode == 0, "sbatch had a non-zero exit code!"
            jobid = res.stdout.strip().replace('Submitted batch job ', '')
            # put this new job in the tracker
            live_jobs[jobid] = time.time()

            running_jobs = update_jobs()
            # clean out the live jobs
            njobs = len(running_jobs)

            ave = total_time/total_done if total_done > 0 else 0
            line = f"{nbatches - len(jobs)}/{nbatches} ({100*np.round((nbatches - len(jobs))/nbatches, 3)}%) batches submitted, {njobs} jobs currently running, {len(done_jobs)} batches done, ~{np.round(ave/60, 2)} mins per job...\r"
            sys.stderr.write(line)
            sys.stderr.flush()
        
        if not len(jobs) and not len(live_jobs):
            break
        time.sleep(sleep)
    return time.time() - t0, done_jobs



@click.command()
@click.argument('config', type=click.File('r'), required=True)
@click.option('--secs-per-job', required=True, type=int, help="number of seconds per simulation")
@click.option('--max-jobs', default=5000, help="max number of jobs before launching more")
@click.option('--seed', required=True, type=int, help='seed to use')
@click.option('--split-dirs', default=3, type=int, help="number of seed digits to use as subdirectory")
@click.option('--slim', default='slim', help='path to SLiM executable')
@click.option('--max-array', default=None, type=int, help='max number of array jobs')
@click.option('--batch-size', default=None, type=int, help='size of number of sims to run in one job')
def generate(config, secs_per_job, max_jobs,  seed, split_dirs=3,
             slim='slim', max_array=None, batch_size=None):
    suffix = 'treeseq.tree'
    config = json.load(config)

    # note: we package all the sim seed-based subdirs into a sims/ directory
    if config['runtype'] == 'grid':
        sampler = ParamGrid
    else:
        assert config['runtype'] == 'samples', "config file must have 'grid' or 'samples' runtype."
        sampler = Sampler

    run = SlimRuns(config, dir=DATADIR, sims_subdir=True, sampler=sampler, 
                   split_dirs=split_dirs, seed=seed)

    # get the existing files
    print("searching for existing simulation results...   ", end='')
    existing = get_files(run.dir, suffix)  # run.dir has the name included
    print("done.")
    print(f"{len(existing):,} result files have been found -- these are ignored.")

    # generate and batch all the sims
    run.generate(suffix=suffix, ignore_files=existing, package_basename=True, package_rep=True)
    total_size = len(run.runs)
    #import pdb;pdb.set_trace()
    if not total_size:
        print("no files need to be generated, exiting successfully")
        sys.exit(0)
    print(f"beginning dispatching of {total_size:,} simulations...")
    job_batches = run.batch_runs(batch_size=batch_size, slim_cmd=slim)

    # turn these into a list
    job_batches = list(job_batches.values())

    # set out the output directory
    sim_dir = make_dirs(DATADIR, config['name'])
    total_time, done_jobs = job_dispatcher(job_batches, max_jobs, batch_size, secs_per_job)
    print(f"\n\ntotal run time: {str(total_time)}")
    with open(f"{config['name']}_stats.pkl", 'wb') as f:
        pickle.dump(done_jobs, f)

if __name__ == "__main__":
    generate()
