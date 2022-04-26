"""
"""
import sys
# todo: maybe remove this middle one -- used for testing in the slim/training
# dir
sys.path.extend(['..', '../../', '../bprime'])

import json
import pickle

TEMPLATE = """



python slurm_job_runner.py getjob {batches} --index $SLURM_ARRAY_TASK_ID

"""

import sys
import click
from bprime.slim import SlimRuns, read_params, time_grower
from bprime.samplers import Sampler


def make_job_script_lines(jobs):
    rows = []
    for job in jobs:
        outfile, cmd = job
        rows.append(cmd)
    return "\n".join(rows)

@click.group()
def cli():
    pass

@cli.command()
@click.argument('batches', type=click.File('rb'), required=True)
@click.option('--index', required=True, type=int, help="batch index number")
def getjob(batches, index):
    job_batches = pickle.load(batches)
    print(make_job_script_lines(job_batches[index]))

@cli.command()
@click.argument('config', type=click.File('r'), required=True)
@click.option('--batches', required=True, type=click.File('wb'), help="pickle file for serialized SlimRuns")
@click.option('--dir', required=True, help="output directory")
@click.option('--seed', required=True, type=int, help='seed to use')
@click.option('--script', type=click.File('w'), default="slurm_array.sh", help='script file')
@click.option('--split-dirs', default=3, type=int, help="number of seed digits to use as subdirectory")
@click.option('--nreps', default=None, help='number of simulations to average over per parameter set')
@click.option('--slim', default='slim', help='path to SLiM executable')
@click.option('--batch-size', default=None, type=int, help='seed to use')
def generate(config, batches, dir, seed, script, split_dirs=3,
             nreps=None, slim='slim', batch_size=None):
    config = json.load(config)
    run = SlimRuns(config, dir=dir, sampler=Sampler, split_dirs=split_dirs, seed=seed)
    run.generate()
    job_batches = run.batch_runs(suffix='treeseq.tree', batch_size=batch_size)
    pickle.dump(job_batches, batches)




def array():
    pass

if __name__ == "__main__":
    cli()
