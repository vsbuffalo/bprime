import sys
sys.path.extend(['..', '../bprime'])

import pickle
import json
import numpy as np
from collections import defaultdict
import click

import tensorflow as tf
from tensorflow import keras
try:
    import tensorflow_addons as tfa
    PROGRESS_BAR_ENABLED = True
except ImportError:
    PROGRESS_BAR_ENABLED = False


from bprime.utils import index_cols
from bprime.learn import load_data, data_to_learnedfunc, fit_dnn
from bprime.learn import LearnedFunction


@click.group()
def cli():
    pass

@cli.command()
@click.argument('jsonfile', required=True)
@click.argument('npzfile', required=True)
@click.option('--outfile', default=None, help="output file (default <jsonfile>_data.pkl")
@click.option('--seed', default=None, help='random seed for test/train split')
def data(jsonfile, npzfile, outfile=None, test_size=0.3, seed=None, match=True):
    func = data_to_learnedfunc(*load_data(jsonfile, npzfile), seed=seed)
    if outfile is None:
        # suffix is handled by LearnedFunction.save
        outfile = jsonfile.replace('.json', '_data')
    func.save(outfile)

@cli.command()
@click.argument('funcfile', required=True)
@click.option('--outfile', default=None,
              help="output filepath (default <funcfile>, "
                   "creating <_dnn.pkl> and <funcfile>_dnn.h5")
@click.option('--n64', default=4, help="number of 64 dense layers")
@click.option('--n32', default=2, help="number of 32 dense layers")
@click.option('--batch-size', default=64, help="batch size")
@click.option('--epochs', default=400, help="number of epochs to run")
@click.option('--test-size', default=0.3, help="proportion to use as test data set")
@click.option('--reshuffle/--no-reshuffle', default=True, help="reshuffle with new seed")
@click.option('--match/--no-match', default=True, help="transform X to match if log10 scale")
@click.option('--progress/--no-progress', default=True, help="show progress")
def fit(funcfile, outfile=None, n64=4, n32=2, batch_size=64,
        epochs=400, test_size=0.3, reshuffle=True, match=True, progress=True):
    if outfile is None:
        outfile = funcfile.replace('_data.pkl', '_dnn')

    func = LearnedFunction.load(funcfile)

    if reshuffle:
        func.reshuffle()

    # split the data into test/train
    func.split(test_size=test_size)

    # transform the features using log10 if the simulation scale is log10
    if match:
        func.scale_features(transforms='match')
    else:
        # just normalize
        func.scale_features(transforms=None)

    model, history = fit_dnn(func, n64, n32, batch_size=batch_size,
                             epochs=epochs,
                             progress=(progress and PROGRESS_BAR_ENABLED))
    func.model = model
    func.history = history.history
    func.save(outfile)

if __name__ == "__main__":
    cli()


