import sys
sys.path.extend(['..', '../..', '../bgspy'])

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


from bgspy.utils import index_cols
from bgspy.learn_utils import load_data, data_to_learnedfunc, fit_dnn
from bgspy.learn import LearnedFunction


@click.group()
def cli():
    pass

@cli.command()
@click.argument('jsonfile', required=True)
@click.argument('npzfile', required=True)
@click.option('--average-reps', default=True, help="whether to average over replicates")
@click.option('--outfile', default=None, help="output file (default <jsonfile>_data.pkl")
@click.option('--seed', default=None, help='random seed for test/train split')
def data(jsonfile, npzfile, average_reps=True, outfile=None, test_size=0.3,
         seed=None, match=True):
    func = data_to_learnedfunc(*load_data(jsonfile, npzfile),
                               average_reps=average_reps, seed=seed)
    if outfile is None:
        # suffix is handled by LearnedFunction.save
        outfile = jsonfile.replace('.json', '_data')
    func.save(outfile)

@cli.command()
@click.argument('funcfile', required=True)
@click.option('--outfile', default=None,
              help="output filepath (default <funcfile>, "
                   "creating <_dnn.pkl> and <funcfile>_dnn.h5")
@click.option('--n128', default=0, help="number of 128 dense layers")
@click.option('--n64', default=0, help="number of 64 neuron dense layers")
@click.option('--n32', default=0, help="number of 32 neuron dense layers")
@click.option('--n8', default=0, help="number of 8 neuron dense layers")
@click.option('--nx', default=2, help="number of x neuron dense layers where x is input size")
@click.option('--activation', default='elu', help="layer activation")
@click.option('--batch-size', default=64, help="batch size")
@click.option('--epochs', default=1000, help="number of epochs to run")
@click.option('--early/--no-early', default=True, help="use early stopping")
@click.option('--test-size', default=0.2, help="proportion to use as test data set")
@click.option('--reseed/--no-reseed', default=True, help="reseed with new seed")
@click.option('--match/--no-match', default=True, help="transform X to match if log10 scale")
@click.option('--progress/--no-progress', default=True, help="show progress")
def fit(funcfile, outfile=None, n128=0, n64=4, n32=2, n8=0, nx=0,
        activation='elu', batch_size=64,
        epochs=1000, early=True, test_size=0.2, reseed=True,
        match=True, progress=True):
    if outfile is None:
        outfile = funcfile.replace('_data.pkl', '_dnn')

    func = LearnedFunction.load(funcfile)

    # if we want to fit the model on a new, fresh split of test/train
    # compared to the data that's loaded in, we reseed.
    if reseed:
        func.reseed()

    # split the data into test/train
    func.split(test_size=test_size)


    normalize_target = True
    if match:
        # transform the features using log10 if the simulation scale is log10
        func.scale_features(transforms='match', normalize_target=normalize_target)
    else:
        # just normalize
        func.scale_features(transforms=None, normalize_target=normalize_target)

    # TODO -- CLI 
    model, history = fit_dnn(func, n128=n128, n64=n64, n32=n32, n8=n8, nx=nx,
                             activation=activation,
                             output_activation='tanh',
                             batch_size=batch_size,
                             epochs=epochs, early_stopping=early,
                             progress=(progress and PROGRESS_BAR_ENABLED))
    func.model = model
    func.history = history.history
    func.save(outfile)





if __name__ == "__main__":
    cli()


