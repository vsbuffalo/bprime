import sys
sys.path.extend(['..', '../..', '../bgspy'])

import pickle
import json
import numpy as np
from collections import defaultdict
import click
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import tensorflow as tf
from tensorflow import keras
try:
    import tensorflow_addons as tfa
    PROGRESS_BAR_ENABLED = True
except ImportError:
    PROGRESS_BAR_ENABLED = False


from bgspy.utils import index_cols
from bgspy.learn_utils import load_data, data_to_learnedfunc, fit_dnn
from bgspy.learn_utils import TargetReweighter
from bgspy.learn import LearnedFunction
from bgspy.utils import random_seed, read_params
from bgspy.msprime import msprime_spike


@click.group()
def cli():
    pass


@cli.command()
@click.argument('jsonfile', required=True)
@click.option('--outfile', default=None, help="output file")
@click.option('--N', required=True, type=int, help='population size')
@click.option('--total', required=True, type=int, help='how many sims')
@click.option('--reps', required=True, type=int, help='how many replicates per parameter set')
@click.option('--seed', default=None, help='random seed for test/train split')
def spike(jsonfile, outfile, n, total, reps, seed):
    """
    Experimental: spike in L=0 neutral sims
    """
    N = n # click lowercases stuff
    if seed is None:
        seed = random_seed()
    else:
        seed = int(seed)
    if outfile is None:
        outfile = f"msprime_spike_{total}total_{reps}reps_{N}N_{seed}seed.npz"
    config = json.load(open(jsonfile))
    params = read_params(config)[0]
    X, y, features, targets, keys = msprime_spike(params, N, total, reps, seed=seed)
    np.savez(outfile, X=X, y=y, features=features, targets=targets, keys=keys)


@cli.command()
@click.argument('jsonfile', required=True)
@click.argument('npzfile', required=True, nargs=-1)
@click.option('--average/--no-average', default=True, help="whether to average over replicates")
@click.option('--outfile', default=None, help="output file (default <jsonfile>_data.pkl")
@click.option('--seed', default=None, help='random seed for test/train split')
def data(jsonfile, npzfile, average=True, outfile=None, test_size=0.3,
         seed=None, match=True):
    func = data_to_learnedfunc(*load_data(jsonfile, *npzfile),
                               average_reps=average, seed=seed)
    if outfile is None:
        # suffix is handled by LearnedFunction.save
        outfile = jsonfile.replace('.json', '_data')
    func.save(outfile)

@cli.command()
@click.argument('funcfile', required=True)
@click.option('--outfile', default=None,
              help="output filepath (default <funcfile>, "
                   "creating <_dnn.pkl> and <funcfile>_dnn.h5")
@click.option('--n16', default=0, help="number of 16 neuron dense layers")
@click.option('--n8', default=0, help="number of 8 neuron dense layers")
@click.option('--n4', default=0, help="number of 4 neuron dense layers")
@click.option('--n2', default=0, help="number of 2 neuron dense layers")
@click.option('--nx', default=2, help="number of x neuron dense layers where x is input size")
@click.option('--var-sample-weights', default=False, 
              help="weight the samples by empirical variances")
@click.option('--l2-penalty', default=None, help="L2 regularizer for weights and biases")
@click.option('--activation', default='elu', help="layer activation")
@click.option('--output-activation', default='sigmoid', help="output activation")
@click.option('--batch-size', default=64, help="batch size")
@click.option('--epochs', default=500, help="number of epochs to run")
@click.option('--early/--no-early', default=True, help="use early stopping")
@click.option('--test-split', default=0.2, help="proportion to use as test data set")
@click.option('--valid-split', default=0.2,
              help="proportion of training data set to use as validation")
@click.option('--reseed', is_flag=True, default=True, help="reseed with new seed")
@click.option('--match', is_flag=True, default=True,
              help="transform X to match if log10 scale")
@click.option('--normalize-target', is_flag=True, default=False,
              help="transform X to match if log10 scale")
@click.option('--progress', is_flag=True, default=True, help="show progress")
def fit(funcfile, outfile=None, n16=0, n8=0, n4=0, n2=0, nx=2,
        var_sample_weights=False,
        l2_penalty=None, activation='elu', 
        output_activation='sigmoid', batch_size=64,
        epochs=500, early=True, test_split=0.2,
        valid_split=0.1, reseed=True, match=True, normalize_target=False,
        progress=True):
    if outfile is None:
        outfile = funcfile.replace('_data.pkl', '_dnn')

    # turning this off, as it seems to do unniversally poorly
    balance_target = False
    func = LearnedFunction.load(funcfile)

    # if we want to fit the model on a new, fresh split of test/train
    # compared to the data that's loaded in, we reseed.
    if reseed:
        func.reseed()

    # split the data into test/train
    func.split(test_split=test_split)

    sample_weight = None
    if var_sample_weights:
        assert func.bhat_var is not None, "LearnedFunction.bhat_var not set!"
        train_idx = func._train_idx
        emp_vars = func.bhat_var[train_idx]
        # calculate the precision
        pipe = Pipeline(steps=[('poly', PolynomialFeatures(degree=3)),
                               ('regr', LinearRegression())])
        pipe.fit(func.y_train[:, None], emp_vars[:, None])
        # weight by the conditional mean
        pred_vars = pipe.predict(func.y_train[:, None])
        sample_weight = 1/pred_vars # precision
        sample_weight = sample_weight / sample_weight.sum()


    if balance_target:
        # the copy is because sklearn annoyingly doesn't like read only data (why?!)
        y = np.copy(func.y_train)
        trw = TargetReweighter(y)
        trw.set_bandwidth(bandwidth)
        sample_weight = trw.weights(y)
        func.trw = trw

    if match:
        # transform the features using log10 if the simulation scale is log10
        func.scale_features(transforms='match',
                            normalize_target=normalize_target)
    else:
        # just normalize
        func.scale_features(transforms=None, normalize_target=normalize_target)

    # currently we use same penalty for weights and biaes
    weight_l2 = None if l2_penalty == "None" else float(l2_penalty)
    bias_l2 = None if l2_penalty == "None" else float(l2_penalty)

    # TODO -- CLI
    model, history = fit_dnn(func, n16=n16, n8=n8, n4=n4, n2=n2, nx=nx, 
                             weight_l2=weight_l2,
                             bias_l2=bias_l2,
                             activation=activation,
                             output_activation=output_activation,
                             valid_split=valid_split,
                             batch_size=batch_size,
                             epochs=epochs, early_stopping=early,
                             sample_weight=sample_weight,
                             progress=(progress and PROGRESS_BAR_ENABLED))
    func.model = model
    func.history = history.history
    # save some time later on
    mse = np.mean((func.predict_test() - func.y_test)**2)
    if func.metadata is not None:
        func.metadata['test_mse'] = mse
    else:
        func.metadata = dict(test_mse=mse)
    func.save(outfile)





if __name__ == "__main__":
    cli()


