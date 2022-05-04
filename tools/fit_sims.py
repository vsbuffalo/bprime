import sys
sys.path.extend(['..', '../..', '../bprime'])

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
from bprime.learn_utils import load_data, data_to_learnedfunc, fit_dnn
from bprime.learn import LearnedFunction


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

    if match:
        # transform the features using log10 if the simulation scale is log10
        func.scale_features(transforms='match')
    else:
        # just normalize
        func.scale_features(transforms=None)

    model, history = fit_dnn(func, n128=n128, n64=n64, n32=n32, n8=n8, nx=nx,
                             activation=activation,
                             batch_size=batch_size,
                             epochs=epochs, early_stopping=early,
                             progress=(progress and PROGRESS_BAR_ENABLED))
    func.model = model
    func.history = history.history
    func.save(outfile)


def make_bgs_model(seqlens, annot, recmap, conv_factor, w, t, chroms=None, name=None):
    """
    Build the BGSModel and the Genome object it uses.
    """
    if name is None:
        # infer the name
        bn = os.path.splitext(os.path.basename(seqlens))[0]
        name = bn.replace('_seqlens', '')
    g = Genome(name, seqlens_file=seqlens, chroms=chroms, conversion_factor=conv_factor)
    g.load_annot(annot)
    g.load_recmap(recmap)
    g.create_segments()
    m = BGSModel(g, w_grid=w, t_grid=t)
    return m

def parse_grid(x):
    try:
        return np.array(list(map(float, x.split(','))))
    except:
        raise ValueError("misformated grid string, needs to be 'x,y'")


@cli.command()
@click.argument('--seqlens', type=str, required=True, help="TSV of sequence lengthss")
@click.argument('--annot', type=str, required=True, help="BED/TSV of conserved sequences")
@click.argument('--recmap', type=str, required=True, help="HapMap or BED-like TSV of recombination rates")
@click.argument('--conv-factor', default=1e-8,
                help="Conversation factor of recmap factors to natural recombination units, M/bp (for "
                     "cM/Mb rates, use 1e-8)")
@click.option('--w', default="1e-7,5e-7,1e-8,1e-9", help="mu weight grid")
@click.option('--t', default="0.1,0.01,0.001,0.0001", help="sel coef grid")
@click.option('--outfile', default=None, help="")
@click.option('--progress/--no-progress', default=True, help="show progress")
def calcb(seqlens, annot, recmap, conv_factor=1e-8, w="1e-7,5e-7,1e-8,1e-9",
          t="0.1,0.01,0.001,0.0001", outfile=None, progress=True):
    """
    Classic B Map calculations.
    """
    pass


@cli.command()
@click.argument('learnedb', type=str, required=True, help="file path name (sans extensions) to the .pkl/h5 model")
@click.argument('--seqlens', type=str, required=True, help="TSV of sequence lengthss")
@click.argument('--name', type=str, help="genome name (otherwise inferred from seqlens file)")
@click.argument('--annot', type=str, required=True, help="BED/TSV of conserved sequences")
@click.argument('--recmap', type=str, required=True, help="HapMap or BED-like TSV of recombination rates")
@click.argument('--conv-factor', default=1e-8,
                help="Conversation factor of recmap rates to M (for "
                     "cM/Mb rates, use 1e-8)")
@click.option('--w', default="1e-7,5e-7,1e-8,1e-9", help="mu weight grid")
@click.option('--t', default="0.1,0.01,0.001,0.0001", help="sel coef grid")
@click.option('--out-dir', default=None, help="output directory (default: cwd)")
@click.option('--progress/--no-progress', default=True, help="show progress")
def predict_files(learnedb, seqlens, annot, recmap, conv_factor=1e-8, w="1e-7,5e-7,1e-8,1e-9",
                  t="0.1,0.01,0.001,0.0001", out_dir=None, progress=True):
    """
    DNN B Map calculations (prediction, step 1)
    Output files necessary to run the DNN prediction across a cluster.
    Will output scripts to run on a SLURM cluster using job arrays.
    """
    m = make_bgs_model(seqlens, annot, recmap, conv_factor,
                       parse_grid(w), parse_grid(t),
                       chroms=None, name=None)

    m.load_learnedb(learnedb)
    m.bfunc.write_BpX_chunks(out_dir)



if __name__ == "__main__":
    cli()


