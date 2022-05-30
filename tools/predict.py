"""
Note -- can we speed up prediction by grid'ing
all the rfs and counting occurences, then predictin on the small
rf grid?
"""
USE_GPU = False
import os
import json
from os.path import join
import numpy as np
import itertools
import re
import glob
import pickle
import click
if USE_GPU:
    try:
        import gpustat
        GPUSTAT_AVAIL = True
    except:
        GPUSTAT_AVAIL = False
import tensorflow as tf
from tensorflow import keras
from bgspy.utils import make_dirs
from bgspy.predict import predict_chunk
from bgspy.learn import LearnedFunction

# for playing nice on GPUs
if USE_GPU and GPUSTAT_AVAIL:
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used'])/float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)

if not USE_GPU:
    # limit CPU usage
    MAX_CPU = 10
    os.environ["OMP_NUM_THREADS"] = f"{MAX_CPU}"
    os.environ["TF_NUM_INTRAOP_THREADS"] = f"{MAX_CPU}"
    os.environ["TF_NUM_INTEROP_THREADS"] = f"{MAX_CPU}"
    tf.config.threading.set_inter_op_parallelism_threads(MAX_CPU)
    tf.config.threading.set_intra_op_parallelism_threads(MAX_CPU)
    tf.config.set_soft_device_placement(True)

CHROM_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+).npy')
CHUNK_MATCHER = re.compile(r'(?P<name>\w+)_(?P<chrom>\w+)_(?P<i>\d+)_(?P<lidx>\d+)_(?P<uidx>\d+).npy')

@click.command()
@click.argument('chunkfile', required=True)
@click.option('--input-dir', required=True, help='main prediction directory')
@click.option('--constrain/--no-constrain', default=True,
              help='whether to compute B using segments along the entire '
              'chromosome, or whether to use the supplied slices in filename')
@click.option('--progress/--no-progress', default=True, help='whether to display progress bar')
@click.option('--output-xps', is_flag=True, default=False, help='output unformatted chunk matrices for debugging')
@click.option('--output-preds', is_flag=True, default=False, help='output B predictions for each segment')
@click.option('--dont-predict', is_flag=True, default=False, help='skip prediction (False if --output-preds)')
def predict(chunkfile, input_dir, constrain=True, progress=True, 
            output_xps=False, output_preds=False, dont_predict=False):
    """

    Note: bounds are determined for a group of models (e.g. all trained on the same
    simulation data, which sets the boudns). Centering and scaling parameters, and
    what features are log transfomed are model fit specific.
    """
    dont_predict = False if output_preds else dont_predict

    out_dir = make_dirs(input_dir, 'preds')
    chunk_dir = make_dirs(input_dir, 'chunks')
    seg_dir = make_dirs(input_dir, 'segments')

    infofile = join(input_dir, 'info.json')
    with open(infofile, 'r') as f:
        info = json.load(f)

    # get the model information dict from info.json
    models = [(m, LearnedFunction.load(m)) for m in info['models']]

    # get the chunk to process and get the right segments file
    chunk_parts = CHUNK_MATCHER.match(os.path.basename(chunkfile)).groupdict()
    chrom = chunk_parts['chrom']
    name = chunk_parts['name']
    seg_file = join(seg_dir, f"{name}_{chrom}.npy")
    assert os.path.exists(seg_file), "no appropriate chromosome segment file found!"

    # deal with the focal position chunk file
    chrom_parts = CHROM_MATCHER.match(os.path.basename(seg_file)).groupdict()

    # get the indices from the filename
    if constrain:
        lidx, uidx = int(chunk_parts['lidx']), int(chunk_parts['uidx'])
    else:
        # use all segments on chromosome
        lidx, uidx = None, None

    chunk_i = int(chunk_parts['i'])
    # contains two columns: the physical and map positions of focal sites
    sites_chunk = np.load(chunkfile)

    # segment info matrix
    Sm = np.load(seg_file)

    # selection and mutation grids
    w, t = np.array(info['w']), np.array(info['t'])

    # run the main prediction function
    B, Xps, Bpreds = predict_chunk(sites_chunk, models, Sm, w, t,
                                   lidx=lidx, uidx=uidx, 
                                   use_haldane=True, # TODO FIXME
                                   output_xps=output_xps, 
                                   output_preds=output_preds,
                                   # skip prediction if we just want matrices
                                   dont_predict=dont_predict,
                                   progress=progress)

    # save real output
    chrom_out_dir = make_dirs(out_dir, chrom)
    outfile = join(chrom_out_dir, os.path.basename(chunkfile))
    np.save(outfile, B.squeeze())

    if output_xps:
        xps_dir = make_dirs(input_dir, 'xps')
        for i, Xp in enumerate(Xps):
            outfile = join(xps_dir, os.path.basename(chunkfile))
            outfile = outfile.replace('.npy', f"_{i}.npy")
            np.save(outfile, Xp.squeeze())

    if output_preds:
        preds_dir = make_dirs(input_dir, 'bpreds')
        for model, preds in Bpreds.items():
            name = os.path.basename(model)
            for i, pred in enumerate(preds):
                outfile = join(preds_dir, os.path.basename(chunkfile))
                outfile = outfile.replace('.npy', f"_{name}_{i}.npy")
                np.save(outfile, pred.squeeze())
        

if __name__ == "__main__":
    predict()

