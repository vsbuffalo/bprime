import sys
sys.path.extend(['..', '../bprime'])

import pickle
import json
import numpy as np
from collections import defaultdict

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from bprime.utils import index_cols
from bprime.sim_utils import fixed_params, get_bounds
from bprime.learn import LearnedFunction, network
from bprime.theory import bgs_segment, bgs_rec


json_file = sys.argv[1]
npz_file = sys.argv[2]
out_file = sys.argv[2].replace('.npz', '_dnn.pkl')
TEST_SPLIT_PROP = 0.3
MATCH = True
VALIDATION_SPLIT = 0.3
NFITS = 3
ARCHS = [{'n64': 4, 'n32': 0}]
# ARCHS = [{'n64': 4, 'n32': 0},
#          {'n64': 4, 'n32': 4},
#          {'n64': 8, 'n32': 4}]

## load data

with open(json_file) as f:
    sim_params = json.load(f)['params']
sim_bounds = get_bounds(sim_params)

sim_data = np.load(npz_file, allow_pickle=True)
assert(len(sim_data['features']) == sim_data['X'].shape[1])

## process data

Xo, y = np.array(sim_data['X']), sim_data['y']

# remove some fixed columns
Xcols = index_cols(sim_data['features'])
keep_cols = ('mu', 's', 'rbp', 'rf', 'L')
X = Xo[:, Xcols(*keep_cols)]
print(f"total samples: {X.shape[0]:,}")

## build the learn func object
domain = {p: sim_bounds[p] for p in keep_cols}
func = LearnedFunction(X, y, domain=domain)

# build a column indexer -- maps feature names to column indices
Xcols = func.col_indexer()

func.split(test_size=TEST_SPLIT_PROP)

# transform the features using log10 if the simulation scale is log10
if MATCH:
    func.scale_features(transforms = 'match')
else:
    # just normalize
    func.scale_features(transforms = 'match')

## DNN
models = defaultdict()
histories = defaultdict()
for arch in ARCHS:
    key = tuple(arch.keys())
    for f in range(NFITS):
        model = network(input_size=5, output_activation='sigmoid', **arch)
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                        patience=50, restore_best_weights=True)
        tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

        history = model.fit(func.X_train, func.y_train,
                            validation_split=VALIDATION_SPLIT,
                            batch_size=64, epochs=400, verbose=0,
                            callbacks=[es, tqdm_callback])
        models[key].append(model)
        histories[key].append(history)

with open(out_file, 'wb') as f:
    res = {'models': models, 'histories': histories, 'func': func}
    pickle.dump(res, f)


