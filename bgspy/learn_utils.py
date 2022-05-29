import warnings
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
try:
    import tensorflow_addons as tfa
    PROGRESS_BAR_ENABLED = True
except ImportError:
    PROGRESS_BAR_ENABLED = False
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from bgspy.utils import index_cols
from bgspy.sim_utils import fixed_params, get_bounds
from bgspy.theory import BGS_MODEL_PARAMS, BGS_MODEL_FUNCS
from bgspy.learn import LearnedFunction


def new_layer(size, activation, weight_l2=None, bias_l2=None):
    if weight_l2 is not None:
        # this is called kernel_regularizer by keras(?!)
        assert isinstance(weight_l2, (int, float))
        weight_l2 = regularizers.L2(weight_l2)
    if bias_l2 is not None:
        assert isinstance(bias_l2, (int, float))
        bias_l2 = regularizers.L2(bias_l2) 
    return layers.Dense(size, activation=activation, 
                        kernel_regularizer=weight_l2,
                        bias_regularizer=bias_l2)



def network(input_size=2, n16=0, n8=0, n4=0, n2=0, nx=2, 
            weight_l2=None, bias_l2=None,
            output_activation='sigmoid', activation='elu'):
    """
    Build a sequential network given the specified layers. nx specifies the
    number of layers with the number of neurons equal to the input size.
    """
    # build network
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_size,)))
    for i in range(nx):
        model.add(new_layer(input_size, activation=activation, 
                            weight_l2=weight_l2, bias_l2=bias_l2))
    for i in range(n2):
        model.add(new_layer(2, activation=activation, 
                            weight_l2=weight_l2, bias_l2=bias_l2))
    for i in range(n4):
        model.add(new_layer(4, activation=activation, 
                            weight_l2=weight_l2, bias_l2=bias_l2))
    for i in range(n8):
        model.add(new_layer(8, activation=activation, 
                            weight_l2=weight_l2, bias_l2=bias_l2))
    for i in range(n16):
        model.add(new_layer(16, activation=activation, 
                            weight_l2=weight_l2, bias_l2=bias_l2))
    model.add(keras.layers.Dense(1, activation=output_activation))
    model.compile(
        optimizer='Adam',
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')],
        #weighted_metrics=[keras.metrics.MeanSquaredError(name='weighted_mse'),
        #                  keras.metrics.MeanAbsoluteError(name='weighted_mae')],
        )
    return model

def load_data(jsonfile, npzfile):
    """
    Load the JSON file with the parameters (and other details) of a simulation,
    and the NPZ file containing the features and targets matrices.

    Returns a tuple of the simulation parameters dict, the simulation data from
    the .npz, and a model name. Meant to be passed as arguments to
    data_to_learnedfunc().
    """
    with open(jsonfile) as f:
        sim_config = json.load(f)
    sim_params, model = sim_config['params'], sim_config['model']
    sim_data = np.load(npzfile, allow_pickle=True)
    assert(len(sim_data['features']) == sim_data['X'].shape[1])
    return sim_params, sim_data, model

def fixed_cols(X, colnames=None):
    """
    Return which columns are fixed. If colnames are provided, return a dict
    like in fixed_params().
    """
    assert X.shape[1] == len(colnames)
    # using np.unique because np.var is less numerically stable
    n = X.shape[1]
    vals = [np.unique(X[:, i]) for i in range(n)]
    idx = [i for i in range(n) if len(vals[i]) == 1]
    if colnames is not None:
        return {colnames[i]: vals[i] for i in idx}
    return idx

def match_features(sim_params, data, features):
    """
    For a set of parameters for simulations (msprime or SLiM) from a JSON
    file, match to the data we received from the simulations (with the feature
    column names provided). They key parts are the data and the column
    labels (which should match the JSON file).

    Returns fixed values dict and variable columns
    """
    assert data.shape[1] == len(features), "number of columns in 'data' ≠ number of features"
    assert set(sim_params.keys()) == set(features), "sim parameters don't match data features"

    # get the fixed columns from sims and data, seed if they match
    param_fixed_cols = fixed_params(sim_params)
    data_fixed_cols = fixed_cols(data, colnames=features)
    x = ', '.join(param_fixed_cols)
    y = ', '.join(data_fixed_cols)
    msg = f"mismatching fixed columns in parameters ({x}) and data ({y})!"
    assert set(param_fixed_cols) == set(data_fixed_cols), msg

    # are the fixed values the same in the parameters and data?
    same_vals = all([param_fixed_cols[k] == data_fixed_cols[k] for k in param_fixed_cols])
    msg = f"fixed parameters differr between parameters ({param_fixed_cols}) and data ({data_fixed_cols})"
    assert same_vals, msg

    return param_fixed_cols, set(features).difference(set(data_fixed_cols))

def check_feature_with_models(features, model):
    """
    Two BGS models are used, 'segment' and 'rec'. We check here
    the features in the data are consistent with one of those two models.
    """
    model_features = BGS_MODEL_PARAMS[model]
    x = ', '.join(model_features)
    # drop N, since we condition on that
    features = [f for f in features if f != 'N']
    y = ', '.join(features)
    msg = (f"feature mismatch between features for model '{model}' ({x}) and "
           f"supplied data features ({y})")
    assert set(features) == set(model_features), msg


def data_to_learnedfunc(sim_params, sim_data, model, seed,
                        average_reps=False):
    """
    Get the bounds of parameters from the simulation parameters dictionary, find
    all fixed and variable parameters, take the product of the selection and
    dominance coefficient columns, subset the data to include only variable
    (non-fixed) params that go into the training as features. This also does
    lots of validation of the simulation data and simulation parameters.

    Returns a LearnedFunction, with the fixed attributes set.
    """

    # raw (original) data -- this contains extraneous columns, e.g.
    # ones that aren't fixed
    Xo, yo = np.array(sim_data['X']), sim_data['y']
    bhat_var = None
    if average_reps:
        keys = sim_data['keys']
        assert np.all(sorted(keys) == keys)
        yd = pd.DataFrame(yo, columns=sim_data['targets'])
        yd['key'] = keys
        Xd = pd.DataFrame(Xo, columns=sim_data['features'])
        Xd['key'] = keys
        Xo_ave = Xd.groupby('key').mean().reset_index().drop('key', axis=1)
        msg ="keys not mapped to unique parameters!"
        Xo_var = Xd.groupby('key').var().reset_index().drop(['key', 'rep'], axis=1).values
        # if there is a 5th column, it's the replicate number -- it does vary
        assert np.allclose(Xo_var, 0), msg
        yo_ave = yd.groupby('key').mean().reset_index().drop('key', axis=1)
        yo_var = yd.groupby('key').var().reset_index()
        bhat_var = yo_var['Bhat'].values
        Xo, yo = Xo_ave.values, yo_ave.values

    # first deal with y -- here we only care about Bhat, so we get that
    if yo.shape[1] > 1:
        ycols = index_cols(sim_data['targets'])
        y = yo[:, ycols('Bhat')]
        yextra = yo[:, ycols(*[f for f in sim_data['targets'] if f != 'Bhat'])]
    else:
        yextra = None
        y = yo

    # we exclude rep number from now on -- not needed
    idx, all_features = zip(*[(i, f) for i, f in enumerate(sim_data['features']) if f != 'rep'])
    Xo = Xo[:, idx]
    Xo_cols = index_cols(all_features)

    # check the features we get are one of the BGS models
    check_feature_with_models(all_features, model)

    # Get the fixed columns/features in the original data. First drop N, which is
    # not in the features matrix since we condition on it.
    sim_params = {k: v for k, v in sim_params.items() if k not in ('N', )}
    fixed_vals, var_cols = match_features(sim_params, data=Xo, features=all_features)

    # build up a matrix of non-fixed features, combining sh
    expected_features = BGS_MODEL_PARAMS[model]
    nfeatures = len(expected_features)
    Xsh = np.empty((Xo.shape[0], nfeatures))
    for i, feature in enumerate(expected_features):
        col = Xo[:, Xo_cols(feature)]
        Xsh[:, i] = col.squeeze()

    # now let's get the fixed columns again since stuff has changed
    new_fixed_vals = fixed_cols(Xsh, expected_features)
    new_var_cols = list(set(expected_features) - set(new_fixed_vals.keys()))

    # the real feature set is fixed columns, sorted by their order in
    # BGS_MODEL_PARAMS
    features = [v for v in expected_features if v in new_var_cols]

    # get the parameter boundaries from params
    sim_bounds = get_bounds(sim_params)
    assert len(sim_bounds) == len(all_features)

    # Now, subset the features matrix with sh merged to include only variable
    # columns
    Xsh_cols = index_cols(expected_features)
    X = Xsh[:, Xsh_cols(*features)]

    ## build the learn func object
    # get the domain of non-fixed parameters
    domain = {p: sim_bounds[p] for p in features}

    func = LearnedFunction(X, y, domain=domain, fixed=fixed_vals, seed=seed)
    func.bhat_var = bhat_var # set the empirical y variance
    func.metadata = {'model': model, 'params': sim_params, 'yextra': yextra}
    return func

def fit_dnn(func, n16, n8, n4, n2, nx, 
            weight_l2=None, bias_l2=None,
            activation='elu',
            output_activation='sigmoid', valid_split=0.2, batch_size=64,
            epochs=500, early_stopping=True, sample_weight=None,
            progress=False):
    """
    Fit a DNN based on data in a LearnedFunction.
    """
    assert func.X_train is not None, "func.X_train is not set -- split test/train data"
    assert func.y_train is not None, "func.y_train is not set -- split test/train data"
    input_size = len(func.features)
    model = network(input_size=input_size, output_activation=output_activation,
                    weight_l2=weight_l2, bias_l2=bias_l2,
                    n16=n16, n8=n8, n4=n4, n2=n2, nx=nx, activation=activation)
    callbacks = []
    if early_stopping:
        #model_file = NamedTemporaryFile() if model_file is None else model_file
        #callbacks.append(keras.callbacks.ModelCheckpoint(
        #                  filepath=model_file.name, save_weights_only=True))
        callbacks.append(keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       mode='min', verbose=1,
                                                       patience=100,
                                                       restore_best_weights=True))

    if progress and PROGRESS_BAR_ENABLED:
        callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False))

    if sample_weight is not None:
        assert sample_weight.size == func.y_train.size, "sample_weight is wrong size"
    history = model.fit(func.X_train, func.y_train,
                        validation_split=valid_split,
                        batch_size=batch_size, epochs=epochs, verbose=0,
                        sample_weight=sample_weight, callbacks=callbacks)
    return model, history

class TargetReweighter:
    """
    Very simple target reweighting. I was implementing a version and found
    that our approach is quite similar to this:
        https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-article-density-based-weighting-for-imbalanced-regression-has-been-accepted-for-the-ecml-pkdd-2021-journal-track/
    """
    def __init__(self, y, kernel='epanechnikov'):
        assert isinstance(y, np.ndarray), "y must be a numpy.ndarray"
        if y.ndim == 1:
            y = y[:, None]
        self.y = y
        self.kde = KernelDensity(kernel=kernel)

    def find_optimum_bandwidth(self, bandwidth_grid, cv=5, n_jobs=1, **kwargs):
        param_grid = {'bandwidth': bandwidth_grid}
        self._gcv = GridSearchCV(estimator=self.kde, param_grid={'bandwidth': bandwidth_grid}, cv=cv, n_jobs=n_jobs, **kwargs)
        self._gcv.fit(self.y)
        self.kde.set_params(**self._gcv.best_params_)
        self.kde.fit(self.y)
        self._best_bandwidth = self._gcv.best_params_['bandwidth']
        self._best_score = self._gcv.best_score_

    def set_bandwidth(self, bandwidth):
        self.kde.set_params(bandwidth=bandwidth)
        self.kde.fit(self.y)

    def weights(self, y=None, min_weight='min'):
        y = y if y is not None else self.y
        if y.ndim == 1:
            y = y[:, None]
        w = np.exp(self.kde.score_samples(y))
        rescaled = (w - min(w)) / (max(w) - min(w))
        yp = 1-rescaled
        # if there's a 0 weighted sample, replace with minimum weight
        if min_weight == 'min':
            yp[yp <= 0] = np.min(yp[yp > 0])
        else:
            assert isinstance(min_weight, float), "min_weight must be 'min' or a float"
            yp[yp <= min_weight] = min_weight
        assert np.all(yp > 0)
        return yp


