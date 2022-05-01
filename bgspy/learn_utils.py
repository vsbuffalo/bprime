import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

def network(input_size=2, n128=0, n64=0, n32=0, n8=0, nx=2,
            output_activation='sigmoid', activation='elu'):
    """
    Build a sequential network given the specified layers. nx specifies the
    number of layers with the number of neurons equal to the input size.
    """
    # build network
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    for i in range(nx):
        model.add(layers.Dense(input_size, activation=activation))
    for i in range(n128):
        model.add(layers.Dense(128, activation=activation))
    for i in range(n64):
        model.add(layers.Dense(64, activation=activation))
    for i in range(n32):
        model.add(layers.Dense(32, activation=activation))
    for i in range(n8):
        model.add(layers.Dense(8, activation=activation))
    model.add(tf.keras.layers.Dense(1, activation=output_activation))
    model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=['MeanAbsoluteError'],
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
                        average_reps=False, unlogL=True, combine_sh=True):
    """
    Get the bounds of parameters from the simulation parameters dictionary, find
    all fixed and variable parameters, take the product of the selection and
    dominance coefficient columns, subset the data to include only variable
    (non-fixed) params that go into the training as features. This also does
    lots of validation of the simulation data and simulation parameters.

    Returns a LearnedFunction, with the fixed attributes set.

    Currently only combine_sh=True is supported.

    If unlogL=True, we fix an issue from log10'd seq lengths with a lower
    bound < 1. This is a hack to get sims to run L=0 since int(log10(L)) = 0
    if L < 1, e.g. a neutral case. So, we fit on linear scale if logL=True
    """

    # raw (original) data -- this contains extraneous columns, e.g.
    # ones that aren't fixed
    Xo, yo = np.array(sim_data['X']), sim_data['y']
    if average_reps:
        keys = sim_data['keys']
        assert np.all(sorted(keys) == keys)
        yd = pd.DataFrame(yo)
        yd['key'] = keys
        Xd = pd.DataFrame(Xo)
        Xd['key'] = keys
        Xo_ave = Xd.groupby('key').mean()
        yo_ave = yd.groupby('key').mean()
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

    # Get the fixed columns/features in the original data (with s, h separately)
    # First drop N, which is not in the features matrix since we condition on it.
    sim_params = {k: v for k, v in sim_params.items() if k not in ('N', )}
    fixed_vals, var_cols = match_features(sim_params, data=Xo, features=all_features)

    # currently we just model t = s*h, check that here
    try:
        assert combine_sh
        assert 'h' in fixed_vals
    except AssertionError:
        msg = f"combine_sh set to False or variable dominance coefficients found in data"
        raise NotImplementedError(msg)

    # build up a matrix of non-fixed features, combining sh
    expected_features = [x for x in BGS_MODEL_PARAMS[model] if x not in ('s', 'h')]
    expected_features.insert(1, 'sh')
    nfeatures = len(expected_features)
    Xsh = np.empty((Xo.shape[0], nfeatures))
    for i, feature in enumerate(expected_features):
        if feature == 'sh':
            col = Xo[:, Xo_cols('s')] * Xo[:, Xo_cols('h')]
        else:
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

    # now, calc the sh bounds and add in to the sim bounds
    s_low, s_high, s_log10 = sim_bounds['s']
    h_low, h_high, h_log10 = sim_bounds['h']
    assert h_low == h_high
    assert not h_log10, "'h' cannot be log10 currently"
    # we match if s is log10 or not
    if s_log10:
        low, high = np.log10(h_low * 10**s_low), np.log10(h_low * 10**s_high)
    else:
        low, high = h_low * s_low, h_low * s_high
    sim_bounds['sh'] = low, high, s_log10
    fixed_vals.pop('h') # we don't need this anymore, h is included in sh

    # Now, subset the features matrix with sh merged to include only variable
    # columns
    Xsh_cols = index_cols(expected_features)
    X = Xsh[:, Xsh_cols(*features)]

    ## build the learn func object
    # get the domain of non-fixed parameters
    domain = {p: sim_bounds[p] for p in features}

    if unlogL:
        lower, upper, log10 = domain['L']
        domain['L'] = int(10**lower), int(10**upper), False

    func = LearnedFunction(X, y, domain=domain, fixed=fixed_vals, seed=seed)
    func.metadata = {'model': model, 'params': sim_params, 'yextra': yextra}
    return func

def fit_dnn(func, n128, n64, n32, n8, nx, activation='elu', output_activation='sigmoid',
            valid_split=0.3, batch_size=64, epochs=400, early_stopping=True,
            progress=False):
    """
    Fit a DNN based on data in a LearnedFunction.
    """
    input_size = len(func.features)
    model = network(input_size=input_size, output_activation=output_activation,
                    n128=n128, n64=n64, n32=n32, n8=n8, nx=nx, activation=activation)
    callbacks = []
    if early_stopping:
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                       patience=100, restore_best_weights=True)
        callbacks.append(es)
    if progress and PROGRESS_BAR_ENABLED:
        callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False))

    history = model.fit(func.X_train, func.y_train,
                        validation_split=valid_split,
                        batch_size=batch_size, epochs=epochs, verbose=0,
                        callbacks=callbacks)
    return model, history



