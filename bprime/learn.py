## learn.py -- classes, etc for DNN learned B functions

import os
import json
import itertools
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    import tensorflow_addons as tfa
    PROGRESS_BAR_ENABLED = True
except ImportError:
    PROGRESS_BAR_ENABLED = False

from bprime.utils import signif, index_cols, dist_to_segment
from bprime.sim_utils import fixed_params, get_bounds, random_seed
from bprime.theory import bgs_segment, bgs_rec, BGS_MODEL_PARAMS, BGS_MODEL_FUNCS
from bprime.learn_utils import get_loss_func


def network(input_size=2, n64=4, n32=2, output_activation='sigmoid'):
    # build network
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    for i in range(n64):
        model.add(layers.Dense(64, activation='elu'))
    for i in range(n32):
        model.add(layers.Dense(32, activation='elu'))
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


def data_to_learnedfunc(sim_params, sim_data, model, seed, combine_sh=True):
    """
    Get the bounds of parameters from the simulation parameters dictionary, find
    all fixed and variable parameters, take the product of the selection and
    dominance coefficient columns, subset the data to include only variable
    (non-fixed) params that go into the training as features. This also does
    lots of validation of the simulation data and simulation parameters.

    Returns a LearnedFunction, with the fixed attributes set.

    Currently onlu combine_sh=True is supported.
    """

    # raw (original) data -- this contains extraneous columns, e.g.
    # ones that aren't fixed
    Xo, y = np.array(sim_data['X']), sim_data['y']
    all_features = sim_data['features']
    Xo_cols = index_cols(all_features)

    # check the features we get are one of the BGS models
    check_feature_with_models(all_features, model)

    # get the fixed columns/features in the original data (with s, h separately)
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

    features = new_var_cols # the real feature set is fixed columns

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
    func = LearnedFunction(X, y, domain=domain, fixed=fixed_vals, seed=seed)
    func.metadata = {'model': model, 'params': sim_params}

    return func

def fit_dnn(func, n64, n32, valid_split=0.3, batch_size=64,
            epochs=400, progress=False):
    """
    Fit a DNN based on data in a LearnedFunction.
    """
    input_size = len(func.features)
    model = network(input_size=input_size, output_activation='sigmoid',
                    n64=n64, n32=n32)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                       patience=50, restore_best_weights=True)
    callbacks = [es]
    if progress and PROGRESS_BAR_ENABLED:
        callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False))

    history = model.fit(func.X_train, func.y_train,
                        validation_split=valid_split,
                        batch_size=batch_size, epochs=epochs, verbose=0,
                        callbacks=callbacks)
    return model, history


class LearnedFunction(object):
    """
    A class for storing a learned function from an ML algorithm. Stores the
    domain of the function, split test/training data, and wraps prediction
    functions of the ML model class. By default LearnedFunction.models is a list
    for ensemble/averaging appraoches.

    LearnedFunction.fixed is a dict of fixed parameters, which doesn't effect
    the learning, etc -- just storange.
    """
    def __init__(self, X, y, domain, fixed=None, seed=None):
        if seed is None:
            seed = random_seed()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        assert len(domain) == X.shape[1]
        assert X.shape[0] == y.shape[0]
        n = X.shape[0]
        self.X = X
        self.y = y
        self.test_size = None
        self.features = {}       # dict of features of X and their column
        self.bounds = {}         # dict of lower, upper boundaries
        self.logscale = {}       # dict of which features are log10 scale
        self.normalized = None   # whether the features have been normalized
        self.transforms = None   # the dict of transforms for parameters
        self.fixed = fixed       # the dict of fixed values

        # Auxillary data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_test_orig = None
        self.X_train_orig = None
        self.y_test_orig = None
        self.y_train_orig = None

        # parse the data domains
        self._parse_domains(domain)

        # storage for model/history
        self.model = None
        self.history = None
        self.metadata = None

    def reshuffle(self, seed=None):
        """
        Reset the RandomState with seed, resplit the data,
        and rescale the features using the arguments used previously.
        """
        if seed is None:
            seed = random_seed()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # store existing transforms (reset during split)
        normalize = self.normalized
        transforms = self.transforms
        self.split(test_size=self.test_size)
        self.scale_features(normalize, transforms)
        return self

    def _parse_domains(self, domain):
        """
        Parse the domains dictionary of feature->(lower, upper, log10)
        tuples.
        """
        i = 0
        for feature, params in domain.items():
            if isinstance(params, dict):
                attrs = 'lower', 'upper', 'log10'
                lower, upper, log10, _ = [params[k] for k in attrs]
            elif isinstance(params, tuple):
                lower, upper, log10 = params
            else:
                raise ValueError("params must be dictionary or tuple")
            self.features[feature] = i
            i += 1
            self.bounds[feature] = (lower, upper)
            assert(isinstance(log10, bool))
            self.logscale[feature] = log10

    def col_indexer(self):
        """
        Return a column indexer for the feature columns of X, which makes
        grabbing columns by name easier.
        """
        return index_cols(self.features.keys())

    @property
    def is_split(self):
        return self.X_test is not None and self.X_train is not None

    def __repr__(self):
        nfeat = len(self.features)
        ntarg = self.y.shape[1]
        rows = [
            f"LearnedFunction with {nfeat} feature(s) and {ntarg} target(s)",
            f" variable feature(s):"
        ]
        for feature in self.features:
            scale = 'linear' if not self.logscale[feature] else 'log10'
            lower, upper = (signif(b, 3) for b in self.bounds[feature])
            trans_func = str(self.transforms[feature]) if self.transforms is not None else None
            row = f"  - {feature} ∈ [{lower}, {upper}] ({scale}, {trans_func})"
            rows.append(row)
        rows.append(f" fixed fixed(s) (based on metadata):")
        for fixed, val in self.fixed.items():
            rows.append(f"  - {fixed} = {val}")
        normed = False if self.normalized is None else self.normalized
        rows.append(f"Features normalized? {normed}")
        rows.append(f"Features split? {self.is_split}")
        if self.is_split:
            rows[-1] += f", test size: {100*np.round(self.test_size, 2)}% (n={self.X_test.shape[0]:,})"

        rows.append(f"Total size: {self.X.shape[0]:,}")
        return "\n".join(rows)

    def split(self, test_size=0.2):
        """
        Make a test/train split. This resets the state of the object
        to initialization (any scale_feature transforms will be reset).
        """
        self.test_size = test_size
        dat = train_test_split(self.X, self.y, test_size=test_size,
                               random_state=self.rng)
        Xtrn, Xtst, ytrn, ytst = dat
        self.X_train = Xtrn
        self.X_test = Xtst
        self.y_train = ytrn.squeeze()
        self.y_test = ytst.squeeze()
        self.normalized = False
        self.transforms = {f: None for f in self.features}
        return self

    def scale_features(self, normalize=True, transforms='match'):
        """
        Normalize (center and scale) features, optionally applying
        a feature transform beforehand. This uses sklearn's StandardScaler
        so inverse transforms are easy.

        If transforms = 'match', inputs are log10 scaled based on whether
        the domain was log10 scale. If transforms is None, no transformations
        are conducted. A dict of manual transforms can also be supplied.

        normalize: boolean whether to center and scale
        feature_transforms: dictionary of feature, transform function pairs
        """
        if not self.is_split:
            raise ValueError("X, y must be split first")
        if self.normalized or not all(x is None for x in self.transforms.values()):
            raise ValueError("X already transformed!")
        if transforms not in (None, 'match'):
            valid_transforms = all(t in self.features for t in transforms.keys())
            if not valid_transforms:
                raise ValueError("'transforms' dict has key not in features")
        # store the pre-transformed daata, which is useful for figures, etc
        self.X_test_orig = self.X_test
        self.X_train_orig = self.X_train
        self.y_test_orig = self.y_test
        self.y_train_orig = self.y_train
        for feature, col_idx in self.features.items():
            trans_func = None
            if transforms == 'match' and self.logscale[feature]:
                trans_func = np.log10
            elif isinstance(transforms, dict):
                trans_func = transforms.get(feature, None)
            if trans_func is not None:
                # do the transform with the given trans_func
                self.X_train[:, col_idx] = trans_func(self.X_train[:, col_idx])
                self.X_test[:, col_idx] = trans_func(self.X_test[:, col_idx])
                self.transforms[feature] = trans_func
        if normalize:
            self.scaler = StandardScaler().fit(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            self.X_train = self.scaler.transform(self.X_train)
            self.normalized = True
        return self

    @property
    def X_test_orig_linear(self):
        "Return LearnedFunction.X_train_orig, transforming log10'd columns back to linear"
        X = np.copy(self.X_test_orig)
        for i, feature in enumerate(self.features):
            if self.logscale[feature]:
                X[:, i] = 10**X[:, i]

        return X


    @property
    def X_train_orig_linear(self):
        "Return LearnedFunction.X_train_orig, transforming log10'd columns back to linear"
        X = np.copy(self.X_train_orig)
        for i, feature in enumerate(self.features):
            if self.logscale[feature]:
                X[:, i] = 10**X[:, i]

        return X

    def save(self, filepath):
        """
        Save the LearnedFunction object at 'filepath.pkl' and 'filepath.h5'.
        Pickle the LearnedFunction object, and the model is saved via that
        object's save method.
        """
        if filepath.endswith('.pkl'):
            filepath = filepath.replace('.pkl', '')
        model = self.model # store the reference
        self.model = None
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self, f)
        self.model = model
        if self.model is not None:
            model.save(f"{filepath}.h5")

    @classmethod
    def load(cls, filepath, load_model=True):
        """
        Load the LearnedFunction object at 'filepath.pkl' and 'filepath.h5'.
        """
        import keras
        if filepath.endswith('.pkl'):
            filepath = filepath.replace('.pkl', '')
        with open(f"{filepath}.pkl", 'rb') as f:
            obj = pickle.load(f)
        model_path = f"{filepath}.h5"
        if load_model and os.path.exists(model_path):
            obj.model = keras.models.load_model(model_path)
        return obj

    @property
    def has_model(self):
        return self.model is not None

    def predict_test(self, **kwargs):
        """
        Predict the test data.
        """
        assert self.has_model
        return self.model.predict(self.X_test, **kwargs).squeeze()

    def test_mae(self):
        return np.mean(np.abs(self.model.predict(self.X_test).squeeze() - self.y_test))

    def test_mse(self):
        return np.mean((self.model.predict(self.X_test).squeeze() - self.y_test)**2)

    def get_bounds(self, feature):
        "Get the bounds, rescaling to linear if log-transformed"
        log10 = self.logscale[feature]
        lower, upper = self.bounds[feature]
        if log10:
            lower, upper = 10**lower, 10**upper
        return lower, upper


    def check_bounds(self, X, correct_bounds=False):
        out_lowers, out_uppers = [], []
        total = 0
        for i, feature in enumerate(self.features):
            lower, upper = self.get_bounds(feature)
            out_lower = X[:, i] < lower
            out_upper = X[:, i] > upper
            total += out_lower.sum() + out_upper.sum()
            if np.any(out_lower) or np.any(out_upper):
                #print(feature, 'lower', X[out_lower, i])
                #print(feature, 'upper', X[out_upper, i])
                if correct_bounds:
                    X[out_lower, i] = lower
                    X[out_upper, i] = upper
                else:
                    if np.any(out_lower):
                        out_lowers.append(feature)
                    if np.any(out_upper):
                        out_uppers.append(feature)

        out_of_bounds = len(out_lowers) or len(out_uppers)
        if not correct_bounds and out_of_bounds:
            lw = ', '.join(out_lowers)
            up = ', '.join(out_uppers)
            perc = 100*np.round(total / np.prod(X.shape), 2)
            msg = f"out of bounds, lower ({lw}) upper ({up}), total = {total} ({perc}%)"
            raise ValueError(msg)
        return X


    def predict(self, X, correct_bounds=True, transforms=True,
                scale_input=True, **kwargs):
        """
        Predict for an input function X (linear space). If transforms is True,
        and transforms in LearnedFunction.transforms dict are applied to match
        those applied from LearnedB.scale_features().
        """
        assert self.has_model
        X = self.check_bounds(X, correct_bounds)
        if transforms:
            for i, (feature, trans_func) in enumerate(self.transforms.items()):
                if trans_func is not None:
                    X[:, i] = trans_func(X[:, i])
        if scale_input:
            X = self.scaler.transform(X)
        return self.model.predict(X, **kwargs).squeeze()

    def predict_train(self, **kwargs):
        """
        Predict the training data.
        """
        assert self.has_model
        return self.model.predict(self.X_train, **kwargs).squeeze()

    def domain_grids(self, n, fix_X=None):
        """
        Create a grid of values across the domain for all features.
        fix_X is a dict of fixed values for the grids. If a feature
        name is in log10 tuple, it will be log10'd.
        """
        valid_features = set(self.features)
        msg = "'fix_X' has a feature not in X's features"
        assert fix_X is None or all([(k in valid_features) for k in fix_X.keys()]), msg
        grids = []
        nx = n
        for feature, (lower, upper) in self.bounds.items():
            is_logscale = self.logscale[feature]
            if fix_X is None or feature not in fix_X:
                if isinstance(n, dict):
                    nx = n[feature]
                grid = np.linspace(lower, upper, nx)
            elif feature in fix_X:
                grid = fix_X[feature]
            else:
                assert False, "should not reach this point"
            if is_logscale:
                grid = 10**grid
            grids.append(grid)
        return grids

    def predict_grid(self, n, fix_X=None, verbose=True):
        """
        Predict a grid of points (useful for visualizing learned function).
        This uses the domain specified by the model.

        Returns:
          - A list of the grid values for each column.
          - A matrix of the mesh grid, flattened into columns (the total number
             of columns, *before* the transformations.
          - A meshgrid that's been normalized and transformed as the original
            data has.
        """
        assert self.has_model
        domain_grids = self.domain_grids(n, fix_X=fix_X)
        if verbose:
            grid_dims = 'x'.join(map(str, n.values()))
            print(f"making {grid_dims} grid...\t", end='')
        mesh = np.meshgrid(*domain_grids)
        if verbose:
            print("done.")
        mesh_array = np.stack(mesh)
        X_meshcols = np.stack([col.flatten() for col in mesh]).T
        X_meshcols_orig = X_meshcols.copy()
        # transform/scale the new mesh
        for feature, col_idx in self.features.items():
            trans_func = self.transforms.get(feature, None)
            if trans_func is not None:
                X_meshcols[:, col_idx] = trans_func(X_meshcols[:, col_idx])
        if self.normalized:
            X_meshcols = self.scaler.transform(X_meshcols)

        predict = self.model.predict(X_meshcols, verbose=int(verbose)).squeeze()
        return domain_grids, X_meshcols_orig, X_meshcols, predict.reshape(mesh[0].shape)




class LearnedB(object):
    """
    A general class that wraps a learned B function.
    """
    def __init__(self, t_grid=None, w_grid=None, genome=None, model='segment'):
        try:
            model = 'bgs_' + model if not model.startswith('bgs_') else model
            params = BGS_MODEL_PARAMS[model]
            bgs_model = BGS_MODEL_FUNCS[model]
        except KeyError:
            allow = ["'"+x.replace('bgs_', '')+"'" for x in BGS_MODEL_PARAMS.keys()]
            raise KeyError(f"model ('{model}') must be either {', '.join(allow)}")
        self.genome = None
        self.func = None
        self._predict = None
        self._X_test_hash = None
        self.bgs_model = bgs_model
        self.params = params
        self.w_grid = w_grid
        self.t_grid = t_grid

    @property
    def dim(self):
        return self.w_grid.shape[0], self.t_grid.shape[0]

    @property
    def tw_mesh(self):
        return np.array(list(itertools.product(self.t_grid, self.w_grid)))

    def theory_B(self, X=None):
        """
        Compute the BGS theory given the right function ('segment' or 'rec')
        on the feature matrix X. E.g. use for X_test_orig_linear or using
        meshgrids.
        """
        if X is None:
            X = self.func.X_test_orig_linear
        assert X.shape[1] == len(self.func.features)
        features = self.func.features
        kwargs = {}
        for i, feature in enumerate(features):
            kwargs[feature] = X[:, i]

        # merge in the fixed params
        kwargs = {**kwargs, **self.func.fixed}
        # we tweak s, and h since sh is set
        kwargs['h'] = 1
        kwargs['s'] = kwargs.pop('sh')
        kwargs.pop('N') # not needed for theory
        return self.bgs_model(**kwargs)


    def predict_test(self):
        """
        Predict X_test, caching the results (invalidation based on hash of
        X_test).
        """
        X_test_hash = hash(self.func.X_test.data.tobytes())
        if self._predict is None or X_test_hash != self._X_test_hash:
            self._X_test_hash = X_test_hash
            predict = self.func.predict_test()
            self._predict = predict
        else:
            print("using cached predictions")
        return self._predict

    def binned_Bhats(self, bins):
        predict = self.predict_test()
        if isinstance(bins, int):
            bins = np.linspace(predict.min(), predict.max(), bins)
        ytest_bins = stats.binned_statistic(predict, self.func.y_test.squeeze(),
                                            bins=bins)
        edges = ytest_bins.bin_edges
        return edges, 0.5*(edges[:-1]+edges[1:]), ytest_bins.statistic

    def Bhat_mse(self, bins):
        _, x, y = self.binned_Bhats(bins)
        return np.mean((x - y)**2)

    def load_func(self, filepath):
        func = LearnedFunction.load(filepath)
        self.func = func

    def train_model(self, filepath):
        pass

    def theory_loss(self, loss='mae', raw=False):
        lossfunc = get_loss_func(loss)
        B = self.theory_B()
        predict = self.predict_test()
        lossvals = lossfunc(B, predict)
        if raw:
            return B, lossvals
        return lossvals.mean()

    def is_valid_grid(self):
        """
        Check that all elements of the X features matrix (minus rec frac)
        are valid and within the bounds of the simulations the function was trained on.
        """
        t_lower, t_upper = self.func.get_bounds('t')
        w_lower, w_upper = self.func.get_bounds('mu')
        assert np.all(t_lower < self.t_grid) and np.all(t_upper > self.t_grid)
        assert np.all(w_lower < self.w_grid) and np.all(w_upper > self.w_grid)
        Ls = list(itertools.chain(self.chrom_seg_L))
        l_lower, l_upper = self.func.get_bounds('L')
        assert np.all(l_lower < Ls) and np.all(l_upper > Ls)
        rbps = list(itertools.chain(self.chrom_seg_rbp))
        rbp_lower, rbp_upper = self.func.get_bounds('rbp')
        assert np.all(rbp_lower < rbps) and np.all(rbp_upper > rbps)

    def _build_pred_matrix(self, chrom):
        """

        Build a prediction matrix for an entire chromosome. This takes the
        segment annotation data (map position recomb rates, segment length) and
        tiles them so a group of data called A repeats,

          X = [[w1, t1, A],
               [w1, t2, A],
               ...

        The columns are the features of the B' training data:
               0,    1,    2,     3,   4
            'sh', 'mu', 'rf', 'rbp', 'L'

        The third column is the recombination fraction away from the
        focal site and needs to change each time the position changes.
        This is done externally; it's set to zero here.
        """
        # size of mesh; all pairwise combinations
        n = self.tw_mesh.shape[0]
        # number of segments
        nsegs = len(self.chrom_idx[chrom])
        X = np.zeros((nsegs*n, 5))
        # repeat the w/t element for each block of segments
        X[:, 0:2] = np.repeat(self.tw_mesh, nsegs, axis=0)
        # tile the annotation data
        X[:, 3] = np.tile(self.chrom_seg_rbp[chrom], n)
        X[:, 4] = np.tile(self.chrom_seg_L[chrom], n)
        return X

    def transform(self, *arg, **kwargs):
        return self.func.scaler.transform(*args, **kwargs)

    def write_chrom_Xs(self, dir, scale=True):
        """
        Write the X chromosome segment data.
        The columns are sh, mu, rf, rbp, and L -- note that rf is blank,
        as this is filled in depending on what the focal position is.
        """
        for chrom, X in self.Xs.items():
            if scale:
                X = self.transform(X)
            np.save(os.path.join(f"chrom_data_{chrom}.npy", X))

    def focal_positions(self, progress=True):
        """
        Focal site by chunk.
        """
        if progress:
            outer_progress_bar = tq.tqdm(total=self.total)
        # inner_progress_bar = tq.tqdm(leave=True)
        while True:
            next_chunk = next(self.mpos_iter)
            # get the next chunk of map positions to process
            chrom, mpos_chunk = next_chunk
            # chromosome segment positions
            chrom_seg_mpos = self.chrom_seg_mpos[chrom]
            # focal map positions
            map_positions = mpos_chunk
            for f in map_positions:
                # compute the rec frac to each segment's start or end
                rf = dist_to_segment(f, chrom_seg_mpos)
                # place rf in X, log10'ing it since that's what we train on
                # and tile it to repeat
                Xrf_col = np.tile(rf, self.tw_mesh.shape[0])
                if self.func.logscale['rf']:
                    Xrf_col = np.log10(Xrf_col)
                assert np.all(np.isfinite(Xrf_col)), "rec frac column not all finite!"
                yield chrom, self.transform(X)[:, 2]
            if progress:
                outer_progress_bar.update(1)
            # inner_progress_bar.reset()


