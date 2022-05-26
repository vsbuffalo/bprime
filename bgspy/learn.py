## learn.py -- classes, etc for DNN learned B functions

import os
from os.path import join, basename
import json
import itertools
import warnings
import pickle
from collections import Counter, defaultdict
import numpy as np
import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from tensorflow import keras

from bgspy.utils import signif, index_cols, dist_to_segment, make_dirs
from bgspy.sim_utils import random_seed
from bgspy.theory import bgs_segment, bgs_rec, BGS_MODEL_PARAMS, BGS_MODEL_FUNCS
from bgspy.loss import get_loss_func
from bgspy.parallel import MapPosChunkIterator
from bgspy.utils import BScores


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
        self.X.setflags(write=False)
        self.y.setflags(write=False)
        self.test_split = None
        self.features = {}       # dict of features of X and their column
        self.bounds = {}         # dict of lower, upper boundaries
        self.logscale = {}       # dict of which features are log10 scale
        self.normalized = None   # whether the features have been normalized
        self.transforms = None   # the dict of transforms for parameters
        fixed = fixed if fixed is not None else {}
        self.fixed = fixed       # the dict of fixed values

        # Auxillary data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # these are the X_test/X_train values *before* transforms/scales
        # have been applied
        self.X_test_raw = None
        self.X_train_raw = None
        self.y_test_raw = None
        self.y_train_raw = None

        # parse the data domains
        self._parse_domains(domain)

        # storage for model/history
        self.model = None
        self.history = None
        self.metadata = None

    def reseed(self, seed=None):
        """
        Reset the RandomState with seed and clear out the random state.
        """
        if seed is None:
            seed = random_seed()
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_test_raw = None
        self.X_train_raw = None

    def _parse_domains(self, domain):
        """
        Parse the domains dictionary of feature->(lower, upper, log10)
        tuples.
        """
        i = 0
        for feature, params in domain.items():
            if isinstance(params, dict):
                attrs = 'lower', 'upper', 'log10'
                lower, upper, log10, *_ = [params[k] for k in attrs]
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
        rows = [
            f"LearnedFunction with {nfeat} feature(s)",
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
            rows[-1] += f", test size: {100*np.round(self.test_split, 2)}% (n={self.X_test.shape[0]:,})"

        rows.append(f"Total size: {self.X.shape[0]:,}")
        return "\n".join(rows)

    def _protect(self):
        protectables = ['X', 'y', 'X_test', 'X_train', 'y_test', 'y_train',
                        'X_test_raw', 'X_train_raw']
        for protectable in protectables:
            val = getattr(self, protectable)
            if val is not None:
                val.setflags(write=False)

    def split(self, test_split=0.2):
        """
        Make a test/train split. This resets the state of the object
        to initialization (any scale_feature transforms will be reset).
        """
        self.test_split = test_split
        dat = train_test_split(self.X, self.y, test_size=test_split,
                               random_state=self.rng)
        Xtrn, Xtst, ytrn, ytst = dat
        self.X_train = Xtrn
        self.X_test = Xtst
        self.y_train = ytrn.squeeze()
        self.y_test = ytst.squeeze()
        # store the pre-transformed daata, which is useful for figures, etc
        self.X_test_raw = np.copy(self.X_test)
        self.X_train_raw = np.copy(self.X_train)

        self.y_test_raw = np.copy(self.y_test)
        self.y_train_raw = np.copy(self.y_train)
        self._protect()
        self.normalized = False
        self.transforms = {f: None for f in self.features}
        return self

    def scale_features(self, normalize=True, normalize_target=False,
                       log10_target=False, transforms='match'):
        """
        Normalize (center and scale) the split features (test/train), optionally
        applying a feature transform beforehand. This uses sklearn's
        StandardScaler so inverse scaling is easy.

        If transforms = 'match', inputs are log10 scaled based on whether the
        domain was log10 scale. If transforms is None, no transformations are
        conducted. A dict of manual transforms can also be supplied.

        normalize: boolean whether to center and scale
        transforms: dictionary of feature, transform function pairs
        """
        if isinstance(transforms, dict):
            raise NotImplementedError("transforms cannot be a dict yet")
        if normalize_target:
            raise NotImplementedError("normalize_target not implemented yet")
        if not self.is_split:
            raise ValueError("X, y must be split first")
        if self.normalized or not all(x is None for x in self.transforms.values()):
            raise ValueError("X already transformed!")
        if transforms not in (None, 'match'):
            valid_transforms = all(t in self.features for t in transforms.keys())
            if not valid_transforms:
                raise ValueError("'transforms' dict has key not in features")
        # all this is done in place(!)
        X_train = np.copy(self.X_train)
        X_test = np.copy(self.X_test)
        y_test = np.copy(self.y_test)
        y_train = np.copy(self.y_train)
        X_train.setflags(write=True)
        X_test.setflags(write=True)
        for feature, col_idx in self.features.items():
            # note that if something is log scaled log10 will be added to the transform
            # trans_func dict!
            trans_func = None
            if transforms == 'match' and self.logscale[feature]:
                trans_func = np.log10
            elif isinstance(transforms, dict):
                trans_func = transforms.get(feature, None)
            if trans_func is not None:
                # do the transform with the given trans_func
                X_train[:, col_idx] = trans_func(X_train[:, col_idx])
                X_test[:, col_idx] = trans_func(X_test[:, col_idx])
                self.transforms[feature] = trans_func
        if normalize:
            self.scaler = StandardScaler().fit(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = self.scaler.transform(X_train)
            self.normalized = True
        if log10_target:
           y_train = np.log10(y_train)
           y_test = np.log10(y_test)
        if normalize_target:
            self.target_scaler = StandardScaler().fit(y_train[:, None])
            self.y_test = self.target_scaler.transform(y_test[:, None])
            self.y_train = self.target_scaler.transform(y_train[:, None])
            self.target_normalized = True

        # if we've done the transforms once, save them to the object
        # and protect them
        self.X_test = X_test
        self.X_train = X_train
        self._protect()
        return self

    def transform_feature_to_match(self, feature, x, correct_bounds=False):
        """
        For linear input feature x, match the transformations/scalers
        in applied to the training data.
        """
        x = np.copy(x)
        x = self.check_feature_bounds(feature, x, correct_bounds=correct_bounds)
        if feature in self.transforms:
            x = self.transforms[feature](x)
        if self.normalized:
            idx = [i for i, k in enumerate(self.features.keys()) if k == feature]
            assert len(idx) == 1
            idx = idx[0]
            mean = self.scaler.mean_[idx]
            scale = self.scaler.scale_[idx]
            x = (x - mean) / scale
        return x

    def transform_X_to_match(self, X, correct_bounds=False):
        """
        For linear input matrix X, match the transformations/scalers
        in applied to the training data X.
        """
        X = np.copy(X)
        X = self.check_bounds(X, correct_bounds=correct_bounds)
        for i, (feature, trans_func) in enumerate(self.transforms.items()):
            if trans_func is not None:
                X[:, i] = trans_func(X[:, i])
        if self.normalized:
            X = self.scaler.transform(X)
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
        if filepath.endswith('.pkl'):
            filepath = filepath.replace('.pkl', '')
        with open(f"{filepath}.pkl", 'rb') as f:
            obj = pickle.load(f)
            # numpy write flags are not preserved when pickling objects(!), so
            # need to fix that here.
            obj._protect()
        model_path = f"{filepath}.h5"
        if load_model and os.path.exists(model_path):
            obj.model = keras.models.load_model(model_path)
        return obj

    @property
    def has_model(self):
        return self.model is not None

    def predict_test(self, **kwargs):
        """
        Predict the test data in LearnedFunction.X_test (note: this has
        already been transformed and scaled). This function expects
        transformed/scaled data -- it's fed directly into model prediction.
        """
        assert self.has_model
        return self.predict(self.X_test_raw, **kwargs).squeeze()

    def test_mae(self):
        return np.mean(np.abs(self.predict(self.X_test_raw).squeeze() - self.y_test))

    def test_mse(self):
        return np.mean((self.predict(self.X_test_raw).squeeze() - self.y_test)**2)

    def get_bounds(self, feature):
        "Get the bounds, rescaling to linear if log-transformed"
        log10 = self.logscale[feature]
        lower, upper = self.bounds[feature]
        if log10:
            lower, upper = 10**lower, 10**upper
        return lower, upper

    def check_feature_bounds(self, feature, x, correct_bounds=False):
        """
        Check (and possibly correct) the bounds of a single feature.
        """
        x = np.copy(x)
        lower, upper = self.get_bounds(feature)
        out_lower = x < lower
        out_upper = x > upper
        if np.any(out_lower) or np.any(out_upper):
            if correct_bounds:
                x[out_lower] = lower
                x[out_upper] = upper
            else:
                raise AssertionError(f"{feature} is out of bounds")
        return x

    def check_bounds(self, X, return_stats=False, correct_bounds=False):
        """
        Check the boundaries of X compared to LearnedFunction.bounds.
        If correct_bounds=True, any out of bounds values are coerced
        to the nearest boundary value. If return_stats=True, this function
        will return statistics of what features have out of bounds values
        and how many (e.g. useful for warnings)
        """
        X = np.copy(X)
        n_lowers, n_uppers = Counter(), Counter()
        for i, feature in enumerate(self.features):
            lower, upper = self.get_bounds(feature)
            out_lower = X[:, i] < lower
            out_upper = X[:, i] > upper
            if np.any(out_lower) or np.any(out_upper):
                #print(feature, 'lower', X[out_lower, i])
                #print(feature, 'upper', X[out_upper, i])
                if correct_bounds:
                    X[out_lower, i] = lower
                    X[out_upper, i] = upper
                else:
                    if np.any(out_lower):
                        n_lowers[feature] += np.sum(out_lower)
                    if np.any(out_upper):
                        n_uppers[feature] += np.sum(out_upper)

        total = sum(n_lowers.values()) + sum(n_uppers.values())
        out_of_bounds = total > 0

        # format the features that have been found to have out of bounds values
        lw = ', '.join(n_lowers)
        up = ', '.join(n_uppers)
        perc = 100*np.round(total / np.prod(X.shape), 2)
        if return_stats:
            return n_lowers, n_uppers
        if not correct_bounds and out_of_bounds:
            msg = f"out of bounds, lower ({lw}) upper ({up}), total = {total} ({perc}%)"
            raise ValueError(msg)
        return X


    def predict(self, X=None, transform_to_match=True, correct_bounds=True,
                **kwargs):
        """
        Predict for an input function X (in the same as simulation space).
        If transform_to_match is True, transforms (e.g. log10's features as
        necessary) and scales everything.

        If X is None, uses traning data X_test_raw.
        """
        if X is None:
            X = self.X_test_raw
        assert self.has_model
        X = np.copy(X) # protect the original object
        X = self.check_bounds(X, correct_bounds=correct_bounds)
        if transform_to_match:
            # bounds already correct...
            X = self.transform_X_to_match(X, correct_bounds=False)
        return self.model.predict(X, **kwargs).squeeze()

    def predict_train(self, **kwargs):
        """
        Predict the training data.
        """
        assert self.has_model
        return self.predict(self.X_train_raw, **kwargs).squeeze()

    def domain_grids(self, n, fix_X=None, manual_domains=None):
        """
        Create a grid of values across the domain for all features.
        fix_X is a dict of fixed values for the grids. If a feature
        name is in log10 tuple, it will be log10'd.

        Note: fix_X is to be supplied on the *linear* scale!
        """
        valid_features = set(self.features)
        msg = "'fix_X' has a feature not in X's features"
        assert fix_X is None or all([(k in valid_features) for k in fix_X.keys()]), msg
        msg = "'manual_domains' has a feature not in X's features"
        assert manual_domains is None or all([(k in valid_features) for k in manual_domains.keys()]), msg
        fix_X = {} if fix_X is None else fix_X
        manual_domains = {} if manual_domains is None else manual_domains
        grids = []
        nx = n
        for feature, (lower, upper) in self.bounds.items():
            is_logscale = self.logscale[feature]
            if feature not in fix_X and feature not in manual_domains:
                if isinstance(n, dict):
                    nx = n[feature]
                grid = np.linspace(lower, upper, nx)
            elif feature in manual_domains:
                assert feature not in fix_X, "feature cannot be in fix_X and manual_domains!"
                lower, upper, nx, log_it = manual_domains[feature]
                grid = np.linspace(lower, upper, nx)
                if log_it:
                    grid = 10**grid
            elif feature in fix_X:
                grid = fix_X[feature]
            else:
                assert False, "should not reach this point"
            if is_logscale and feature not in fix_X and feature not in manual_domains:
                grid = 10**grid
            grids.append(grid)
        return grids

    def predict_grid(self, n, fix_X=None, manual_domains=None,
                     correct_bounds=False, verbose=True):
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
        domain_grids = self.domain_grids(n, fix_X=fix_X, manual_domains=manual_domains)
        #if verbose:
        #    grid_dims = 'x'.join(map(str, n.values()))
        #    print(f"making {grid_dims} grid...\t", end='')
        mesh = np.meshgrid(*domain_grids)
        if verbose:
            print("done.")
        mesh_array = np.stack(mesh)
        X_meshcols = np.stack([col.flatten() for col in mesh]).T
        X_meshcols_raw = X_meshcols.copy()
        # transform/scale the new mesh
        X_meshcols = self.transform_X_to_match(X_meshcols, correct_bounds=correct_bounds)
        predict = self.model.predict(X_meshcols, verbose=int(verbose)).squeeze()
        return domain_grids, X_meshcols_raw, X_meshcols, predict.reshape(mesh[0].shape)


class LearnedB(object):
    """
    A general class that wraps a learned B function, mostly for wrapping
    code for speciality loss functions, comparison to BGS theory, etc
    """
    def __init__(self, t_grid=None, w_grid=None, model='segment'):
        try:
            model = 'bgs_' + model if not model.startswith('bgs_') else model
            params = BGS_MODEL_PARAMS[model]
            bgs_model = BGS_MODEL_FUNCS[model]
        except KeyError:
            allow = ["'"+x.replace('bgs_', '')+"'" for x in BGS_MODEL_PARAMS.keys()]
            raise KeyError(f"model ('{model}') must be either {', '.join(allow)}")
        self.func = None
        self._predict = None
        self._X_test_hash = None
        model = model if not model.startswith('bgs_') else model.replace('bgs_', '')
        self.bgs_model = bgs_model
        self.bgs_model_name = model
        self.params = params  # expected parameters under one of the BGS models
        self.w_grid = w_grid
        self.t_grid = t_grid

    @property
    def dim(self):
        return self.w_grid.shape[0], self.t_grid.shape[0]

    @property
    def wt_mesh(self):
        return np.array(list(itertools.product(self.w_grid, self.t_grid)))

    def is_valid_learnedfunc(self):
        assert self.func is not None, "LearnedB.func is None"
        features = tuple(self.func.features.keys())
        msg = ("LearnedFunction.features do not match the parameter "
              "order of bgs_rec or bgs_segment!")
        assert features == self.params, msg
        return True

    def theory_B(self, X=None):
        """
        Compute the BGS theory given the right function ('segment' or 'rec')
        on the feature matrix X. E.g. use for X_test_raw or using
        meshgrids.
        """
        self.is_valid_learnedfunc()
        if X is None:
            X = self.func.X_test_raw
        assert X.shape[1] == len(self.func.features)
        features = self.func.features
        kwargs = {}
        for i, feature in enumerate(features):
            kwargs[feature] = X[:, i]
        return self.bgs_model(**kwargs)

    def predict_train(self):
        return self.func.predict_train()

    def predict_test(self):
        """
        Predict X_test_raw.
        """
        predict = self.func.predict_test()
        return predict

    def predict_datum(self, **kwargs):
        msg = f"kwargs: {', '.join(kwargs.keys())}, features: {', '.join(self.func.features.keys())}"
        assert set(kwargs.keys()) == set(self.func.features.keys()), msg
        # make a prediction matrix
        X = np.array([[kwargs[k] for k in self.func.features.keys()]])
        return float(self.func.predict(X))

    def binned_Bhats(self, bins, which='test'):
        assert which in ('test', 'train'), "which must be 'test' or 'train'"
        if which == 'test':
            predict = self.predict_test()
        else:
            predict = self.predict_train()
        if isinstance(bins, int):
            bins = np.linspace(predict.min(), predict.max(), bins)
        y = self.func.y_test if which == 'test' else self.func.y_train
        y = y.squeeze()
        y_bins = stats.binned_statistic(predict, y, bins=bins)
        edges = y_bins.bin_edges
        return edges, 0.5*(edges[:-1]+edges[1:]), y_bins.statistic

    def Bhat_mse(self, bins):
        # x is the prediction midpoint, y is the mean y_test
        _, x, y = self.binned_Bhats(bins)
        return np.nanmean((x - y)**2)

    def load_func(self, filepath):
        func = LearnedFunction.load(filepath)
        self.func = func

    def save(self, filepath):
        """
        Save the LearnedB (and any contained LearnedFunction) object at
        'filepath.pkl' and 'filepath.h5'.  Pickle the LearnedFunction object,
        and the model is saved via that object's save method.
        """
        if filepath.endswith('.pkl'):
            filepath = filepath.replace('.pkl', '')
        has_func = self.func is not None
        if has_func:
            # only need to worry about H5'ing the func.model if there's a func
            model = self.model # store the reference
            self.model = None
        # save this object
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self, f)
        if has_func:
            self.model = model
        if self.model is not None:
            model.save(f"{filepath}.h5")

    @classmethod
    def load(cls, filepath, load_model=True):
        """
        Load the LearnedB object at 'filepath.pkl' and 'filepath.h5'.
        """
        if filepath.endswith('.pkl'):
            filepath = filepath.replace('.pkl', '')
        with open(f"{filepath}.pkl", 'rb') as f:
            obj = pickle.load(f)
            if isinstance(obj, LearnedFunction):
                raise ValueError("This is a LearnedFunction, not LearnedB; use LearnedFunction.load()")
            # numpy write flags are not preserved when pickling objects(!), so
            # need to fix that here.
            obj._protect()
        model_path = f"{filepath}.h5"
        if load_model and os.path.exists(model_path):
            obj.func.model = keras.models.load_model(model_path)
        return obj

    def predict_loss(self, loss='mae', raw=False):
        lossfunc = get_loss_func(loss)
        predict = self.predict_test()
        ytest = self.func.y_test
        lossvals = lossfunc(ytest, predict)
        if raw:
            return B, lossvals
        return lossvals.mean()

    def theory_loss(self, X=None, loss='mae', raw=False):
        lossfunc = get_loss_func(loss)
        B = self.theory_B(X)
        predict = self.func.predict(X)
        lossvals = lossfunc(B, predict)
        if raw:
            return B, lossvals
        return lossvals.mean()

    def is_valid_grid(self):
        """
        Check that wt grid is within the training bounds.

        Note that there is also checking with LearnedFunction.check_bounds(),
        but this is a bit lower-level (e.g. on actualy input feature matrix X).
        """
        t_lower, t_upper = self.func.get_bounds('sh')
        w_lower, w_upper = self.func.get_bounds('mu')
        assert np.all(t_lower <= self.t_grid) and np.all(t_upper >= self.t_grid), "sh grid out of training bounds"
        assert np.all(w_lower <= self.w_grid) and np.all(w_upper >= self.w_grid), "mu grid out of training bounds"

    def is_valid_segments(self):
        """
        Check that segment values are within training bounds.
        Note this is less of a big deal if they're not -- LearnedFunction can
        correct bounds by forcing out of band X values to the boundary value.

        Note that there is also checking with LearnedFunction.check_bounds(),
        but this is a bit lower-level (e.g. on actualy input feature matrix X).
        """
        Ls = np.fromiter(itertools.chain(*self.segments.L.values()), dtype=float)
        l_lower, l_upper = self.func.get_bounds('L')
        assert np.all(l_lower < Ls) and np.all(l_upper > Ls), "segment lengths out of training bounds"
        rbps = np.fromiter(itertools.chain(*self.segments.rbp.values()), dtype=float)
        rbp_lower, rbp_upper = self.func.get_bounds('rbp')
        assert np.all(rbp_lower < rbps) and np.all(rbp_upper > rbps), "segment rec rates out of training bounds"

    def fix_bounds(self, x, feature):
        """
        Fix the bounds of a vector of a given feature.
        """
        x = np.copy(x)
        lower, upper = self.func.get_bounds(feature)
        x[x < lower] = lower
        x[x > upper] = upper
        return x

    def transform(self, *arg, **kwargs):
        return self.func.scaler.transform(*args, **kwargs)

