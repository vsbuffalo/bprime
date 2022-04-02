## learn.py -- classes, etc for DNN learned B functions

import os
import itertools
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from bprime.utils import signif, index_cols, dist_to_segment
from bprime.sim_utils import random_seed

class LearnedFunction(object):
    """
    A class for storing a learned function from an ML algorithm. Stores the
    domain of the function, split test/training data, and wraps prediction
    functions of the ML model class. By default LearnedFunction.models is a list
    for ensemble/averaging appraoches.
    """
    def __init__(self, X, y, domain):
        assert(len(domain) == X.shape[1])
        assert(X.shape[0] == y.shape[0])
        self.X = X
        self.y = y
        self.test_size = None
        self.features = {}       # dict of features of X and their column
        self.bounds = {}         # dict of lower, upper boundaries
        self.logscale = {}       # dict of which features are log10 scale
        self.normalized = None   # whether the features have been normalized
        self.transforms = None    # the dict of transforms for parameters

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
            f" feature(s):"
        ]
        for feature in self.features:
            scale = 'linear' if not self.logscale[feature] else 'log10'
            lower, upper = (signif(b, 3) for b in self.bounds[feature])
            trans_func = str(self.transforms[feature]) if self.transforms is not None else None
            row = f"  - {feature} ∈ [{lower}, {upper}] ({scale}, {trans_func})"
            rows.append(row)
        normed = False if self.normalized is None else self.normalized
        rows.append(f"Features normalized? {normed}")
        rows.append(f"Features split? {self.is_split}")
        if self.is_split:
            rows[-1] += f", test size: {100*np.round(self.test_size, 2)}%"
        return "\n".join(rows)

    def split(self, test_size=0.1, random_state=None):
        """
        Make a test/train split. This resets the state of the object
        to initialization (any scale_feature transforms will be reset).
        """
        self.test_size = test_size
        if random_state is None:
            random_state = random_seed()
        dat = train_test_split(self.X, self.y, test_size=test_size,
                               random_state=random_state)
        self.split_random_state = random_state
        Xtrn, Xtst, ytrn, ytst = dat
        self.X_train = Xtrn
        self.X_test = Xtst
        self.y_train = ytrn
        self.y_test = ytst
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
            self.X_test_scaler = StandardScaler().fit(self.X_test)
            self.X_test = self.X_test_scaler.transform(self.X_test)
            self.X_train_scaler = StandardScaler().fit(self.X_train)
            self.X_train = self.X_train_scaler.transform(self.X_train)
            self.normalized = True
        return self

    def save(self, filepath):
        """
        Save the LearnedFunction object at 'filepath.pkl' and 'filepath.h5'.
        Pickle the LearnedFunction object, and the model is saved via that
        object's save method.
        """
        model = self.model # store the reference
        self.model = None
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self, f)
        self.model = model
        if self.model is not None:
            model.save(f"{filepath}.h5")

    @classmethod
    def load(cls, filepath):
        """
        Load the LearnedFunction object at 'filepath.pkl' and 'filepath.h5'.
        """
        import keras
        with open(f"{filepath}.pkl", 'rb') as f:
            obj = pickle.load(f)
        obj.model = keras.models.load_model(f"{filepath}.h5")
        return obj

    def add_model(self, model):
        """
        Add a model (this needs to be trained outside the class).
        """
        self.model = model

    def predict_test(self, **kwargs):
        """
        Predict the test data.
        """
        return self.model.predict(self.X_test, **kwargs).squeeze()


    def check_bounds(self, X, correct_bounds=False):
        out_lowers, out_uppers = [], []
        total = 0
        for i, feature in enumerate(self.features):
            lower, upper = self.bounds[feature]
            log10 = self.logscale[feature]
            if log10:
                lower, upper = 10**lower, 10**upper
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
        X = self.check_bounds(X, correct_bounds)
        if transforms:
            for i, (feature, trans_func) in enumerate(self.transforms.items()):
                if trans_func is not None:
                    X[:, i] = trans_func(X[:, i])
        if scale_input:
            X = self.X_test_scaler.transform(X)
        return self.model.predict(X, **kwargs).squeeze()

    def predict_train(self, **kwargs):
        """
        Predict the training data.
        """
        return self.model.predict(self.X_train, **kwargs).squeeze()

    def domain_grids(self, n, fix_X=None, log10=None):
        """
        Create a grid of values across the domain for all features.
        fix_X is a dict of fixed values for the grids. If a feature
        name is in log10 tuple, it will be log10'd.
        """
        valid_features = set(self.features)
        msg = "'fix_X' has a feature not in X's features"
        assert fix_X is None or all([(k in valid_features) for k in fix_X.keys()]), msg
        msg = "'log10' has a feature not in X's features"
        assert log10 is None or all([(k in valid_features) for k in log10]), msg
        grids = []
        nx = n
        for feature, (lower, upper) in self.bounds.items():
            is_logscale = self.logscale[feature]
            if log10 is not None and feature in log10:
                is_logscale = True
                lower = np.log10(lower)
                upper = np.log10(upper)
            if fix_X is None or feature not in fix_X:
                if isinstance(n, dict):
                    nx = n[feature]
                grid = np.linspace(lower, upper, nx)
            elif feature in fix_X:
                grid = fix_X[feature]
            else:
                assert(False)
            if is_logscale:
                grid = 10**grid
            grids.append(grid)
        return grids

    def predict_grid(self, n, fix_X=None, log10=None, verbose=True):
        """
        Predict a grid of points (useful for visualizing learned function).
        This uses the domain specified by the model.

        Returns:
          - A list of the grid values for each column.
          - A matrix of the mesh grid, flattened into columns (the total number
             of columns.
        """
        domain_grids = self.domain_grids(n, fix_X=fix_X, log10=log10)
        if verbose:
            grid_dims = 'x'.join(map(str, n.values()))
            print(f"making {grid_dims} grid...\t", end='')
        mesh = np.meshgrid(*domain_grids)
        if verbose:
            print("done.")
        mesh_array = np.stack(mesh)
        X_meshcols = np.stack([col.flatten() for col in mesh]).T
        X_meshcols_orig = X_meshcols[:]
        # transform/scale the new mesh
        for feature, col_idx in self.features.items():
            trans_func = self.transforms.get(feature, None)
            if trans_func is not None:
                X_meshcols[:, col_idx] = trans_func(X_meshcols[:, col_idx])
        if self.normalized:
            X_meshcols = self.X_test_scaler.transform(X_meshcols)

        predict = self.model.predict(X_meshcols, verbose=int(verbose)).squeeze()
        return domain_grids, X_meshcols_orig, X_meshcols, predict.reshape(mesh[0].shape)

class LearnedB(object):
    """
    A general class that wraps a learned B function.
    """
    def __init__(self, learned_func, w_grid, t_grid):
        bgs_cols = ('sh', 'mu', 'rf', 'rbp', 'L')
        assert tuple(learned_func.features.keys()) == bgs_cols
        self.func = learned_func
        self.t_w_mesh = None
        self.w_grid = w_grid
        self.t_grid = t_grid
        self._make_grid_predictor(w_grid, t_grid)
        self._dim = (w_grid.shape[0], t_grid.shape[0])

    def _make_grid_predictor(self, w_grid, t_grid):
        func = self.func
        self.tw_mesh = np.array(list(itertools.product(t_grid, w_grid)))

    def predict_across_wtmesh(self, rf, rbp, L):
        # TODO domain checking
        ngrid = self.tw_mesh.shape[0]
        X = np.array([rbp, rf, L]).T
        X = np.repeat(X, ngrid, axis=0)
        Xg = np.repeat(self.tw_mesh, ngrid, axis=0)
        __import__('pdb').set_trace()
        X = np.concatenate((Xg, X), axis=1)
        n = X.shape[0]
        m = []
        t_w_mesh = self.t_w_mesh
        assert t_w_mesh is not None
        i = 0
        for t_w in t_w_mesh:
            print(f"mesh point {i}/{len(t_w_mesh)}")
            i += 1
            Xn = np.concatenate([np.repeat([t_w], n, axis=0), X], axis=1)
            print(t_w, X.shape, Xn.shape)
            m.append(np.log10(self.func.model.predict(Xn)))
        return np.array(m).reshape((n, *self._dim))

    def calc_Bp_chunk_worker(self, args):
        map_positions, seg_mpos, X = args
        Bs = []
        nw, nt = len(self.w_grid), len(self.t_grid)
        nseg = len(seg_mpos)
        max_mpos = dist_to_segment(map_positions[-1], seg_mpos)
        X[:, 3] = np.tile(max_mpos, self.tw_mesh.shape[0])
        idx = X[:, 3] < 0.2
        X = X[idx, :]
        chop_idx = max_mpos < 0.2
        nseg = np.sum(chop_idx)
        seg_mpos = seg_mpos[chop_idx]
        for f in map_positions:
            rf = dist_to_segment(f, seg_mpos)
            # return rf
            # __import__('pdb').set_trace()
            X[:, 3] = np.tile(rf, self.tw_mesh.shape[0])
			# TODO ignores features matrix!
            B = self.func.predict(X, verbose=True).reshape((nw, nt, nseg))
            Bs.append(B.sum(axis=2))
        print('done!')
        return Bs

def calc_Bp_chunk_worker(args):
    import keras
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model_file = "../data/dnn_models/fullbgs.h5"
    model = keras.models.load_model(model_file)

    map_positions, seg_mpos, w_grid, t_grid, tw_mesh, X = args
    Bs = []
    nw, nt = len(w_grid), len(t_grid)
    nseg = len(seg_mpos)
    max_mpos = dist_to_segment(map_positions[-1], seg_mpos)
    X[:, 3] = np.tile(max_mpos, tw_mesh.shape[0])
    idx = X[:, 3] < 0.2
    X = X[idx, :]
    chop_idx = max_mpos < 0.2
    nseg = np.sum(chop_idx)
    seg_mpos = seg_mpos[chop_idx]
    for f in map_positions:
        rf = dist_to_segment(f, seg_mpos)
        # return rf
        # __import__('pdb').set_trace()
        X[:, 3] = np.log10(np.tile(rf, tw_mesh.shape[0]))
        # TODO ignores features matrix!
        B = model.predict(X, verbose=True).reshape((nw, nt, nseg))
        Bs.append(B.sum(axis=2))
    print('done!')
    return Bs
