## learn.py -- classes, etc for DNN learned B functions

import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
        self.transform = None    # the dict of transforms for parameters

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
            trans_func = str(self.transform[feature]) if self.transform is not None else None
            row = f"  - {feature} ∈ [{lower}, {upper}] ({scale}, {trans_func})"
            rows.append(row)
        normed = False if self.normalized is None else self.normalized
        rows.append(f"Features normalized? {normed}")
        rows.append(f"Features split? {self.is_split}")
        if self.is_split:
            rows[-1] += f", test size: {100*np.round(self.test_size, 2)}%"
        return "\n".join(rows)

    def split(self, test_size=0.1, random_state=None):
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
        self.transform = {f: None for f in self.features}
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
        if self.normalized or not all(x is None for x in self.transform.values()):
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
                self.transform[feature] = trans_func
        if normalize:
            self.X_test_scaler = StandardScaler().fit(self.X_test)
            self.X_test = self.X_test_scaler.transform(self.X_test)
            self.X_train_scaler = StandardScaler().fit(self.X_train)
            self.X_train = self.X_train_scaler.transform(self.X_train)
            self.normalized = True
        return self

    def add_model(self, model):
        """
        Add a model (this needs to be trained outside the class).
        """
        self.model = model

    def predict_test(self):
        return self.model.predict(self.X_test).squeeze()

    def predict(self, X, scale_input=True):
        if scale_input:
            X = self.X_test_scaler.transform(X)
        return self.model.predict(X).squeeze()

    def predict_train(self):
        return self.model.predict(self.X_train).squeeze()

    def domain_grids(self, n, fix_X=None, log10=None):
        """
        Create a grid of values across the domain for all features.
        fix_X is a dict of fixed values for the grids. If a feature
        name is in log10 tuple, it will be log10'd.
        """
        valid_features = set(self.features)
        print(valid_features)
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

    def predict_grid(self, n, fix_X=None, log10=None):
        """
        Predict a grid of points (useful for visualizing learned function).
        This uses the domain specified by the model.

        Returns:
          - A list of the grid values for each column.
          - A matrix of the mesh grid, flattened into columns (the total number
             of columns.
        """
        domain_grids = self.domain_grids(n, fix_X=fix_X, log10=log10)
        mesh = np.meshgrid(*domain_grids)
        mesh_array = np.stack(mesh)
        X_meshcols = np.stack([col.flatten() for col in mesh]).T
        X_meshcols_orig = X_meshcols[:]
        # transform/scale the new mesh
        for feature, col_idx in self.features.items():
            trans_func = self.transform.get(feature, None)
            if trans_func is not None:
                X_meshcols[:, col_idx] = trans_func(X_meshcols[:, col_idx])
        if self.normalized:
            X_meshcols = self.X_test_scaler.transform(X_meshcols)

        predict = self.model.predict(X_meshcols).squeeze()
        return domain_grids, X_meshcols_orig, X_meshcols, predict.reshape(mesh[0].shape)

class LearnedB(object):
    """
    A general class that wraps a learned B function.
    """
    def __init__(self, learned_func, w_grid, t_grid):
        bgs_cols = ('sh', 'mu', 'rbp', 'rf', 'L')
        assert tuple(learned_func.features.keys()) == bgs_cols
        self.func = learned_func
        self.t_w_mesh = None
        self.w_grid = w_grid
        self.t_grid = t_grid
        self._make_grid_predictor(w_grid, t_grid)
        self._dim = (w_grid.shape[0], t_grid.shape[0])

    def _make_grid_predictor(self, w_grid, t_grid):
        func = self.func
        self.t_w_mesh = list(itertools.product(10**t_grid, 10**w_grid))

    def predict(self, rf, rbp, L):
        # TODO domain checking
        X = np.array([rf, rbp, L]).T
        n = X.shape[0]
        m = []
        t_w_mesh = self.t_w_mesh
        assert t_w_mesh is not None
        for t_w in t_w_mesh:
            Xn = np.concatenate([np.repeat([t_w], n, axis=0), X], axis=1)
            m.append(np.log10(self.func.model.predict(Xn)))
        return np.array(m).reshape((n, *self._dim))

    def calc_Bp_chunk_worker(self, args):
        map_positions, seg_mpos, seg_rbp, seg_L, features_matrix = args
        Bs = []
        for f in map_positions:
            rf = dist_to_segment(f, seg_mpos)
            # TODO ignores features_matrix!
            B = self.predict(rf, seg_rbp, seg_L)
            Bs.append(B)
        return Bs


