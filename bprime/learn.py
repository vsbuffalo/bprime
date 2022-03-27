import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from bprime.utils import signif, index_cols

class LearnedFunction(object):
    def __init__(self, X, y, domain):
        assert(len(domain) == X.shape[1])
        assert(X.shape[0] == y.shape[0])
        self.X = X
        self.y = y
        self.features = {}
        self.bounds = {}
        self.logscale = {}
        self.test_size = None
        self.normalized = None
        self.transform = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_test_orig = None
        self.X_train_orig = Noneself.y_train
        self.y_test_orig = None
        self.y_train_orig = None
        i = 0
        # parse domains
        # note: not doing anythign with type currently
        for feature, params in domain.items():
            if isinstance(params, dict):
                attrs = 'lower', 'upper', 'log10', 'type'
                lower, upper, log10, _ = [params[k] for k in attrs]
            elif isinstance(params, tuple):
                lower, upper, log10, _ = params
            else:
                raise ValueError("params must be dictionary or tuple")
            self.features[feature] = i
            i += 1
            self.bounds[feature] = (lower, upper)
            assert(isinstance(log10, bool))
            self.logscale[feature] = log10

    def col_indexer(self):
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
        dat = train_test_split(self.X, self.y, test_size=test_size,
                               random_state=random_state)
        Xtrn, Xtst, ytrn, ytst = dat
        self.X_train = Xtrn
        self.X_test = Xtst
        self.y_train = ytrn
        self.y_test = ytst
        self.normalized = False
        self.transform = {f: None for f in self.features}
        return self

    def scale_features(self, normalize=True, transforms=None):
        """
        Normalize (center and scale) features, optionally applying
        a feature transform beforehand. This uses sklearn's StandardScaler
        so inverse transforms are easy.

        normalize: boolean whether to center and scale
        feature_transforms: dictionary of feature, transform function pairs
        """
        if not self.is_split:
            raise ValueError("X, y must be split first")
        if self.normalized or not all(x is None for x in self.transform.values()):
            raise ValueError("X already transformed!")
        if not all(t in self.features for t in transforms.keys()):
            raise ValueError("'transforms' dict has key not in features")
        self.X_test_orig = self.X_test
        self.X_train_orig = self.X_train
        self.y_test_orig = self.y_test
        self.y_train_orig = self.y_train
        for feature, col_idx in self.features.items():
            trans_func = transforms.get(feature, None)
            if trans_func is not None:
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
        TODO: could have an option to put this through the scaling inverse
        function.
        """
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
             of columsn
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



