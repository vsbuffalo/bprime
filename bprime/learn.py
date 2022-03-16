import numpy as np
from sklearn.preprocessing import StandardScaler

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
        self.transformed = None

        i = 0
        for feature, (lower, upper, log10, type) in domain.items():
            self.features[feature] = i
            i += 1
            self.bounds[feature] = (lower, upper)
            assert(isinstance(log10, bool))
            self.logscale[feature] = log10

    def split(self, test_size=0.1):
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

    def scale_features(self, normalize=True, feature_transforms=None):
        """
        Normalize (center and scale) features, optionally applying
        a feature transform beforehand. This uses sklearn's StandardScaler
        so inverse transforms are easy.

        normalize: boolean whether to center and scale
        feature_transforms: dictionary of feature, transform function pairs
        """
        for feature, col_idx in self.features.items():
            trans_func = feature_transforms.get(feature, None)
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

    def add_model(self, model):
        """
        Add a model (this needs to be trained outside the class).
        """
        self.model = model

    def predict_test(self):
        return self.model.predict(self.X_test).squeeze()

    def predict_train(self):
        return self.model.predict(self.X_train).squeeze()

    def domain_grids(self, n):
        """
        TODO: could have an option to put this through the scaling inverse
        function.
        """
        grids = []
        for feature, (lower, upper) in self.bounds.items():
            is_logscale = self.logscale[feature]
            grid = np.linspace(lower, upper, n)
            if is_logscale:
                grid = 10**grid
            grids.append(grid)
        return grids

    def predict_grid(self, n):
        """
        Predict a grid of points (useful for visualizing learned function).
        This uses the domain specified by the model.
        """
        domain_grids = self.domain_grids(n)
        mesh = np.meshgrid(*domain_grids)
        mesh_array = np.stack(mesh)
        X_mesh = np.stack([col.flatten() for col in mesh]).T
        # transform/scale the new mesh
        for feature, col_idx in self.feature.items():
            trans_func = self.transform.get(feature, None)
            if trans_func is not None:
                X_mesh[:, col_idx] = trans_func(X_mesh[:, col_idx])
        if self.normalized:
            X_mesh = self.X_test_scaler.transform(X_mesh)

        predict = self.model.predict(X_mesh).squeeze()
        return domain_grids, X_mesh, predict



