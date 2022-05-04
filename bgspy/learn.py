## learn.py -- classes, etc for DNN learned B functions

import os
import json
import itertools
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from tensorflow import keras

from bgspy.utils import signif, index_cols, dist_to_segment
from bgspy.sim_utils import random_seed
from bgspy.theory import bgs_segment, bgs_rec, BGS_MODEL_PARAMS, BGS_MODEL_FUNCS
from bgspy.loss import get_loss_func


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
        # these are the X_test/X_train values *before* transforms/scales
        # have been applied
        self.X_test_raw = None
        self.X_train_raw = None
        # raw y's not needed currently, no y transforms used
        # self.y_test_raw = None
        # self.y_train_raw = None

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
            rows[-1] += f", test size: {100*np.round(self.test_size, 2)}% (n={self.X_test.shape[0]:,})"

        rows.append(f"Total size: {self.X.shape[0]:,}")
        return "\n".join(rows)

    def _protect(self):
        protectables = ['X', 'y', 'X_test', 'X_train', 'y_test', 'y_train',
                        'X_test_raw', 'X_train_raw']
        for protectable in protectables:
            val = getattr(self, protectable)
            if val is not None:
                val.setflags(write=False)

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
        # store the pre-transformed daata, which is useful for figures, etc
        self.X_test_raw = np.copy(self.X_test)
        self.X_train_raw = np.copy(self.X_train)
        self._protect()
        self.normalized = False
        self.transforms = {f: None for f in self.features}
        return self

    def scale_features(self, normalize=True, transforms='match'):
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
        # if we've done the transforms once, save them to the object
        # and protect them
        self.X_test = X_test
        self.X_train = X_train
        self._protect()
        return self

    def transform_feature_to_match(self, feature, x):
        """
        For linear input feature x, match the transformations/scalers
        in applied to the training data.
        """
        if feature in self.trans_func:
            x = self.trans_func[feature](x)
        if self.normalized:
            mean = self.func.scaler.mean_
            scale = self.func.scaler.scale_
            x = (x - mean) / scale
        return x

    def transform_X_to_match(self, X, correct_bounds=False):
        """
        For linear input matrix X, match the transformations/scalers
        in applied to the training data X.
        """
        X = np.copy(X)
        X = self.check_bounds(X, correct_bounds)
        if transforms:
            for i, (feature, trans_func) in enumerate(self.transforms.items()):
                if trans_func is not None:
                    X[:, i] = trans_func(X[:, i])
        if scale_input:
            X = self.scaler.transform(X)


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


    def check_bounds(self, X, correct_bounds=False):
        # mutates in place!
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
            X = transform_X_to_match(X, correct_bounds=False)
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
        self.genome = genome
        self.func = None
        self._predict = None
        self._X_test_hash = None
        model = model if not model.startswith('bgs_') else model.replace('bgs_', '')
        self.bgs_model = model
        self.params = params
        self.w_grid = w_grid
        self.t_grid = t_grid

    @property
    def dim(self):
        return self.w_grid.shape[0], self.t_grid.shape[0]

    @property
    def wt_mesh(self):
        return np.array(list(itertools.product(self.w_grid, self.t_grid)))

    def theory_B(self, X=None):
        """
        Compute the BGS theory given the right function ('segment' or 'rec')
        on the feature matrix X. E.g. use for X_test_raw or using
        meshgrids.
        """
        if X is None:
            X = self.func.X_test_raw
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
        if 'N' in kwargs:
            kwargs.pop('N') # not needed for theory
        return self.bgs_model(**kwargs)


    def predict_test(self):
        """
        Predict X_test_raw, caching the results (invalidation
        based on hash of X_test).
        """
        X_test_hash = hash(self.func.X_test_raw.data.tobytes())
        if self._predict is None or X_test_hash != self._X_test_hash:
            self._X_test_hash = X_test_hash
            predict = self.func.predict_test()
            self._predict = predict
        return self._predict

    def predict_datum(self, **kwargs):
        msg = f"kwargs: {', '.join(kwargs.keys())}, features: {', '.join(self.func.features.keys())}"
        assert set(kwargs.keys()) == set(self.func.features.keys()), msg
        # make a prediction matrix
        X = np.array([[kwargs[k] for k in self.func.features.keys()]])
        return float(self.func.predict(X))

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
            # numpy write flags are not preserved when pickling objects(!), so
            # need to fix that here.
            obj._protect()
        model_path = f"{filepath}.h5"
        if load_model and os.path.exists(model_path):
            obj.func.model = keras.models.load_model(model_path)
        return obj

    def train_model(self, filepath):
        pass

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

        The columns are the features of the B' training data match those in
        the BGS_MODEL_PARAMS['bgs_segment'] or self.params
               0,    1,    2,     3,   4
            'mu',  'sh', 'L', 'rbp', 'rf'

        The last column is the recombination fraction away from the
        focal site and needs to change each time the position changes.
        This is done externally; it's set to zero here.

        Note, each output needs to be log10 transformed as according to the
        LearnedB.func.trans_func, then scaled using LearnedFunction.func.scaler
        """
        assert self.bgs_model == 'segment'
        # size of mesh; all pairwise combinations
        n = self.wt_mesh.shape[0]

        # get number of segments
        segments = self.genome.segments
        chrom_idx = {c: segments.index[c] for c in seqlens}
        nsegs = len(self.chrom_idx[chrom])

        X = np.zeros((nsegs*n, 5))
        # repeat the w/t element for each block of segments
        X[:, 0:2] = np.repeat(self.wt_mesh, nsegs, axis=0)

        # add in the the segment annotation data
        X[:, 2] = np.tile(self.chrom_seg_L[chrom], n)
        X[:, 3] = np.tile(self.chrom_seg_rbp[chrom], n)
        X = self.func.transform_X_to_match(X)
        X[:, 4] = np.nan # NaN out rf column, just in case
        return X

    def transform(self, *arg, **kwargs):
        return self.func.scaler.transform(*args, **kwargs)

    def write_BpX_chunks(self, dir, **kwargs):
        """
        Write the B' X chunks (all the features X for a chromosome
        needed to predict B') to a directory for distributed prediction.
        There are two types of tiles:

            1. 'chrom_data_{chrom}.npy'
            2. 'chunk_data_{id}.npy'

        The chromosome data contains the feature matrix X, all transformed


        Write the X chromosome segment and the focal position data.

        The columns are mu,s,L,rbp,rh -- note that rf is blank,
        as this is filled in depending on what the focal position is.
        """
        for chrom, X in self.Xs.items():
            np.save(os.path.join(dir, f"chrom_data_{chrom}.npy", X))

        for i, (chrom, X_chunk) in enumerate(self.focal_positions(**kwargs)):
            np.save(os.path.join(dir, f"chunk_data_{chrom}.npy", X_chunk))


    def focal_positions(self, step=1000, nchunks=100, progress=True):
        """
        Get the focal positions for each position that B is calculated at.

        nchunks is how many chunks across the genome to break this up
        to (for parallel processing)
        """
        chunks = MapPosChunkIterator(self.genome, self.w_grid, self.t_grid,
                                         step=step, nchunks=nchunks)
        if progress:
            outer_progress_bar = tq.tqdm(total=self.total)
        # inner_progress_bar = tq.tqdm(leave=True)
        while True:
            next_chunk = next(chunks.mpos_iter)
            # get the next chunk of map positions to process
            chrom, mpos_chunk = next_chunk
            # chromosome segment positions
            chrom_seg_mpos = chunks.chrom_seg_mpos[chrom]
            # focal map positions
            map_positions = mpos_chunk
            for f in map_positions:
                # compute the rec frac to each segment's start or end
                rf = dist_to_segment(f, chrom_seg_mpos)
                # place rf in X, log10'ing it since that's what we train on
                # and tile it to repeat
                Xrf_col = np.tile(rf, self.wt_mesh.shape[0])
                Xrf_col = self.func.transform_feature_to_match('rf', Xrf_col)
                assert np.all(np.isfinite(Xrf_col)), "rec frac column not all finite!"
                yield chrom, Xrf_col
            if progress:
                outer_progress_bar.update(1)
            # inner_progress_bar.reset()

