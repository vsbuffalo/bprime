import copy
import pytest
import numpy as np
from bgspy.learn import LearnedB, LearnedFunction
from bgspy.theory import bgs_segment

# this is a test case from a random chunk of chr10 from predictions
# fed into the ML B approach. Columns are mu, s, L, rbp, rf --
# the same order as expected (specified in bgspy.theory.BGS_MODEL_PARAMS
Xp = np.array([[1.0000e-09, 1.0000e-03, 3.3400e+02, 2.8873e-10, 4.7425e-04],
               [1.0000e-09, 1.0000e-04, 8.0900e+02, 1.6206e-08, 5.1120e-02],
               [1.0000e-08, 1.0000e-03, 1.0000e+03, 1.8624e-09, 1.2862e-02],
               [1.0000e-08, 1.0000e-02, 1.0000e+03, 2.0333e-08, 6.3086e-02],
               [1.0000e-08, 1.0000e-04, 1.9500e+02, 4.8172e-10, 1.8595e-02],
               [1.0000e-08, 1.0000e-02, 1.0000e+03, 3.3601e-08, 4.6281e-02],
               [1.0000e-09, 1.0000e-04, 8.4100e+02, 5.5386e-09, 2.9605e-02],
               [1.0000e-09, 1.0000e-02, 4.4400e+02, 1.7687e-10, 4.7748e-02],
               [1.0000e-09, 2.0000e-02, 2.5500e+02, 6.1183e-09, 5.3264e-02],
               [1.0000e-08, 1.0000e-03, 1.0000e+03, 1.5244e-10, 1.1342e-02],
               [1.0000e-07, 1.0000e-02, 6.7600e+02, 9.1097e-09, 1.0493e-02],
               [1.0000e-09, 1.0000e-03, 3.3000e+01, 1.2818e-08, 9.3163e-02],
               [1.0000e-09, 1.0000e-04, 1.0000e+03, 5.4405e-09, 4.2521e-02],
               [1.0000e-08, 2.0000e-02, 2.7700e+02, 1.7699e-08, 3.3569e-02],
               [1.0000e-07, 1.0000e-04, 1.0000e+03, 1.6668e-09, 5.1079e-02],
               [1.0000e-07, 1.0000e-02, 1.6000e+01, 1.8927e-08, 7.3314e-03],
               [1.0000e-08, 1.0000e-02, 1.2900e+02, 1.0219e-10, 9.0873e-02],
               [1.0000e-09, 1.0000e-04, 7.8000e+01, 2.8858e-09, 5.3245e-02],
               [1.0000e-09, 2.0000e-02, 1.9600e+02, 4.9232e-09, 5.3160e-02],
               [1.0000e-09, 1.0000e-04, 8.4000e+01, 1.0000e-10, 5.5641e-02],
               [1.0000e-08, 1.0000e-04, 1.0000e+03, 4.2427e-09, 2.7192e-02],
               [1.0000e-09, 1.0000e-02, 1.0000e+03, 1.2653e-08, 7.5517e-02],
               [1.0000e-09, 2.0000e-02, 4.6000e+01, 7.7780e-09, 6.0303e-02],
               [1.0000e-07, 1.0000e-04, 1.7500e+02, 4.0149e-10, 1.2043e-02],
               [1.0000e-09, 1.0000e-02, 4.5800e+02, 1.0154e-08, 9.5790e-02],
               [1.0000e-09, 1.0000e-02, 4.2300e+02, 3.3984e-09, 9.3825e-02],
               [1.0000e-08, 1.0000e-03, 3.4800e+02, 1.2185e-09, 2.9772e-02],
               [1.0000e-08, 1.0000e-03, 2.8500e+02, 1.0000e-07, 4.6131e-02],
               [1.0000e-09, 1.0000e-04, 1.6300e+02, 7.6761e-10, 2.3491e-02],
               [1.0000e-08, 1.0000e-02, 2.3200e+02, 2.0136e-09, 2.0534e-02]])

## fixtures
@pytest.fixture
def fake_bgs_segment_domains():
     "This mimics a parameter JSON file"
     params = dict(mu=dict(lower=-9, upper=-7, log10=True),
                   sh=dict(lower=-5, upper=-1, log10=True),
                   L=dict(lower=1, upper=1000, log10=False),
                   rbp=dict(lower=-8, upper=-7, log10=True),
                   rf=dict(lower=-9, upper=-8, log10=True))
     return params

@pytest.fixture
def fake_training_data(fake_bgs_segment_domains):
    params = fake_bgs_segment_domains
    rng = np.random.default_rng(1)
    n = 100
    X = np.empty((n, len(params)))
    data = dict()
    for i, feature in enumerate(params):
         is_log10 = params[feature]['log10']
         lower, upper = params[feature]['lower'], params[feature]['upper']
         val = rng.uniform(lower, upper)
         if is_log10:
             val = 10**val
         X[:, i] = val
         data[feature] = val
    # create y
    for i in range(n):
        y = rng.exponential(bgs_segment(**data), n)
    return X, y

@pytest.fixture
def learned_func(fake_bgs_segment_domains, fake_training_data):
    domain = fake_bgs_segment_domains
    X, y = fake_training_data
    return LearnedFunction(X, y, domain)

class Test_LearnedFunction:
    def test_init(self, fake_bgs_segment_domains, fake_training_data):
        domain = fake_bgs_segment_domains
        X, y = fake_training_data
        func = LearnedFunction(X, y, domain)
        expected_bounds = {f: (v['lower'], v['upper']) for f, v in domain.items()}
        assert func.bounds == expected_bounds, "bounds wrong"
        expected_logscale = {f: v['log10'] for f, v in domain.items()}
        assert func.logscale == expected_logscale
        assert func.features == {'mu': 0, 'sh': 1, 'L': 2, 'rbp': 3, 'rf': 4}


class Test_LearnedB:
    def test_is_valid(self, learned_func):
        func = learned_func
        b = LearnedB(model='segment')
        # test if exception is raised if no func set
        with pytest.raises(AssertionError):
            b.is_valid_learnedfunc()


        # test if exception is raised if parameters don't have right order
        func_wrong = copy.deepcopy(func)
        wrong_order = ('sh', 'mu', 'L', 'rbp', 'rf')
        func_wrong.features = {f: func.features[f] for f in wrong_order}
        b.func = func_wrong
        with pytest.raises(AssertionError) as excinfo:
            b.is_valid_learnedfunc()
            assert "parameter order of" in str(excinfo.value)

        # now, a good one
        b.func = func
        assert b.is_valid_learnedfunc()


    def test_theory_B(self, learned_func):
        func = learned_func
        b = LearnedB(model='segment')
        with pytest.raises(AssertionError):
            b.is_valid_learnedfunc()

        b.func = func
        actual = b.theory_B(X=Xp)
        desired = bgs_segment(*Xp.T)
        np.testing.assert_allclose(actual, desired)

        actual = b.theory_B(X=Xp)
        desired = []
        for i in range(Xp.shape[0]):
            desired.append(bgs_segment(*Xp[i, :]))
        np.testing.assert_allclose(actual, desired)


