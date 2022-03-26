import numpy as np
from bprime.utils import signif

class UniformSampler(object):
    def __init__(self, ranges, total=None, seed=None, add_seed=True,
                 seed_max=2**32, signif_digits=4):
        assert isinstance(total, int) or total is None
        assert isinstance(seed, int) or seed is None
        assert isinstance(ranges, dict), "'ranges' must be a dict"
        self.seed = seed
        self.add_seed = add_seed
        self.seed_max = seed_max
        self.rng = np.random.default_rng(seed)
        self.ranges = ranges
        self.total = total
        self.samples_remaining = total
        self.samples = []
        self.signif_digits = signif_digits

    def __iter__(self):
        return self

    @property
    def __len__(self):
        return len(self.samples)

    def as_matrix(self, return_cols=True):
        assert len(self.samples) > 0, "No samples have been taken."
        mat = np.array([[row[k] for k in self.ranges.keys()] for row
                        in self.samples])
        if not return_cols:
            return mat
        return mat, list(self.ranges.keys())

    def __next__(self):
        if self.total is not None and self.samples_remaining == 0:
            raise StopIteration
        assert(self.samples_remaining >= 0)
        param = {}
        for key, (lower, upper, log10) in self.ranges.items():
            val_type = type(lower)
            assert type(lower) is type(upper)
            if lower == upper:
                # fixed parameter
                val = val_type(lower)
                if log10:
                    val = 10**lower
                param[key] = val
                continue
            # log10 or floats get uniform float
            if val_type is float or log10:
                sample = self.rng.uniform(lower, upper)
            elif val_type is int and not log10:
                # if the type is int and not log10, discrete uniform
                sample = self.rng.integers(lower, upper)
            else:
                raise ValueError("val_type must be float or int")
            if log10:
                sample = signif(10**sample, self.signif_digits)
            if val_type is int:
                sample = int(sample)
            else:
                sample = signif(sample, self.signif_digits)
            param[key] = sample
        if self.add_seed:
            seed = self.rng.integers(0, self.seed_max)
            param['seed'] = seed

        #assert(len(param) == len(self.ranges.keys()) + int(self.add_seed))

        self.samples_remaining -= 1
        self.samples.append(param)
        return param

    def __repr__(self):
        rows = [f"UniformSampler with {self.samples_remaining}/{self.total} samples remaining, seed={self.seed}"]
        for key, (lower, upper, log10) in self.ranges.items():
            val_type = type(lower)
            assert type(lower) is type(upper)
            scale = 'linear'
            if log10:
               scale = 'log10'
            type_str = {float: 'float', int: 'int'}[val_type]
            if isinstance(lower, float) or isinstance(upper, float):
                lower, upper = [signif(lower, self.signif_digits), signif(upper, self.signif_digits)]
            row = f"{key} ∈ [{lower}, {upper}] ({scale}, {type_str})"
            rows.append(row)
        return "\n".join(rows)


