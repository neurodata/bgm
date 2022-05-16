#%%

import numpy as np
from joblib import Parallel, delayed


def random_function(rng):
    return rng.integers(1000)


rngs = [np.random.default_rng(82), np.random.default_rng(83)]
par = Parallel(n_jobs=2)
par(delayed(random_function)(rng) for rng in rngs)
