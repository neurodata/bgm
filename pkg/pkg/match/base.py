#%%
import time
from functools import wraps

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.utils import check_random_state


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        sec = te - ts
        output = f"Function {f.__name__} took {sec:.3f} seconds."
        print(output)
        return result

    return wrap


class BaseMatchSolver:
    def __init__(
        self,
        rng=None,
        init="barycenter",
        verbose=False,
        shuffle_input=True,
        maximize=True,
        maxiter=30,
        tol=0.01,
    ):
        if init == "barycenter":
            init = 1.0

        self.rng = check_random_state(rng)
        self.init = init
        self.verbose = verbose
        self.shuffle_input = shuffle_input
        self.maximize = maximize
        self.maxiter = maxiter
        self.tol = tol

        if maximize:
            self.obj_func_scalar = -1
        else:
            self.obj_func_scalar = 1

    def status(self):
        if hasattr(self, "n_iter"):
            return f"[Iteration: {self.n_iter}]"
        else:
            return "[Pre-loop]"

    def print(self, msg):
        if self.verbose:
            status = self.status()
            print(status + " " + msg)

    def check_converged(self, P, P_new):
        return np.linalg.norm(P - P_new) / np.sqrt(self.n_unseed) < self.tol

    def solve(self):
        self.check_outlier_cases()
        self.set_reference_frame()

        P = self.initialize()
        self.compute_constant_terms()
        for n_iter in range(self.maxiter):
            self.n_iter = n_iter

            Q, permutation = self.compute_step_direction(P)
            alpha = self.compute_step_size(P, Q, permutation)

            # take a step in this direction
            P_new = alpha * P + (1 - alpha) * Q

            if self.check_converged(P, P_new):
                self.converged = True
                break
            P = P_new

        self.finalize(P)

    def initialize(self):
        self.print("Initializing")
        if isinstance(self.init, float):
            n_unseed = self.n_unseed
            rng = self.rng
            J = np.ones((n_unseed, n_unseed)) / n_unseed
            # DO linear combo from barycenter
            K = rng.uniform(size=(n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            P = J * self.init + K * (1 - self.init)  # TODO check how defined in paper
        elif isinstance(self.init, np.ndarray):
            raise NotImplementedError()
            # TODO fix below
            # P0 = np.atleast_2d(P0)
            # _check_init_input(P0, n_unseed)
            # invert_inds = np.argsort(nonseed_B)
            # perm_nonseed_B = np.argsort(invert_inds)
            # P = P0[:, perm_nonseed_B]

        self.converged = False
        return P

    def finalize(self, P):
        self.print("Finalizing permutation")
        P = self.unset_reference_frame(P)
        self.P_final_ = P

        _, permutation = linear_sum_assignment(P, maximize=self.maximize)
        self.permutation_ = permutation

        score = self.compute_score(permutation)
        self.score_ = score

    def unset_reference_frame(self, P):
        reverse_perm = self._reverse_permutation
        P = P[:, reverse_perm]
        return P


# REF: https://github.com/microsoft/graspologic/blob/dev/graspologic/match/qap.py
def _doubly_stochastic(P: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
            np.abs(P_eps.sum(axis=0) - 1) < tol
        ).all():
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps
