import numpy as np
from numba import jit
from ..match import BaseMatchSolver
from scipy.optimize import linear_sum_assignment


class BisectedGraphMatchSolver(BaseMatchSolver):
    def __init__(
        self,
        adjacency,
        indices1,
        indices2,
        similarity=None,
        partial_match=None,
        rng=None,
        init="barycenter",
        verbose=False,
        shuffle_input=True,
        maximize=True,
        maxiter=30,
        tol=0.01,
    ):
        # TODO more input checking
        super().__init__(
            rng=rng,
            init=init,
            verbose=verbose,
            shuffle_input=shuffle_input,
            maximize=maximize,
            maxiter=maxiter,
            tol=tol,
        )
        # TODO input validation
        # TODO seeds
        # A, B, partial_match = _common_input_validation(A, B, partial_match)

        # TODO similarity
        # if S is None:
        #     S = np.zeros((A.shape[0], B.shape[1]))
        # S = np.atleast_2d(S)

        # TODO padding
        A = adjacency[np.ix_(indices1, indices1)]
        B = adjacency[np.ix_(indices2, indices2)]
        AB = adjacency[np.ix_(indices1, indices2)]
        BA = adjacency[np.ix_(indices2, indices1)]

        if init == "barycenter":
            init = 1.0

        self.A = A.copy()
        self.B = B.copy()
        self.AB = AB.copy()
        self.BA = BA.copy()

        # self.S = S
        # self.partial_match = partial_match

        # self.n = .shape[0]  # number of vertices in graphs
        # self.n_seeds = partial_match.shape[0]  # number of seeds
        self.n_unseed = A.shape[0]

    # TODO
    def check_outlier_cases(self):
        pass

    # side_perm = self.rng.permutation(self.n_unseed, 2 * self.n_unseed)
    # perm = np.concatenate((np.arange(self.n_unseed), side_perm))
    # TODO
    def set_reference_frame(self):
        if self.shuffle_input:
            perm = self.rng.permutation(self.n_unseed)

            self._reverse_permutation = np.argsort(perm)

            self.B = self.B[perm][:, perm]
            self.AB = self.AB[:, perm]  # permute columns only
            self.BA = self.BA[perm]  # permute rows only
            # TODO permute seeds and anything else that could be added
        else:
            self._reverse_permutation = np.arange(self.n_unseed)

    def compute_constant_terms(self):
        # only happens with seeds
        pass

    def compute_step_direction(self, P):
        self.print("Computing step direction")
        grad_fp = self.compute_gradient(P)
        Q, permutation = self.solve_assignment(grad_fp)
        return Q, permutation

    def solve_assignment(self, grad_fp):
        self.print("Solving assignment problem")
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, permutation = linear_sum_assignment(grad_fp, maximize=self.maximize)
        Q = np.eye(self.n_unseed)[permutation]
        return Q, permutation

    # permutation is here as a dummy for now
    def compute_step_size(self, P, Q, permutation):
        self.print("Computing step size")
        a, b = _compute_coefficients(P, Q, self.A, self.B, self.AB, self.BA)
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * self.obj_func_scalar])
        return alpha

    def compute_gradient(self, P):
        self.print("Computing gradient")
        gradient = _compute_gradient(P, self.A, self.B, self.AB, self.BA)
        return gradient

    def compute_score(*args):
        return 0


@jit(nopython=True)
def _compute_gradient(P, A, B, AB, BA):
    return A @ P @ B.T + A.T @ P @ B + AB @ P.T @ BA.T + BA.T @ P.T @ AB


@jit(nopython=True)
def _compute_coefficients(P, Q, A, B, AB, BA):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares
    a_cross = np.trace(AB.T @ R @ BA @ R)
    b_cross = np.trace(AB.T @ R @ BA @ Q) + np.trace(AB.T @ Q @ BA @ R)
    a_intra = np.trace(A @ R @ B.T @ R.T)
    b_intra = np.trace(A @ Q @ B.T @ R.T + A @ R @ B.T @ Q.T)

    a = a_cross + a_intra
    b = b_cross + b_intra

    return a, b
