#%%
import time
from functools import wraps

import numpy as np
from sklearn.utils import check_random_state
from graspologic.match.qap import _doubly_stochastic  # TODO fix
from numba import jit
from scipy.optimize import linear_sum_assignment


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

        self.P_final_ = P

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
            # TODO fix below
            P0 = np.atleast_2d(P0)
            _check_init_input(P0, n_unseed)
            invert_inds = np.argsort(nonseed_B)
            perm_nonseed_B = np.argsort(invert_inds)
            P = P0[:, perm_nonseed_B]

        self.converged = False
        return P

    def finalize(self, P):
        self.print("Finalizing permutation")
        _, permutation = linear_sum_assignment(self.P_final_, maximize=self.maximize)
        self.permutation_ = permutation
        self.unset_reference_frame()

        score = self.compute_score(self.permutation_)
        self.score_ = score


class FAQSolver(BaseMatchSolver):
    def __init__(
        self,
        A,
        B,
        S=None,
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
        A, B, partial_match = _common_input_validation(A, B, partial_match)
        if S is None:
            S = np.zeros((A.shape[0], B.shape[1]))
        S = np.atleast_2d(S)

        if init == "barycenter":
            init = 1.0

        self.A = A
        self.B = B
        self.S = S
        self.partial_match = partial_match

        self.n = A.shape[0]  # number of vertices in graphs
        self.n_seeds = partial_match.shape[0]  # number of seeds
        self.n_unseed = self.n - self.n_seeds

    def set_reference_frame(self):
        """Deals with seeds and random shuffle permutations before optimization.

        Note that random shuffle permutations before optimization matter due to how
        linear_sum_assignment solves ties based on the order of input matrices.
        """
        self.print("Setting reference frame")

        n = self.n
        partial_match = self.partial_match
        shuffle_input = self.shuffle_input
        rng = self.rng
        nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])
        perm_S = np.copy(nonseed_B)
        if shuffle_input:
            nonseed_B = rng.permutation(nonseed_B)
            self.nonseed_B = nonseed_B
            # shuffle_input to avoid results from inputs that were already matched

        nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
        perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
        perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

        S = self.S
        A = self.A
        B = self.B
        n_seeds = self.n_seeds
        S = S[:, perm_B]

        # definitions according to Seeded Graph Matching [2].
        A11, A12, A21, A22 = _split_matrix(A[perm_A][:, perm_A], n_seeds)
        B11, B12, B21, B22 = _split_matrix(B[perm_B][:, perm_B], n_seeds)
        S22 = S[perm_S, n_seeds:]

        self.perm_A = perm_A
        self.perm_B = perm_B
        self.perm_S = perm_S

        self.A11 = A11
        self.A12 = A12
        self.A21 = A21
        self.A22 = A22

        self.B11 = B11
        self.B12 = B12
        self.B21 = B21
        self.B22 = B22
        self.S22 = S22

    def unset_reference_frame(self):
        self.print("Unsetting reference frame")
        perm = np.concatenate(
            (np.arange(self.n_seeds), self.permutation_ + self.n_seeds)
        )
        unshuffled_perm = np.zeros(self.n, dtype=int)
        unshuffled_perm[self.perm_A] = self.perm_B[perm]
        self.permutation_ = unshuffled_perm

        P_unseed = self.P_final_
        P_final = np.zeros((self.n, self.n))
        # Identity permutation
        P_final[np.arange(self.n_seeds), np.arange(self.n_seeds)] = 1
        # Permutation we fit for the unseeded vertices
        P_final[self.n_seeds :, self.n_seeds :] = P_unseed

        self.P_final_ = self.P_final_

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
            # TODO fix below
            P0 = np.atleast_2d(P0)
            _check_init_input(P0, n_unseed)
            invert_inds = np.argsort(nonseed_B)
            perm_nonseed_B = np.argsort(invert_inds)
            P = P0[:, perm_nonseed_B]

        self.converged = False
        return P

    def compute_gradient(self, P):
        self.print("Computing gradient")
        # [1] Algorithm 1 Line 3 - compute the gradient of f(P) = -tr(APB^tP^t)
        grad_fp = _compute_gradient(P, self.A22, self.B22, self.const_sum)
        return grad_fp

    def solve_assignment(self, grad_fp):
        self.print("Solving assignment problem")
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, permutation = linear_sum_assignment(grad_fp, maximize=self.maximize)
        Q = np.eye(self.n_unseed)[permutation]
        return Q, permutation

    def compute_step_direction(self, P):
        self.print("Computing step direction")
        grad_fp = self.compute_gradient(P)
        Q, permutation = self.solve_assignment(grad_fp)
        return Q, permutation

    def compute_step_size(self, P, Q, permutation=None):
        self.print("Computing step size")
        # [1] Algorithm 1 Line 5 - compute the step size

        if permutation is None:
            a, b = _compute_coefficients(
                P,
                Q,
                self.A21,
                self.B21,
                self.A12,
                self.B12,
                self.A22,
                self.B22,
                self.S22,
            )
        else:
            a, b = _compute_coefficients_permutation(
                P,
                Q,
                self.A21,
                self.B21,
                self.A12,
                self.B12,
                self.A22,
                self.B22,
                self.S22,
                permutation,
            )

        # critical point of ax^2 + bx + c is at x = -d/(2*e)
        # if a * obj_func_scalar > 0, it is a minimum
        # if minimum is not in [0, 1], only endpoints need to be considered
        # print((b + a) * self.obj_func_scalar)
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * self.obj_func_scalar])
        return alpha

    def compute_constant_terms(self):
        self.print("Computing constant terms")
        const_sum = self.A21 @ self.B21.T + self.A12.T @ self.B12 + self.S22
        if isinstance(const_sum, csr_matrix):
            const_sum = const_sum.toarray()
        self.const_sum = const_sum

    def check_outlier_cases(self):
        return 0
        # TODO
        if n == 0 or partial_match.shape[0] == n:
            # Cannot assume partial_match is sorted.
            partial_match = np.row_stack(sorted(partial_match, key=lambda x: x[0]))
            score = _calc_score(A, B, S, partial_match[:, 1])

    def finalize(self, P):
        self.print("Finalizing permutation")
        _, permutation = linear_sum_assignment(self.P_final_, maximize=self.maximize)
        self.permutation_ = permutation
        self.unset_reference_frame()

        score = self.compute_score(self.A, self.B, self.S, self.permutation_)
        self.score_ = score


# NOPYTHON = True


# @jit(nopython=NOPYTHON)
# def _compute_gradient(P, A22, B22, const_sum):
#     return const_sum + A22 @ P @ B22.T + A22.T @ P @ B22


# @jit(nopython=NOPYTHON)
# def _compute_coefficients(P, Q, A21, B21, A12, B12, A22, B22, S22):
#     # [1] Algorithm 1 Line 5 - compute the step size
#     # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
#     # terms as ax**2 + bx + c. c does not affect location of minimum
#     # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
#     # apply where possible for efficiency.
#     R = P - Q
#     b21 = ((R.T @ A21) * B21).sum()
#     b12 = ((R.T @ A12.T) * B12.T).sum()
#     AR22 = A22.T @ R
#     BR22 = B22 @ R.T
#     b22a = (AR22 * (Q @ B22.T)).sum()
#     b22b = (A22 * (Q @ BR22)).sum()
#     s = (S22 * R).sum()
#     a = (AR22.T * BR22).sum()
#     b = b21 + b12 + b22a + b22b + s
#     return a, b


# @jit(nopython=NOPYTHON)
# def _compute_coefficients_permutation(
#     P, Q, A21, B21, A12, B12, A22, B22, S22, permutation
# ):
#     # [1] Algorithm 1 Line 5 - compute the step size
#     # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
#     # terms as ax**2 + bx + c. c does not affect location of minimum
#     # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
#     # apply where possible for efficiency.
#     R = P - Q
#     b21 = ((R.T @ A21) * B21).sum()
#     b12 = ((R.T @ A12.T) * B12.T).sum()
#     AR22 = A22.T @ R
#     BR22 = B22 @ R.T
#     b22a = (AR22 * B22.T[permutation]).sum()
#     b22b = (A22 * BR22[permutation]).sum()
#     s = (S22 * R).sum()
#     a = (AR22.T * BR22).sum()
#     b = b21 + b12 + b22a + b22b + s
#     return a, b


# # class GOATSolver(FAQSolver):
# #     def __init__(
# #         self,
# #         A,
# #         B,
# #         S=None,
# #         partial_match=None,
# #         rng=None,
# #         init="barycenter",
# #         verbose=False,
# #         shuffle_input=True,
# #         maximize=True,
# #         maxiter=30,
# #         tol=0.01,
# #         ot_tol=5e-2,
# #         ot_reg=100,
# #     ):
# #         super().__init__(
# #             A,
# #             B,
# #             S=S,
# #             partial_match=partial_match,
# #             rng=rng,
# #             init=init,
# #             verbose=verbose,
# #             shuffle_input=shuffle_input,
# #             maximize=maximize,
# #             maxiter=maxiter,
# #             tol=tol,
# #         )
# #         self.ot_tol = ot_tol
# #         self.ot_reg = ot_reg

# #     def solve_assignment(self, grad_fp):
# #         return alap(grad_fp, self.maximize, self.ot_reg, self.ot_tol), None


# # def _split_matrix(X, n):
# #     upper, lower = X[:n], X[n:]
# #     return upper[:, :n], upper[:, n:], lower[:, :n], lower[:, n:]


# # def _doubly_stochastic(P, tol=1e-3, max_iter=1000):
# #     # Adapted from @btaba implementation
# #     # https://github.com/btaba/sinkhorn_knopp
# #     # of Sinkhorn-Knopp algorithm
# #     # https://projecteuclid.org/euclid.pjm/1102992505

# #     c = 1 / P.sum(axis=0)
# #     r = 1 / (P @ c)
# #     P_eps = P

# #     for it in range(max_iter):
# #         if it % 100 == 0:  # only check every so often to speed up
# #             if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
# #                 np.abs(P_eps.sum(axis=0) - 1) < tol
# #             ).all():
# #                 # All column/row sums ~= 1 within threshold
# #                 break

# #         c = 1 / (r @ P)
# #         r = 1 / (P @ c)
# #         P_eps = r[:, None] * P * c
# #     return P_eps


# # def alap(P, maximize, reg, tol):
# #     power = 1 if maximize else -1
# #     lamb = reg / np.max(np.abs(P))
# #     P = np.exp(lamb * power * P)

# #     P_eps = _doubly_stochastic(P, tol)

# #     return P_eps


# # def _compute_score(A, B, S, perm):
# #     # equivalent to objective function but avoids matmul
# #     return np.sum(A * B[perm][:, perm]) + np.sum(S[np.arange(len(S)), perm])


# # def _common_input_validation(A, B, partial_match):
# #     # A = np.atleast_2d(A)
# #     # B = np.atleast_2d(B)

# #     if partial_match is None:
# #         partial_match = np.array([[], []]).T
# #     partial_match = np.atleast_2d(partial_match).astype(int)

# #     msg = None
# #     if A.shape[0] != A.shape[1]:
# #         msg = "`A` must be square"
# #     elif B.shape[0] != B.shape[1]:
# #         msg = "`B` must be square"
# #     elif A.ndim != 2 or B.ndim != 2:
# #         msg = "`A` and `B` must have exactly two dimensions"
# #     elif A.shape != B.shape:
# #         msg = "`A` and `B` matrices must be of equal size"
# #     elif partial_match.shape[0] > A.shape[0]:
# #         msg = "`partial_match` can have only as many seeds as there are nodes"
# #     elif partial_match.shape[1] != 2:
# #         msg = "`partial_match` must have two columns"
# #     elif partial_match.ndim != 2:
# #         msg = "`partial_match` must have exactly two dimensions"
# #     elif (partial_match < 0).any():
# #         msg = "`partial_match` must contain only positive indices"
# #     elif (partial_match >= A.shape[0]).any():
# #         msg = "`partial_match` entries must be less than number of nodes"
# #     elif not len(set(partial_match[:, 0])) == len(partial_match[:, 0]) or not len(
# #         set(partial_match[:, 1])
# #     ) == len(partial_match[:, 1]):
# #         msg = "`partial_match` column entries must be unique"

# #     if msg is not None:
# #         raise ValueError(msg)

# #     return A, B, partial_match


# # def graph_match(
# #     A,
# #     B,
# #     S=None,
# #     partial_match=None,
# #     solver="ot",
# #     n_init=1,
# #     rng=None,
# #     init="barycenter",
# #     verbose=False,
# #     shuffle_input=True,
# #     maximize=True,
# #     maxiter=30,
# #     tol=0.01,
# #     ot_tol=5e-2,
# #     ot_reg=100,
# # ):
# #     for init in range(n_init):
# #         if solver == "ot":
# #             current_solver = GOATSolver(
# #                 A,
# #                 B,
# #                 S=S,
# #                 partial_match=partial_match,
# #                 rng=rng,
# #                 init=init,
# #                 verbose=verbose,
# #                 shuffle_input=shuffle_input,
# #                 maximize=maximize,
# #                 maxiter=maxiter,
# #                 tol=tol,
# #                 ot_tol=ot_tol,
# #                 ot_reg=ot_reg,
# #             )
# #         elif solver == "lap":
# #             current_solver = FAQSolver(
# #                 A,
# #                 B,
# #                 S=S,
# #                 partial_match=partial_match,
# #                 rng=rng,
# #                 init=init,
# #                 verbose=verbose,
# #                 shuffle_input=shuffle_input,
# #                 maximize=maximize,
# #                 maxiter=maxiter,
# #                 tol=tol,
# #             )
# #         solver.solve()


# # # from graspologic.simulations import er_corr
# # # from scipy.sparse import csr_matrix


# # # n_sims = 20
# # # n = 1000
# # # A, B = er_corr(n, np.log(n) / n, 1.0)

# # # currtime = time.time()
# # # for i in range(n_sims):
# # #     solver = FAQSolver(A, B)
# # #     solver.solve()


# # # print(f"{time.time() - currtime:.3f} seconds elapsed with NOPYTHON={NOPYTHON}.")


# # # # solver.P_final_
# # # # perm = solver.permutation_
# # # # match_ratio = (np.arange(n) == perm).mean()
# # # # print(solver.n_iter)
# # # # print(match_ratio)
