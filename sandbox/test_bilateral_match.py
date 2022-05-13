#%%
import numpy as np
from pkg.match import BaseMatchSolver
from numba import jit
from scipy.optimize import linear_sum_assignment
from graspologic.simulations import er_corr
import pandas as pd


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

    # TODO
    def set_reference_frame(self):
        pass

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

    def unset_reference_frame(self):
        pass

    def compute_score(*args):
        return 0

    # def finalize(self, P):
    #     self.print("Finalizing permutation")
    #     self.P = P
    #     pass
    # _, permutation = linear_sum_assignment(self.P_final_, maximize=True)
    # self.permutation_ = permutation
    # self.unset_reference_frame()

    # score = _compute_score(self.A, self.B, self.S, self.permutation_)
    # self.score_ = score


class GraphMatchSolver(BisectedGraphMatchSolver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.AB = np.zeros_like(self.AB)
        self.BA = np.zeros_like(self.BA)


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


from tqdm import tqdm

n_side = 10
n_sims = 1000
ipsi_rho = 0.8
p = 0.3
rows = []
for contra_rho in np.linspace(0, 1, 11):
    for sim in tqdm(range(n_sims)):
        A, B = er_corr(n_side, p, ipsi_rho, directed=True)
        AB, BA = er_corr(n_side, p, contra_rho, directed=True)
        indices_A = np.arange(n_side)
        indices_B = np.arange(n_side, 2 * n_side)
        adjacency = np.zeros((2 * n_side, 2 * n_side))
        adjacency[np.ix_(indices_A, indices_A)] = A
        adjacency[np.ix_(indices_B, indices_B)] = B
        adjacency[np.ix_(indices_A, indices_B)] = AB
        adjacency[np.ix_(indices_B, indices_A)] = BA

        side_perm = np.random.permutation(n_side) + n_side
        perm = np.concatenate((indices_A, side_perm))
        adjacency = adjacency[np.ix_(perm, perm)]
        undo_perm = np.argsort(side_perm)

        for Solver, method in zip(
            [BisectedGraphMatchSolver, GraphMatchSolver], ["BGM", "GM"]
        ):
            solver = Solver(adjacency, indices_A, indices_B)
            solver.solve()
            solver.P_final_
            match_ratio = (solver.permutation_ == undo_perm).mean()

            rows.append(
                {
                    "ipsi_rho": ipsi_rho,
                    "contra_rho": contra_rho,
                    "match_ratio": match_ratio,
                    "sim": sim,
                    "method": method,
                }
            )

results = pd.DataFrame(rows)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from pkg.plot import set_theme

set_theme()
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(data=results, x="contra_rho", y="match_ratio", hue="method", ax=ax)
ax.set_ylabel("Match ratio")
ax.set_xlabel("Contralateral edge correlation")
sns.move_legend(ax, loc="upper left", title="Method")

#%%

from graspologic.match import GraphMatch

match_ratios = []
for sim in tqdm(range(n_sims)):
    A, B = er_corr(n_side, p, ipsi_rho, directed=True)

    gm = GraphMatch()
    gm.fit(A, B)
    perm = gm.perm_inds_
    match_ratio = (perm == np.arange(A.shape[0])).mean()
    match_ratios.append(match_ratio)

np.mean(match_ratios)
