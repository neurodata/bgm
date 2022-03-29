#%%
# Simulation

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from graspologic.simulations import er_corr
from pkg.io import FIG_PATH, OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import method_palette, set_theme
from tqdm import tqdm


DISPLAY_FIGS = True

FILENAME = "simulations"

OUT_PATH = OUT_PATH / FILENAME

FIG_PATH = FIG_PATH / FILENAME


def glue(name, var, **kwargs):
    default_glue(name, var, FILENAME, **kwargs)


def gluefig(name, fig, **kwargs):
    savefig(name, foldername=FILENAME, **kwargs)

    glue(name, fig, figure=True)

    if not DISPLAY_FIGS:
        plt.close()


t0 = time.time()
set_theme()
rng = np.random.default_rng(8888)


#%%
n_side = 10
n_sims = 1000
ipsi_rho = 0.8
p = 0.3
rows = []
for contra_rho in np.linspace(0, 1, 11):
    for sim in tqdm(range(n_sims)):
        # simulate the correlated subgraphs
        A, B = er_corr(n_side, p, ipsi_rho, directed=True)
        AB, BA = er_corr(n_side, p, contra_rho, directed=True)

        # construct the full network
        indices_A = np.arange(n_side)
        indices_B = np.arange(n_side, 2 * n_side)
        adjacency = np.zeros((2 * n_side, 2 * n_side))
        adjacency[np.ix_(indices_A, indices_A)] = A
        adjacency[np.ix_(indices_B, indices_B)] = B
        adjacency[np.ix_(indices_A, indices_B)] = AB
        adjacency[np.ix_(indices_B, indices_A)] = BA

        # permute one hemisphere
        side_perm = rng.permutation(n_side) + n_side
        perm = np.concatenate((indices_A, side_perm))
        adjacency = adjacency[np.ix_(perm, perm)]
        undo_perm = np.argsort(side_perm)

        # run the matching
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


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.lineplot(
    data=results,
    x="contra_rho",
    y="match_ratio",
    hue="method",
    ax=ax,
    palette=method_palette,
)
ax.set_ylabel("Match ratio")
ax.set_xlabel("Contralateral edge correlation")
sns.move_legend(ax, loc="upper left", title="Method")
gluefig("match_ratio_by_contra_rho", fig)

# #%%
# contra_rho = 0.8
# A, B = er_corr(n_side, p, ipsi_rho, directed=True)
# AB, BA = er_corr(n_side, p, contra_rho, directed=True)

# # construct the full network
# indices_A = np.arange(n_side)
# indices_B = np.arange(n_side, 2 * n_side)
# adjacency = np.zeros((2 * n_side, 2 * n_side))
# adjacency[np.ix_(indices_A, indices_A)] = A
# adjacency[np.ix_(indices_B, indices_B)] = B
# adjacency[np.ix_(indices_A, indices_B)] = AB
# adjacency[np.ix_(indices_B, indices_A)] = BA

# # permute one hemisphere
# side_perm = np.random.permutation(n_side) + n_side
# perm = np.concatenate((indices_A, side_perm))
# adjacency = adjacency[np.ix_(perm, perm)]
# undo_perm = np.argsort(side_perm)

# #%%
# from graspologic.plot import heatmap

# heatmap(adjacency)

#%%


#%%

# from graspologic.match import GraphMatch

# match_ratios = []
# for sim in tqdm(range(n_sims)):
#     A, B = er_corr(n_side, p, ipsi_rho, directed=True)

#     gm = GraphMatch()
#     gm.fit(A, B)
#     perm = gm.perm_inds_
#     match_ratio = (perm == np.arange(A.shape[0])).mean()
#     match_ratios.append(match_ratio)

# np.mean(match_ratios)
