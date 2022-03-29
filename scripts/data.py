#%% [markdown]
# # Matching when including the contralateral connections
#%% [markdown]
# ## Preliminaries
#%%
from pkg.utils import get_seeds


import datetime
import time
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import seaborn as sns
from numba import jit

from giskard.plot import matched_stripplot
from pkg.data import load_maggot_graph
from pkg.io import savefig
from pkg.plot import set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs

from giskard.plot import adjplot, matrixplot
from pkg.io import OUT_PATH
from myst_nb import glue as default_glue
from pkg.data import load_maggot_graph, load_matched


t0 = time.time()


FILENAME = "bilateral_match_data"

DISPLAY_FIGS = True

OUT_PATH = OUT_PATH / FILENAME


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


#%% [markdown]
# ### Load the data


#%%
left_adj, left_nodes = load_matched("left")
right_adj, right_nodes = load_matched("right")
left_nodes["inds"] = range(len(left_nodes))
right_nodes["inds"] = range(len(right_nodes))
seeds = get_seeds(left_nodes, right_nodes)
all_nodes = pd.concat((left_nodes, right_nodes))
all_nodes["inds"] = range(len(all_nodes))

left_nodes.iloc[seeds[0]]["pair_id"]

#%%
mg = load_maggot_graph()
mg = mg.node_subgraph(all_nodes.index)
adj = mg.sum.adj

#%%
# mg = mg[mg.nodes["paper_clustered_neurons"]]
# mg = mg[mg.nodes["hemisphere"].isin(["L", "R"])]
# lp_inds, rp_inds = get_paired_inds(mg.nodes)

# # ll_mg, rr_mg, lr_mg, rl_mg = mg.bisect()

# left_in_right = ll_mg.nodes["pair"].isin(rr_mg.nodes.index)
# left_in_right_idx = left_in_right[left_in_right].index
# right_in_left = rr_mg.nodes["pair"].isin(ll_mg.nodes.index)
# right_in_left_idx = right_in_left[right_in_left].index
# left_in_right_pair_ids = ll_mg.nodes.loc[left_in_right_idx, "pair_id"]
# right_in_left_pair_ids = rr_mg.nodes.loc[right_in_left_idx, "pair_id"]
# valid_pair_ids = np.intersect1d(left_in_right_pair_ids, right_in_left_pair_ids)
# n_pairs = len(valid_pair_ids)
# mg.nodes["valid_pair_id"] = False
# mg.nodes.loc[mg.nodes["pair_id"].isin(valid_pair_ids), "valid_pair_id"] = True
# mg.nodes.sort_values(
#     ["hemisphere", "valid_pair_id", "pair_id"], inplace=True, ascending=False
# )
# mg.nodes["_inds"] = range(len(mg.nodes))
# adj = mg.sum.adj
# left_nodes = mg.nodes[mg.nodes["hemisphere"] == "L"].copy()
# left_inds = left_nodes["_inds"]
# right_nodes = mg.nodes[mg.nodes["hemisphere"] == "R"].copy()
# right_inds = right_nodes["_inds"]

max_n_side = max(len(left_nodes), len(right_nodes))

#%%
left_inds = all_nodes.iloc[: len(left_nodes)]["inds"]
right_inds = all_nodes.iloc[len(left_nodes) :]["inds"]


def pad(A, size):
    # naive padding for now
    A_padded = np.zeros((size, size))
    rows = A.shape[0]
    cols = A.shape[1]
    A_padded[:rows, :cols] = A
    return A_padded


ll_adj = pad(adj[np.ix_(left_inds, left_inds)], max_n_side)
rr_adj = pad(adj[np.ix_(right_inds, right_inds)], max_n_side)
lr_adj = pad(adj[np.ix_(left_inds, right_inds)], max_n_side)
rl_adj = pad(adj[np.ix_(right_inds, left_inds)], max_n_side)

for i in range(max_n_side - len(right_inds)):
    right_nodes = right_nodes.append(
        pd.Series(name=-i - 1, dtype="float"), ignore_index=False
    )

n = max_n_side

full_adj = np.zeros((2 * n, 2 * n))
full_adj[np.ix_(left_inds, left_inds)] = ll_adj
full_adj[np.ix_(right_inds, right_inds)] = rr_adj
full_adj[np.ix_(left_inds, right_inds)] = lr_adj
full_adj[np.ix_(right_inds, left_inds)] = rl_adj

full_left_inds = np.arange(n)
full_right_inds = np.arange(n) + n

#%% [markdown]
# ### Run the graph matching experiment

from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver

# seeds = np.
n_sims = 2
seeds = rng.randint(np.iinfo(np.int32).max, size=n_sims)

for sim, seed in enumerate(seeds):
    for Solver, method in zip(
        [BisectedGraphMatchSolver, GraphMatchSolver], ["BGM", "GM"]
    ):
        solver = Solver(adjacency, indices_A, indices_B)
        solver.solve()
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

# solver = BisectedGraphMatchSolver(full_adj, full_left_inds, full_right_inds, rng=88)
# solver.solve()
# perm1 = solver.permutation_

#%%
(perm1 == np.arange(n)).mean()


#%%
last_results_idx = results.groupby(["between_term", "init"])["iter"].idxmax()
last_results = results.loc[last_results_idx].copy()

#%%
# TODO save the results

from giskard.plot import matched_stripplot

matched_stripplot(data=last_results, x="between_term", y="match_ratio", match="init")

# %%
from scipy.stats import wilcoxon

between_results = last_results[last_results["between_term"] == True]
no_between_results = last_results[last_results["between_term"] == False]

wilcoxon(
    between_results["match_ratio"].values, no_between_results["match_ratio"].values
)
