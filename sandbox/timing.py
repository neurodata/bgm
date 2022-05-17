#%% [markdown]
# # Timing

#%%
import datetime
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.match import GraphMatchSolver
from giskard.plot import matched_stripplot
from pkg.data import load_split_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import method_palette, set_theme
from scipy.stats import wilcoxon
from tqdm import tqdm
from scipy.sparse import csr_matrix

FILENAME = "check_repro"

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

#%% [markdown]
# ## Load processed data, run matching experiment
#%%


def get_hemisphere_indices(nodes):
    nodes = nodes.copy()
    nodes["_inds"] = np.arange(len(nodes))
    left_nodes = nodes[nodes["hemisphere"] == "L"]
    right_nodes = nodes[nodes["hemisphere"] == "R"]
    assert (left_nodes["pair"].values == right_nodes["pair"].values).all()
    left_indices = left_nodes["_inds"].values
    right_indices = right_nodes["_inds"].values
    return left_indices, right_indices


RERUN_SIMS = True
dataset = "maggot_subset"

n_sims = 20
rows = []
for sparse in [True, False]:
    rng = np.random.default_rng(8888)

    adj, nodes = load_split_connectome(dataset)
    if sparse:
        adj = csr_matrix(adj)
    n_nodes = len(nodes)
    # n_edges = np.count_nonzero(adj)

    left_inds, right_inds = get_hemisphere_indices(nodes)
    A = adj[left_inds][:, left_inds]
    B = adj[right_inds][:, right_inds]
    AB = adj[left_inds][:, right_inds]
    BA = adj[right_inds][:, left_inds]

    n_side = len(left_inds)
    seeds = rng.integers(np.iinfo(np.uint32).max, size=n_sims)

    for sim, seed in enumerate(tqdm(seeds, leave=False)):
        for method in ["GM", "BGM"]:
            if method == "GM":
                solver = GraphMatchSolver(A, B, rng=seed)
            elif method == "BGM":
                solver = GraphMatchSolver(A, B, AB=AB, BA=BA, rng=seed)
            run_start = time.time()
            solver.solve()
            match_ratio = (solver.permutation_ == np.arange(n_side)).mean()
            elapsed = time.time() - run_start
            rows.append(
                {
                    "match_ratio": match_ratio,
                    "sim": sim,
                    "method": method,
                    "seed": seed,
                    "converged": solver.converged,
                    "n_iter": solver.n_iter,
                    "score": solver.score_,
                    "elapsed": elapsed,
                    "sparse": sparse,
                }
            )
    results = pd.DataFrame(rows)

#%%
sparse_results = results[results["sparse"]]
nonsparse_results = results[~results["sparse"]]
set_theme()
fig, axs = plt.subplots(1, 4, figsize=(12, 8), constrained_layout=True)
for i, key in enumerate(["match_ratio", "elapsed"]):
    ax = axs[2 * i]
    diff = (
        sparse_results.groupby(["method", "seed"])[key].mean()
        - nonsparse_results.groupby(["method", "seed"])[key].mean()
    )
    sns.stripplot(y=diff, ax=ax)
    ax.axhline(0, color="black")
    ax.set_ylabel(f"{key} (sparse - nonsparse)")

    ax = axs[2 * i + 1]
    sns.histplot(y=diff, ax=ax)
    ax.axhline(0, color="black")
    ax.set(ylabel="", yticks=[], xlabel="", xticks=[ax])


#%%
# #%%
# from giskard.match import GraphMatchSolver

# sparse = True
adj, nodes = load_split_connectome(dataset)
sparse = True
if sparse:
    adj = csr_matrix(adj)
print(adj.dtype)
# n_nodes = len(nodes)
# # n_edges = np.count_nonzero(adj)

# left_inds, right_inds = get_hemisphere_indices(nodes)
# A = adj[left_inds][:, left_inds]
# B = adj[right_inds][:, right_inds]
# AB = adj[left_inds][:, right_inds]
# BA = adj[right_inds][:, left_inds]

# solver = GraphMatchSolver(A, B, rng=seed)
# print(solver._sparse)
# # print(solver.B)
# #
# # solver._seeded
# # solver.compute_constant_terms()
# solver.solve()
