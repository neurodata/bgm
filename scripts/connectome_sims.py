#%% [markdown]
# # Connectome data

#%%
import datetime
from re import S
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pkg.data import load_split_connectome
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.plot import matched_stripplot, method_palette, set_theme
from pkg.utils import get_hemisphere_indices
from tqdm.autonotebook import tqdm

FILENAME = "connectome_sims"

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
rng = np.random.default_rng(8888)

#%% [markdown]
# ## Load processed data, run matching experiment
#%%


def remove_edges(adjacency, effect_size=100, rng=None, max_tries=None):
    if rng is None:
        rng = np.random.default_rng()

    n_nonzero = np.count_nonzero(adjacency)
    if effect_size > n_nonzero:
        return None

    row_inds, col_inds = np.nonzero(adjacency)

    select_edges = rng.choice(len(row_inds), size=effect_size, replace=False)
    select_row_inds = row_inds[select_edges]
    select_col_inds = col_inds[select_edges]
    adjacency[select_row_inds, select_col_inds] = 0
    return adjacency


# TODO
# @jit(nopython=True)
# https://numba-how-to.readthedocs.io/en/latest/numpy.html

# TODO add an induced option
# TODO deal with max number of edges properly
# TODO put down edges all at once rather than this silly thing
def add_edges(adjacency, effect_size=100, rng=None, max_tries=None):
    if rng is None:
        rng = np.random.default_rng()

    n_source = adjacency.shape[0]
    n_target = adjacency.shape[1]
    n_possible = n_source * n_target
    if effect_size > n_possible:  # technicall should be - n if on main diagonal
        return

    n_edges_added = 0
    tries = 0
    while n_edges_added < effect_size and tries < max_tries:
        i = rng.integers(n_source)
        j = rng.integers(n_target)
        tries += 1
        if i != j and adjacency[i, j] == 0:
            adjacency[i, j] = 1
            n_edges_added += 1

    if tries == max_tries and effect_size != 0:
        msg = (
            "Maximum number of tries reached when adding edges, number added was"
            " less than specified."
        )
        raise UserWarning(msg)

    return adjacency


def shuffle_edges(adjacency, effect_size=100, rng=None, max_tries=None):
    if rng is None:
        rng = np.random.default_rng()

    adjacency = remove_edges(
        adjacency, effect_size=effect_size, rng=rng, max_tries=max_tries
    )

    if adjacency is None:
        return

    adjacency = add_edges(
        adjacency, effect_size=effect_size, rng=rng, max_tries=max_tries
    )

    return adjacency


def shuffle_edges(adjacency, effect_size=100, rng=None, max_tries=None):
    adjacency = adjacency.copy()

    if max_tries is None:
        max_tries = effect_size * 1000

    if rng is None:
        rng = np.random.default_rng()

    n_nonzero = np.count_nonzero(adjacency)
    if effect_size > n_nonzero:
        return None

    row_inds, col_inds = np.nonzero(adjacency)

    select_edges = rng.choice(len(row_inds), size=effect_size, replace=False)
    select_row_inds = row_inds[select_edges]
    select_col_inds = col_inds[select_edges]
    edge_weights = list(adjacency[select_row_inds, select_col_inds])
    adjacency[select_row_inds, select_col_inds] = 0

    n_source = adjacency.shape[0]
    n_target = adjacency.shape[1]
    n_possible = n_source * n_target
    if effect_size > n_possible:  # technicall should be - n if on main diagonal
        return None

    n_edges_added = 0
    tries = 0
    while n_edges_added < effect_size and tries < max_tries:
        i = rng.integers(n_source)
        j = rng.integers(n_target)
        tries += 1
        if i != j and adjacency[i, j] == 0:
            adjacency[i, j] = edge_weights.pop()
            n_edges_added += 1

    return adjacency


#%%

from graspologic.match import graph_match

RERUN_SIMS = True
datasets = ["male_chem", "herm_chem", "specimen_148", "specimen_107"]

n_sims = 10
glue("n_initializations", n_sims)

p_shuffles = np.linspace(0, 0.9, 10)  # [0, 0.25, 0.5, 0.75]
contra_weight_ratios = {}
results_by_dataset = {}

rows = []
with tqdm(total=len(datasets) * len(p_shuffles) * len(p_shuffles) * n_sims * 2) as pbar:
    for dataset in datasets:
        adj, nodes = load_split_connectome(dataset)
        n_nodes = len(nodes)

        left_inds, right_inds = get_hemisphere_indices(nodes)
        A = adj[left_inds][:, left_inds]
        n_edges_ipsi = np.count_nonzero(A)
        AB = adj[left_inds][:, right_inds]
        n_edges_contra = np.count_nonzero(AB)

        n_side = A.shape[0]

        # B = adj[right_inds][:, right_inds]
        # BA = adj[right_inds][:, left_inds]
        for p_shuffle_ipsi in p_shuffles:
            # for p_shuffle_contra in p_shuffles:
            effect_size_ipsi = int(np.floor(p_shuffle_ipsi * n_edges_ipsi))
            B = shuffle_edges(A, effect_size=effect_size_ipsi)

            effect_size_contra = int(np.floor(p_shuffle_contra * n_edges_contra))
            BA = shuffle_edges(AB, effect_size=effect_size_contra)

            seed = rng.integers(np.iinfo(np.uint32).max)
            seeds = rng.integers(np.iinfo(np.uint32).max, size=n_sims)

            for sim, seed in enumerate(seeds):
                for method in ["GM", "BGM"]:
                    run_start = time.time()
                    if method == "GM":
                        # solver = GraphMatchSolver(A, B, rng=seed)
                        indices_A, indices_B, score, misc = graph_match(
                            A, B, rng=seed
                        )
                    elif method == "BGM":
                        # solver = GraphMatchSolver(A, B, AB=AB, BA=BA, rng=seed)
                        indices_A, indices_B, score, misc = graph_match(
                            A, B, AB=AB, BA=BA, rng=seed
                        )
                    elapsed = time.time() - run_start
                    match_ratio = (indices_B == np.arange(n_side)).mean()
                    rows.append(
                        {
                            "match_ratio": match_ratio,
                            "sim": sim,
                            "method": method,
                            "seed": seed,
                            "elapsed": elapsed,
                            "converged": misc[0]["converged"],
                            "n_iter": misc[0]["n_iter"],
                            "score": score,
                            "dataset": dataset,
                            "p_shuffle_ipsi": p_shuffle_ipsi,
                            "p_shuffle_contra": p_shuffle_contra,
                        }
                    )
                    pbar.update(1)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH / "match_results.csv")

    # else:
    #     results = pd.read_csv(OUT_PATH / f"{dataset}_match_results.csv", index_col=0)


#%%

set_theme(font_scale=1)

fig, axs = plt.subplots(
    3, len(datasets), figsize=(len(datasets) * 5, 5 * 3), sharex=True, sharey=True
)

for j, dataset in enumerate(datasets):
    dataset_results = results[results["dataset"] == dataset]
    all_square_results = {}
    for i, method in enumerate(["GM", "BGM"]):
        method_dataset_results = dataset_results[dataset_results["method"] == method]
        square_results = method_dataset_results.pivot_table(
            index="p_shuffle_contra",
            columns="p_shuffle_ipsi",
            values="match_ratio",
            aggfunc=np.mean,
        ).iloc[::-1]
        all_square_results[method] = square_results
        ax = axs[i, j]
        sns.heatmap(
            square_results,
            ax=ax,
            cbar=False,
            vmin=0,
            vmax=1,
            square=True,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
        )
    ratio = all_square_results["BGM"] - all_square_results["GM"]

    ax = axs[2, j]
    sns.heatmap(
        ratio,
        ax=ax,
        cbar=False,
        vmin=-1,
        vmax=1,
        square=True,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
    )
