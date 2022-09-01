#%% [markdown]
# # Connectome data

#%%
import datetime
from secrets import choice
import time
from tkinter import N

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


def shuffle_incident_edges(adjacency, choice_inds, rng=None):
    adjacency = adjacency.copy()
    n = adjacency.shape[0]

    if rng is None:
        rng = np.random.default_rng()

    for ind in choice_inds:
        # permute the incident row/outgoing connections
        perm = rng.permutation(n)
        adjacency[ind, :] = adjacency[ind, :][perm]

        # permute the incident column/incoming connections
        perm = rng.permutation(n)
        adjacency[:, ind] = adjacency[:, ind][perm]

    return adjacency


#%%

from graspologic.match import graph_match

RERUN_SIMS = False
datasets = ["specimen_107", "specimen_148", "herm_chem", "male_chem", "maggot_subset"]
n_sims = 50
glue("n_initializations", n_sims)

p_shuffles = np.linspace(0, 0.25, 6)  # [0, 0.25, 0.5, 0.75]

rows = []
if RERUN_SIMS:
    with tqdm(total=len(datasets) * len(p_shuffles) * n_sims * 2) as pbar:
        for dataset in datasets:
            adj, nodes = load_split_connectome(dataset)
            left_inds, right_inds = get_hemisphere_indices(nodes)
            n_nodes = len(nodes)
            A = adj[left_inds][:, left_inds]
            B = adj[right_inds][:, right_inds]
            AB = adj[left_inds][:, right_inds]
            BA = adj[right_inds][:, left_inds]
            n_side = A.shape[0]
            for p_shuffle in p_shuffles:
                n_shuffle = int(np.floor(n_side * p_shuffle))
                p_shuffle = n_shuffle / n_side
                for sim in range(n_sims):
                    choice_inds = rng.choice(n_side, size=n_shuffle, replace=False)
                    non_choice_inds = np.setdiff1d(np.arange(n_side), choice_inds)

                    A_perturbed = shuffle_incident_edges(A, choice_inds, rng=rng)
                    B_perturbed = shuffle_incident_edges(B, choice_inds, rng=rng)
                    AB_perturbed = shuffle_incident_edges(AB, choice_inds, rng=rng)
                    BA_perturbed = shuffle_incident_edges(BA, choice_inds, rng=rng)

                    for method in ["GM", "BGM"]:
                        run_start = time.time()
                        if method == "GM":
                            # solver = GraphMatchSolver(A, B, rng=seed)
                            indices_A, indices_B, score, misc = graph_match(
                                A_perturbed, B_perturbed, rng=rng
                            )
                        elif method == "BGM":
                            # solver = GraphMatchSolver(A, B, AB=AB, BA=BA, rng=seed)
                            indices_A, indices_B, score, misc = graph_match(
                                A_perturbed,
                                B_perturbed,
                                AB=AB_perturbed,
                                BA=BA_perturbed,
                                rng=rng,
                            )

                        elapsed = time.time() - run_start
                        match_ratio_full = (indices_B == np.arange(n_side)).mean()

                        match_ratio = (indices_B == np.arange(n_side))[
                            non_choice_inds
                        ].mean()

                        rows.append(
                            {
                                "match_ratio": match_ratio,
                                "match_ratio_full": match_ratio_full,
                                "method": method,
                                "elapsed": elapsed,
                                "converged": misc[0]["converged"],
                                "n_iter": misc[0]["n_iter"],
                                "score": score,
                                "dataset": dataset,
                                "p_shuffle": p_shuffle,
                                "n_shuffle": n_shuffle,
                                "sim": sim,
                            }
                        )
                        pbar.update(1)

        results = pd.DataFrame(rows)
        results.to_csv(OUT_PATH / "match_results.csv")
else:
    results = pd.read_csv(OUT_PATH / "match_results.csv", index_col=0)

#%%

from pkg.plot import dashes

set_theme(font_scale=1)

fig, axs = plt.subplots(1, 5, figsize=(15, 5), constrained_layout=True)

nice_dataset_map = {
    "herm_chem": "C. elegans\nhermaphrodite",
    "male_chem": "C. elegans\nmale",
    "maggot": "Maggot",
    "maggot_subset": "D. melanogaster\n larva subset",
    "specimen_107": "P. pacificus\npharynx 1",
    "specimen_148": "P. pacificus\npharynx 2",
}

for i, dataset in enumerate(datasets):
    dataset_results = results[results["dataset"] == dataset]
    ax = axs.flat[i]
    sns.lineplot(
        data=dataset_results,
        x="p_shuffle",
        y="match_ratio",
        hue="method",
        palette=method_palette,
        style="method",
        dashes=dashes,
        ax=ax,
    )
    ax.set_title(nice_dataset_map[dataset])
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel('Proportion of\nnodes unmatched')
    if i > 0:
        ax.get_legend().remove()
        ax.set_ylabel("")
    else:
        ax.set_ylabel("Matching accuracy\n(matchable nodes only)")
        sns.move_legend(ax, "upper right", frameon=True, title="Method")
gluefig("unmatchable_accuracy", fig)


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
