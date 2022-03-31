#%% [markdown]
# # Connectome data

#%%
import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import adjplot, matched_stripplot, matrixplot
from numba import jit
from pkg.data import load_maggot_graph, load_matched
from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import method_palette, set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs, get_seeds
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
from pkg.data import load_split_connectome
from tqdm import tqdm


FILENAME = "connectomes"

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


RERUN_SIMS = False
datasets = ["maggot", "herm_chem", "male_chem"]

n_sims = 25
glue("n_initializations", n_sims)

results_by_dataset = {}
for dataset in datasets:
    adj, nodes = load_split_connectome(dataset)
    n_nodes = len(nodes)
    glue(f"{dataset}_n_nodes", n_nodes)
    if RERUN_SIMS:
        left_inds, right_inds = get_hemisphere_indices(nodes)
        n_side = len(left_inds)
        seeds = rng.integers(np.iinfo(np.int32).max, size=n_sims)
        rows = []
        for sim, seed in enumerate(tqdm(seeds)):
            for Solver, method in zip(
                [BisectedGraphMatchSolver, GraphMatchSolver], ["BGM", "GM"]
            ):
                run_start = time.time()
                solver = Solver(adj, left_inds, right_inds, rng=seed)
                solver.solve()
                match_ratio = (solver.permutation_ == np.arange(n_side)).mean()
                elapsed = time.time() - run_start
                rows.append(
                    {
                        "match_ratio": match_ratio,
                        "sim": sim,
                        "method": method,
                        "seed": seed,
                        "elapsed": elapsed,
                        "converged": solver.converged,
                        "n_iter": solver.n_iter,
                        "score": solver.score_,
                    }
                )

        results = pd.DataFrame(rows)
        results.to_csv(OUT_PATH / f"{dataset}_match_results.csv")
    else:
        results = pd.read_csv(OUT_PATH / f"{dataset}_match_results.csv", index_col=0)
    results_by_dataset[dataset] = results

#%%

set_theme(font_scale=1.2)
scale = 5
jitter = 0.25
meanline_width = 0.35
n_datasets = len(datasets)
order = ["GM", "BGM"]
nice_dataset_map = {
    "herm_chem": "C. elegans hermaphrodite",
    "male_chem": "C. elegans male",
    "maggot": "Maggot",
}

fig, axs = plt.subplots(
    1, len(datasets), figsize=(n_datasets * scale, scale), sharey=True
)

for i, (dataset, results) in enumerate(results_by_dataset.items()):
    ax = axs[i]
    matched_stripplot(
        data=results,
        x="method",
        y="match_ratio",
        match="sim",
        order=order,
        hue="method",
        palette=method_palette,
        ax=ax,
        jitter=jitter,
        legend=False,
    )

    ax.tick_params(which="both", length=7)
    ax.set_ylabel("Match ratio")
    ax.set_xlabel("Method")
    ax.set_title(nice_dataset_map[dataset])

    ticklabels = ax.get_xticklabels()
    for ticklabel in ticklabels:
        method = ticklabel.get_text()
        ticklabel.set_color(method_palette[method])

    gm_results = results[results["method"] == "GM"]
    bgm_results = results[results["method"] == "BGM"]

    stat, pvalue = wilcoxon(
        gm_results["match_ratio"].values, bgm_results["match_ratio"].values
    )
    glue(f"{dataset}_match_ratio_pvalue", pvalue, form="pvalue")

    for i, method in enumerate(order):
        mean_match_ratio = results[results["method"] == method]["match_ratio"].mean()
        ax.plot(
            [i - meanline_width, i + meanline_width],
            [mean_match_ratio, mean_match_ratio],
            color=method_palette[method],
        )
        ax.text(
            i + meanline_width + 0.05,
            mean_match_ratio,
            f"{mean_match_ratio:0.2f}",
            color=method_palette[method],
            va="center",
            ha="left",
            fontsize="medium",
        )
        glue(f"{dataset}_{method}_mean_match_ratio", mean_match_ratio)

    ax.set_xlim((-0.5, 1.5))
    ax.set_yticks([0.45, 0.6, 0.75, 0.9])
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

gluefig("match_ratio_comparison", fig)
