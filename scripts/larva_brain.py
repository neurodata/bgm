#%% [markdown]
# # Matching when including the contralateral connections
#%% [markdown]
# ## Preliminaries
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


FILENAME = "larva_brain"

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

assert len(left_nodes) == len(right_nodes)

#%%
mg = load_maggot_graph()
mg = mg.node_subgraph(all_nodes.index)
adj = mg.sum.adj

n = len(left_nodes)
left_inds = np.arange(n)
right_inds = np.arange(n) + n

glue("n_nodes", n)


#%% [markdown]
# ### Run the graph matching experiment

n_sims = 25
glue("n_initializations", n_sims)

RERUN_SIMS = False
if RERUN_SIMS:
    seeds = rng.integers(np.iinfo(np.int32).max, size=n_sims)
    rows = []
    for sim, seed in enumerate(seeds):
        for Solver, method in zip(
            [BisectedGraphMatchSolver, GraphMatchSolver], ["BGM", "GM"]
        ):
            run_start = time.time()
            solver = Solver(adj, left_inds, right_inds, rng=seed)
            solver.solve()
            match_ratio = (solver.permutation_ == np.arange(n)).mean()
            elapsed = time.time() - run_start
            print(f"{elapsed:.3f} seconds elapsed.")
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
    results.to_csv(OUT_PATH / "larva_comparison.csv")
else:
    results = pd.read_csv(OUT_PATH / "larva_comparison.csv", index_col=0)

results.head()

#%%


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
matched_stripplot(
    data=results,
    x="method",
    y="match_ratio",
    match="sim",
    order=["GM", "BGM"],
    hue="method",
    palette=method_palette,
    ax=ax,
    jitter=0.25,
)
sns.move_legend(ax, "upper left", title="Method")


mean1 = results[results["method"] == "GM"]["match_ratio"].mean()
mean2 = results[results["method"] == "BGM"]["match_ratio"].mean()

ax.set_yticks([mean1, mean2])
ax.set_yticklabels([f"{mean1:.2f}", f"{mean2:.2f}"])
ax.tick_params(which="both", length=7)
ax.set_ylabel("Match ratio")
ax.set_xlabel("Method")

gluefig("match_ratio_larva", fig)

# %%

bgm_results = results[results["method"] == "BGM"]
gm_results = results[results["method"] == "GM"]

stat, pvalue = wilcoxon(
    bgm_results["match_ratio"].values, gm_results["match_ratio"].values
)

glue("match_ratio_pvalue", pvalue, form="pvalue")

mean_bgm = bgm_results["match_ratio"].mean()
glue("mean_match_ratio_bgm", mean_bgm)

mean_gm = gm_results["match_ratio"].mean()
glue("mean_match_ratio_gm", mean_gm)
