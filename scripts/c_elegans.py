import datetime
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from giskard.plot import adjplot, matched_stripplot, matrixplot

from pkg.io import OUT_PATH
from pkg.io import glue as default_glue
from pkg.io import savefig
from pkg.match import BisectedGraphMatchSolver, GraphMatchSolver
from pkg.plot import method_palette, set_theme
from pkg.utils import get_paired_inds, get_paired_subgraphs, get_seeds
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

FILENAME = "c_elegans"

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


#%%


data_path = "bgm/data/worm_wiring/male_chem_adj.csv"

adj_df = pd.read_csv(data_path, index_col=0).fillna(0)

node_ids = np.union1d(adj_df.index, adj_df.columns)

adj_df = adj_df.reindex(index=node_ids, columns=node_ids).fillna(0)

adj_df
#%%

exceptions = ["vBWM", "dgl", "dBWM"]
node_rows = []

for node_id in node_ids:
    is_sided = True
    if not ((node_id[-1] == "L") or (node_id[-1] == "R")):
        is_exception = False
        for exception in exceptions:
            if exception in node_id:
                is_exception = True
        if not is_exception:
            is_sided = False

    if is_sided:
        # node_id_no_side = node_id.strip("0123456789")
        left_pos = node_id.rfind("L")
        right_pos = node_id.rfind("R")
        is_right = bool(np.argmax((left_pos, right_pos)))
        side_indicator_loc = right_pos if is_right else left_pos
        node_class = node_id[:side_indicator_loc] + node_id[side_indicator_loc + 1 :]
        hemisphere = "R" if is_right else "L"
        node_rows.append(
            {"node_id": node_id, "class": node_class, "hemisphere": hemisphere}
        )

nodes = pd.DataFrame(node_rows).set_index("node_id")
nodes
#%%

counts = nodes.groupby("class").size()
singleton_classes = counts[counts != 2].index

bads = nodes[nodes["class"].isin(singleton_classes)]

nodes = nodes[~nodes["class"].isin(singleton_classes)]

# %%
nodes = nodes.sort_values(["hemisphere", "class"])
left_nodes = nodes[nodes["hemisphere"] == "L"]
right_nodes = nodes[nodes["hemisphere"] == "R"]
assert (left_nodes["class"].values == right_nodes["class"].values).all()

#%%
ll_adj_df = adj_df.reindex(index=left_nodes.index, columns=left_nodes.index)
rr_adj_df = adj_df.reindex(index=right_nodes.index, columns=right_nodes.index)
lr_adj_df = adj_df.reindex(index=left_nodes.index, columns=right_nodes.index)
rl_adj_df = adj_df.reindex(index=right_nodes.index, columns=left_nodes.index)
adj_df = adj_df.reindex(index=nodes.index, columns=nodes.index)
#%%
from graspologic.plot import adjplot

adjplot(adj_df.values, plot_type="scattermap")

#%%
from pkg.match import GraphMatchSolver

n_side = len(left_nodes)
left_inds = np.arange(n_side)
right_inds = np.arange(n_side) + n_side

solver = GraphMatchSolver(adj_df.values, left_inds, right_inds)
solver.solve()
(solver.permutation_ == np.arange(n_side)).mean()

#%%

from pkg.match import BisectedGraphMatchSolver

solver = BisectedGraphMatchSolver(adj_df.values, left_inds, right_inds)
solver.solve()
(solver.permutation_ == np.arange(n_side)).mean()

#%%

rng = 

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
